#!/usr/bin/env python3
"""Parquet-format profiling pass: per engine, the fastest format for each column.

For one scale factor and a candidate set of encodings x compressions, each
candidate format is one uniform variant dataset of the same rows, named
shuffle-<encoding>-<compression> under $DATA_DIR/fmt_sf<SF>/. No TPC-H queries
run. Instead every engine scans and decodes every column of every variant with
one column probe, `SELECT min(c), max(c) FROM <table>` (ablation/probe_*.py),
and the fastest eligible format per (engine, column) wins.

  format_profile.py [--sf 5] [--encodings "plain dict delta"]
                    [--compressions "snappy zstd lz4raw"] [--rounds 3]
                    [--engines "rapids polars_gpu duckdb_cpu sirius"]
                    [--tol 0.05] [--out DIR]                (make ablation)

Stages:
  1. variants   generate the missing ones with datagen/ndsgen (one seed, so
                every variant holds the same rows in the same order and only
                the format differs; each is published only when complete);
                existing ones are validated and reused.
  2. probes     per round, per variant (staged disk -> ramdisk once), per engine:
                one worker process probes every column in a shuffled order,
                engine caches off. A probe that fails, times out
                ($PROBE_TIMEOUT s, default 30 + SF/2) or leaves the engine's
                native path (FALLBACK) makes that format ineligible for that
                column on that engine. -> <out>/format_profile.csv
  3. decide     per engine and column, the format to use -> <out>/format_map.json;
                prints the results: seconds per engine x format, and the map.

Outputs (the per-round, per-worker probe logs stay in
<out>/format_profile_logs/, not a result):

  format_profile.csv  one row per (engine, scale_factor, table, column,
      encoding, compression): status (OK if OK in every round, else the
      non-OK status, e.g. TIMEOUT), seconds (median over rounds; empty unless
      OK) and bytes (the column's compressed size in that format). A run
      replaces the rows of the engines it ran at that SF.
  format_map.json  {"sf<SF>": {"scale_factor", "tolerance", "engines":
      {engine: {table: {column: {"encoding", "compression"} | null}}}}}.
      Eligible: status OK. Formats within TOL of the fastest count as tied
      and the one with the fewest bytes wins; null = no eligible format.

Env: DATA_DIR (datasets), SCRATCH (engine scratch goes to $SCRATCH/probe),
SHM (ramdisk root, default /dev/shm), GPU or CUDA_VISIBLE_DEVICES (the one GPU
every GPU engine uses), PROBE_TIMEOUT, GEN_PARALLEL / GEN_BATCH (dbgen chunking
for new variants, as run.sh), and the image's tool variables.
"""
import argparse
import csv
import glob
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABLES = ["lineitem", "orders", "partsupp", "part", "customer", "supplier", "nation", "region"]
ENCODINGS = {  # ndsgen flags per encoding
    "plain": "--dictionary false --writer-version v1",
    "dict": "--dictionary true --writer-version v1",
    "delta": "--dictionary false --writer-version v2",
}
COMPRESSIONS = ["snappy", "zstd", "lz4raw"]
ENGINES = ["rapids", "polars_gpu", "duckdb_cpu", "sirius"]
DEFAULT = ("dict", "snappy")   # the format the generator writes for the benchmark
GPU_ENGINES = {"rapids", "polars_gpu", "sirius"}

PROFILE_CSV = "format_profile.csv"
MAP_JSON = "format_map.json"
FIELDS = ["engine", "scale_factor", "table", "column", "encoding", "compression",
          "status", "seconds", "bytes"]


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def variant(enc, comp):
    return f"shuffle-{enc}-{comp}"


def data_dir():
    d = os.environ.get("DATA_DIR")
    if not d:
        sys.exit("DATA_DIR is not set")
    return d


# ------------------------------------------------------------------ variants
def ensure_variants(sf, variants):
    root = os.path.join(data_dir(), f"fmt_sf{sf}")
    par = int(os.environ.get("GEN_PARALLEL", max(20, 2 * sf)))
    batch = int(os.environ.get("GEN_BATCH", 25))
    for v in variants:
        pqdir = os.path.join(root, v, "parquet")
        if os.path.isdir(pqdir):
            log(f"variant {v}: exists")
        else:
            enc, comp = v.split("-")[1:]
            log(f"variant {v}: generating (PARALLEL={par} BATCH={batch})")
            ndsgen = os.environ.get("NDSGEN") or sys.exit("NDSGEN is not set (run through make / docker/run.sh)")
            os.makedirs(root, exist_ok=True)
            with open(os.path.join(root, f"{v}.ndsgen.log"), "w") as gen_log:
                subprocess.run([ndsgen, "--sf", str(sf), "--parallel", str(par), "--batch", str(batch),
                                "--out", os.path.join(root, v), "--compression", comp,
                                *ENCODINGS[enc].split()], check=True, stderr=gen_log)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "results", "validate_dataset.py"),
                            pqdir, str(sf)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"variant {v} failed validation:\n{r.stdout}{r.stderr}")
    return {v: os.path.join(root, v, "parquet") for v in variants}


# ------------------------------------------------------------------ sizes
def _table_sizes(job):
    v, pqdir, t = job
    agg = {}
    for path in glob.glob(f"{pqdir}/{t}/*.parquet"):
        md = pq.ParquetFile(path).metadata
        for rg in range(md.num_row_groups):
            r = md.row_group(rg)
            for c in range(r.num_columns):
                cc = r.column(c)
                agg[cc.path_in_schema] = agg.get(cc.path_in_schema, 0) + cc.total_compressed_size
    return {(v, t, c): b for c, b in agg.items()}


def column_sizes(paths):
    """{(variant, table, column): compressed bytes}, from the parquet footers."""
    jobs = [(v, p, t) for v, p in paths.items() for t in TABLES]
    sizes = {}
    with ProcessPoolExecutor(min(32, len(jobs))) as ex:
        for part in ex.map(_table_sizes, jobs):
            sizes.update(part)
    return sizes


def write_profile(sf, engines, runs, sizes, out):
    """Rewrite format_profile.csv: the rows of `engines` at `sf` become this
    run's, aggregated over the rounds probed so far; all other rows stay."""
    path = os.path.join(out, PROFILE_CSV)
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != FIELDS:
                sys.exit(f"{path}: columns {rd.fieldnames} != {FIELDS}; not merging into it")
            rows = [r for r in rd if not (r["engine"] in engines and int(r["scale_factor"]) == sf)]
    for (e, v, t, c), rs in runs.items():
        enc, comp = v.split("-")[1:]
        bad = [st for st, _ in rs if st != "OK"]
        secs = "" if bad else f"{statistics.median(s for _, s in rs):.4f}"
        rows.append(dict(engine=e, scale_factor=sf, table=t, column=c, encoding=enc,
                         compression=comp, status=bad[0] if bad else "OK", seconds=secs,
                         bytes=sizes[(v, t, c)]))
    rows.sort(key=lambda r: (ENGINES.index(r["engine"]), int(r["scale_factor"]),
                             TABLES.index(r["table"]), r["column"],
                             list(ENCODINGS).index(r["encoding"]), COMPRESSIONS.index(r["compression"])))
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ probes
def columns(pqdir):
    cols = []
    for t in TABLES:
        schema = pq.read_schema(sorted(glob.glob(f"{pqdir}/{t}/*.parquet"))[0])
        cols += [(t, c) for c in schema.names if c != "ignore"]
    return cols


def stage(src, shm, sf):
    ram = os.path.join(shm, f"tpch_fmt_sf{sf}")
    shutil.rmtree(ram, ignore_errors=True)
    os.makedirs(ram)
    dst = os.path.join(ram, "parquet")
    subprocess.run(["cp", "-r", src, dst], check=True)
    # Sirius' GPU reader rejects 0-row part-files; drop them for every engine (as run.sh).
    for f in glob.glob(f"{dst}/*/*.parquet"):
        if pq.ParquetFile(f).metadata.num_rows == 0:
            os.remove(f)
    return ram, dst


def run_worker(engine, pqdir, cols, run_dir, timeout, env):
    """Probe cols with one worker; restart it after a timeout or crash.
    Returns {(table, column): (status, seconds, detail)}."""
    results, remaining, attempt, no_prime = {}, list(cols), 0, False
    while remaining:
        attempt += 1
        cols_file = os.path.join(run_dir, f"columns.{attempt}.txt")
        out_csv = os.path.join(run_dir, f"probes.{attempt}.csv")
        with open(cols_file, "w") as f:
            f.writelines(f"{t} {c}\n" for t, c in remaining)
        # cur = the probe in flight: ("probe", table, column) or ("prime", ...)
        state = {"cur": None, "since": time.time()}
        logf = open(os.path.join(run_dir, f"worker.{attempt}.log"), "w")
        wenv = dict(env, PROBE_NO_PRIME="1" if no_prime else "0")
        proc = subprocess.Popen([os.path.join(HERE, "probe.sh"), engine, pqdir, cols_file, out_csv],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=wenv, start_new_session=True)

        def read():
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                word = line.split(" ", 1)[0]
                if word in ("PROBE-START", "PRIME-START"):
                    t, c = line.split()[1].split(".", 1)
                    state["cur"] = ("probe" if word == "PROBE-START" else "prime", t, c)
                    state["since"] = time.time()
                elif word in ("PROBE-END", "PRIME-END", "PROBE-ABORT"):
                    state["cur"] = None
        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        timed_out = None
        try:
            while proc.poll() is None:
                time.sleep(0.5)
                # a probe in flight may take `timeout`; worker startup up to 15 min
                limit = timeout if state["cur"] else max(timeout, 900)
                if time.time() - state["since"] > limit:
                    timed_out = state["cur"] or ("startup",)
                    os.killpg(proc.pid, 9)
                    proc.wait()
        finally:
            if proc.poll() is None:   # the driver itself is being stopped
                os.killpg(proc.pid, 9)
        reader.join(10)
        logf.close()

        progress = 0
        if os.path.exists(out_csv):
            with open(out_csv) as f:
                for t, c, status, secs, detail in csv.reader(f):
                    results[(t, c)] = (status, float(secs) if secs else None, detail)
                    progress += 1
        if timed_out and timed_out[0] == "probe":
            results[timed_out[1:]] = ("TIMEOUT", float(timeout), f"exceeded {timeout}s")
            progress += 1
        elif timed_out and timed_out[0] == "prime" and not no_prime:
            log(f"      priming probe {timed_out[1]}.{timed_out[2]} timed out; restarting without priming")
            no_prime = True
            progress += 1       # not a result, but a change worth another attempt
        elif state["cur"] and state["cur"][0] == "probe":   # died inside this probe
            results[state["cur"][1:]] = ("FAIL", None, f"worker exited rc={proc.returncode} during probe")
            progress += 1
        remaining = [tc for tc in remaining if tc not in results]
        if remaining and progress == 0:
            why = (f"worker {timed_out[0]} timed out" if timed_out
                   else f"worker exited rc={proc.returncode} before probing")
            for tc in remaining:
                results[tc] = ("FAIL", None, why)
            remaining = []
        elif remaining:
            log(f"      worker stopped ({'timeout' if timed_out else f'rc={proc.returncode}'}); "
                f"restarting on {len(remaining)} columns")
    return results


def probes(sf, paths, sizes, engines, rounds, out, seed):
    shm = os.environ.get("SHM", "/dev/shm")
    # spark-rapids hangs outright on some delta+zstd reads; a probe normally
    # takes seconds, so the cap stays short and grows with the data.
    timeout = int(os.environ.get("PROBE_TIMEOUT", 30 + sf // 2))
    if not os.environ.get("SCRATCH"):
        sys.exit("SCRATCH is not set")
    scratch = os.path.join(os.environ["SCRATCH"], "probe")
    env = dict(os.environ, PROBE_SCRATCH=scratch)
    gpu = os.environ.get("GPU") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if GPU_ENGINES & set(engines) and not gpu:
        sys.exit("set GPU (or CUDA_VISIBLE_DEVICES) to the one free GPU the GPU engines may use")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    runs = {}   # (engine, variant, table, column) -> [(status, seconds)] per round
    for rnd in range(1, rounds + 1):
        for v, src in paths.items():
            t0 = time.time()
            ram, pqdir = stage(src, shm, sf)
            cols = columns(pqdir)
            log(f"round {rnd} / {v}: staged in {time.time() - t0:.0f}s, {len(cols)} columns")
            try:
                for e in engines:
                    run_dir = os.path.join(out, "format_profile_logs", f"sf{sf}", f"r{rnd}", v, e)
                    shutil.rmtree(run_dir, ignore_errors=True)
                    os.makedirs(run_dir)
                    order = list(cols)
                    random.Random(f"{seed}:{sf}:{v}:{rnd}:{e}").shuffle(order)
                    env["PROBE_LOG_DIR"] = run_dir
                    t0 = time.time()
                    res = run_worker(e, pqdir, order, run_dir, timeout, env)
                    for (t, c), (st, s, _) in res.items():
                        runs.setdefault((e, v, t, c), []).append((st, s))
                    write_profile(sf, engines, runs, sizes, out)
                    bad = {}
                    for st, _, _ in res.values():
                        if st != "OK":
                            bad[st] = bad.get(st, 0) + 1
                    log(f"    {e}: {len(res) - sum(bad.values())}/{len(res)} OK"
                        f"{' ' + str(bad) if bad else ''}, {time.time() - t0:.0f}s")
            finally:
                shutil.rmtree(ram, ignore_errors=True)


# ------------------------------------------------------------------ decide
def decide(sf, tol, out):
    with open(os.path.join(out, PROFILE_CSV)) as f:
        rows = [r for r in csv.DictReader(f) if int(r["scale_factor"]) == sf]
    engines = [e for e in ENGINES if any(r["engine"] == e for r in rows)]
    cols = sorted({(r["table"], r["column"]) for r in rows}, key=lambda tc: (TABLES.index(tc[0]), tc[1]))
    fmts = sorted({(r["encoding"], r["compression"]) for r in rows},
                  key=lambda f: (list(ENCODINGS).index(f[0]), COMPRESSIONS.index(f[1])))
    ok = {}   # (engine, table, column) -> {(encoding, compression): (seconds, bytes)}
    for r in rows:
        if r["status"] == "OK":
            ok.setdefault((r["engine"], r["table"], r["column"]), {})[
                (r["encoding"], r["compression"])] = (float(r["seconds"]), int(r["bytes"]))

    picks = {}   # (engine, table, column) -> (encoding, compression) | None
    for e in engines:
        for t, c in cols:
            elig = ok.get((e, t, c), {})
            if elig:
                best = min(s for s, _ in elig.values())
                tied = [f for f, (s, _) in elig.items() if s <= best * (1 + tol)]
                picks[(e, t, c)] = min(tied, key=lambda f: (elig[f][1], elig[f][0], f))
            else:
                picks[(e, t, c)] = None
    decisions = {}
    for (e, t, c), f in picks.items():
        decisions.setdefault(e, {}).setdefault(t, {})[c] = (
            None if f is None else {"encoding": f[0], "compression": f[1]})
    mpath = os.path.join(out, MAP_JSON)
    doc = {}
    if os.path.exists(mpath):
        with open(mpath) as f:
            doc = json.load(f)
    doc[f"sf{sf}"] = dict(scale_factor=sf, tolerance=tol, engines=decisions)
    with open(mpath + ".tmp", "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(mpath + ".tmp", mpath)

    # The results: per engine, the probe seconds summed over all columns for
    # each uniform format (n/a: not OK on every column), then the map's.
    log(f"results SF{sf} -> {out}/{PROFILE_CSV}, {mpath}")
    print(f"  {f'seconds summed over {len(cols)} columns':34s}" + "".join(f"{e:>12s}" for e in engines))
    for f in fmts:
        cells = []
        for e in engines:
            col_s = [ok.get((e, t, c), {}).get(f) for t, c in cols]
            cells.append(f"{sum(x[0] for x in col_s):12.2f}" if all(col_s) else f"{'n/a':>12s}")
        print(f"  {'-'.join(f):34s}" + "".join(cells))
    mine = {e: [picks[(e, t, c)] for t, c in cols] for e in engines}
    print(f"  {'format map':34s}" + "".join(
        f"{sum(ok[(e, t, c)][picks[(e, t, c)]][0] for t, c in cols if picks[(e, t, c)]):12.2f}"
        for e in engines))
    print(f"  {'map columns != ' + '-'.join(DEFAULT):34s}" + "".join(
        f"{sum(1 for f in mine[e] if f and f != DEFAULT):12d}" for e in engines))
    print(f"  {'map columns with no OK format':34s}" + "".join(
        f"{sum(1 for f in mine[e] if f is None):12d}" for e in engines), flush=True)


def main():
    # SIGTERM -> SystemExit, so a stopped pass still kills its worker and ramdisk copy
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sf", type=int, default=5)
    p.add_argument("--encodings", default="plain dict delta")
    p.add_argument("--compressions", default="snappy zstd lz4raw")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--engines", default=" ".join(ENGINES))
    p.add_argument("--tol", type=float, default=0.05)
    p.add_argument("--out", default=os.path.join(ROOT, "results"))
    p.add_argument("--seed", default="0", help="column-order shuffle seed")
    a = p.parse_args()
    encs, comps, engines = a.encodings.split(), a.compressions.split(), a.engines.split()
    bad = [x for x in encs if x not in ENCODINGS] + [x for x in comps if x not in COMPRESSIONS] + \
          [x for x in engines if x not in ENGINES]
    if bad:
        sys.exit(f"unknown encoding/compression/engine: {bad}")
    os.makedirs(a.out, exist_ok=True)
    variants = [variant(e, c) for e in encs for c in comps]
    log(f"format profile SF{a.sf}: variants {variants}, engines {engines}, rounds {a.rounds} -> {a.out}")
    paths = ensure_variants(a.sf, variants)
    probes(a.sf, paths, column_sizes(paths), engines, a.rounds, a.out, a.seed)
    decide(a.sf, a.tol, a.out)


if __name__ == "__main__":
    main()
