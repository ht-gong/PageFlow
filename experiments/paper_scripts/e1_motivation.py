#!/usr/bin/env python3
"""E1 -- Motivation: how the four baseline engines (Sirius, cudf-Polars,
Spark-RAPIDS, DuckDB) scale, from scale factors whose data is held in memory to
ones read from parquet, SF30 to SF1000; the rows fig_motivation_geomean draws
(``figures/make_motivation_geomean_fig.py``, SF30-700), and the baselines of
fig_baselines (SF1000) and fig_scaling (SF100, 400, 700, 1000).

The suite, per engine, through the baselines harness (``baselines/run.sh``,
``baselines/AGENTS.md``), one engine key per (engine, SF):

    polars_gpu, rapids, duckdb_cpu   <engine>_resident below SF100, <engine> from it
    sirius                           sirius_host at every SF: the second Sirius
                                     build (SIRIUS_HOST_REF), the query's columns
                                     pinned in its host tier before the timed run

ONE ENGINE, READY DATA.  The driver runs ``--engine`` alone, as the harness
requires, on ``$DATA_DIR/<engine>/sf<SF>`` and its ``sf1`` (the warm pass's):
datasets made from the engine's format map (``baselines/AGENTS.md``, "Before
every benchmark"; ``make -C baselines engine-data CMD="make <engine> <SF>"``),
Sirius's for ``sirius_host``.  It never generates data: a scale factor whose
dataset is missing, or was not made from this engine's map (its
``NDSGEN.json``), is not run, and the driver exits 3.  Without ``--engine`` it
prints the whole suite's plan -- every engine and scale factor with runs left,
and whether its data is ready -- and runs nothing.

First the driver reads ``baselines/results/all_results.csv``, the file the
runs merge into.  A query with no row is run.  A query that has one is rerun
only with permission to overwrite it: asked once for all of them, or given as
``--overwrite``; ``failed`` reruns only the rows whose status is not ``OK``.
With no terminal to ask on and no ``--overwrite``, existing rows are kept.

REPS, EACH ITS OWN PROCESS.  Each query's timed run is repeated ``--reps``
times (default 2).  Every rep is a process of its own, as every run of the
harness is: it starts the engine, runs the untimed SF1 warm pass and then the
one timed run, and exits.  The reps are interleaved -- the whole query list,
then the whole list again -- in one ``run.sh`` call, which stages the dataset
once.  ``run.sh`` keeps the rows unmerged (``RUN_CSV_DIR``); they stay in
``--out`` (``reps_<key>_sf<SF>.csv``), and ``all_results.csv`` gets one row per
query through ``baselines/merge_results.py``: OK with the reps' mean seconds
when every rep is OK with one row count; else the first failing rep's status,
seconds and error (prefixed ``rep i/N``), or FAIL when a rep left no row or the
reps' row counts differ.

SETTINGS.  Every query's process is killed after ``--timeout`` seconds
(default 3600) and recorded ``TIMEOUT`` (``QUERY_TIMEOUT``, ``RAPIDS_TIMEOUT``
and ``SIRIUS_TIMEOUT``).  ``polars_gpu_resident`` runs over a prefetching
managed-memory pool (``POLARS_RESIDENT_MR=managed_pool``), so a query whose
working set exceeds the GPU pages to host memory instead of failing.  The
figures' rows took ``--reps 2``, then ``--overwrite failed`` reruns of the
cells with failures, up to four more times each.  Spark-RAPIDS writes its
shuffle under ``$SCRATCH``: its SF1000 Q9 needs about 500 GB free there, or it
is killed ``KILLED_DISK``.

``run.sh`` runs through a launcher: ``baselines/docker/run.sh`` (the image)
by default, or a native stand-in on a box without Docker, given as
``--launcher`` or ``BASELINES_LAUNCHER`` (environment or ``local.env``).

Exit: 0 when every requested row is there; 1 when a run left one missing or
``run.sh`` failed; 3 when a scale factor was skipped for its data.

    python3 experiments/paper_scripts/e1_motivation.py            # the suite's plan
    python3 experiments/paper_scripts/e1_motivation.py --engine sirius \\
        [--sf 100] [--reps 2] [--timeout 3600] [--data-root DIR] [--dry-run] \\
        [--overwrite ask|all|failed|none] [--gpu 0] [--q 1 6] [--launcher CMD]
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "helpers"))
import common  # noqa: E402

BASELINES = os.path.join(common.ROOT, "baselines")
RESULTS_CSV = os.path.join(BASELINES, "results", "all_results.csv")
RUN_LOG = os.path.join(BASELINES, "results", "run.log")

SFS = [30, 50, 100, 200, 400, 700, 1000]     # the figures'
RESIDENT_BELOW = 100    # resident mode below this SF, parquet from it

# Never passed on: settings under which run.sh's rows would not reach
# all_results.csv, the file this driver reads to decide what is left to run,
# and the spill pass's, whose instrumentation a runner turns on whenever
# SPILL_STATS_DIR is set.  The driver sets the ones its own calls need.
DROPPED_ENV = ("RUN_CSV_DIR", "TPCH_MERGE", "SPILL", "SPILL_SMOKE",
               "SPILL_STATS_DIR")


def engine_key(engine: str, sf: int) -> str:
    if engine == "sirius":
        return "sirius_host"
    return f"{engine}_resident" if sf < RESIDENT_BELOW else engine


def qnum(query: str) -> int:
    return int("".join(c for c in query if c.isdigit()))


def read_results() -> dict:
    """``{(engine key, sf, query number): status}`` from all_results.csv."""
    have = {}
    try:
        with open(RESULTS_CSV, newline="") as f:
            for r in csv.DictReader(f):
                have[(r["engine"], int(r["scale_factor"]), qnum(r["query"]))] = r["status"]
    except FileNotFoundError:
        pass
    return have


def aggregate(rows: list[dict], todo: list[int], reps: int) -> list[list]:
    """One row per query of ``todo`` (merge_results.py's input columns) from
    its reps' rows, in run order: OK with the mean seconds when every rep is
    OK with one row count; else the first failing rep's status and seconds,
    its error prefixed ``rep i/N``; FAIL when a rep left no row (a runner's
    own no-row check sees an earlier rep's row) or the row counts differ."""
    by = collections.defaultdict(list)
    for r in rows:
        by[qnum(r["query"])].append(r)
    out = []
    for q in todo:
        rs = by.get(q, [])
        bad = next(((i, r) for i, r in enumerate(rs, 1) if r["status"] != "OK"), None)
        counts = sorted({r["result_rows_or_error"] for r in rs})
        if bad:
            i, r = bad
            row = [r["status"], r["seconds"],
                   f"rep {i}/{reps}: {r['result_rows_or_error']}"[:90]]
        elif len(rs) < reps:
            row = ["FAIL", "NA", f"{reps - len(rs)} of {reps} reps left no row"]
        elif len(counts) > 1:
            row = ["FAIL", "NA", "row counts differ across reps: " + "/".join(counts)]
        else:
            secs = statistics.mean(float(r["seconds"]) for r in rs)
            row = ["OK", f"{secs:.3f}", counts[0]]
        out.append([f"query{q}", *row])
    return out


def ask_overwrite(existing: dict) -> str:
    """The overwrite policy for the rows in ``existing``
    (``{(sf, key): {query: status}}``), asked on the terminal."""
    print(f"\n{sum(map(len, existing.values()))} of the requested rows are "
          f"already in {RESULTS_CSV}:")
    for (sf, key), rows in existing.items():
        st = collections.Counter(rows.values())
        print(f"  {key:<22} SF{sf:<5} {len(rows):>2} rows  "
              + " ".join(f"{s}:{n}" for s, n in sorted(st.items())))
    if not sys.stdin.isatty():
        print("no terminal to ask on: keeping them "
              "(--overwrite all|failed reruns them)")
        return "none"
    ans = input("Overwrite them? [a]ll / [f]ailed only (status not OK) / "
                "[N]one: ").strip().lower()
    return {"a": "all", "all": "all",
            "f": "failed", "failed": "failed"}.get(ans, "none")


def left(have: dict, key: str, sf: int, queries: list[int], policy: str) -> list[int]:
    """The queries of ``queries`` still to run for ``(key, sf)`` under ``policy``."""
    return [q for q in queries
            if (key, sf, q) not in have or policy == "all"
            or (policy == "failed" and have[(key, sf, q)] != "OK")]


def plan(engine: str, sfs: list[int], queries: list[int], policy: str,
         root: str) -> list:
    """``[(sf, key, queries to run, data problems)]`` for ``engine``: the
    scale factors with queries left under ``policy``."""
    have = read_results()
    calls = []
    for sf in sfs:
        key = engine_key(engine, sf)
        todo = left(have, key, sf, queries, policy)
        if todo:
            calls.append((sf, key, todo, common.dataset_problems(root, sf, engine)))
    return calls


def print_plan(engine: str, calls: list, reps: int) -> None:
    for sf, key, todo, problems in calls:
        print(f"  {engine:<11} SF{sf:<5} {key:<20} x{reps}: {' '.join(map(str, todo))}"
              "\n              "
              + ("data ready" if not problems else "NOT RUN: " + "; ".join(problems)))


def summary(engine: str, sfs: list[int], queries: list[int]) -> list[str]:
    """Per cell, the requested queries' rows in all_results.csv now."""
    have = read_results()
    lines = [f"  {'engine':<20} {'SF':<6} {'OK':>5} {'other':<28} missing"]
    for sf in sfs:
        key = engine_key(engine, sf)
        st = collections.Counter(have[(key, sf, q)] for q in queries
                                 if (key, sf, q) in have)
        miss = [q for q in queries if (key, sf, q) not in have]
        other = " ".join(f"{s}:{n}" for s, n in sorted(st.items()) if s != "OK")
        lines.append(f"  {key:<20} SF{sf:<4} {st['OK']:>2}/{len(queries):<2} "
                     f"{other:<28} {' '.join(map(str, miss)) or '-'}")
    return lines


def suite(args, sfs: list[int], queries: list[int]) -> int:
    """Without --engine: the whole suite's plan, every engine, nothing run."""
    policy = "none" if args.overwrite == "ask" else args.overwrite
    print(f"E1 suite: {len(common.BASELINE_ENGINES)} engines x SF"
          f"{'/'.join(map(str, sfs))} (resident below SF{RESIDENT_BELOW}; "
          f"Sirius host-pinned throughout); {args.reps} timed reps; "
          f"overwrite: {policy};  cells with runs left:")
    n = 0
    for engine in common.BASELINE_ENGINES:
        root = common.engine_root(engine)
        if not root:
            sys.exit("no data root: set DATA_DIR (environment or local.env)")
        calls = plan(engine, sfs, queries, policy, root)
        print_plan(engine, calls, args.reps)
        n += len(calls)
    print(f"{n} of {len(common.BASELINE_ENGINES) * len(sfs)} (engine, SF) "
          f"cells left; run each with --engine <engine> --sf <SF>")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", choices=common.BASELINE_ENGINES,
                    help="the engine to run; without it, print the whole "
                    "suite's plan and run nothing")
    ap.add_argument("--sf", type=int, nargs="+", default=SFS,
                    help=f"scale factors (default the figures': {SFS}); "
                    f"resident below SF{RESIDENT_BELOW}")
    ap.add_argument("--reps", type=int, default=2,
                    help="timed runs per query and cell, each its own "
                    "process (default 2); their mean is merged")
    ap.add_argument("--timeout", type=int, default=3600,
                    help="kill a query's process after this many seconds "
                    "(default 3600; 0: the runners' own limits)")
    ap.add_argument("--data-root",
                    help="the engine's datasets (default $DATA_DIR/<engine>)")
    ap.add_argument("--overwrite", choices=["ask", "all", "failed", "none"],
                    default="ask", help="rows already in all_results.csv "
                    "(default: ask)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, with each scale factor's data, and stop")
    ap.add_argument("--q", type=int, nargs="+", default=common.QUERIES)
    ap.add_argument("--gpu", default=common.box_setting("GPU") or "0",
                    help="default: GPU (environment or local.env), else 0")
    ap.add_argument("--launcher",
                    default=common.box_setting("BASELINES_LAUNCHER")
                    or os.path.join(BASELINES, "docker", "run.sh"),
                    help="runs run.sh in the baselines' environment "
                    "(default: BASELINES_LAUNCHER, else docker/run.sh)")
    ap.add_argument("--out", default=common.default_out("e1_motivation"),
                    help="relative to the repository root")
    args = ap.parse_args()
    queries = sorted(set(args.q))
    sfs = sorted(set(args.sf))
    if args.engine is None:
        if args.data_root:
            sys.exit("--data-root names one engine's datasets: give --engine too")
        return suite(args, sfs, queries)
    root = args.data_root or common.engine_root(args.engine)
    if not root:
        sys.exit("no data root: set DATA_DIR (environment or local.env) or "
                 "pass --data-root")

    have = read_results()
    existing = {}
    for sf in sfs:
        key = engine_key(args.engine, sf)
        rows = {q: have[(key, sf, q)] for q in queries if (key, sf, q) in have}
        if rows:
            existing[(sf, key)] = rows
    policy = args.overwrite
    if policy == "ask":
        # A dry run asks nothing: it plans as if every existing row is kept.
        policy = ask_overwrite(existing) if existing and not args.dry_run else "none"

    calls = plan(args.engine, sfs, queries, policy, root)
    print(f"\n{args.engine} on {root};  overwrite: {policy};  timeout "
          f"{args.timeout or 'runner default'} s;  {len(calls)} cell(s):")
    print_plan(args.engine, calls, args.reps)
    if args.dry_run or not calls:
        return 0

    launcher = shlex.split(args.launcher)
    out_dir = common.host_dir(args.out)
    env = {k: v for k, v in os.environ.items() if k not in DROPPED_ENV}
    env["GPU"] = str(args.gpu)
    # Both launchers take the data root from the environment (docker/run.sh
    # mounts it), and run.sh reads $DATA_DIR/sf<SF> and $DATA_DIR/sf1.
    env["DATA_DIR"] = root
    env["POLARS_RESIDENT_MR"] = "managed_pool"
    if args.timeout > 0:
        for v in ("QUERY_TIMEOUT", "RAPIDS_TIMEOUT", "SIRIUS_TIMEOUT"):
            env[v] = str(args.timeout)
    # The timed runs' rows stay unmerged (RUN_CSV_DIR), in a directory both
    # launchers can see (docker/run.sh shares /dev/shm), until aggregated.
    runs = os.path.join(common.box_setting("SHM") or "/dev/shm",
                        f"e1_motivation-{os.getpid()}")
    os.makedirs(runs)
    run_sh = os.path.join(BASELINES, "run.sh")
    failed_calls, skipped = 0, []
    try:
        for sf, key, todo, problems in calls:
            if problems:
                skipped.append(sf)
                continue
            # The reps interleaved: the whole list, then the whole list again.
            qs = " ".join(map(str, todo * args.reps))
            print(f"\n[{time.strftime('%H:%M:%S')}] SF{sf}: ENGINES={key} "
                  f"{args.reps} reps of QUERIES=\"{' '.join(map(str, todo))}\"",
                  flush=True)
            # Exits 1 unless every query is OK; the rows say how it went.
            rc = subprocess.call([*launcher, run_sh, str(sf)],
                                 env=dict(env, RUN_CSV_DIR=runs, ENGINES=key, QUERIES=qs))
            # run.sh truncates its log on every call.
            if os.path.exists(RUN_LOG):
                shutil.copy(RUN_LOG, os.path.join(out_dir, f"run_{key}_sf{sf}.log"))
            run_csv = os.path.join(runs, f"{key}_sf{sf}.csv")
            if not os.path.exists(run_csv):
                print(f"[{time.strftime('%H:%M:%S')}] exit {rc}, no rows: "
                      f"nothing merged", flush=True)
                failed_calls += 1
                continue
            reps_csv = os.path.join(out_dir, f"reps_{key}_sf{sf}.csv")
            shutil.copy(run_csv, reps_csv)
            with open(run_csv, newline="") as f:
                rows = aggregate(list(csv.DictReader(f)), todo, args.reps)
            mean_csv = os.path.join(out_dir, f"mean_{key}_sf{sf}.csv")
            with open(mean_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["query", "status", "seconds", "result_rows_or_error"])
                w.writerows(rows)
            # all_results.csv's one writer upserts the means.
            failed_calls += subprocess.call(
                [sys.executable, os.path.join(BASELINES, "merge_results.py"),
                 key, str(sf), mean_csv]) != 0
            ok = sum(r[1] == "OK" for r in rows)
            print(f"[{time.strftime('%H:%M:%S')}] {ok}/{len(rows)} OK over "
                  f"{args.reps} reps (raw: {reps_csv})", flush=True)
    finally:
        shutil.rmtree(runs, ignore_errors=True)

    lines = summary(args.engine, sfs, queries)
    if skipped:
        lines.append(f"  not run, data not ready: SF{', SF'.join(map(str, skipped))}")
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{RESULTS_CSV} now (logs in {out_dir}):")
    print("\n".join(lines))
    have = read_results()
    ran = [sf for sf in sfs if sf not in skipped]
    missing = sum((engine_key(args.engine, sf), sf, q) not in have
                  for sf in ran for q in queries)
    if missing or failed_calls:
        return 1
    return 3 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
