#!/usr/bin/env python3
"""E2 -- The end-to-end ablation: UVM, then TQP-Vortex with each mechanism
added, at one scale factor, partition budget and chunk size, over the 22
queries.

Four settings, each run ``--reps`` times; each rep is its own ``main.py``
process over the suite (its own pinning and warmup), and the mean over the
reps is reported:

  uvm           --uvm: every column and intermediate in CUDA managed memory,
                the driver migrating pages (no pool, cache or compression)
  tqp_raw       TQP-Vortex, no compression: --compression RAW (scans and
                intermediates), searches off, double-buffered (overlap)
  tqp_ans       --compression ANS, searches off
  tqp_adaptive  --compression adaptive --compression_search read-and-write
                --sizing_search mid_stream

Every setting shares ``--SF``, ``--chunk_size_mb``, ``--partition_budget_mb``
and ``--cmp_pool_gb`` (under --uvm, the GPU budget); the TQP settings use
the LRU cache over a ``--slot_pool_gb`` region.  Every other engine setting
is its default: the ambient ``TQP_*`` environment is dropped, so a setting
runs only what its flags say (``--show`` prints each setting's command and
environment).  Timings are ``query_times.csv`` rows (AGENTS.md, TIMING).

Results go to ``--csv`` (default ``paper_scripts/results/ablation_results.csv``),
one row per (setting, query).  A setting already in the CSV at the same SF,
chunk size and partition budget is skipped; ``--overwrite`` reruns it and
replaces its rows.  Each setting's logs and perf reports go to
``--work/<setting>/``.

Runs natively (no Docker on the box), like ``bench/run_tqp.sh`` under
``TQP_NATIVE=1``: ``--python`` (default ``/opt/envs/tqp/bin/python``, with
this checkout's modules built by ``bench/native_env.sh modules``), the
engine's CUDA device 0 = ``--gpu``, its CPUs bound to the GPU's NUMA node,
``TQP_NUMA_BIND=0``, ``OMP_NUM_THREADS=32`` and a 64 MiB stack.  A process
that dies (or makes no progress for ``--hang_min``) records its occurrence
FAIL and the rest of the rep's list is run again in a new process.  The
engine's CUDA order is the PCI bus order, and each process is checked to be
on ``--gpu``'s bus (the run stops otherwise).

    python3 experiments/paper_scripts/e2_end_to_end_ablation.py --show
    python3 experiments/paper_scripts/e2_end_to_end_ablation.py \\
        [--gpu 1] [--settings tqp_raw tqp_ans] [--overwrite] [--wait_idle]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "helpers"))
import common  # noqa: E402
import perf_json  # noqa: E402

ROOT = common.ROOT


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    flags: tuple[str, ...]


def _tqp(compression: str, compression_search: str = "off",
         sizing_search: str = "off") -> tuple[str, ...]:
    return ("--buffer_strategy", "lru", "--compression", compression,
            "--compression_search", compression_search,
            "--sizing_search", sizing_search)


SETTINGS: list[Setting] = [
    Setting("uvm", "UVM", ("--uvm",)),
    Setting("tqp_raw", "TQP, no compression", _tqp("RAW")),
    Setting("tqp_ans", "TQP, ANS", _tqp("ANS")),
    Setting("tqp_adaptive", "TQP, adaptive + searches",
            _tqp("adaptive", "read-and-write", "mid_stream")),
    # Diagnostic splits of tqp_adaptive (not in the default --settings).
    Setting("tqp_adaptive_static", "TQP, adaptive, searches off",
            _tqp("adaptive")),
    Setting("tqp_adaptive_cs", "TQP, adaptive + compression search",
            _tqp("adaptive", "read-and-write")),
    Setting("tqp_adaptive_sz", "TQP, adaptive + sizing search",
            _tqp("adaptive", "off", "mid_stream")),
]
DEFAULT_SETTINGS = ["uvm", "tqp_raw", "tqp_ans", "tqp_adaptive"]
BY_KEY = {s.key: s for s in SETTINGS}

# What a setting's process gets of the ambient environment: never a TQP_*
# variable (each changes the engine; the defaults are the experiment) or a
# CUDA device choice, except those set below.
KEEP_TQP = ("TQP_QFAIL_TRACE",)

FIELDS = ["setting", "label", "sf", "chunk_size_mb", "partition_budget_mb",
          "query", "status", "n_ok", "reps", "rep_seconds", "mean_s",
          "error", "restarts", "flags", "commit", "gpu", "finished_utc"]


def setting_cmd(s: Setting, args, qids: list[int], perf_dir: str) -> list[str]:
    cmd = [args.python, os.path.join(ROOT, "main.py"),
           "--SF", str(args.sf), "--q", *map(str, qids),
           "--chunk_size_mb", str(args.chunk_size_mb),
           "--partition_budget_mb", str(args.partition_budget_mb),
           "--cmp_pool_gb", str(args.cmp_pool_gb), "--no_debug", *s.flags]
    if s.key != "uvm":
        cmd += ["--slot_pool_gb", f"{args.slot_pool_gb:g}"]
        cmd += args.engine_flags.split()
    return cmd + ["--perf_report", perf_dir]


def setting_env(args) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("TQP_") and k not in KEEP_TQP)
           and k not in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")}
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=str(args.gpu), TQP_TABLES_PATH=args.data,
               TQP_NUMA_BIND="0", OMP_NUM_THREADS="32",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    return env


def numa_prefix(gpu: str) -> list[str]:
    cpus = common.node_cpus(gpu)
    return ["taskset", "-c", cpus] if cpus and shutil.which("taskset") else []


def gpu_busy(gpu: str) -> str:
    """The compute processes on ``gpu``, empty when it is idle."""
    uuid = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return "; ".join(line for line in apps.splitlines() if uuid and uuid in line)


def _stack_64m():
    resource.setrlimit(resource.RLIMIT_STACK, (64 << 20, 64 << 20))


def _results_rows(perf_dir: str) -> list[tuple[int, str, float | None, str]]:
    """``query_results.csv``'s rows: one per occurrence of the real run, in
    order, (qid, status, seconds, rows or error)."""
    path = os.path.join(perf_dir, "query_results.csv")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            head, _, rest = line.rstrip("\n").partition(",")
            status, _, rest = rest.partition(",")
            secs, _, rest = rest.partition(",")
            detail = rest.rsplit(",", 1)[0]
            if head.isdigit():
                rows.append((int(head), status,
                             float(secs) if secs else None, detail))
    return rows


def gpu_bus(gpu: str) -> str:
    return subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=pci.bus_id",
         "--format=csv,noheader"], capture_output=True,
        text=True).stdout.strip().lower()


def bus_of_pid(pid: int) -> str | None:
    """The PCI bus of the GPU ``pid`` holds a context on, None until it has
    one."""
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in apps.splitlines():
        p, _, bus = (x.strip() for x in line.partition(","))
        if p == str(pid):
            return bus.lower()
    return None


def run_rep(s: Setting, args, work: str, rep: int):
    """Rep ``rep`` of the setting: the queries in one main.py process, a
    new process for the rest after a crash or a hang (no log output for
    ``--hang_min``).  Returns ([(qid, status, seconds, detail)], restarts)."""
    perf_dir = os.path.join(work, s.key, f"rep{rep}")
    log_path = os.path.join(work, s.key, f"main_rep{rep}.log")
    os.makedirs(perf_dir, exist_ok=True)
    perf_json.clear_run_outputs(perf_dir)
    # main.py rewrites query_results.csv when it starts, so each process's
    # rows are moved to query_results_all.csv before the next one starts.
    qr = os.path.join(perf_dir, "query_results.csv")
    qr_all = os.path.join(perf_dir, "query_results_all.csv")
    for path in (qr, qr_all):
        if os.path.exists(path):
            os.remove(path)
    open(log_path, "w").close()
    want_bus = gpu_bus(args.gpu)
    outcomes: list[tuple[int, str, float | None, str]] = []
    restarts = 0
    while len(outcomes) < len(args.q):
        todo = args.q[len(outcomes):]
        cmd = numa_prefix(args.gpu) + setting_cmd(s, args, todo, perf_dir)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, text=True,
                                    env=setting_env(args),
                                    preexec_fn=_stack_64m,
                                    start_new_session=True)
        limit = args.hang_min * 60 if args.hang_min > 0 else float("inf")
        size, deadline = 0, time.time() + limit
        why, bus_ok = None, False
        while proc.poll() is None:
            time.sleep(5)
            if not bus_ok and (bus := bus_of_pid(proc.pid)):
                if bus != want_bus:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    sys.exit(f"{s.key}: main.py is on GPU {bus}, not "
                             f"--gpu {args.gpu} ({want_bus}); stopped")
                bus_ok = True
                print(f"    {s.key} rep {rep}: on GPU {bus} "
                      f"(CPUs {' '.join(numa_prefix(args.gpu)[2:]) or 'unbound'})",
                      flush=True)
            n = os.path.getsize(log_path)
            if n > size:
                size, deadline = n, time.time() + limit
            if time.time() > deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                why = f"no progress for {args.hang_min:g} min (killed)"
        rows = _results_rows(perf_dir)
        outcomes += rows[:len(todo)]
        if os.path.exists(qr):
            with open(qr, encoding="utf-8") as src, \
                    open(qr_all, "a", encoding="utf-8") as dst:
                dst.writelines(line for line in src
                               if line[:1].isdigit())
            os.remove(qr)
        print(f"    {s.key} rep {rep}: {len(outcomes)}/{len(args.q)} queries, "
              f"exit {proc.returncode}", flush=True)
        if len(outcomes) < len(args.q):
            # The query in flight when the process ended is a FAIL; the
            # rest go to a new process.
            why = why or f"process exited {proc.returncode}"
            q = args.q[len(outcomes)]
            outcomes.append((q, "FAIL", None, why))
            restarts += 1
            print(f"    {s.key} rep {rep}: Q{q} {why}; restarting with the "
                  "rest", flush=True)
    res = os.path.join(ROOT, "results.out")
    if os.path.exists(res):
        os.replace(res, os.path.join(perf_dir, "results_all.out"))
    return outcomes, restarts


def run_setting(s: Setting, args, work: str):
    """Every rep of the setting, each in its own process(es)."""
    outcomes, restarts = [], 0
    for rep in range(1, args.reps + 1):
        o, r = run_rep(s, args, work, rep)
        outcomes += o
        restarts += r
    return outcomes, restarts


def summarize(s: Setting, args, outcomes, restarts, commit: str) -> list[dict]:
    rows = []
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%F %T")
    for q in args.q:
        mine = [o for o in outcomes if o[0] == q]
        ok = [o[2] for o in mine if o[1] == "OK" and o[2] is not None]
        errors = sorted({o[3][:160] for o in mine if o[1] != "OK"})
        status = ("OK" if len(ok) == args.reps
                  else "PARTIAL" if ok else "FAIL")
        rows.append({
            "setting": s.key, "label": s.label, "sf": args.sf,
            "chunk_size_mb": args.chunk_size_mb,
            "partition_budget_mb": args.partition_budget_mb,
            "query": q, "status": status, "n_ok": len(ok), "reps": args.reps,
            "rep_seconds": ";".join(
                f"{o[2]:.6f}" if o[1] == "OK" and o[2] is not None else "NA"
                for o in mine),
            "mean_s": f"{sum(ok) / len(ok):.6f}" if ok else "",
            "error": " | ".join(errors), "restarts": restarts,
            "flags": " ".join(setting_cmd(s, args, [], "<perf>")[2:]),
            "commit": commit, "gpu": args.gpu, "finished_utc": stamp,
        })
    return rows


def _key(row) -> tuple:
    return (row["setting"], str(row["sf"]), str(row["chunk_size_mb"]),
            str(row["partition_budget_mb"]))


def read_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = {s.key: i for i, s in enumerate(SETTINGS)}
    rows = sorted(rows, key=lambda r: (int(r["sf"]), int(r["chunk_size_mb"]),
                                       int(r["partition_budget_mb"]),
                                       order.get(r["setting"], 99),
                                       int(r["query"])))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def show(args, todo: list[Setting]) -> None:
    env = setting_env(args)
    print("Environment of every setting (beside the ambient non-TQP_ one):")
    for k in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "TQP_TABLES_PATH", "TQP_NUMA_BIND",
              "OMP_NUM_THREADS", *KEEP_TQP):
        if k in env:
            print(f"  {k}={env[k]}")
    print(f"  stack 64 MiB; CPUs: {' '.join(numa_prefix(args.gpu)) or 'unbound'}")
    for s in SETTINGS:
        mark = ("run" if s in todo else "skip (in the CSV)"
                if s.key in args.settings else "not selected")
        print(f"\n{s.key} [{mark}]  {s.label}")
        print(f"  {args.reps} processes of: " + " ".join(
            setting_cmd(s, args, ["<1..22>"],
                        f"<work>/{s.key}/rep<k>")[1:]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", nargs="+", choices=list(BY_KEY),
                    default=DEFAULT_SETTINGS)
    ap.add_argument("--overwrite", action="store_true",
                    help="rerun the settings that are already in the CSV and "
                         "replace their rows")
    ap.add_argument("--sf", type=int, default=1000)
    ap.add_argument("--q", type=int, nargs="+", default=common.QUERIES)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--chunk_size_mb", type=int, default=512)
    ap.add_argument("--partition_budget_mb", type=int, default=512)
    ap.add_argument("--cmp_pool_gb", type=int, default=29,
                    help="the arena; under --uvm the GPU budget (default 29: "
                         "30 leaves too little outside the arena on a 32 GB "
                         "card)")
    ap.add_argument("--slot_pool_gb", type=float, default=20.0,
                    help="the TQP settings' LRU cache region")
    ap.add_argument("--gpu", default=common.default_gpu())
    ap.add_argument("--data", default=None,
                    help="data root holding sf<SF>/parquet (default "
                         "TQP_TABLES_PATH, else TQP_DATA_DIR / DATA_DIR)")
    ap.add_argument("--python", default="/opt/envs/tqp/bin/python"
                    if os.path.exists("/opt/envs/tqp/bin/python")
                    else sys.executable)
    ap.add_argument("--csv", default=os.path.join(HERE, "results",
                                                  "ablation_results.csv"))
    ap.add_argument("--work", default=os.path.join(
        ROOT, "out", "e2_end_to_end_ablation"))
    ap.add_argument("--hang_min", type=float, default=10.0,
                    help="kill a process whose log has had no output for "
                         "this long (under --no_debug a query prints its "
                         "start and nothing until it ends), record its query "
                         "FAIL and go on in a new process; 0: no limit")
    ap.add_argument("--only_failed", action="store_true",
                    help="for a setting already in the CSV, run only its "
                         "queries that are not OK there (or missing), and "
                         "replace just their rows; the work goes to "
                         "<work>/retry/<setting>/"),
    ap.add_argument("--engine_flags", default="",
                    help="extra main.py flags for the TQP settings (a "
                         "diagnostic sweep, e.g. a --sizing_config); they "
                         "are not part of the CSV key")
    ap.add_argument("--wait_idle", action="store_true",
                    help="wait for the GPU to be idle instead of stopping")
    ap.add_argument("--show", "--dry-run", action="store_true", dest="show",
                    help="print each setting's command and environment and "
                         "which would run; run nothing")
    args = ap.parse_args()
    args.data = (args.data or os.environ.get("TQP_TABLES_PATH")
                 or common.data_dir())
    args.csv = os.path.abspath(args.csv)

    existing = read_csv(args.csv)
    have = {_key(r) for r in existing}

    def key_of(k):
        return _key({"setting": k, "sf": args.sf,
                     "chunk_size_mb": args.chunk_size_mb,
                     "partition_budget_mb": args.partition_budget_mb})

    # --only_failed: each setting's queries that are not OK in the CSV.
    queries = {k: list(args.q) for k in args.settings}
    if args.only_failed:
        for k in args.settings:
            ok = {int(r["query"]) for r in existing
                  if _key(r) == key_of(k) and r["status"] == "OK"}
            queries[k] = [q for q in args.q if q not in ok]
        todo = [BY_KEY[k] for k in args.settings if queries[k]]
        for k in args.settings:
            print(f"{k}: {'rerun ' + ' '.join(f'Q{q}' for q in queries[k]) if queries[k] else 'every query OK'}")
    else:
        todo = [BY_KEY[k] for k in args.settings
                if args.overwrite or key_of(k) not in have]
    skipped = [k for k in args.settings if BY_KEY[k] not in todo]
    if skipped and not args.only_failed:
        print(f"In {args.csv} already (--overwrite to rerun): "
              f"{' '.join(skipped)}")
    if args.show:
        show(args, todo)
        return 0
    if not todo:
        return 0
    if not os.path.isdir(os.path.join(args.data, f"sf{args.sf}", "parquet")):
        sys.exit(f"no sf{args.sf}/parquet under {args.data}")

    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short",
                             "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    if subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "HEAD", "--",
                       ".", ":!experiments"]).returncode:
        commit += "+dirty"
    for s in todo:
        while busy := gpu_busy(args.gpu):
            if not args.wait_idle:
                sys.exit(f"GPU {args.gpu} is busy ({busy}); --wait_idle waits")
            time.sleep(60)
        print(f"[{time.strftime('%F %T')}] {s.key}: {s.label}", flush=True)
        sargs = argparse.Namespace(**vars(args))
        sargs.q = queries[s.key]
        work = (os.path.join(args.work, "retry") if args.only_failed
                else args.work)
        outcomes, restarts = run_setting(s, sargs, work)
        rows = summarize(s, sargs, outcomes, restarts, commit)
        ok = sum(r["status"] == "OK" for r in rows)
        print(f"[{time.strftime('%F %T')}] {s.key}: {ok}/{len(rows)} queries "
              f"OK in every rep", flush=True)
        # Re-read, so a concurrent edit of other settings' rows survives;
        # replace the setting's rows (--only_failed: those queries' only).
        mine = _key(rows[0])
        redo = {str(q) for q in sargs.q}
        write_csv(args.csv, [r for r in read_csv(args.csv)
                             if _key(r) != mine or (args.only_failed
                                                    and r["query"] not in redo)]
                  + rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
