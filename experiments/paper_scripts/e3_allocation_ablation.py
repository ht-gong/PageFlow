#!/usr/bin/env python3
"""E3 -- The allocation ablation: the free-list baseline, with a reserve and
without, against TQP-Vortex, at one scale factor and partition budget over
the 22 queries; every setting BITCOMP-compressed, searches off, double-buffered
(overlap: ``TQP_SERIAL_TICKS`` is never set), and the gather's index
decoded whole (``TQP_NO_DECODE_WINDOWS=1``: no window-by-window index decode;
the sort's payload windows stay on).

Three settings:

  freelist_reserve  --buffer_strategy freelist --freelist_reserve_gb 15
                    (``--reserve_gb``), 64 MiB input chunks
  freelist_full     --buffer_strategy freelist, no reserve (the whole arena
                    is the free list's), 512 MiB input chunks
  tqp               TQP-Vortex: --buffer_strategy lru over a
                    ``--slot_pool_gb`` cache region, 512 MiB input chunks

Queries may fail here, so a setting's time per query is the mean of its
first ``--need`` (default 2) successful runs: the suite runs once, then
again, and then only the queries still short of ``--need`` successes run,
in further passes, until each has them or has been tried ``--max_tries``
times (0: no cap).  Each pass is its own ``main.py`` process (its own
pinning and warmup), restarted on the rest of its list after a crash or
``--hang_min`` without output, as in E2.

Every other setting is E2's (``e2_end_to_end_ablation.py``): ``--SF``,
``--partition_budget_mb``, ``--cmp_pool_gb``, the scrubbed environment, the
NUMA binding and the GPU bus check, ``query_times.csv`` timing.  Results go
to ``--csv`` (default ``paper_scripts/results/allocation_ablation_results.csv``),
one row per (setting, query), rewritten after every pass; a setting already
there with every query settled is skipped (``--overwrite`` reruns it).  Logs
and perf reports go to ``--work/<setting>/try<k>/``.

    python3 experiments/paper_scripts/e3_allocation_ablation.py --show
    python3 experiments/paper_scripts/e3_allocation_ablation.py \\
        [--gpu 1] [--settings tqp] [--overwrite] [--wait_idle]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "helpers"))
sys.path.insert(0, HERE)
import common  # noqa: E402
import perf_json  # noqa: E402
import e2_end_to_end_ablation as e2  # noqa: E402

ROOT = common.ROOT

# Set for every E3 process (E2's environment otherwise): the gather's index
# is decoded whole, not window by window.
E3_ENV = {"TQP_NO_DECODE_WINDOWS": "1"}


def setting_env(args) -> dict[str, str]:
    return {**e2.setting_env(args), **E3_ENV}


@dataclass(frozen=True)
class Setting:
    key: str
    label: str
    chunk_size_mb: int
    flags: tuple[str, ...]


def settings(reserve_gb: float) -> list[Setting]:
    codec = ("--compression", "BITCOMP", "--compression_search", "off",
           "--sizing_search", "off")
    return [
        Setting("freelist_reserve", f"Free list, {reserve_gb:g} GiB reserve",
                64, ("--buffer_strategy", "freelist",
                     "--freelist_reserve_gb", f"{reserve_gb:g}", *codec)),
        Setting("freelist_full", "Free list, whole arena", 512,
                ("--buffer_strategy", "freelist", *codec)),
        Setting("tqp", "TQP-Vortex", 512, ("--buffer_strategy", "lru", *codec)),
    ]


KEYS = [s.key for s in settings(0)]

FIELDS = ["setting", "label", "sf", "chunk_size_mb", "partition_budget_mb",
          "query", "status", "n_ok", "need", "tries", "try_seconds", "mean_s",
          "error", "restarts", "flags", "commit", "gpu", "finished_utc"]


def setting_cmd(s: Setting, args, qids, perf_dir: str) -> list[str]:
    cmd = [args.python, os.path.join(ROOT, "main.py"),
           "--SF", str(args.sf), "--q", *map(str, qids),
           "--chunk_size_mb", str(s.chunk_size_mb),
           "--partition_budget_mb", str(args.partition_budget_mb),
           "--cmp_pool_gb", str(args.cmp_pool_gb), "--no_debug", *s.flags]
    if s.key == "tqp":
        cmd += ["--slot_pool_gb", f"{args.slot_pool_gb:g}"]
    return cmd + ["--perf_report", perf_dir]


def run_pass(s: Setting, args, qids: list[int], perf_dir: str):
    """``qids`` in one main.py process, a new process for the rest after a
    crash or a hang.  Returns ([(qid, status, seconds, detail)], restarts)."""
    log_path = os.path.join(perf_dir, "main.log")
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
    want_bus = e2.gpu_bus(args.gpu)
    outcomes: list[tuple[int, str, float | None, str]] = []
    restarts = 0
    while len(outcomes) < len(qids):
        todo = qids[len(outcomes):]
        cmd = e2.numa_prefix(args.gpu) + setting_cmd(s, args, todo, perf_dir)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log,
                                    stderr=subprocess.STDOUT, text=True,
                                    env=setting_env(args),
                                    preexec_fn=e2._stack_64m,
                                    start_new_session=True)
        limit = args.hang_min * 60 if args.hang_min > 0 else float("inf")
        size, deadline = 0, time.time() + limit
        why, bus_ok = None, False
        while proc.poll() is None:
            time.sleep(5)
            if not bus_ok and (bus := e2.bus_of_pid(proc.pid)):
                if bus != want_bus:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    sys.exit(f"{s.key}: main.py is on GPU {bus}, not "
                             f"--gpu {args.gpu} ({want_bus}); stopped")
                bus_ok = True
            n = os.path.getsize(log_path)
            if n > size:
                size, deadline = n, time.time() + limit
            if time.time() > deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                why = f"no progress for {args.hang_min:g} min (killed)"
        outcomes += e2._results_rows(perf_dir)[:len(todo)]
        if os.path.exists(qr):
            with open(qr, encoding="utf-8") as src, \
                    open(qr_all, "a", encoding="utf-8") as dst:
                dst.writelines(line for line in src if line[:1].isdigit())
            os.remove(qr)
        if len(outcomes) < len(qids):
            # The query in flight when the process ended is a FAIL; the
            # rest go to a new process.
            why = why or f"process exited {proc.returncode}"
            q = qids[len(outcomes)]
            outcomes.append((q, "FAIL", None, why))
            restarts += 1
            print(f"    {s.key}: Q{q} {why}; restarting with the rest",
                  flush=True)
    res = os.path.join(ROOT, "results.out")
    if os.path.exists(res):
        # A move, not os.replace: in the experiments container ROOT is the
        # image's and perf_dir a mounted out/, another filesystem.
        shutil.move(res, os.path.join(perf_dir, "results_all.out"))
    return outcomes, restarts


def summarize(s: Setting, args, outcomes, restarts, commit: str) -> list[dict]:
    rows = []
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%F %T")
    for q in args.q:
        mine = [o for o in outcomes if o[0] == q]
        ok = [o[2] for o in mine if o[1] == "OK" and o[2] is not None][:args.need]
        errors = sorted({o[3][:160] for o in mine if o[1] != "OK"})
        status = ("OK" if len(ok) == args.need
                  else "PARTIAL" if ok else "FAIL")
        rows.append({
            "setting": s.key, "label": s.label, "sf": args.sf,
            "chunk_size_mb": s.chunk_size_mb,
            "partition_budget_mb": args.partition_budget_mb,
            "query": q, "status": status, "n_ok": len(ok), "need": args.need,
            "tries": len(mine),
            "try_seconds": ";".join(
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


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = {k: i for i, k in enumerate(KEYS)}
    rows = sorted(rows, key=lambda r: (int(r["sf"]),
                                       int(r["partition_budget_mb"]),
                                       order.get(r["setting"], 99),
                                       int(r["query"])))
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def run_setting(s: Setting, args, commit: str) -> list[dict]:
    """Passes over the queries short of ``--need`` successes (and under
    ``--max_tries``) until none is left; the CSV is rewritten after each."""
    outcomes, restarts, k = [], 0, 0
    rows: list[dict] = []
    while True:
        tries = {q: sum(o[0] == q for o in outcomes) for q in args.q}
        oks = {q: sum(o[0] == q and o[1] == "OK" for o in outcomes)
               for q in args.q}
        todo = [q for q in args.q if oks[q] < args.need
                and (args.max_tries <= 0 or tries[q] < args.max_tries)]
        if not todo:
            return rows
        k += 1
        print(f"[{time.strftime('%F %T')}] {s.key} try {k}: "
              f"{' '.join(f'Q{q}' for q in todo)}", flush=True)
        o, r = run_pass(s, args, todo,
                        os.path.join(args.work, s.key, f"try{k}"))
        outcomes += o
        restarts += r
        print(f"[{time.strftime('%F %T')}] {s.key} try {k}: "
              + " ".join(f"Q{q}:{st}" for q, st, _, _ in o), flush=True)
        rows = summarize(s, args, outcomes, restarts, commit)
        mine = _key(rows[0])
        write_csv(args.csv, [r for r in e2.read_csv(args.csv)
                             if _key(r) != mine] + rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", nargs="+", choices=KEYS, default=KEYS)
    ap.add_argument("--overwrite", action="store_true",
                    help="rerun the settings that are already settled in the "
                         "CSV and replace their rows")
    ap.add_argument("--sf", type=int, default=1000)
    ap.add_argument("--q", type=int, nargs="+", default=common.QUERIES)
    ap.add_argument("--need", type=int, default=2,
                    help="successful runs per query to average")
    ap.add_argument("--max_tries", type=int, default=5,
                    help="runs per query at most, failed ones included "
                         "(0: no cap)")
    ap.add_argument("--reserve_gb", type=float, default=15.0,
                    help="freelist_reserve's --freelist_reserve_gb")
    ap.add_argument("--partition_budget_mb", type=int, default=512)
    ap.add_argument("--cmp_pool_gb", type=int, default=29)
    ap.add_argument("--slot_pool_gb", type=float, default=20.0,
                    help="the tqp setting's LRU cache region")
    ap.add_argument("--gpu", default=common.default_gpu())
    ap.add_argument("--data", default=None,
                    help="data root holding sf<SF>/parquet (default "
                         "TQP_TABLES_PATH, else TQP_DATA_DIR / DATA_DIR)")
    ap.add_argument("--python", default="/opt/envs/tqp/bin/python"
                    if os.path.exists("/opt/envs/tqp/bin/python")
                    else sys.executable)
    ap.add_argument("--csv", default=os.path.join(
        HERE, "results", "allocation_ablation_results.csv"))
    ap.add_argument("--work", default=os.path.join(
        ROOT, "out", "e3_allocation_ablation"))
    ap.add_argument("--hang_min", type=float, default=10.0,
                    help="kill a process whose log has had no output for "
                         "this long, record its query FAIL and go on in a "
                         "new process; 0: no limit")
    ap.add_argument("--wait_idle", action="store_true",
                    help="wait for the GPU to be idle instead of stopping")
    ap.add_argument("--show", "--dry-run", action="store_true", dest="show",
                    help="print each setting's command and environment; "
                         "run nothing")
    args = ap.parse_args()
    args.data = (args.data or os.environ.get("TQP_TABLES_PATH")
                 or common.data_dir())
    args.csv = os.path.abspath(args.csv)
    by_key = {s.key: s for s in settings(args.reserve_gb)}

    existing = e2.read_csv(args.csv)

    def settled(s: Setting) -> bool:
        mine = [r for r in existing if _key(r) == _key({
            "setting": s.key, "sf": args.sf, "chunk_size_mb": s.chunk_size_mb,
            "partition_budget_mb": args.partition_budget_mb})]
        return len(mine) == len(args.q) and all(
            r["status"] == "OK" or (args.max_tries > 0
                                    and int(r["tries"]) >= args.max_tries)
            for r in mine)

    todo = [by_key[k] for k in args.settings
            if args.overwrite or not settled(by_key[k])]
    skipped = [k for k in args.settings if by_key[k] not in todo]
    if skipped:
        print(f"In {args.csv} already (--overwrite to rerun): "
              f"{' '.join(skipped)}")
    if args.show:
        env = setting_env(args)
        print("Environment of every setting (beside the ambient non-TQP_ one):")
        for k in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES",
                  "TQP_TABLES_PATH", "TQP_NUMA_BIND", "OMP_NUM_THREADS",
                  *e2.KEEP_TQP, *E3_ENV):
            if k in env:
                print(f"  {k}={env[k]}")
        print(f"  stack 64 MiB; CPUs: "
              f"{' '.join(e2.numa_prefix(args.gpu)) or 'unbound'}")
        print(f"  mean of {args.need} OK runs per query, at most "
              f"{args.max_tries or 'unlimited'} tries")
        for s in by_key.values():
            mark = "run" if s in todo else (
                "skip (in the CSV)" if s.key in args.settings
                else "not selected")
            print(f"\n{s.key} [{mark}]  {s.label}")
            print("  " + " ".join(setting_cmd(
                s, args, ["<queries>"], f"<work>/{s.key}/try<k>")[1:]))
        return 0
    if not todo:
        return 0
    if not os.path.isdir(os.path.join(args.data, f"sf{args.sf}", "parquet")):
        sys.exit(f"no sf{args.sf}/parquet under {args.data}")

    commit = common.source_commit()
    for s in todo:
        while busy := e2.gpu_busy(args.gpu):
            if not args.wait_idle:
                sys.exit(f"GPU {args.gpu} is busy ({busy}); --wait_idle waits")
            time.sleep(60)
        print(f"[{time.strftime('%F %T')}] {s.key}: {s.label}", flush=True)
        rows = run_setting(s, args, commit)
        ok = sum(r["status"] == "OK" for r in rows)
        print(f"[{time.strftime('%F %T')}] {s.key}: {ok}/{len(rows)} queries "
              f"with {args.need} OK runs", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
