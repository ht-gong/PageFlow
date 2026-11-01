#!/usr/bin/env python3
"""E4 -- Where the time goes in E2's TQP settings: each query's PCIe volume,
compute and PCIe utilization, and an exclusive split of its timed window, at
E2's scale factor, partition budget and chunk size.

Three settings of ``e2_end_to_end_ablation.py``, with E2's flags and
environment (``setting_cmd``, ``setting_env``: the ambient ``TQP_*`` dropped,
the GPU's NUMA node, ``TQP_NUMA_BIND=0``, a 64 MiB stack):

  tqp_raw       no compression, searches off
  tqp_ans       ANS, searches off
  tqp_adaptive  adaptive codecs + compression search + sizing search

ONE PROFILED PROCESS PER SETTING over the queries, as an E2 rep, under
Nsight Systems (CUDA + NVTX trace, no CPU sampling) with ``--cuda_profile``:
main.py's ``profiler`` NVTX range around each query's timed run is the
capture range, repeated, so each query gets its own report
(``<work>/<setting>/trace_p<process>.<n>.nsys-rep``) and the warmup none.
A process that dies, or writes no log for ``--hang_min``, records its query
FAIL and the rest run in a new process (E2's rule).  Beside it,
``helpers/gpm_sampler.py`` samples the GPU every ``--interval`` s (NVML GPM:
SM utilization, PCIe RX/TX) into ``<work>/<setting>/gpm.csv``, under
``--python`` (its ``nvidia-ml-py`` is requirements-dev.txt's).

PER QUERY, over its ``Query N`` range (the timed window, plus the output
frame's conversion), from the trace:

  kernel_busy   some kernel running                    -> compute utilization
  h2d / d2h     some host->device / device->host copy  -> PCIe utilization
  pcie_busy     either one                               (time)
  d2d_busy      a device->device copy or a memset
  the exclusive split, summing to the window:
    compute_only  kernel, no PCIe copy     overlap   kernel and a PCIe copy
    pcie_only     PCIe copy, no kernel     d2d_only  only a D2D copy or memset
    idle          nothing on the GPU (host work, launches, waits)
  h2d_gb / d2h_gb / d2d_gb  the copies' bytes (PCIe volume), and the link
                bandwidth utilization max(h2d, d2h) / (window x ``--link_gbs``)

and from the sampler (the hardware's view, which also sees what the trace
cannot: zero-copy UVA reads by kernels): ``gpm_h2d_gb`` / ``gpm_d2h_gb``
(PCIe RX/TX integrated over the window) and ``gpm_sm_util`` (mean SM %).

A trace inflates host time, so ``approx_time`` here is not E2's timing; the
CSV carries E2's mean beside it (``e2_mean_s``) and the shares, not the
seconds, are the result.  Rows go to ``--csv`` (default
``results/ablation_profile.csv``); a setting already there is skipped unless
``--overwrite``.  The summary (``results/ablation_profile_summary.csv``,
printed) aggregates each setting over the queries OK in every setting:
shares of the summed windows, bytes and bandwidths.  ``--analyze_only``
redoes the rows from the traces already in ``--work``.  ``--only_failed``
reruns, for each setting in the CSV, its queries that are not OK there, in a
new profiled process (``<work>/retry/<setting>/``), and replaces just their
rows.

    python3 experiments/paper_scripts/e4_ablation_profile.py --show
    python3 experiments/paper_scripts/e4_ablation_profile.py [--gpu 1] \\
        [--settings tqp_raw tqp_ans tqp_adaptive] [--overwrite] [--wait_idle]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "helpers"), HERE]
import common  # noqa: E402
import e2_end_to_end_ablation as e2  # noqa: E402
import perf_json  # noqa: E402

ROOT = common.ROOT
SETTINGS = ["tqp_raw", "tqp_ans", "tqp_adaptive"]
NSYS = ["nsys", "profile", "--trace=cuda,nvtx", "--sample=none",
        "--cpuctxsw=none", "--capture-range=nvtx", "--nvtx-capture=profiler",
        "--capture-range-end=repeat",
        "--env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0", "--force-overwrite=true"]
# CUPTI memcpy kinds (ENUM_CUDA_MEMCPY_OPER): the unified-memory ones too.
H2D, D2H, D2D = {1, 11}, {2, 12}, {8, 10, 13}

FIELDS = ["setting", "query", "status", "approx_time", "e2_mean_s", "window_s",
          "kernel_busy_s", "h2d_busy_s", "d2h_busy_s", "pcie_busy_s",
          "d2d_busy_s", "compute_only_s", "overlap_s", "pcie_only_s",
          "d2d_only_s", "idle_s", "h2d_gb", "d2h_gb", "d2d_gb", "n_kernels",
          "n_copies", "compute_util", "pcie_time_util", "pcie_bw_util",
          "gpm_h2d_gb", "gpm_d2h_gb", "gpm_sm_util", "gpm_samples", "trace",
          "flags", "commit", "gpu", "finished_utc"]


# --- running ----------------------------------------------------------------

def e2_args(args, qids):
    return argparse.Namespace(
        python=args.python, sf=args.sf, q=list(qids),
        chunk_size_mb=args.chunk_size_mb,
        partition_budget_mb=args.partition_budget_mb,
        cmp_pool_gb=args.cmp_pool_gb, slot_pool_gb=args.slot_pool_gb,
        gpu=args.gpu, data=args.data)


def profiled_cmd(s, args, qids, perf_dir, trace):
    return (e2.numa_prefix(args.gpu) + NSYS + ["-o", trace]
            + e2.setting_cmd(s, e2_args(args, qids), qids, perf_dir)
            + ["--cuda_profile"])


def bus_of_group(pgid: int) -> str | None:
    """The PCI bus of a GPU context held by a process of group ``pgid`` (the
    profiler and the engine under it), None until there is one."""
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_bus_id",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    for line in apps.splitlines():
        pid, _, bus = (x.strip() for x in line.partition(","))
        try:
            if os.getpgid(int(pid)) == pgid:
                return bus.lower()
        except (ProcessLookupError, ValueError):
            continue
    return None


def start_sampler(args, path):
    uuid = subprocess.run(
        ["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
         "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    out = open(path, "w")
    proc = subprocess.Popen(
        [args.python, os.path.join(HERE, "helpers", "gpm_sampler.py"), uuid,
         str(args.interval)], stdout=out,
        stderr=open(path + ".err", "w"), start_new_session=True)
    return proc, out


def run_setting(s, args, work):
    """The setting's queries in one profiled process, a new one for the rest
    after a crash or a hang.  Returns [(qid, status, seconds, detail)]."""
    perf_dir = os.path.join(work, "perf")
    log_path = os.path.join(work, "main.log")
    os.makedirs(perf_dir, exist_ok=True)
    perf_json.clear_run_outputs(perf_dir)
    for f in glob.glob(os.path.join(work, "trace_p*")) + [
            os.path.join(perf_dir, "query_results.csv"),
            os.path.join(perf_dir, "query_results_all.csv")]:
        if os.path.exists(f):
            os.remove(f)
    open(log_path, "w").close()
    want_bus = e2.gpu_bus(args.gpu)
    sampler, sampler_out = start_sampler(args, os.path.join(work, "gpm.csv"))
    outcomes, k = [], 0
    try:
        while len(outcomes) < len(args.q):
            k += 1
            todo = args.q[len(outcomes):]
            cmd = profiled_cmd(s, args, todo, perf_dir,
                               os.path.join(work, f"trace_p{k}"))
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n")
                log.flush()
                proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log,
                                        stderr=subprocess.STDOUT, text=True,
                                        env=e2.setting_env(e2_args(args, todo)),
                                        preexec_fn=e2._stack_64m,
                                        start_new_session=True)
            limit = args.hang_min * 60 if args.hang_min > 0 else float("inf")
            size, deadline, why, bus_ok = 0, time.time() + limit, None, False
            while proc.poll() is None:
                time.sleep(5)
                if not bus_ok and (bus := bus_of_group(proc.pid)):
                    if bus != want_bus:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        sys.exit(f"{s.key}: main.py is on GPU {bus}, not --gpu "
                                 f"{args.gpu} ({want_bus}); stopped")
                    bus_ok = True
                    print(f"    {s.key} process {k}: on GPU {bus} (CPUs "
                          f"{' '.join(e2.numa_prefix(args.gpu)[2:]) or 'unbound'})",
                          flush=True)
                n = os.path.getsize(log_path)
                if n > size:
                    size, deadline = n, time.time() + limit
                if time.time() > deadline:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    why = f"no progress for {args.hang_min:g} min (killed)"
            qr = os.path.join(perf_dir, "query_results.csv")
            rows = e2._results_rows(perf_dir)
            outcomes += rows[:len(todo)]
            if os.path.exists(qr):
                with open(qr, encoding="utf-8") as src, open(
                        os.path.join(perf_dir, "query_results_all.csv"), "a",
                        encoding="utf-8") as dst:
                    dst.writelines(l for l in src if l[:1].isdigit())
                os.remove(qr)
            print(f"    {s.key} process {k}: {len(outcomes)}/{len(args.q)} "
                  f"queries, exit {proc.returncode}", flush=True)
            if len(outcomes) < len(args.q):
                why = why or f"process exited {proc.returncode}"
                q = args.q[len(outcomes)]
                outcomes.append((q, "FAIL", None, why))
                print(f"    {s.key}: Q{q} {why}; restarting with the rest",
                      flush=True)
    finally:
        sampler.send_signal(signal.SIGTERM)
        sampler.wait(timeout=30)
        sampler_out.close()
    res = os.path.join(ROOT, "results.out")
    if os.path.exists(res):
        # A move, not os.replace: in the experiments container ROOT is the
        # image's and perf_dir a mounted out/, another filesystem.
        shutil.move(res, os.path.join(perf_dir, "results_all.out"))
    return outcomes


# --- analysis ---------------------------------------------------------------

def merged(ivs, lo, hi):
    """The union of the intervals ``ivs`` clipped to [lo, hi], sorted."""
    out = []
    for a, b in sorted((max(a, lo), min(b, hi)) for a, b in ivs
                       if b > lo and a < hi):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def length(ivs):
    return sum(b - a for a, b in ivs)


def split(k, p, d, lo, hi):
    """Exclusive seconds of [lo, hi] by which of the merged interval lists
    ``k`` (kernels), ``p`` (PCIe copies), ``d`` (D2D) cover each instant."""
    ev = []
    for i, ivs in enumerate((k, p, d)):
        for a, b in ivs:
            ev += [(a, i, 1), (b, i, -1)]
    ev.sort()
    on = [0, 0, 0]
    acc = dict(compute_only=0, overlap=0, pcie_only=0, d2d_only=0, idle=0)
    t = lo
    for when, i, step in ev + [(hi, 0, 0)]:
        dt = when - t
        if dt > 0:
            K, P, D = (x > 0 for x in on)
            key = ("overlap" if K and P else "compute_only" if K else
                   "pcie_only" if P else "d2d_only" if D else "idle")
            acc[key] += dt
        on[i] += step
        t = when
    return acc


def gpm_rows(path):
    rows = []
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((float(r["time"]), float(r["pcie_rx_mibs"] or 0),
                                 float(r["pcie_tx_mibs"] or 0),
                                 float(r["sm_util"] or 0)))
                except (KeyError, ValueError):
                    continue
    return rows


def gpm_window(samples, t0, t1, interval):
    """(H2D GB, D2H GB, mean SM %, n) of the samples whose interval ends in
    (t0, t1]: RX is host->device."""
    mine = [s for s in samples if t0 < s[0] <= t1]
    if not mine:
        return None, None, None, 0
    mib = 1 << 20
    return (sum(s[1] for s in mine) * interval * mib / 1e9,
            sum(s[2] for s in mine) * interval * mib / 1e9,
            sum(s[3] for s in mine) / len(mine), len(mine))


def analyze_report(rep, samples, interval, link_gbs):
    """{qid: metrics} of the ``Query N`` ranges in one report."""
    db = rep[:-len(".nsys-rep")] + ".sqlite"
    subprocess.run(["nsys", "export", "--type", "sqlite",
                    "--force-overwrite=true", "-o", db, rep],
                   capture_output=True, check=True)
    out = {}
    try:
        c = sqlite3.connect(db)
        tabs = {r[0] for r in c.execute(
            "select name from sqlite_master where type='table'")}
        epoch = c.execute("select utcEpochNs from "
                          "TARGET_INFO_SESSION_START_TIME").fetchone()[0]
        names = dict(c.execute("select id, value from StringIds")) \
            if "StringIds" in tabs else {}
        kern = (c.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL")
                .fetchall() if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs else [])
        cps = (c.execute("select start, end, copyKind, bytes from "
                         "CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
               if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else [])
        mset = (c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET")
                .fetchall() if "CUPTI_ACTIVITY_KIND_MEMSET" in tabs else [])
        for start, end, text, tid in c.execute(
                "select start, end, text, textId from NVTX_EVENTS "
                "where end is not null"):
            text = text or names.get(tid, "")
            m = re.fullmatch(r"Query (\d+)", text or "")
            if not m:
                continue
            lo, hi = start, end
            k = merged(kern, lo, hi)
            h = merged([(a, b) for a, b, kind, _ in cps if kind in H2D], lo, hi)
            dn = merged([(a, b) for a, b, kind, _ in cps if kind in D2H], lo, hi)
            p = merged(h + dn, lo, hi)
            d = merged([(a, b) for a, b, kind, _ in cps if kind in D2D] + mset,
                       lo, hi)
            win = (hi - lo) / 1e9

            def gb(kinds):
                return sum(n for a, b, kind, n in cps
                           if kind in kinds and lo <= a < hi) / 1e9
            h2d, d2h = gb(H2D), gb(D2H)
            ex = split(k, p, d, lo, hi)
            g = gpm_window(samples, (epoch + lo) / 1e9, (epoch + hi) / 1e9,
                           interval)
            out[int(m.group(1))] = dict(
                window_s=win, kernel_busy_s=length(k) / 1e9,
                h2d_busy_s=length(h) / 1e9, d2h_busy_s=length(dn) / 1e9,
                pcie_busy_s=length(p) / 1e9, d2d_busy_s=length(d) / 1e9,
                **{f"{key}_s": v / 1e9 for key, v in ex.items()},
                h2d_gb=h2d, d2h_gb=d2h, d2d_gb=gb(D2D),
                n_kernels=sum(1 for a, b in kern if lo <= a < hi),
                n_copies=sum(1 for a, *_ in cps if lo <= a < hi),
                compute_util=length(k) / 1e9 / win,
                pcie_time_util=length(p) / 1e9 / win,
                pcie_bw_util=max(h2d, d2h) / (win * link_gbs),
                gpm_h2d_gb=g[0], gpm_d2h_gb=g[1], gpm_sm_util=g[2],
                gpm_samples=g[3], trace=os.path.basename(rep))
    finally:
        if os.path.exists(db):
            os.remove(db)
    return out


def analyze_setting(s, args, work, outcomes, commit):
    samples = gpm_rows(os.path.join(work, "gpm.csv"))
    metrics = {}
    reps = sorted(glob.glob(os.path.join(work, "trace_p*.nsys-rep")),
                  key=lambda p: [int(x) for x in re.findall(r"\d+", os.path.basename(p))])
    for rep in reps:
        metrics.update(analyze_report(rep, samples, args.interval, args.link_gbs))
    e2_mean = {int(r["query"]): r["mean_s"] for r in e2.read_csv(args.e2_csv)
               if r["setting"] == s.key and int(r["sf"]) == args.sf
               and int(r["chunk_size_mb"]) == args.chunk_size_mb
               and int(r["partition_budget_mb"]) == args.partition_budget_mb}
    by_q = {o[0]: o for o in outcomes}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%F %T")
    flags = " ".join(e2.setting_cmd(s, e2_args(args, []), [], "<perf>")[2:])
    rows = []
    for q in args.q:
        o = by_q.get(q, (q, "FAIL", None, "no row"))
        m = metrics.get(q, {})
        status = o[1] if m or o[1] != "OK" else "OK_NO_TRACE"
        row = {"setting": s.key, "query": q, "status": status,
               "approx_time": "" if o[2] is None else f"{o[2]:.6f}",
               "e2_mean_s": e2_mean.get(q, ""), "flags": flags,
               "commit": commit, "gpu": args.gpu, "finished_utc": stamp}
        for key, v in m.items():
            row[key] = (f"{v:.6f}" if isinstance(v, float) else
                        "" if v is None else v)
        rows.append(row)
    return rows


def outcomes_from(work):
    rows = []
    path = os.path.join(work, "perf", "query_results_all.csv")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                head, _, rest = line.rstrip("\n").partition(",")
                status, _, rest = rest.partition(",")
                secs, _, rest = rest.partition(",")
                if head.isdigit():
                    rows.append((int(head), status, float(secs) if secs else None,
                                 rest.rsplit(",", 1)[0]))
    return rows


# --- results ----------------------------------------------------------------

def write_rows(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = {k: i for i, k in enumerate(SETTINGS)}
    rows = sorted(rows, key=lambda r: (order.get(r["setting"], 9), int(r["query"])))
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)


def summary(args):
    rows = [r for r in e2.read_csv(args.csv) if r["setting"] in SETTINGS]
    have = sorted({r["setting"] for r in rows}, key=SETTINGS.index)
    ok = {(r["setting"], int(r["query"])) for r in rows if r["status"] == "OK"
          and r["window_s"]}
    common_q = [q for q in args.q if all((s, q) in ok for s in have)]
    out = []
    for s in have:
        mine = [r for r in rows if r["setting"] == s and int(r["query"]) in common_q]
        tot = lambda k: sum(float(r[k] or 0) for r in mine)  # noqa: E731
        win = tot("window_s")
        if not win:
            continue
        h2d, d2h = tot("h2d_gb"), tot("d2h_gb")
        out.append({
            "setting": s, "queries": len(mine), "window_s": round(win, 2),
            "e2_total_s": round(tot("e2_mean_s"), 2),
            **{f"{k}_pct": round(100 * tot(f"{k}_s") / win, 1) for k in (
                "compute_only", "overlap", "pcie_only", "d2d_only", "idle")},
            "compute_util_pct": round(100 * tot("kernel_busy_s") / win, 1),
            "pcie_time_util_pct": round(100 * tot("pcie_busy_s") / win, 1),
            "h2d_time_pct": round(100 * tot("h2d_busy_s") / win, 1),
            "d2h_time_pct": round(100 * tot("d2h_busy_s") / win, 1),
            "d2d_time_pct": round(100 * tot("d2d_busy_s") / win, 1),
            "h2d_gb": round(h2d, 1), "d2h_gb": round(d2h, 1),
            "d2d_gb": round(tot("d2d_gb"), 1),
            "h2d_gbs": round(h2d / win, 2), "d2h_gbs": round(d2h / win, 2),
            "pcie_bw_util_pct": round(100 * max(h2d, d2h) / (win * args.link_gbs), 1),
            "gpm_h2d_gb": round(tot("gpm_h2d_gb"), 1),
            "gpm_d2h_gb": round(tot("gpm_d2h_gb"), 1),
            "gpm_sm_util_pct": round(sum(float(r["gpm_sm_util"] or 0) * float(r["window_s"])
                                         for r in mine) / win, 1),
        })
    path = os.path.join(os.path.dirname(args.csv), "ablation_profile_summary.csv")
    if out:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    print(f"\nOver the {len(common_q)} queries OK in every setting "
          f"({' '.join(f'Q{q}' for q in common_q)}); % of the summed windows:")
    cols = ["queries", "window_s", "e2_total_s", "compute_only_pct",
            "overlap_pct", "pcie_only_pct", "d2d_only_pct", "idle_pct",
            "compute_util_pct", "pcie_time_util_pct", "h2d_gb", "d2h_gb",
            "h2d_gbs", "pcie_bw_util_pct", "gpm_h2d_gb", "gpm_d2h_gb",
            "gpm_sm_util_pct"]
    print(f"  {'':<22}" + "".join(f"{s:>14}" for s in (r["setting"] for r in out)))
    for c in cols:
        print(f"  {c:<22}" + "".join(f"{r[c]:>14}" for r in out))
    print(f"  -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--settings", nargs="+", choices=SETTINGS, default=SETTINGS)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--sf", type=int, default=1000)
    ap.add_argument("--q", type=int, nargs="+", default=common.QUERIES)
    ap.add_argument("--chunk_size_mb", type=int, default=512)
    ap.add_argument("--partition_budget_mb", type=int, default=512)
    ap.add_argument("--cmp_pool_gb", type=int, default=29)
    ap.add_argument("--slot_pool_gb", type=float, default=20.0)
    ap.add_argument("--gpu", default=common.default_gpu())
    ap.add_argument("--data", default=None,
                    help="data root holding sf<SF>/parquet (default "
                         "TQP_TABLES_PATH, else TQP_DATA_DIR / DATA_DIR)")
    ap.add_argument("--python", default="/opt/envs/tqp/bin/python"
                    if os.path.exists("/opt/envs/tqp/bin/python") else sys.executable)
    ap.add_argument("--interval", type=float, default=0.1,
                    help="GPM sampling interval, s")
    ap.add_argument("--link_gbs", type=float, default=25.0,
                    help="achievable PCIe bandwidth per direction, GB/s "
                         "(Gen4 x16: 31.5 theoretical)")
    ap.add_argument("--hang_min", type=float, default=15.0)
    ap.add_argument("--csv", default=os.path.join(HERE, "results",
                                                  "ablation_profile.csv"))
    ap.add_argument("--e2_csv", default=os.path.join(HERE, "results",
                                                     "ablation_results.csv"))
    ap.add_argument("--work", default=os.path.join(ROOT, "out", "e4_ablation_profile"))
    ap.add_argument("--analyze_only", action="store_true",
                    help="redo the rows from the traces in --work; run nothing")
    ap.add_argument("--only_failed", action="store_true",
                    help="rerun each setting's non-OK queries in the CSV and "
                         "replace just their rows")
    ap.add_argument("--wait_idle", action="store_true")
    ap.add_argument("--show", "--dry-run", action="store_true", dest="show")
    args = ap.parse_args()
    args.data = (args.data or os.environ.get("TQP_TABLES_PATH")
                 or common.data_dir())
    args.csv = os.path.abspath(args.csv)

    existing = e2.read_csv(args.csv)
    have = {r["setting"] for r in existing}
    redo = {k: [q for q in args.q if not any(
                r["setting"] == k and int(r["query"]) == q and r["status"] == "OK"
                for r in existing)] for k in args.settings}
    if args.only_failed:
        todo = [k for k in args.settings if k in have and redo[k]]
        for k in args.settings:
            print(f"{k}: " + (f"rerun {' '.join(f'Q{q}' for q in redo[k])}"
                              if k in todo else "nothing to rerun"))
    else:
        todo = [k for k in args.settings
                if args.overwrite or args.analyze_only or k not in have]
    if args.show:
        env = e2.setting_env(e2_args(args, []))
        for k in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "TQP_TABLES_PATH",
                  "TQP_NUMA_BIND", "OMP_NUM_THREADS"):
            print(f"  {k}={env[k]}")
        for k in args.settings:
            print(f"\n{k} [{'run' if k in todo else 'in the CSV'}]:\n  " + " ".join(
                profiled_cmd(e2.BY_KEY[k], args, ["<1..22>"],
                             f"<work>/{k}/perf", f"<work>/{k}/trace_p1")))
        print(f"\n  sampler: {args.python} helpers/gpm_sampler.py <uuid> {args.interval}")
        return 0
    if not shutil.which("nsys"):
        sys.exit("nsys is not on PATH")
    if subprocess.run([args.python, "-c", "import pynvml"],
                      capture_output=True).returncode:
        sys.exit(f"{args.python} cannot import pynvml, which the GPM sampler needs "
                 "(nvidia-ml-py, requirements-dev.txt; a native venv: "
                 "bench/native_env.sh venv)")
    commit = common.source_commit()
    for k in todo:
        s = e2.BY_KEY[k]
        sargs = argparse.Namespace(**vars(args))
        if args.only_failed:
            sargs.q = redo[k]
            work = os.path.join(args.work, "retry", k)
        else:
            work = os.path.join(args.work, k)
        os.makedirs(work, exist_ok=True)
        if args.analyze_only:
            outcomes = outcomes_from(work)
        else:
            while busy := e2.gpu_busy(args.gpu):
                if not args.wait_idle:
                    sys.exit(f"GPU {args.gpu} is busy ({busy}); --wait_idle waits")
                time.sleep(60)
            print(f"[{time.strftime('%F %T')}] {k}: profiling", flush=True)
            outcomes = run_setting(s, sargs, work)
        print(f"[{time.strftime('%F %T')}] {k}: analysing the traces", flush=True)
        rows = analyze_setting(s, sargs, work, outcomes, commit)
        ok = sum(r["status"] == "OK" for r in rows)
        print(f"[{time.strftime('%F %T')}] {k}: {ok}/{len(rows)} OK with a trace",
              flush=True)
        mine = {str(q) for q in sargs.q}
        write_rows(args.csv, [r for r in e2.read_csv(args.csv)
                              if r["setting"] != k or (args.only_failed
                                                       and r["query"] not in mine)]
                   + rows)
    summary(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
