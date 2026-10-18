#!/usr/bin/env python
"""rapids's spill-pass record builder (AGENTS.md, "Spill pass"; plan §3.1).

  eventlog_stats.py record <q_dir> --sf SF --query N --evidence-dir DIR
  eventlog_stats.py settings <evidence_dir> --sf SF --spark-dir DIR
  eventlog_stats.py passes <event log | q_dir>          (debug: every pass's totals)

The timed run of one query process (Spark 3.5.8 + RAPIDS 26.04.2), from <q_dir>:
  local-*.zstd[.inprogress]  the zstd Spark event log. run_tpch_queries.py tags
              every job with the local properties tpch.pass (warm | setup | gc |
              timed) and tpch.query, and names SQL executions "tpch:<pass>:q<N>";
              stages are attributed through StageSubmitted.Properties (these
              survive broadcast-exchange jobs, which replace the job group), SQL
              executions and their driver-side metric updates through
              SQLExecutionStart.description. A killed JVM's truncated log is
              read up to its last complete line.
  runner.json the RMM scoped maximum of each pass (py4j) and its wall times.
  monitor.csv 1 Hz samples: shuffle_* and temp_* bytes under spark.local.dir,
              the RAPIDS host spill store, RMM's allocation, the JVM's /proc io.
  driver.log  "Dropping event from queue eventLog" (an incomplete log) and the
              DeviceMemoryEventHandler INFO lines ("Device store spilled N
              bytes", conf/log4j2-spill.properties), whose sum over the timed
              run must equal gpuSpillToHostBytes.

Byte fields say what they count: logical = uncompressed (packed GPU / serialized
table bytes), wire = as stored on disk (Spark's io codec, lz4).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import spill_results as sr  # noqa: E402

ENGINE = "rapids"
# org.apache.spark.sql.rapids.GpuTaskMetrics accumulators -> (field, how)
GPU_ACC = {
    "gpuSpillToHostBytes": ("spill_gpu_to_host_logical_bytes", "sum"),
    "gpuSpillToDiskBytes": ("spill_host_to_disk_wire_bytes", "sum"),
    "gpuSpillToHostTime": ("spill_gpu_to_host_s", "sum"),
    "gpuSpillToDiskTime": ("spill_host_to_disk_s", "sum"),
    "gpuReadSpillFromHostTime": ("restore_from_host_s", "sum"),
    "gpuReadSpillFromDiskTime": ("restore_from_disk_s", "sum"),
    "gpuRetryCount": ("gpu_retry_count", "sum"),
    "gpuSplitAndRetryCount": ("gpu_split_retry_count", "sum"),
    "gpuRetryBlockTime": ("gpu_retry_block_s", "sum"),
    "gpuRetryComputationTime": ("gpu_retry_lost_compute_s", "sum"),
    "gpuDiskWriteSavedBytes": ("shuffle_partial_file_kept_in_host_bytes", "sum"),
    "gpuMaxDeviceMemoryBytes": ("gpu_peak_task_bytes", "max"),
    "gpuMaxTaskFootprint": ("gpu_peak_task_footprint_bytes", "max"),
    "gpuMaxHostMemoryBytes": ("host_peak_rapids_alloc_bytes", "max"),
    "gpuMaxPinnedMemoryBytes": ("host_peak_pinned_bytes", "max"),
    "gpuMaxPageableMemoryBytes": ("host_peak_pageable_bytes", "max"),
    "gpuMaxDiskMemoryBytes": ("disk_peak_spill_bytes", "max"),
    "gpuMaxConcurrentGpuTasks": ("gpu_max_concurrent_tasks", "max"),
}
# SQL metrics: (plan node regex, metric name) -> field (summed); the number of
# updated "data size" accumulators is the number of exchanges that ran (an
# accumulator belongs to one plan-node instance).
SQL_ACC = [
    (r"^GpuColumnarExchange|^GpuShuffleExchange", "data size", "shuffle_d2h_logical_bytes", "n_shuffle_exchanges"),
    (r"^GpuColumnarExchange|^GpuShuffleExchange", "data read size", "shuffle_h2d_logical_bytes", None),
    (r"^GpuBroadcastExchange|^BroadcastExchange", "data size", "broadcast_logical_bytes", "n_broadcast_exchanges"),
]
_SIZE = re.compile(r"\((-?\d+) bytes\)")
_TIME = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$")
LOG_TS = re.compile(r"^(\d\d/\d\d/\d\d \d\d:\d\d:\d\d) ")
STORE_SPILLED = re.compile(r"DeviceMemoryEventHandler: .*Device store spilled (\d+) bytes")
RMM_POOL = re.compile(r"Initializing RMM (\w+) pool size = ([\d.]+) (\w+)")


def _num(v):
    """Accumulable value: int, '123', '1.2MB (1289750 bytes)' or '00:00:01.234'."""
    if isinstance(v, (int, float)):
        return v
    if not isinstance(v, str):
        return None
    m = _SIZE.search(v)
    if m:
        return int(m.group(1))
    m = _TIME.match(v)
    if m:
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    try:
        return float(v) if "." in v else int(v)
    except ValueError:
        return None


def _lines(path):
    """Event log lines (zstd through pyarrow, else plain); the truncated tail of a
    SIGKILLed writer is dropped."""
    if ".zstd" in os.path.basename(path):
        import pyarrow as pa
        with pa.CompressedInputStream(pa.OSFile(path), "zstd") as f:
            buf = b""
            while True:
                try:
                    chunk = f.read(1 << 16)
                except OSError:          # a truncated zstd frame
                    break
                if not chunk:
                    break
                *done, buf = (buf + chunk).split(b"\n")
                yield from done
    else:
        with open(path, "rb") as f:
            yield from f


def _plan(node, acc_node):
    for m in node.get("metrics", []):
        acc_node[m["accumulatorId"]] = (node.get("nodeName", ""), m["name"])
    for c in node.get("children", []):
        _plan(c, acc_node)


def parse(path):
    """({(pass, query): {field: value}}, meta) of one event log."""
    stage_pass, exec_pass, acc_node = {}, {}, {}
    acc_stage, sql_task, sql_driver, acc_exec = {}, defaultdict(float), defaultdict(float), {}
    rec = defaultdict(lambda: defaultdict(float))
    meta = {"bad_lines": 0, "app_end": False, "event_log": path, "spark_props": {}}
    for raw in _lines(path):
        try:
            e = json.loads(raw)
        except ValueError:
            meta["bad_lines"] += 1
            continue
        ev = e.get("Event", "")
        if ev == "SparkListenerStageSubmitted":
            p = e.get("Properties") or {}
            if p.get("tpch.pass"):
                stage_pass[e["Stage Info"]["Stage ID"]] = (p["tpch.pass"], int(p.get("tpch.query", -1)))
        elif ev == "SparkListenerStageCompleted":
            k = stage_pass.get(e["Stage Info"]["Stage ID"])
            if k:
                rec[k]["n_stages"] += 1
        elif ev.endswith("SQLExecutionStart") or ev.endswith("SQLAdaptiveExecutionUpdate"):
            if ev.endswith("SQLExecutionStart"):
                m = re.match(r"tpch:(\w+):q(\d+)$", e.get("description") or "")
                if m:
                    exec_pass[e["executionId"]] = (m.group(1), int(m.group(2)))
            _plan(e["sparkPlanInfo"], acc_node)
        elif ev.endswith("DriverAccumUpdates"):
            for aid, v in e["accumUpdates"]:
                sql_driver[aid] += v
                acc_exec[aid] = e["executionId"]
        elif ev == "SparkListenerEnvironmentUpdate":
            props = e.get("Spark Properties") or {}
            meta["spark_props"] = dict(props) if isinstance(props, dict) else {k: v for k, v in props}
        elif ev == "SparkListenerLogStart":
            meta["spark_version"] = e.get("Spark Version")
        elif ev == "SparkListenerApplicationEnd":
            meta["app_end"] = True
        elif ev == "SparkListenerTaskEnd":
            k = stage_pass.get(e["Stage ID"])
            if k is None:
                continue
            r, ti, tm = rec[k], e["Task Info"], e.get("Task Metrics") or {}
            r["n_tasks"] += 1
            r["n_failed_tasks"] += bool(ti.get("Failed"))
            r["n_killed_tasks"] += bool(ti.get("Killed"))
            if tm:
                sr_, sw = tm.get("Shuffle Read Metrics") or {}, tm.get("Shuffle Write Metrics") or {}
                r["spark_memory_bytes_spilled"] += tm.get("Memory Bytes Spilled", 0)
                r["spark_disk_bytes_spilled"] += tm.get("Disk Bytes Spilled", 0)
                r["scan_input_wire_bytes"] += (tm.get("Input Metrics") or {}).get("Bytes Read", 0)
                r["shuffle_write_wire_bytes"] += sw.get("Shuffle Bytes Written", 0)
                r["shuffle_write_batches"] += sw.get("Shuffle Records Written", 0)
                r["shuffle_write_s"] += sw.get("Shuffle Write Time", 0) / 1e9
                r["shuffle_read_wire_bytes"] += sr_.get("Local Bytes Read", 0) + sr_.get("Remote Bytes Read", 0)
                r["shuffle_read_batches"] += sr_.get("Total Records Read", 0)
                r["shuffle_fetch_wait_s"] += sr_.get("Fetch Wait Time", 0) / 1e3
                r["result_bytes"] += tm.get("Result Size", 0)
                r["executor_run_s"] += tm.get("Executor Run Time", 0) / 1e3
            for a in ti.get("Accumulables", []):
                v = _num(a.get("Update"))
                if v is None:
                    continue
                if a.get("Metadata") == "sql":
                    sql_task[a["ID"]] += v
                    acc_stage.setdefault(a["ID"], e["Stage ID"])
                elif a.get("Name") in GPU_ACC:
                    f, how = GPU_ACC[a["Name"]]
                    r[f] = r[f] + v if how == "sum" else max(r[f], v)
    for aid, v in list(sql_task.items()) + list(sql_driver.items()):
        node = acc_node.get(aid)
        k = stage_pass.get(acc_stage[aid]) if aid in acc_stage else exec_pass.get(acc_exec.get(aid))
        if node is None or k is None:
            continue
        for pat, metric, f, count in SQL_ACC:
            if node[1] == metric and re.search(pat, node[0]):
                rec[k][f] += v
                if count and v > 0:
                    rec[k][count] += 1
    for r in rec.values():   # Spark's spill counters = RAPIDS GPU spill + CPU-operator spill
        r["cpu_spill_memory_bytes"] = r["spark_memory_bytes_spilled"] - r["spill_gpu_to_host_logical_bytes"]
        r["cpu_spill_disk_bytes"] = r["spark_disk_bytes_spilled"] - r["spill_host_to_disk_wire_bytes"]
    return rec, meta


def monitor_peaks(path, label):
    """Peaks of PassStats's 1 Hz samples of one pass ("timed:q<N>"): shuffle and
    spill files under spark.local.dir, both together, the host spill store,
    RMM's allocation; the JVM's rchar/wchar over the pass."""
    out, first, last = {}, None, None
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["pass"] != label:
                continue
            vals = {}
            for c in ("shuffle_bytes", "spill_bytes", "host_store_bytes", "rmm_bytes"):
                if r.get(c) not in (None, ""):
                    vals[c] = int(float(r[c]))
            if "shuffle_bytes" in vals and "spill_bytes" in vals:
                vals["disk_bytes"] = vals["shuffle_bytes"] + vals["spill_bytes"]
            for c, v in vals.items():
                out[f"peak_{c}"] = max(out.get(f"peak_{c}", 0), v)
            out["samples"] = out.get("samples", 0) + 1
            if r.get("rchar"):
                first = first or r
                last = r
    if first and last:
        for c in ("rchar", "wchar"):
            out[f"io_{c}_bytes"] = int(last[c]) - int(first[c])
    return out


def driver_facts(path, t0, t1):
    """From the driver log: dropped event-log events, the INFO "Device store
    spilled" sum inside [t0, t1] (log timestamps have 1 s resolution; passes are
    seconds apart), the RMM pool."""
    out = {"dropped_events": 0, "store_spilled_b": 0, "store_spilled_lines": 0}
    with open(path, errors="replace") as f:
        for ln in f:
            if "Dropping event from queue eventLog" in ln or re.search(r"Dropped \d+ events from eventLog", ln):
                out["dropped_events"] += 1
            m = RMM_POOL.search(ln)
            if m:
                unit = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "KiB": 2**10, "MiB": 2**20,
                        "GiB": 2**30}.get(m.group(3), 0)
                out["rmm_pool"], out["rmm_pool_b"] = m.group(1), int(float(m.group(2)) * unit) or None
            m = STORE_SPILLED.search(ln)
            ts = LOG_TS.match(ln)
            if m and ts and t0 is not None:
                t = datetime.strptime(ts.group(1), "%y/%m/%d %H:%M:%S").timestamp()
                if int(t0) <= t <= t1 + 1:
                    out["store_spilled_b"] += int(m.group(1))
                    out["store_spilled_lines"] += 1
    return out


def event_log(q_dir):
    logs = sorted(glob.glob(os.path.join(q_dir, "local-*")) + glob.glob(os.path.join(q_dir, "app-*")),
                  key=os.path.getmtime)
    if not logs:
        raise FileNotFoundError(f"no event log in {q_dir}")
    return logs[-1]


def native(q_dir, num):
    """The flat record of the timed run of query <num>."""
    path = event_log(q_dir)
    rec, meta = parse(path)
    n = {k: (int(v) if float(v).is_integer() else round(v, 3)) for k, v in rec.get(("timed", num), {}).items()}
    n["has_timed"] = ("timed", num) in rec
    n["event_log"] = os.path.basename(path)
    n["log_inprogress"] = path.endswith(".inprogress")
    n["app_end"] = meta["app_end"]
    n["bad_lines"] = meta["bad_lines"]
    runner = os.path.join(q_dir, "runner.json")
    p = {}
    if os.path.exists(runner):
        with open(runner) as f:
            p = (json.load(f).get("passes") or {}).get(f"timed:q{num}", {})
    n.update({f"runner_{k}": v for k, v in p.items()})
    t0, t1 = p.get("t0"), p.get("t1")
    mon = os.path.join(q_dir, "monitor.csv")
    if os.path.exists(mon):
        n.update({f"mon_{k}": v for k, v in monitor_peaks(mon, f"timed:q{num}").items()})
    drv = os.path.join(q_dir, "driver.log")
    if os.path.exists(drv):
        n.update({f"drv_{k}": v for k, v in driver_facts(drv, t0, t1).items()})
    n["log_complete"] = int(n["app_end"] and n["bad_lines"] == 0 and n.get("drv_dropped_events", 0) == 0)
    return n


def rows(n, status):
    """The contract rows and identity checks of a native record."""
    g = lambda k: n.get(k, 0)
    if not n.get("has_timed"):
        raise LookupError("no timed pass in the event log")
    q = "OK" if n.get("log_complete") and status not in ("TIMEOUT", "KILLED_DISK") else "PARTIAL"
    el = "eventlog:"
    both = lambda kind, key, **kw: sr.row(kind, g(key), g(key), quality=q, **kw)
    # What was surely written: the host -> disk counter (it misses the bypass)
    # or the sampled temp_* peak, whichever is larger.
    disk_files = sr.row("spill_disk_files", wire_b=max(g("spill_host_to_disk_wire_bytes"),
                                                       n.get("mon_peak_spill_bytes", 0)),
                        quality="LOWER_BOUND", source="max(gpuSpillToDiskBytes, sampled temp_* peak)")
    sampled = lambda kind, key, src: (sr.row(kind, n[key], n[key], quality="LOWER_BOUND", source=src)
                                      if key in n else sr.na(kind, src))
    peak = n.get("runner_gpu_peak_process_bytes")
    out = [
        sr.row("scan", wire_b=g("scan_input_wire_bytes"), count=g("n_tasks"), quality=q,
               source=el + "Input Bytes Read"),
        both("shuffle_d2h", "shuffle_d2h_logical_bytes", count=g("n_shuffle_exchanges"),
             source=el + "GpuColumnarExchange data size"),
        both("broadcast", "broadcast_logical_bytes", count=g("n_broadcast_exchanges"),
             source=el + "GpuBroadcastExchange data size"),
        sr.row("shuffle_write", wire_b=g("shuffle_write_wire_bytes"), count=g("shuffle_write_batches"),
               seconds=g("shuffle_write_s"), quality=q, source=el + "Shuffle Bytes Written"),
        sr.row("shuffle_read", wire_b=g("shuffle_read_wire_bytes"), count=g("shuffle_read_batches"),
               quality=q, source=el + "Shuffle Local+Remote Bytes Read"),
        both("shuffle_h2d", "shuffle_h2d_logical_bytes", source=el + "GpuColumnarExchange data read size"),
        both("spill_gpu_out", "spill_gpu_to_host_logical_bytes", seconds=g("spill_gpu_to_host_s"),
             source=el + "gpuSpillToHostBytes"),
        disk_files,
        sr.row("spill_host_disk", wire_b=g("spill_host_to_disk_wire_bytes"), seconds=g("spill_host_to_disk_s"),
               quality=q, source=el + "gpuSpillToDiskBytes"),
        sr.row("restore_host", seconds=g("restore_from_host_s"), quality=q,
               source=el + "gpuReadSpillFromHostTime (bytes NA)"),
        sr.row("restore_disk", seconds=g("restore_from_disk_s"), quality=q,
               source=el + "gpuReadSpillFromDiskTime (bytes NA)"),
        sr.row("peak_gpu", peak, peak, quality="OK", source="py4j Rmm scoped maximum")
        if isinstance(peak, (int, float)) else sr.na("peak_gpu", "py4j Rmm scoped maximum"),
        sampled("peak_host", "mon_peak_host_store_bytes", "sampled host spill store"),
        sampled("peak_disk", "mon_peak_disk_bytes", "sampled shuffle_* + temp_* under spark.local.dir"),
        sampled("peak_disk_spill", "mon_peak_spill_bytes", "sampled temp_* under spark.local.dir"),
        sr.row("retries", count=g("gpu_retry_count") + g("gpu_split_retry_count"), quality=q,
               source=el + "gpuRetryCount + gpuSplitAndRetryCount"),
        sr.row("cpu_spill", max(0, g("cpu_spill_memory_bytes")), max(0, g("cpu_spill_disk_bytes")), quality=q,
               source=el + "Spark spill counters minus the GPU ones"),
    ]
    checks = [sr.check("Memory Bytes Spilled = gpuSpillToHostBytes", g("spark_memory_bytes_spilled"),
                       g("spill_gpu_to_host_logical_bytes"))]
    if "drv_store_spilled_b" in n:
        # within 0.1 %: the handler does not log every spill (SF1 q18 under
        # forced spill: 530 B of 43.7 MB were counted but never logged)
        checks.append(sr.check("INFO Device store spilled = gpuSpillToHostBytes", n["drv_store_spilled_b"],
                               g("spill_gpu_to_host_logical_bytes"), tol=1e-3))
    return out, checks


def cmd_record(a):
    status, pt = sr.pass_time(os.path.join(a.evidence_dir, "pass.csv"), a.query)
    try:
        n = native(a.q_dir, a.query)
        with open(os.path.join(a.q_dir, "record.json"), "w") as f:
            json.dump(n, f, indent=1)
        rs, checks = rows(n, status)
        rec = sr.record(ENGINE, a.sf, a.query, status, rs + [pt], checks)
    except (OSError, LookupError, ValueError, KeyError) as e:
        rec = sr.stub(ENGINE, a.sf, a.query, a.evidence_dir, f"{type(e).__name__}: {e}")
    sr.append_record(a.evidence_dir, rec)


def cmd_settings(a):
    """The budgets and confs the pass ran under, from a query's event log
    (Spark properties) and driver log (the RMM pool)."""
    meta, pool = {}, {}
    for q_dir in sorted(glob.glob(os.path.join(a.evidence_dir, "q*"))):
        try:
            meta = parse(event_log(q_dir))[1]
        except (OSError, LookupError, ValueError):
            continue
        drv = os.path.join(q_dir, "driver.log")
        pool = driver_facts(drv, None, None) if os.path.exists(drv) else {}
        break
    props = meta.get("spark_props", {})
    host = props.get("spark.rapids.memory.host.spillStorageSize")
    budgets = {"gpu_b": pool.get("rmm_pool_b"), "host_b": _bytes(host), "disk_b": None}
    keep = {k: v for k, v in props.items() if k.startswith(("spark.rapids.", "spark.sql.shuffle", "spark.sql.adaptive",
                                                             "spark.sql.files", "spark.shuffle", "spark.local.dir",
                                                             "spark.driver.memory", "spark.master", "spark.io."))}
    sr.write_settings(a.evidence_dir, ENGINE, a.sf, budgets, [a.spark_dir],
                      versions={"spark": meta.get("spark_version"),
                                "rapids_jar": os.path.basename(os.environ.get("RAPIDS_JAR", ""))},
                      spark_conf=keep, rmm_pool=pool.get("rmm_pool"), smoke=os.environ.get("SPILL_SMOKE") == "1")


def _bytes(v):
    m = re.fullmatch(r"\s*(\d+)\s*([kmgt]?)b?\s*", str(v or "").lower())
    return int(m.group(1)) * 1024 ** "_kmgt".index(m.group(2) or "_") if m else None


def cmd_passes(a):
    path = event_log(a.log) if os.path.isdir(a.log) else a.log
    rec, meta = parse(path)
    for (p, q), r in sorted(rec.items()):
        print(json.dumps({"pass": p, "query": q, **{k: v for k, v in sorted(r.items())}}))
    print(json.dumps({k: v for k, v in meta.items() if k != "spark_props"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("q_dir")
    r.add_argument("--sf", type=int, required=True)
    r.add_argument("--query", type=int, required=True)
    r.add_argument("--evidence-dir", required=True)
    s = sub.add_parser("settings")
    s.add_argument("evidence_dir")
    s.add_argument("--sf", type=int, required=True)
    s.add_argument("--spark-dir", required=True)
    p = sub.add_parser("passes")
    p.add_argument("log")
    a = ap.parse_args()
    {"record": cmd_record, "settings": cmd_settings, "passes": cmd_passes}[a.cmd](a)


if __name__ == "__main__":
    main()
