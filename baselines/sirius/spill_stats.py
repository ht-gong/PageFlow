#!/usr/bin/env python
"""Sirius's spill-pass record builder (AGENTS.md, "Spill pass"; plan §3.3).

  spill_stats.py record <q_dir> --sf SF --query N --evidence-dir DIR
  spill_stats.py settings <evidence_dir> --sf SF --config CFG

The timed run of one query process, from three sources in <q_dir>, all cut to
the timed run:
  quent/      quent ndjson with the state machines only (no batch events, no
              NVTX capture). Data batches are attributed producer pipeline ->
              operator -> plan -> Query, else by the Query whose Init..Exit
              holds their construction; the timed Query is the one labelled
              timed_q<N> (sirius_set_query_label).
  log/        the DEBUG Sirius log: `[downgrade] ... done` totals (demand vs
              memory monitor; the identities below), `[gpu_pool]/[host_pool]
              QueryEnd` peaks (process lifetime) and OOM reschedules, inside the
              timed Query's window.
  stdout.txt  /proc/<duckdb>/io printed around the timed run (PROCIO_TIMED_*).
Bytes are cuCascade's batch sizes (cudf alloc size on the GPU, packed size on
host and disk): no scratch, no hash tables.

`record` appends the query's record to DIR/records.jsonl (spill_results.py),
keeps the flat native record as <q_dir>/record.json and gzips the raw evidence.
run_tpch_sirius.py writes the per-query config (write_config) and SQL; this
module's loaders read plain or gzipped files.

stdlib only: the image's python has no YAML library, so write_config edits the
block-style config line by line (yaml_set).
"""
import argparse
import collections
import glob
import gzip
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import spill_results as sr  # noqa: E402

ENGINE = "sirius"
# Forced spill for `make spill-smoke` (SPILL_SMOKE=1): a 768 MiB GPU budget and
# a 64 MiB host tier make SF1 q9 move GPU -> host -> disk and back (plan §3.3).
SMOKE = {
    "sirius.memory.gpu.usage_limit_fraction": None,
    "sirius.memory.gpu.usage_limit_bytes": "768Mi",
    "sirius.memory.host.capacity_bytes": "64Mi",
    "sirius.memory.host.initial_number_pools": "1",
    "sirius.memory.host.pool_size": "8",
    "sirius.memory.disk.capacity_bytes": "8GB",
}


# ---- config ------------------------------------------------------------------

_KEY = re.compile(r"^(?P<ind> *)(?P<key>[A-Za-z_][\w-]*):(?P<rest>.*)$")


def yaml_set(text, updates):
    """Set the scalar keys of a block-style YAML document by dotted path
    ({path: value}; None deletes). A missing key is added at the end of its
    parent's block, which must exist. Flow-style maps are left alone."""
    lines = text.splitlines()
    todo = dict(updates)
    out, stack, ends = [], [], {}            # stack: (indent, key); ends: path -> (index after block, child indent)
    for line in lines:
        m = _KEY.match(line)
        if not m or line.lstrip().startswith("#"):
            out.append(line)
            continue
        ind = len(m["ind"])
        while stack and stack[-1][0] >= ind:
            stack.pop()
        path = ".".join([k for _, k in stack] + [m["key"]])
        parent = ".".join(k for _, k in stack)
        if path in todo:
            v = todo.pop(path)
            if v is not None:
                out.append(f"{m['ind']}{m['key']}: {v}")
        else:
            out.append(line)
        ends[parent] = (len(out), ind)
        stack.append((ind, m["key"]))
        # every ancestor's block now ends after this line
        anc = parent
        while anc:
            ends[anc] = (len(out), ends[anc][1])
            anc = anc.rpartition(".")[0]
    for path, v in todo.items():
        if v is None:
            continue
        parent, _, key = path.rpartition(".")
        if parent not in ends:
            raise ValueError(f"yaml_set: no block {parent!r} for {path!r}")
        at, ind = ends[parent]
        out.insert(at, " " * ind + f"{key}: {v}")
        for p, (i, n) in ends.items():
            if i >= at:
                ends[p] = (i + 1, n)
    return "\n".join(out) + "\n"


def write_config(base, out, quent_dir, smoke=False):
    """The query's config: the effective one (after env.sh) with quent on, state
    machines only, into quent_dir; with smoke, the forced-spill budgets too."""
    upd = {"sirius.telemetry.enable_quent": "true",
           "sirius.telemetry.enable_batch_events": "false",
           "sirius.telemetry.exporter": "ndjson",
           "sirius.telemetry.output_directory": json.dumps(quent_dir)}
    if smoke:
        upd.update(SMOKE)
    with open(base) as f:
        text = f.read()
    with open(out, "w") as f:
        f.write(yaml_set(text, upd))


def yaml_get(text, path):
    """A scalar of a block-style YAML document by dotted path, else None."""
    stack = []
    for line in text.splitlines():
        m = _KEY.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        ind = len(m["ind"])
        while stack and stack[-1][0] >= ind:
            stack.pop()
        if ".".join([k for _, k in stack] + [m["key"]]) == path:
            return m["rest"].split("#")[0].strip().strip("\"'") or None
        stack.append((ind, m["key"]))
    return None


def size_bytes(v):
    """'128Gi', '40GB', '512Mi', '805306368' -> bytes."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]i?)?B?\s*", str(v or ""))
    if not m:
        return None
    mult = {None: 1, "K": 10**3, "M": 10**6, "G": 10**9, "T": 10**12,
            "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}[m[2]]
    return int(float(m[1]) * mult)


# ---- quent ---------------------------------------------------------------------

def _open(path):
    return gzip.open(path, "rt", errors="ignore") if path.endswith(".gz") else open(path, errors="ignore")


def load(d, kind):
    out = []
    for f in sorted(glob.glob(os.path.join(d, kind, "*.ndjson")) + glob.glob(os.path.join(d, kind, "*.ndjson.gz"))):
        with _open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:     # the truncated tail of a killed process
                        pass
    return out


def state_of(e):
    s = e["data"].get("state") if isinstance(e["data"], dict) else None
    if s is None:
        return None, None
    if isinstance(s, str):
        return s, {}
    k = next(iter(s))
    return k, s[k]


def session_dirs(path):
    if glob.glob(os.path.join(path, "model.qmi*")):
        return [path]
    return sorted(p for p in glob.glob(os.path.join(path, "*")) if glob.glob(os.path.join(p, "model.qmi*")))


TIER_RE = re.compile(r"tier=(\w+)")


def analyze(sd):
    """Per Query of a quent session: (queries in Init order, {query id: Counter})."""
    mem, limits = {}, collections.Counter()
    for e in load(sd, "memory"):
        k, v = state_of(e)
        if k == "MemoryInitializing":
            m = TIER_RE.search(v["instance_name"])
            mem[e["id"]] = m.group(1).lower() if m else v["instance_name"]
        elif k == "MemoryOperating" and e["id"] in mem:
            limits[f"limit_{mem[e['id']]}_b"] += v.get("capacity_bytes") or 0
    queries = collections.OrderedDict()
    for e in load(sd, "query"):
        k, v = state_of(e)
        q = queries.setdefault(e["id"], dict(id=e["id"], label=None, init=None, executing=None, exit=None))
        if k == "Init":
            q["label"], q["init"] = v.get("instance_name"), e["timestamp"]
        elif k == "Executing":
            q["executing"] = e["timestamp"]
        elif k == "Exit":
            q["exit"] = e["timestamp"]
    plan_q = {}
    for e in load(sd, "plan"):
        d = e["data"].get("Declaration")
        if d:
            plan_q[e["id"]] = (d.get("parent") or {}).get("query_id")
    op = {}
    for e in load(sd, "operator"):
        d = e["data"].get("Declaration")
        if d:
            op[e["id"]] = (d["instance_name"], plan_q.get(d["plan_id"]))
    qlist = sorted(queries.values(), key=lambda q: q["init"] or 0)

    def query_at(ts):
        for q in qlist:
            if q["init"] is not None and ts >= q["init"] and (q["exit"] is None or ts <= q["exit"]):
                return q["id"]
        return None

    batches = collections.defaultdict(list)
    for e in load(sd, "data_batch"):
        batches[e["id"]].append(e)
    stats = {q["id"]: collections.Counter() for q in qlist}
    stats[None] = collections.Counter()
    intervals = collections.defaultdict(list)      # query id -> [(t0, t1, tier, bytes)]
    ids_seen = collections.Counter()
    for bid, evs in batches.items():
        evs.sort(key=lambda e: e["data"]["seq"])
        k0, v0 = state_of(evs[0])
        prod = v0.get("producer_pipeline_uuid") if k0 == "Constructed" else None
        ids_seen[v0.get("data_batch_id") if k0 == "Constructed" else bid] += 1
        name, qid = op.get(prod, (None, None))
        if qid is None:
            qid = query_at(evs[0]["timestamp"])
        st = stats.setdefault(qid, collections.Counter())
        kind = ("scan_out" if name and name.startswith("GPU_SCAN") else
                "result" if name and name.startswith("RESULT_COLLECTOR") else
                "intermediate" if name else "unattributed")
        st["batches"] += 1
        born, was_on_gpu, prev = None, False, None      # prev = (ts, tiers, bytes)
        seq = [(e["timestamp"],) + state_of(e) for e in evs]
        for i, (ts, k, v) in enumerate(seq):
            if prev is not None:
                for tier in prev[1]:
                    intervals[qid].append((prev[0], ts, tier, prev[2]))
                prev = None
            if k == "Stationary":
                tier = mem.get(v["memory"]["resource_id"], "?")
                b = (v["memory"].get("capacity") or {}).get("capacity_bytes", 0)
                if born is None:
                    born = tier
                    if tier == "gpu":
                        st["produced_gpu_b"] += b
                        st[f"produced_gpu_{kind}_b"] += b
                        st[f"produced_gpu_{kind}_n"] += 1
                    else:
                        st[f"born_{tier}_{kind}_b"] += b
                        if kind == "result" and tier == "host":
                            st["result_d2h_b"] += b
                            st["result_d2h_n"] += 1
                if tier == "gpu":
                    was_on_gpu = True
                prev = (ts, (tier,), b)
            elif k == "InTransit":
                src = mem.get(v["source_memory"]["resource_id"], "?")
                dst = mem.get(v["dest_memory"]["resource_id"], "?")
                b = (v["source_memory"].get("capacity") or {}).get("capacity_bytes", 0)
                landed, t_land = False, ts                # did it land on dst?
                for ts2, k2, v2 in seq[i + 1:]:
                    if k2 == "Stationary":
                        landed = mem.get(v2["memory"]["resource_id"]) == dst
                        t_land = ts2
                        break
                if not landed:
                    st["failed_moves"] += 1
                    st[f"failed_{src}_{dst}_b"] += b
                else:
                    key = {("gpu", "host"): "spill_gpu_host", ("gpu", "disk"): "spill_gpu_disk",
                           ("host", "disk"): "spill_host_disk", ("disk", "host"): "restore_disk_host",
                           ("host", "gpu"): "restore_host_gpu" if was_on_gpu else "scan_h2d",
                           ("disk", "gpu"): "restore_disk_gpu" if was_on_gpu else "scan_disk_h2d",
                           }.get((src, dst), f"move_{src}_{dst}")
                    st[key + "_b"] += b
                    st[key + "_n"] += 1
                    st[key + "_s"] += (t_land - ts) / 1e9
                prev = (ts, (src, dst), b)
            elif k in ("Destructed", "Exit"):
                prev = None
    dup_ids = sum(1 for c in ids_seen.values() if c > 1)

    # Peak residency per tier inside each Query's Executing..Exit (a batch in
    # transit counts in both tiers).
    for q in qlist:
        lo, hi = q["executing"] or q["init"], q["exit"] or float("inf")
        ev = []
        for t0, t1, tier, b in intervals.get(q["id"], []):
            a0, a1 = max(t0, lo), min(t1, hi)
            if a1 > a0 and b:
                ev.append((a0, 1, tier, b))
                ev.append((a1, 0, tier, b))
        ev.sort(key=lambda x: (x[0], x[1]))           # ends before starts at equal times
        cur, peak = collections.Counter(), collections.Counter()
        for _, is_start, tier, b in ev:
            cur[tier] += b if is_start else -b
            peak[tier] = max(peak[tier], cur[tier])
        for tier, p in peak.items():
            stats[q["id"]][f"peak_{tier}_b"] = p

    for e in load(sd, "task"):                         # demand-driven downgrades
        k, v = state_of(e)
        if k == "Downgrading":
            st = stats.setdefault(query_at(e["timestamp"]), collections.Counter())
            st["downgrading_tasks"] += 1
            st["shortfall_b"] += v.get("shortfall_bytes", 0)
    for qid in stats:
        stats[qid].update(limits)
    return qlist, stats, dup_ids


# ---- log -----------------------------------------------------------------------

TS = r"^\[(?P<ts>[0-9-]+ [0-9:.]+)\]"
DG_RE = re.compile(TS + r" \[debug\] \[downgrade_executor\.cpp:\d+\] \[downgrade\] \[(?P<src>\w+):(?P<dev>-?\d+)\] "
                   r"request (?P<mon>monitor )?done: (?P<nb>\d+) batches, (?P<b>\d+) bytes .*?"
                   r"\| to_host: (?P<hb>\d+)/(?P<hB>\d+) batches/bytes, to_disk: (?P<db>\d+)/(?P<dB>\d+)")
TEL_RE = re.compile(TS + r".*\bquery (?P<w>\d+) telemetry_query=(?P<u>[0-9a-f]{32})")
POOL_RE = re.compile(TS + r".*\[(?P<pool>gpu_pool|host_pool)\] \w+:-?\d+ (?P<edge>QueryBegin|QueryEnd) "
                     r".*?window=(?P<w>\d+) .*?allocated=(?P<a>\d+) bytes peak=(?P<p>\d+) bytes")
OOM_RE = re.compile(TS + r".*GPU Pipeline Executor: reschedule \(retry")


def to_ns(ts):
    """A log timestamp as epoch ns. spdlog writes the process's local time
    (its default), so it is read in this process's time zone: the builder
    runs right after the CLI, in its environment (run.sh)."""
    return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1e9


def log_facts(log_path, q):
    """The timed Query's part of the log: downgrade totals (the log has ms
    resolution: 2 ms slack), pool peaks of its window, OOM reschedules. Two
    streaming passes: the first finds the Query's log window."""
    c = collections.Counter()
    lo, hi = q["init"] - 2e6, (q["exit"] or float("inf")) + 2e6
    uuid_hex = q["id"].replace("-", "")
    win, pools = None, {}
    with _open(log_path) as fh:
        for line in fh:
            if "telemetry_query=" in line:
                m = TEL_RE.search(line)
                if m and m["u"] == uuid_hex:
                    win = m["w"]
    with _open(log_path) as fh:
        for line in fh:
            if "[downgrade]" in line and "done:" in line:
                m = DG_RE.search(line)
                if m and lo <= to_ns(m["ts"]) <= hi:
                    src = m["src"].lower()
                    c[f"log_{src}_{'monitor' if m['mon'] else 'demand'}_b"] += int(m["b"])
                    c[f"log_{src}_to_host_b"] += int(m["hB"])
                    c[f"log_{src}_to_disk_b"] += int(m["dB"])
            elif "_pool]" in line:
                m = POOL_RE.search(line)
                if m and m["w"] == win:
                    pools[(m["pool"], m["edge"])] = (int(m["a"]), int(m["p"]))
            elif "reschedule" in line:
                m = OOM_RE.search(line)
                if m and lo <= to_ns(m["ts"]) <= hi:
                    c["oom_reschedules"] += 1
    out = dict(c, log_window=win)
    for pool in ("gpu_pool", "host_pool"):
        b, e = pools.get((pool, "QueryBegin")), pools.get((pool, "QueryEnd"))
        if e:
            out[f"{pool}_peak_b"] = e[1]
            # the lifetime peak grew during the timed run: it is the timed run's
            out[f"{pool}_peak_is_timed"] = bool(b and e[1] > b[1])
    return out


def procio(stdout_path):
    """{tag: {counter: value}} of the PROCIO_<tag> blocks of the CLI's stdout."""
    blocks, tag = {}, None
    with open(stdout_path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("PROCIO_"):
                tag = line[len("PROCIO_"):]
                blocks[tag] = {}
            elif tag and re.match(r"^\w+: \d+$", line):
                k, v = line.split(": ")
                blocks[tag][k] = int(v)
            else:
                tag = None
    return blocks


# ---- the record ----------------------------------------------------------------

def native(q_dir, num):
    """The flat record of the timed run (the prototype's sirius_spill_record)."""
    sd = session_dirs(os.path.join(q_dir, "quent"))
    if not sd:
        raise FileNotFoundError(f"no quent session in {q_dir}/quent")
    qlist, stats, dups = analyze(sd[-1])
    label = f"timed_q{num}"
    timed = next((q for q in qlist if q["label"] == label), None)
    if timed is None:
        raise LookupError(f"no Query labelled {label} in {sd[-1]}")
    rec = dict(stats.get(timed["id"], {}))
    rec.update(query_uuid=timed["id"], label=label, n_queries_in_process=len(qlist),
               duplicate_batch_ids=dups, complete=timed["exit"] is not None,
               exec_s=(timed["exit"] - timed["executing"]) / 1e9 if timed["exit"] and timed["executing"] else None)
    logs = sorted(glob.glob(os.path.join(q_dir, "log", "sirius*.log*")))
    if logs:
        rec.update(log_facts(logs[-1], timed))
    rec["has_log"] = bool(logs)
    out = os.path.join(q_dir, "stdout.txt")
    io = procio(out) if os.path.exists(out) else {}
    b, e = io.get("TIMED_BEGIN"), io.get("TIMED_END")
    if b and e:
        rec["io_rchar_b"] = e["rchar"] - b["rchar"]
        rec["io_wchar_b"] = e["wchar"] - b["wchar"]
        # disk-tier restores are pread()s too; what is left is the parquet read
        rec["scan_read_b"] = rec["io_rchar_b"] - rec.get("restore_disk_gpu_b", 0) - rec.get("restore_disk_host_b", 0)
    return rec


def rows(n, status):
    """The contract rows and the identity checks of a native record."""
    g = lambda k: n.get(k, 0)
    cut = status in ("TIMEOUT", "KILLED_DISK") or not n.get("complete")
    q = "PARTIAL" if cut else "OK"
    src = "quent:DataBatch InTransit"

    def mv(kind, key):
        return sr.row(kind, g(key + "_b"), g(key + "_b"), g(key + "_n"), round(g(key + "_s"), 3), q, src)
    has_log = n.get("has_log")
    logq = q if has_log else "NA"
    lg = lambda *ks: sum(g(k) for k in ks) if has_log else None
    pool = lambda p: (sr.row(f"pool_peak_{p}", n[f"{p}_pool_peak_b"], n[f"{p}_pool_peak_b"],
                             quality="OK" if n.get(f"{p}_pool_peak_is_timed") else "LIFETIME",
                             source=f"log:[{p}_pool] QueryEnd peak")
                      if f"{p}_pool_peak_b" in n else sr.na(f"pool_peak_{p}", f"log:[{p}_pool]"))
    out = [
        sr.row("scan", g("produced_gpu_scan_out_b"), n.get("scan_read_b"), g("produced_gpu_scan_out_n"),
               quality=q, source="quent:GPU_SCAN batches; /proc rchar minus disk restores"),
        sr.row("materialized", g("produced_gpu_intermediate_b"), g("produced_gpu_intermediate_b"),
               g("produced_gpu_intermediate_n"), quality=q, source="quent:GPU-born batches"),
        mv("spill_gpu_host", "spill_gpu_host"),
        mv("spill_gpu_disk", "spill_gpu_disk"),
        mv("spill_host_disk", "spill_host_disk"),
        mv("restore_host_gpu", "restore_host_gpu"),
        mv("restore_disk_gpu", "restore_disk_gpu"),
        mv("restore_disk_host", "restore_disk_host"),
        sr.row("result_d2h", g("result_d2h_b"), g("result_d2h_b"), g("result_d2h_n"), quality=q,
               source="quent:host-born result batches"),
        sr.row("spill_demand", lg("log_gpu_demand_b", "log_host_demand_b"),
               lg("log_gpu_demand_b", "log_host_demand_b"), quality=logq, source="log:[downgrade] request done"),
        sr.row("spill_monitor", lg("log_gpu_monitor_b", "log_host_monitor_b"),
               lg("log_gpu_monitor_b", "log_host_monitor_b"), quality=logq,
               source="log:[downgrade] request monitor done"),
        *[sr.row(f"peak_{t}", g(f"peak_{t}_b"), g(f"peak_{t}_b"), quality=q, source="quent:Stationary sweep")
          for t in ("gpu", "host", "disk")],
        pool("gpu"), pool("host"),
        sr.row("retries", count=g("oom_reschedules"), quality=q, source="log:OOM reschedule")
        if has_log else sr.na("retries", "log:OOM reschedule"),
    ]
    checks = []
    if has_log:
        checks = [
            sr.check("quent GPU>host = log to_host", g("spill_gpu_host_b"),
                     g("log_gpu_to_host_b") + g("log_host_to_host_b")),
            sr.check("quent to disk = log to_disk", g("spill_gpu_disk_b") + g("spill_host_disk_b"),
                     g("log_gpu_to_disk_b") + g("log_host_to_disk_b")),
        ]
    return out, checks


def gzip_evidence(q_dir):
    """gzip the query's logs, quent ndjson and SQL in place (the loaders read both)."""
    pats = ("log/*.log", "quent/*/*/*.ndjson", "*.sql")
    for p in [f for pat in pats for f in glob.glob(os.path.join(q_dir, pat))]:
        with open(p, "rb") as fi, gzip.open(p + ".gz", "wb", compresslevel=6) as fo:
            while True:
                b = fi.read(1 << 20)
                if not b:
                    break
                fo.write(b)
        os.remove(p)


def cmd_record(a):
    """Append the query's record; a query without usable evidence (it failed
    before quent started, say) gets the all-NA stub with the reason."""
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
    gzip_evidence(a.q_dir)


def cmd_settings(a):
    """Budgets as Sirius resolved them (quent MemoryOperating capacities of a
    query, else the config), the disk tier dir, the config's hash."""
    with open(a.config) as f:
        text = f.read()
    lim = {}
    for p in sorted(glob.glob(os.path.join(a.evidence_dir, "q*", "record.json"))):
        with open(p) as f:
            r = json.load(f)
        if r.get("limit_gpu_b"):
            lim = r
            break
    budgets = {"gpu_b": lim.get("limit_gpu_b"),
               "host_b": lim.get("limit_host_b") or size_bytes(yaml_get(text, "sirius.memory.host.capacity_bytes")),
               "disk_b": lim.get("limit_disk_b") or size_bytes(yaml_get(text, "sirius.memory.disk.capacity_bytes"))}
    keys = ("memory.gpu.usage_limit_fraction", "memory.gpu.usage_limit_bytes", "memory.gpu.reservation_limit_fraction",
            "memory.gpu.downgrade_trigger_fraction", "memory.gpu.downgrade_stop_fraction",
            "memory.host.downgrade_trigger_fraction", "memory.host.downgrade_stop_fraction",
            "operator_params.scan_task_batch_size", "operator_params.hash_partition_bytes",
            "operator_params.concat_batch_bytes", "operator_params.max_build_hash_table_bytes",
            "executor.pipeline.num_threads", "executor.downgrade.num_threads")
    params = {k: yaml_get(text, "sirius." + k) for k in keys}
    sr.write_settings(a.evidence_dir, ENGINE, a.sf, budgets,
                      [yaml_get(text, "sirius.memory.disk.downgrade_root_dirs")], [a.config],
                      {"sirius_ref": os.environ.get("SIRIUS_REF", ""), "duckdb": os.environ.get("SIRIUS_DUCKDB", "")},
                      params={k: v for k, v in params.items() if v is not None},
                      io=os.environ.get("SIRIUS_IO", ""), smoke=os.environ.get("SPILL_SMOKE") == "1")


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
    s.add_argument("--config", required=True)
    a = ap.parse_args()
    {"record": cmd_record, "settings": cmd_settings}[a.cmd](a)


if __name__ == "__main__":
    main()
