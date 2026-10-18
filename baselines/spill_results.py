#!/usr/bin/env python
"""The spill pass's results contract (AGENTS.md, "Spill pass"): per (engine,
scale factor, query), what the timed run moved off the GPU, brought back and
held, in results/spill_profile.csv; the settings each engine ran under in
results/spill_settings.json.

  spill_results.py merge <engine> <scale_factor> <evidence_dir> [results_dir]
  spill_results.py check [--expect-spill] <evidence_dir> [<evidence_dir> ...]
  spill_results.py stub <engine> <scale_factor> <query> <evidence_dir> [--error MSG]

<evidence_dir> is one engine's spill pass at one SF, $SPILL_STATS_DIR
($SPILL_ROOT/sf<SF>/<engine>, set by run.sh):
  pass.csv        the pass's own rows (query,status,seconds,result_rows_or_error)
  records.jsonl   one line per query from the engine's record builder; the last
                  line of a query wins
  settings.json   write_settings()
  q<N>/           the query's raw evidence (event log, quent, logs)

merge validates each record against KINDS and upserts it: a record replaces
every row of its (engine, scale_factor, query), every other row is kept.
Atomic; refused on a header mismatch. check validates records and evaluates the
identities their builders assert, without merging (`make spill-smoke`). stub
appends the record of a query whose builder failed: every kind NA.

Record builders (rapids/eventlog_stats.py, polars/spill_stats.py,
sirius/spill_stats.py, and tqp_vortex's ../bench/tqp_spill_stats.py) import
this module for row(), record(), pass_result(), append_record() and
write_settings().
"""
import csv
import datetime
import hashlib
import json
import os
import re
import socket
import subprocess
import sys

ENGINE_ORDER = ["duckdb_cpu", "sirius", "polars_gpu", "rapids", "tqp_vortex"]
FIELDS = ["engine", "scale_factor", "query", "kind", "run_status", "quality",
          "logical_b", "wire_b", "count", "seconds", "source"]
RUN_STATUS = ("OK", "FAIL", "TIMEOUT", "KILLED_DISK")
# OK; NA: not measured (the instrumentation cannot, or the builder failed);
# LOWER_BOUND; LIFETIME: a process-lifetime peak that may include the warm
# pass; PARTIAL: from a killed process or a truncated log.
QUALITY = ("OK", "LIFETIME", "LOWER_BOUND", "PARTIAL", "NA")
CATEGORIES = (
    "scan_in",          # base-table bytes read (wire = stored, logical = decoded)
    "scan_uva",         # base-table bytes read zero-copy
    "design_d2h",       # intermediates the design puts off the GPU ...
    "design_h2d",       # ... and their read-back, by copy ...
    "design_uva",       # ... or zero-copy
    "design_disk_w",    # by-design disk materialization (Spark shuffle files)
    "design_disk_r",
    "design_d2d",       # on-GPU repartition, fanout and cache copies (context)
    "pressure_d2h",     # GPU -> host spill forced by memory pressure
    "pressure_d2disk",  # GPU -> disk spill
    "pressure_h2disk",  # host -> disk spill
    "restore_h2d",      # spilled data host -> GPU
    "restore_disk",     # spilled data disk -> host or GPU
    "result_d2h",       # the result copy (context)
    "materialized",     # intermediates produced, each engine's own definition (context)
    "peak_gpu", "peak_host", "peak_disk",   # residency peaks of the timed run
    "retries",          # OOM retries or reschedules (count)
    "context",          # engine-specific extras no headline reads
)

# Each engine's closed list of kinds, {kind: category}, in CSV order. Every
# record has every kind of its engine, zeros included. Readers aggregate by
# category, never by kind name. Sources: docs/engine_spill_plan.md §3.
KINDS = {
    "sirius": {
        "scan": "scan_in",                 # wire: /proc rchar of the timed run minus disk restores; logical: GPU_SCAN-born batches
        "materialized": "materialized",    # the other GPU-born batches
        "spill_gpu_host": "pressure_d2h",  # quent DataBatch InTransit bytes, deduplicated by batch
        "spill_gpu_disk": "pressure_d2disk",
        "spill_host_disk": "pressure_h2disk",
        "restore_host_gpu": "restore_h2d",
        "restore_disk_gpu": "restore_disk",
        "restore_disk_host": "restore_disk",
        "result_d2h": "result_d2h",        # the host-born result batch
        "spill_demand": "context",         # DEBUG [downgrade] totals: reservation-driven ...
        "spill_monitor": "context",        # ... and memory-monitor-driven
        "peak_gpu": "peak_gpu",            # quent Stationary sweep (batch bytes; no scratch, no hash tables)
        "peak_host": "peak_host",
        "peak_disk": "peak_disk",
        "pool_peak_gpu": "context",        # INFO pool peaks, process lifetime
        "pool_peak_host": "context",
        "retries": "retries",              # OOM reschedules
    },
    "polars_gpu": {
        "scan": "scan_in",                 # logical bytes of the scan actors' chunks
        "spill_d2h": "pressure_d2h",       # rapidsmpf copy-device-to-{host,pinned_host}-bytes
        "restore_h2d": "restore_h2d",      # copy-{host,pinned_host}-to-device-bytes
        "shuffle": "design_d2d",           # shuffle inserts (full volume on one rank)
        "allgather": "design_d2d",
        "fanout_d2d": "design_d2d",        # copy-device-to-device (fanout deep copies)
        "materialized": "materialized",    # operator output chunks
        "dev_alloc": "context",            # device bytes allocated through the spill-limited resource
        "peak_gpu": "peak_gpu",            # StatisticsResourceAdaptor peak of the pass
        "peak_host": "peak_host",          # pinned pool, LIFETIME; NA when spill is pageable
        "fallbacks": "context",            # single-partition fallback warnings (count)
        "gpu_callbacks": "context",        # cudf-polars callbacks (count; 0 = CPU fallback)
    },
    "rapids": {
        "scan": "scan_in",                 # task metric Input Bytes Read (wire)
        "shuffle_d2h": "design_d2h",       # GpuColumnarExchange "data size" (logical)
        "broadcast": "design_d2h",         # GpuBroadcastExchange "data size" (logical)
        "shuffle_write": "design_disk_w",  # Shuffle Bytes Written (wire, lz4)
        "shuffle_read": "design_disk_r",   # Shuffle Local + Remote Bytes Read (wire); a reused exchange is read twice
        "shuffle_h2d": "design_h2d",       # "data read size" (logical)
        "spill_gpu_out": "pressure_d2h",   # gpuSpillToHostBytes (logical); includes the GPU -> disk bypass
        "spill_disk_files": "pressure_h2disk",  # bytes written to temp_* spill files: bypass + host -> disk
        "spill_host_disk": "context",      # gpuSpillToDiskBytes (wire): misses the bypass
        "restore_host": "restore_h2d",     # bytes NA: gpuReadSpillFromHostTime only
        "restore_disk": "restore_disk",    # bytes NA: gpuReadSpillFromDiskTime only
        "peak_gpu": "peak_gpu",            # RMM scoped maximum of the pass (py4j)
        "peak_host": "peak_host",          # host spill store, sampled; also holds shuffle staging and broadcast copies
        "peak_disk": "peak_disk",          # shuffle_* + temp_* under spark.local.dir, sampled
        "peak_disk_spill": "context",      # temp_* alone, sampled
        "retries": "retries",              # gpuRetryCount + gpuSplitAndRetryCount
        "cpu_spill": "context",            # Spark's spill counters minus the GPU ones (expected 0)
    },
    # The movement ledger (IO/telemetry/movement.py: logical = decoded, wire =
    # stored bytes, a granule's decoded bytes pro-rated from its records';
    # count = pages of one kind, none for zero-copy reads and sums) and the
    # residency peaks (IO/telemetry/residency.py) of the perf report. No disk
    # tier and no retries: an OOM fails the query. Under --buffer_strategy cpu
    # (the bench's) the cache kinds and the pressure kinds are 0 by
    # construction; the cache modes move them. Each consumer of a region reads
    # it, and a read leaves it where it is: the read-back kinds can exceed what
    # was written home.
    "tqp_vortex": {
        "scan_h2d": "scan_in",             # base-table runs copied H2D from the pinned pool (the parquet is loaded before the timer)
        "scan_uva": "scan_uva",            # base-table rows read zero-copy (rows x width: a lower bound on PCIe bytes)
        "spill_bypass": "design_d2h",      # every intermediate region, written home to the pinned pool when produced
        "spill_mirror": "pressure_d2h",    # cache modes: cleaning copies of dirty granules, only while pressure holds the cleaning gate open (as Sirius's memory-monitor spills)
        "spill_evict": "pressure_d2h",     # cache modes: dirty granules a reclaim copies home
        "spill_rejected": "pressure_d2h",  # cache modes: the part of a cached region the cache refused
        "restore_h2d": "design_h2d",       # intermediate runs read back H2D (once per consumer)
        "restore_evicted_h2d": "restore_h2d",  # cache modes: cached regions' home pieces read back H2D (once per consumer), written by the three pressure kinds
        "restore_uva": "design_uva",       # intermediate rows read zero-copy
        "cache_hit": "design_d2d",         # cache modes: slot pieces read D2D
        "resident_d2d": "design_d2d",      # GPU regions outside the cache read D2D
        "cache_fill": "design_d2d",        # cache modes: granules written into slots
        "avoided_dead": "context",         # cache modes: reclaimed dead granules, no copy
        "avoided_clean": "context",        # cache modes: reclaimed clean granules, no copy
        "born": "materialized",            # cache_fill + spill_bypass + spill_rejected: each pipe output once
        "result_d2h": "result_d2h",        # the output step's copy of the result (outside the timer, as the step is)
        "output_read": "context",          # the output step's reads of the result columns
        "peak_gpu_arena": "peak_gpu",      # the allocator arena's high-water mark (whole 5 MiB blocks; cuDF and the slot slab included)
        "peak_host_intermediate": "peak_host",  # pinned-pool blocks held for intermediates: the columns' chunks, the unwritten rest of each open chunk included (up to 16 MiB per open column); whole 0.5 MiB blocks
        "peak_pipe_gpu": "context",        # the largest per-pipe arena peak (peak_gpu_bytes)
        "peak_host_pool": "context",       # the pinned pool over every tag, the scanned tables included
        "host_scan": "context",            # the pinned pool bytes the scanned tables held when the timer started
        "slot_slab": "context",            # cache modes: the slot pool's slab in the arena (0 without a cache)
        "peak_slot": "context",            # cache modes: the most slots in use at once (0 without a cache)
    },
}
# Every engine: the spill pass's own timed run (seconds = its wall time, count =
# result rows). Context only, never compared with all_results.csv.
COMMON_KINDS = {"pass_time": "context"}

# The headline numbers of a query (plan §2): sums of categories, logical bytes
# with wire beside them.
HEADLINES = {
    "off_gpu": ("design_d2h", "pressure_d2h", "pressure_d2disk"),
    "pressure_spill": ("pressure_d2h", "pressure_d2disk"),
    "disk_spill": ("pressure_h2disk", "pressure_d2disk"),
    "design_disk": ("design_disk_w",),
    "read_back": ("design_h2d", "design_uva", "restore_h2d", "restore_disk"),
}
CHECK_OPS = ("==", ">=", "<=")
_NUM = ("logical_b", "wire_b", "count", "seconds")


def kinds(engine):
    """{kind: category} every record of the engine must have, in CSV order."""
    return {**KINDS[engine], **COMMON_KINDS}


def category(engine, kind):
    return {**KINDS[engine], **COMMON_KINDS}[kind]


def row(kind, logical_b=None, wire_b=None, count=None, seconds=None, quality="OK", source=""):
    """One measurement. Bytes known only one way stay empty the other way (Spark
    shuffle files: wire only); uncompressed movement has wire_b = logical_b."""
    return {"kind": kind, "quality": quality, "logical_b": logical_b, "wire_b": wire_b,
            "count": count, "seconds": seconds, "source": source}


def na(kind, source=""):
    return row(kind, quality="NA", source=source)


def check(name, a, b, op="==", tol=0.0):
    """An identity the builder expects between two of its measurements; `check`
    (spill-smoke) fails on it. tol is relative."""
    return {"name": name, "a": a, "b": b, "op": op, "tol": tol}


def pass_result(pass_csv, num):
    """(run_status, seconds, rows) of query <num> in the runner's pass.csv, the
    last row of the query; FAIL with no time if it has none."""
    status, secs, rows = "FAIL", None, None
    try:
        with open(pass_csv) as f:
            for r in csv.DictReader(f):
                if r["query"] == f"query{num}":
                    status = r["status"]
                    secs = _float(r["seconds"])
                    v = r.get("result_rows_or_error") or ""
                    rows = int(v) if status == "OK" and v.isdigit() else None
    except OSError:
        pass
    return (status if status in RUN_STATUS else "FAIL"), secs, rows


def pass_time(pass_csv, num):
    """The pass_time row of query <num>, and its run_status."""
    status, secs, rows = pass_result(pass_csv, num)
    r = row("pass_time", count=rows, seconds=secs, source="pass.csv")
    if secs is None:
        r = na("pass_time", "pass.csv")
    return status, r


def record(engine, sf, num, run_status, rows, checks=(), error=None):
    """One records.jsonl line. Raises ValueError unless it satisfies the contract."""
    rec = {"engine": engine, "scale_factor": int(sf), "query": f"query{int(num)}",
           "run_status": run_status, "rows": list(rows), "checks": list(checks)}
    if error:
        rec["error"] = str(error)[:300]
    problems = validate(rec)
    if problems:
        raise ValueError(f"{engine} query{num}: " + "; ".join(problems))
    return rec


def stub(engine, sf, num, evidence_dir, error):
    """The record of a query whose builder failed: every kind NA, pass_time
    from pass.csv."""
    status, pt = pass_time(os.path.join(evidence_dir, "pass.csv"), num)
    rows = [na(k, "builder failed") for k in KINDS[engine]] + [pt]
    return record(engine, sf, num, status, rows, error=error)


def append_record(evidence_dir, rec):
    with open(os.path.join(evidence_dir, "records.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def validate(rec):
    """The contract violations of one record (empty if none)."""
    p = []
    engine = rec.get("engine")
    if engine not in KINDS:
        return [f"unknown engine {engine!r}"]
    if rec.get("run_status") not in RUN_STATUS:
        p.append(f"run_status {rec.get('run_status')!r}")
    if not re.fullmatch(r"query\d+", str(rec.get("query"))):
        p.append(f"query {rec.get('query')!r}")
    seen = [r.get("kind") for r in rec.get("rows", [])]
    dup = sorted({k for k in seen if seen.count(k) > 1})
    if dup:
        p.append(f"duplicate kinds {dup}")
    have, need = set(seen), set(kinds(engine))
    if need - have:
        p.append(f"missing kinds {sorted(need - have)}")
    if have - need:
        p.append(f"unknown kinds {sorted(have - need)}")
    for r in rec.get("rows", []):
        q = r.get("quality")
        if q not in QUALITY:
            p.append(f"{r.get('kind')}: quality {q!r}")
            continue
        vals = [r.get(f) for f in _NUM]
        bad = [f for f, v in zip(_NUM, vals) if v is not None
               and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0)]
        if bad:
            p.append(f"{r.get('kind')}: not a non-negative number: {bad}")
        if q == "NA" and any(v is not None for v in vals):
            p.append(f"{r.get('kind')}: NA with a value")
        if q != "NA" and all(v is None for v in vals):
            p.append(f"{r.get('kind')}: {q} without a value")
    for c in rec.get("checks", []):
        if c.get("op") not in CHECK_OPS or not c.get("name"):
            p.append(f"check {c!r}")
    return p


def check_result(c):
    """True / False, or None when a side is missing."""
    a, b, tol = c["a"], c["b"], c.get("tol") or 0.0
    if a is None or b is None:
        return None
    if c["op"] == "==":
        return abs(a - b) <= tol * max(abs(a), abs(b))
    if c["op"] == ">=":
        return a >= b * (1 - tol)
    return a <= b * (1 + tol)


def headline(engine, rows):
    """{name: (logical_b, wire_b, quality)} of one query's rows (dicts with kind,
    quality, logical_b, wire_b): each side summed over the parts that measured
    it. None where the engine has no kind in the categories (no such tier: "—",
    not 0). The quality is that of the logical side (of the wire side where no
    part has logical bytes): its parts' worst, and LOWER_BOUND when a part did
    not measure it (NA, or bytes not counted)."""
    rank = {q: i for i, q in enumerate(QUALITY)}
    by_kind = {r["kind"]: r for r in rows}
    out = {}
    for name, cats in HEADLINES.items():
        mine = [k for k, c in KINDS[engine].items() if c in cats]
        if not mine:
            out[name] = None
            continue
        parts = [by_kind.get(k) or na(k) for k in mine]
        has = {f: [r for r in parts if r["quality"] != "NA" and _float(r[f]) is not None]
               for f in ("logical_b", "wire_b")}
        sums = [sum(_float(r[f]) for r in has[f]) if has[f] else None for f in ("logical_b", "wire_b")]
        basis = has["logical_b"] or has["wire_b"]
        if not basis:
            out[name] = (None, None, "NA")
            continue
        worst = max((r["quality"] for r in basis), key=rank.get)
        if len(basis) < len(parts) and rank[worst] < rank["LOWER_BOUND"]:
            worst = "LOWER_BOUND"
        out[name] = (sums[0], sums[1], worst)
    return out


def _float(v):
    if v in (None, "", "NA"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cuda_bus_id():
    """The PCI bus id of CUDA's device 0 in this process, the engine's GPU:
    CUDA numbers the devices CUDA_VISIBLE_DEVICES leaves visible in its own
    order (CUDA_DEVICE_ORDER), nvidia-smi every GPU in PCI order, so
    an index can name another GPU. From the CUDA driver, without a context;
    None without a driver or a device."""
    import ctypes
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    dev, bus = ctypes.c_int(), ctypes.create_string_buffer(32)
    if (cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(dev), 0)
            or cuda.cuDeviceGetPCIBusId(bus, len(bus), dev)):
        return None
    return bus.value.decode()


def _gpu():
    """The GPU this process sees (CUDA's device 0, :func:`cuda_bus_id`; else
    the first of CUDA_VISIBLE_DEVICES), from nvidia-smi."""
    idx = cuda_bus_id() or (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    q = "name,memory.total,pci.bus_id,driver_version"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits",
                              "-i", idx], capture_output=True, text=True, timeout=30).stdout.strip()
        name, mem, bus, drv = [x.strip() for x in out.split(",")]
        return {"name": name, "memory_b": int(mem) << 20, "pci_bus_id": bus, "driver": drv}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def write_settings(evidence_dir, engine, sf, budgets, spill_dirs, configs=(), versions=None, **extra):
    """settings.json of one engine's pass: the GPU, host and disk budgets in
    bytes ({"gpu_b", "host_b", "disk_b"}, None where the engine has no such
    tier), the spill directories, a sha256 of each config file, versions, the
    GPU, the box and the date; extra keys as given."""
    s = {"engine": engine, "scale_factor": int(sf), "budgets": budgets,
         "spill_dirs": list(spill_dirs), "configs": {p: _sha256(p) for p in configs},
         "versions": versions or {}, "gpu": _gpu(), "box": socket.gethostname(),
         "date": datetime.date.today().isoformat(), "evidence": os.path.abspath(evidence_dir)}
    s.update(extra)
    with open(os.path.join(evidence_dir, "settings.json"), "w") as f:
        json.dump(s, f, indent=1)
    return s


def read_records(evidence_dir):
    """{query: record} of an evidence dir, the last line of each query."""
    recs = {}
    with open(os.path.join(evidence_dir, "records.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[r.get("query")] = r
    return recs


def _qnum(q):
    return int("".join(c for c in str(q) if c.isdigit()) or 0)


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}" if not v.is_integer() else str(int(v))
    return str(v)


def summarize(rec):
    """One line per record: status, headline GB, failed checks."""
    def gb(v):
        if v is None:
            return "—"
        b, wire = (v[0], "") if v[0] is not None else (v[1], " wire")
        if b is None:
            return "NA"
        size = f"{b / 1e9:.2f} GB" if b >= 1e9 else f"{b / 1e6:.1f} MB"
        return size + wire + ("" if v[2] == "OK" else f" ({v[2]})")
    h = headline(rec["engine"], rec["rows"])
    res = [check_result(c) for c in rec["checks"]]
    bad = [c["name"] for c, r in zip(rec["checks"], res) if r is False]
    return (f"{rec['query']:8} {rec['run_status']:11} off-GPU {gb(h['off_gpu'])} | "
            f"pressure {gb(h['pressure_spill'])} | disk {gb(h['disk_spill'])} | "
            f"checks {res.count(True)}/{len(res)} ok" + (f", FAILED: {bad}" if bad else "")
            + (f" | builder error: {rec['error']}" if rec.get("error") else ""))


def merge(engine, sf, evidence_dir, resdir):
    out_csv = os.path.join(resdir, "spill_profile.csv")
    recs = read_records(evidence_dir)
    problems = []
    for q, r in recs.items():
        if r.get("engine") != engine or int(r.get("scale_factor", -1)) != sf:
            problems.append(f"{q}: record of {r.get('engine')} sf{r.get('scale_factor')}")
        problems += [f"{q}: {x}" for x in validate(r)]
    if problems:
        sys.exit("spill_results.py merge: refused, " + "; ".join(problems))
    rows = []
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            rd = csv.DictReader(f)
            if rd.fieldnames != FIELDS:
                sys.exit(f"spill_results.py merge: refused, {out_csv} header {rd.fieldnames} != {FIELDS}")
            rows = [r for r in rd if not (r["engine"] == engine and int(r["scale_factor"]) == sf
                                          and r["query"] in recs)]
    for q, rec in recs.items():
        for r in rec["rows"]:
            rows.append({"engine": engine, "scale_factor": sf, "query": q, "kind": r["kind"],
                         "run_status": rec["run_status"], "quality": r["quality"],
                         **{f: _cell(r[f]) for f in _NUM}, "source": r.get("source", "")})

    def key(r):
        e = r["engine"]
        order = list({**KINDS.get(e, {}), **COMMON_KINDS})
        return (ENGINE_ORDER.index(e) if e in ENGINE_ORDER else len(ENGINE_ORDER),
                int(r["scale_factor"]), _qnum(r["query"]),
                order.index(r["kind"]) if r["kind"] in order else len(order), r["kind"])
    rows.sort(key=key)
    os.makedirs(resdir, exist_ok=True)
    tmp = out_csv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    os.replace(tmp, out_csv)

    sfile = os.path.join(evidence_dir, "settings.json")
    if os.path.exists(sfile):
        settings_json = os.path.join(resdir, "spill_settings.json")
        allset = {}
        if os.path.exists(settings_json):
            with open(settings_json) as f:
                allset = json.load(f)
        with open(sfile) as f:
            allset.setdefault(f"sf{sf}", {})[engine] = json.load(f)
        allset = {k: allset[k] for k in sorted(allset, key=_qnum)}
        with open(settings_json + ".tmp", "w") as f:
            json.dump(allset, f, indent=1, sort_keys=False)
            f.write("\n")
        os.replace(settings_json + ".tmp", settings_json)
    else:
        print(f"warning: no {sfile}; spill_settings.json not updated")
    for q in sorted(recs, key=_qnum):
        print(summarize(recs[q]))
    print(f"merged {engine} sf{sf}: {len(recs)} queries -> {out_csv} ({len(rows)} rows total)")


def check_dirs(dirs, expect_spill):
    """Validate and evaluate the checks of each evidence dir; True if all hold."""
    ok = True
    for d in dirs:
        try:
            recs = read_records(d)
        except OSError as e:
            print(f"{d}: no records ({e})")
            ok = False
            continue
        engine = next(iter(recs.values()))["engine"] if recs else "?"
        print(f"== {engine}: {d} ({len(recs)} queries)")
        spilled = False
        for q in sorted(recs, key=_qnum):
            r = recs[q]
            problems = validate(r)
            if problems:
                print(f"{q:8} CONTRACT: {problems}")
                ok = False
                continue
            print(summarize(r))
            if r["run_status"] != "OK" or r.get("error") or any(
                    check_result(c) is False for c in r["checks"]):
                ok = False
            h = headline(engine, r["rows"])
            spilled |= bool(h["pressure_spill"] and (h["pressure_spill"][0] or 0) > 0)
        if expect_spill and not spilled:
            print(f"{engine}: no query spilled under the forced-spill settings")
            ok = False
    return ok


def main():
    a = sys.argv[1:]
    cmd = a.pop(0) if a else ""
    if cmd == "merge" and len(a) in (3, 4):
        resdir = a[3] if len(a) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
        merge(a[0], int(a[1]), a[2], resdir)
    elif cmd == "check" and a:
        expect = "--expect-spill" in a
        sys.exit(0 if check_dirs([x for x in a if x != "--expect-spill"], expect) else 1)
    elif cmd == "stub" and len(a) >= 4:
        err = a[a.index("--error") + 1] if "--error" in a else "record builder failed"
        append_record(a[3], stub(a[0], a[1], a[2], a[3], err))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
