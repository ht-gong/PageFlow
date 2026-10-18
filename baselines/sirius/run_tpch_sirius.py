#!/usr/bin/env python
"""Run TPC-H queries 1-22 through Sirius (GPU) on the SF500 parquet via DuckDB.

Sirius (github.com/sirius-db/sirius) is a GPU-native SQL engine that loads as a
DuckDB extension and transparently intercepts plain SQL, running supported
operators on the GPU (cuDF/RMM/cuCascade) with out-of-core tiered spilling
(GPU -> pinned host -> disk). We drive its bundled `duckdb` binary, which has
the extension statically linked and auto-loading, via `-f <file.sql>`.

Mirrors the rapids/polars runners so the baselines are comparable:
  * reads the SAME marker-delimited query stream the rapids run used
    (results/queries/stream_qualification.sql, through results/queries/stream.py
    with q11's FRACTION for $TPCH_SF), so the SQL is identical;
  * the dataset is read from the ramdisk parquet (/dev/shm) as DuckDB views;
  * protocol (as every engine, see AGENTS.md; run_sirius.sh starts one process
    per query): an untimed warm pass first runs the query on the SF1 dataset in
    $WARM_PARQUET (GPU init, cuDF kernel JIT, reader setup; its output is
    discarded), then the views point at the target dataset and the query runs
    once, timed by the CLI's `.timer on` (engine startup excluded). An error
    anywhere in the process, the warm pass included, fails the query;
  * results are written incrementally to a CSV so a watchdog kill never loses
    prior rows. A companion *_detail.csv adds the GPU/fallback flag.

LOAD_MODE=resident (engine key sirius_resident, see AGENTS.md): before each
pass, untimed, the query's columns (pin_columns.py) are pinned in GPU memory
with `CALL pin_table(..., tier='gpu')`, and DuckDB CPU fallback is off. Sirius
serves a scan from a pin only if every column it projects was pinned, and falls
back to the parquet reader silently otherwise, so a timed run whose log lacks
an "assigned pinned entry" line for every pinned table is FAIL.

  run_tpch_sirius.py <parquet_dir> <stream.sql> <out_csv> [SUBSET] [append]
    SUBSET  comma-separated query numbers, e.g. "9" or "1,2,3"  (default 1..22)

Environment:
  SIRIUS_DUCKDB        path to the built duckdb binary
  SIRIUS_ENVLIB        its pixi env's lib dir (libcudf, rmm, cudart, ...); put on
                       the binary's LD_LIBRARY_PATH (its RPATH names it as well)
  SIRIUS_CONFIG_FILE   path to the gpu_execution YAML config (required by Sirius)
  WARM_PARQUET         the SF1 dataset of the warm pass
  TPCH_SF              the target scale factor (q11's FRACTION; the warm pass
                       runs the same text)
  SIRIUS_TIMEOUT       per-query subprocess timeout in seconds (default 1800)
  SIRIUS_LOG_DIR       if set, Sirius writes its spdlog file here; we grep it to
                       tell whether the query actually ran on GPU or fell back
                       (and, resident, whether the pins served its scans:
                       required then).
  LOAD_MODE            parquet (default) or resident
  SPILL_STATS_DIR      the spill pass (AGENTS.md): each query runs with its own
                       config (quent on, state machines only), a DEBUG log,
                       sirius_set_query_label() before each pass and /proc io
                       brackets around the timed run; its SQL, config, log,
                       quent output and stdout go to $SPILL_STATS_DIR/q<N>/,
                       which spill_stats.py turns into the query's record.
                       SPILL_SMOKE=1 adds the forced-spill budgets.
"""
import os
import re
import sys
import time
import glob
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "results", "queries"))
from stream import queries as stream_queries
from pin_columns import QUERY_COLUMNS

PARQUET = sys.argv[1].rstrip("/")
STREAM = sys.argv[2]
OUT_CSV = sys.argv[3]
SUBSET = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 and sys.argv[4] else list(range(1, 23))
APPEND = len(sys.argv) > 5 and sys.argv[5] == "append"

DUCKDB = os.environ["SIRIUS_DUCKDB"]
WARM = (os.environ.get("WARM_PARQUET") or sys.exit(
    "WARM_PARQUET is not set (the SF1 dataset of the warm pass)")).rstrip("/")
SF = os.environ.get("TPCH_SF") or sys.exit("TPCH_SF is not set (the target scale factor)")
TIMEOUT = int(os.environ.get("SIRIUS_TIMEOUT", "1800"))
LOG_DIR = os.environ.get("SIRIUS_LOG_DIR", "")
DETAIL_CSV = os.environ.get("SIRIUS_DETAIL_CSV", os.path.splitext(OUT_CSV)[0] + "_detail.csv")
SPILL_STATS_DIR = os.environ.get("SPILL_STATS_DIR", "")
if SPILL_STATS_DIR:
    import spill_stats
RESIDENT = os.environ.get("LOAD_MODE", "parquet") == "resident"
if RESIDENT and not LOG_DIR:
    sys.exit("LOAD_MODE=resident needs SIRIUS_LOG_DIR (the pin check reads the Sirius log)")

TABLES = ["customer", "lineitem", "nation", "orders",
          "part", "partsupp", "region", "supplier"]

RUN_MARK = "__SIRIUS_TIMED_RUN__"
# Resident: an untimed statement just before the timed run, whose QueryBegin
# line marks where the timed run's part of the Sirius log starts.
LOG_MARK = "sirius_timed_run_begin"
RUN_TIME_RE = re.compile(r"Run Time \(s\): real ([0-9]+\.[0-9]+)")
# stderr noise that must not be miscounted as CSV result rows. `mbind: Operation
# not permitted` is emitted whenever Sirius grows a NUMA-pinned host pool in a
# container without CAP_SYS_NICE (docker/run.sh grants it; `make doctor` checks).
NOISE_RE = re.compile(r"mbind:|Operation not permitted|terminate called|what\(\):|^\[[0-9]{4}-|^\s*$")


def view_sql(root):
    """CREATE OR REPLACE VIEW over the ramdisk parquet under root (one sub-dir of
    part-*.parquet per table, as produced by the NDS-H transcode pipeline)."""
    lines = []
    for t in TABLES:
        files = sorted(glob.glob(f"{root}/{t}/*.parquet")) or [f"{root}/{t}.parquet"]
        lst = ", ".join(f"'{f}'" for f in files)
        lines.append(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet([{lst}]);")
    return "\n".join(lines)


def pin_sql(root, num):
    """Resident: pin the query's columns of each table it reads, on the GPU. The
    glob expands to the same sorted file list as view_sql()'s, which is what lets
    the views' scans match the pin."""
    return "\n".join(
        f"CALL pin_table('{root}/{t}/*.parquet', tier='gpu', name='{t}', "
        f"cols=[{', '.join(repr(c) for c in cols)}]);"
        for t, cols in QUERY_COLUMNS[num].items())


def unpin_sql(num):
    return "\n".join(f"CALL unpin_table('{t}');" for t in QUERY_COLUMNS[num])


def build_sql(num, stmts):
    query = ";\n".join(stmts) + ";"
    # Spill pass: quent labels for both passes, and the duckdb process's /proc
    # io counters printed around the timed run (spill_stats.py). It skips the
    # resident engines (run.sh).
    spill = bool(SPILL_STATS_DIR)
    label = lambda p: [f"CALL sirius_set_query_label('{p}_q{num}');"] if spill else []
    procio = lambda t: [f".print PROCIO_{t}", ".shell cat /proc/$PPID/io"] if spill else []
    if not RESIDENT:
        parts = [view_sql(WARM),
                 # Warm pass: the same query on SF1, untimed (before `.timer on`
                 # and the run marker), its result rows discarded.
                 ".output /dev/null",
                 *label("warm"),
                 query,
                 ".output",
                 view_sql(PARQUET)]
    else:
        # The same, with the query's columns pinned on the GPU before each pass
        # (untimed), and no CPU fallback: a fallback would read the parquet.
        parts = ["SET enable_duckdb_fallback=false;",
                 view_sql(WARM),
                 ".output /dev/null",
                 pin_sql(WARM, num),
                 query,
                 unpin_sql(num),
                 view_sql(PARQUET),
                 pin_sql(PARQUET, num),
                 f"SELECT '{LOG_MARK}';",
                 ".output"]
    parts += [*procio("TIMED_BEGIN"),
              *label("timed"),
              ".mode csv",
              ".headers off",
              ".timer on",
              f".print {RUN_MARK}",
              query]
    if spill:
        parts += [".timer off", *procio("TIMED_END")]
    return "\n".join(parts) + "\n"


def parse_output(text):
    """The timed run, from the CLI's stdout: everything after the run marker.
    Returns (seconds, rows): the sum of its statements' 'Run Time' values (total
    wall time) and its CSV result rows (non-timer lines); None without a marker
    (the CLI stops at the first error, so a failed warm pass leaves none)."""
    secs, rows, seen = 0.0, 0, False
    for line in text.splitlines():
        if line.startswith(RUN_MARK):
            seen = True
            continue
        if not seen:
            continue
        if line.startswith("PROCIO_TIMED_END"):     # the spill pass's closing bracket
            break
        m = RUN_TIME_RE.search(line)
        if m:
            secs += float(m.group(1))
        elif line.strip() and not NOISE_RE.search(line):
            rows += 1                     # a CSV result row of the timed run
    return (secs, rows) if seen else None


def log_segment(log_before):
    """The part of the newest Sirius log written during this run (Sirius starts
    a new, dated file each day; then the whole new file belongs to this run)."""
    log = newest_log()
    if not log:
        return None
    start = log_before[1] if log == log_before[0] else 0
    try:
        with open(log, errors="ignore") as f:
            return f.read()[start:]
    except OSError:
        return None


def pins_unserved(num, log_before):
    """Resident: the pinned tables whose pin served none of the timed run's
    scans (a scan that falls back reads the parquet instead), or every table if
    the timed run's part of the log is missing. Sirius logs a served scan, and
    an error for a pin that matched but could not serve, but nothing for a pin
    skipped for a missing column; so a table scanned twice passes when either
    scan is served. pin_columns.py covering every scan closes that gap."""
    tables = set(QUERY_COLUMNS[num])
    txt = log_segment(log_before)
    i = txt.rfind(LOG_MARK) if txt else -1
    if i < 0:
        return sorted(tables)
    timed = txt[i:]
    served = set(re.findall(r"assigned pinned entry '(\w+)'", timed))
    if "error while trying to assign cached entries" in timed:
        served = set()
    return sorted(tables - served)


def newest_log():
    logs = sorted(glob.glob(os.path.join(LOG_DIR, "sirius*.log")), key=os.path.getmtime) if LOG_DIR else []
    return logs[-1] if logs else None


def gpu_or_fallback(log_before):
    """Best-effort: inspect the newest Sirius log written during this run to see
    whether Sirius handled the query on GPU or DuckDB CPU fallback kicked in.
    log_before = (path, size) of the newest log before the run; Sirius starts a
    new (dated) file each day, and then the whole new file belongs to this run."""
    txt = log_segment(log_before)
    if txt is None:
        return "?"
    low = txt.lower()
    # A real CPU fallback shows up as a scan/plan error that Sirius drains and
    # hands back to DuckDB. Do NOT match the substring "fallback" alone -- the
    # init line "pinned memory resource configured (... fallback node=0)" is
    # NUMA config noise, not an execution fallback.
    if "draining after error" in low or "error executing query" in low:
        return "fallback"
    # Count GPU vs total query completions in this segment. Transparent GPU runs
    # log "Transparent GPU execution: query completed" per GPU-executed statement.
    gpu_done = low.count("transparent gpu execution: query completed")
    if gpu_done > 0:
        return "gpu"
    return "?"


def spill_setup(num, env):
    """Spill pass: the query's evidence dir, config (quent on) and DEBUG log;
    returns the SQL file's path."""
    global LOG_DIR
    qdir = os.path.join(SPILL_STATS_DIR, f"q{num}")
    shutil.rmtree(qdir, ignore_errors=True)
    for d in ("log", "quent"):
        os.makedirs(os.path.join(qdir, d))
    cfg = os.path.join(qdir, "sirius.yaml")
    spill_stats.write_config(os.environ["SIRIUS_CONFIG_FILE"], cfg, os.path.join(qdir, "quent"),
                             smoke=os.environ.get("SPILL_SMOKE") == "1")
    LOG_DIR = os.path.join(qdir, "log")
    env.update(SIRIUS_CONFIG_FILE=cfg, SIRIUS_LOG_DIR=LOG_DIR, SIRIUS_LOG_LEVEL="debug",
               # no NVTX capture: it writes ~50 MB per process and costs 15-25 %
               NVTX_INJECTION64_PATH="/nonexistent/nvtx-off.so")
    return os.path.join(qdir, f"q{num}.sql")


def keep_output(num, stdout, stderr):
    """Spill pass: the CLI's output, for the /proc brackets."""
    qdir = os.path.join(SPILL_STATS_DIR, f"q{num}")
    for name, text in (("stdout.txt", stdout), ("stderr.txt", stderr)):
        if isinstance(text, bytes):
            text = text.decode(errors="replace")
        with open(os.path.join(qdir, name), "w") as f:
            f.write(text or "")


def run_query(num, stmts):
    sql = build_sql(num, stmts)
    env = dict(os.environ)
    tmp = spill_setup(num, env) if SPILL_STATS_DIR else f"/tmp/sirius_q{num}.sql"
    with open(tmp, "w") as f:
        f.write(sql)
    log = newest_log()
    log_before = (log, os.path.getsize(log) if log else 0)
    if os.environ.get("SIRIUS_ENVLIB"):
        env["LD_LIBRARY_PATH"] = os.environ["SIRIUS_ENVLIB"]
    cmd = [DUCKDB, "-f", tmp]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=TIMEOUT, env=env)
        out = p.stdout + "\n" + p.stderr
    except subprocess.TimeoutExpired as e:
        if SPILL_STATS_DIR:
            keep_output(num, e.stdout, e.stderr)
        out = (e.stdout or "") + "\n" + (e.stderr or "") if isinstance(e.stdout, str) else ""
        return dict(status="TIMEOUT", secs=time.time() - t0, rows="killed_timeout", gpu="?")
    if SPILL_STATS_DIR:
        keep_output(num, p.stdout, p.stderr)
    run = parse_output(p.stdout)
    err = None
    for pat in ("Error:", "Invalid Error", "IO Error", "Catalog Error", "Parser Error",
                "Binder Error", "Out of Memory", "std::bad_alloc", "CUDA", "RMM error"):
        m = re.search(rf"^.*{re.escape(pat)}.*$", out, re.MULTILINE)
        if m:
            err = m.group(0).strip()[:120].replace(",", ";")
            break
    if run is None:
        return dict(status="FAIL", secs=time.time() - t0, rows=(err or "no_output")[:120], gpu="?")
    secs, rows = run
    if err and (p.returncode != 0):
        return dict(status="FAIL", secs=secs, rows=err, gpu="?")
    if RESIDENT:
        unserved = pins_unserved(num, log_before)
        if unserved:
            return dict(status="FAIL", secs=secs, rows="pin_not_served:" + "+".join(unserved), gpu="?")
    return dict(status="OK", secs=secs, rows=rows, gpu=gpu_or_fallback(log_before))

def main():
    print(f"Sirius run | duckdb={DUCKDB}")
    print(f"config={os.environ.get('SIRIUS_CONFIG_FILE','<none>')} | warm={WARM} | timeout={TIMEOUT}s"
          f" | load={'resident (GPU pins)' if RESIDENT else 'parquet'}")
    if not os.path.exists(DUCKDB):
        sys.exit(f"ERROR: duckdb binary not found at {DUCKDB} (build Sirius first)")
    queries = stream_queries(SF, STREAM)

    write_header = not (APPEND and os.path.exists(OUT_CSV))
    f = open(OUT_CSV, "a" if APPEND else "w")
    fd = open(DETAIL_CSV, "a" if (APPEND and os.path.exists(DETAIL_CSV)) else "w")
    if write_header:
        f.write("query,status,seconds,result_rows_or_error\n"); f.flush()
        fd.write("query,status,seconds,rows,gpu\n"); fd.flush()

    print(f"\n{'query':10} {'status':8} {'secs':>9} {'rows':>9}  gpu")
    ok = 0
    for n in SUBSET:
        if n not in queries:
            continue
        name = f"query{n}"
        r = run_query(n, queries[n])
        print(f"{name:10} {r['status']:8} {r['secs']:9.2f} {str(r['rows']):>9}  {r['gpu']}")
        f.write(f"{name},{r['status']},{r['secs']:.3f},{r['rows']}\n"); f.flush()
        fd.write(f"{name},{r['status']},{r['secs']:.3f},{r['rows']},{r['gpu']}\n"); fd.flush()
        if r["status"] == "OK":
            ok += 1
    f.close(); fd.close()
    print(f"\n{ok}/{len([n for n in SUBSET if n in queries])} queries OK | csv -> {OUT_CSV}")
    sys.exit(0 if ok == len([n for n in SUBSET if n in queries]) else 1)

if __name__ == "__main__":
    main()
