#!/usr/bin/env python3
"""Usage: run_tpch_duckdb.py <parquet_dir> <stream.sql> <out_csv> [queries] [append]

DuckDB CPU baseline over the shared TPC-H parquet dataset.

Every table is a view over read_parquet() of the ramdisk copy, and DuckDB's
external file cache is off, so every query reads and decodes its parquet from
the ramdisk; nothing is loaded into memory ahead of the timed run.

LOAD_MODE=resident (engine key duckdb_cpu_resident, see AGENTS.md) loads every
table into a native DuckDB table instead, untimed, before each pass: the timed
run reads no parquet.

Protocol (as every engine, see AGENTS.md; run_duckdb.sh starts one process per
query): an untimed warm pass first runs the same queries on the SF1 dataset in
$WARM_PARQUET, then the views point at <parquet_dir> and each query runs once,
timed (engine startup excluded). A query that fails in the warm pass is
recorded FAIL and not timed. The queries come from results/queries/stream.py,
with q11's FRACTION for $TPCH_SF.

Emits the merge_results.py input schema -- `query,status,seconds,
result_rows_or_error`, one row per query -- so run_duckdb.sh can fold it into
results/all_results.csv.
"""
import csv
import os
import re
import sys
import time

import duckdb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "queries"))
from stream import queries as stream_queries


TABLES = ["region", "nation", "supplier", "customer",
          "part", "partsupp", "orders", "lineitem"]


def create_views(con, parquet):
    """One view per table over read_parquet() of the parquet dataset (replacing
    any earlier view of that name)."""
    base = parquet.rstrip("/")
    for t in TABLES:
        con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{base}/{t}/*.parquet')")


def create_tables(con, parquet):
    """Resident mode: one native in-memory table per table, loaded from the
    parquet dataset (replacing any earlier table of that name)."""
    base = parquet.rstrip("/")
    t0 = time.time()
    for t in TABLES:
        con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_parquet('{base}/{t}/*.parquet')")
    print(f"loaded {len(TABLES)} tables from {base} in {time.time() - t0:.1f}s "
          f"(excluded from query timings)", flush=True)


def connect():
    """In-memory connection with the benchmark settings; returns (con, threads)."""
    con = duckdb.connect()
    # honour cgroup/taskset affinity -- os.cpu_count() reports the whole box and
    # would oversubscribe when the container is pinned to a subset.
    default_threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    threads = int(os.environ.get("DUCKDB_THREADS", default_threads))
    con.execute(f"PRAGMA threads={threads}")
    mem = os.environ.get("DUCKDB_MEMORY_LIMIT")
    if mem:
        con.execute(f"PRAGMA memory_limit='{mem}'")
    if os.environ.get("DUCKDB_TEMP_DIR"):
        con.execute(f"PRAGMA temp_directory='{os.environ['DUCKDB_TEMP_DIR']}'")
    # no in-memory copy of parquet file bytes across queries: every query reads
    # the ramdisk
    con.execute("SET enable_external_file_cache = false")
    return con, threads


def run_query(con, stmts):
    # return final select row count
    rows = 0
    for stmt in stmts:
        res = con.execute(stmt)
        code = re.sub(r"(?m)^\s*--.*\n?", "", stmt).strip().lower()
        if code.startswith(("select", "with")):
            rows = len(res.fetchall())
        else:
            res.fetchall()
    return rows


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)

    # get args
    parquet, stream, out_csv = sys.argv[1:4]
    subset = [int(q) for q in (sys.argv[4].replace(",", " ").split()
                               if len(sys.argv) > 4 and sys.argv[4] else range(1, 23))]
    append = len(sys.argv) > 5 and sys.argv[5] == "append"
    warm = os.environ.get("WARM_PARQUET") or sys.exit(
        "WARM_PARQUET is not set (the SF1 dataset of the warm pass)")
    sf = os.environ.get("TPCH_SF") or sys.exit("TPCH_SF is not set (the target scale factor)")

    resident = os.environ.get("LOAD_MODE", "parquet") == "resident"
    load = create_tables if resident else create_views

    con, threads = connect()
    print(f"duckdb {duckdb.__version__} sf={sf} threads={threads} "
          f"{'tables loaded from' if resident else 'views over'} {parquet} queries={subset}", flush=True)
    queries = stream_queries(sf, stream)

    # Warm pass: the same queries on SF1, untimed (q11 keeps the target SF's
    # FRACTION, so the warm pass runs the timed run's text).
    load(con, warm)
    warm_failed = {}
    t0 = time.time()
    for q in subset:
        tq = time.time()
        try:
            run_query(con, queries[q])
        except Exception as e:
            warm_failed[q] = (time.time() - tq, str(e).splitlines()[0][:120])
    print(f"SF1 warm pass done in {time.time() - t0:.1f}s (excluded from query timings)", flush=True)
    load(con, parquet)

    write_header = not (append and os.path.exists(out_csv))
    with open(out_csv, "a" if append else "w", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["query", "status", "seconds", "result_rows_or_error"])

        for q in subset:
            if q in warm_failed:
                dt, msg = warm_failed[q]
                w.writerow([f"query{q}", "FAIL", f"{dt:.3f}", f"SF1 warm pass: {msg}"])
                print(f"query{q:<3d} FAIL SF1 warm pass: {msg}", flush=True)
                f.flush()
                continue
            try:
                t0 = time.time()
                rows = run_query(con, queries[q])
                dt = time.time() - t0
                w.writerow([f"query{q}", "OK", f"{dt:.3f}", rows])
                print(f"query{q:<3d} OK   {dt:8.3f}s rows={rows}", flush=True)
            except Exception as e:
                msg = str(e).splitlines()[0][:120]
                w.writerow([f"query{q}", "FAIL", "NA", msg])
                print(f"query{q:<3d} FAIL {msg}", flush=True)
            f.flush()

if __name__ == "__main__":
    main()
