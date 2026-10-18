#!/usr/bin/env python3
"""duckdb_cpu column probes (see probe_common.py). The runner's connection and
views over read_parquet() (DuckDB's external file cache off), so every probe
reads its column from the parquet files."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "duckdb"))
import probe_common as pc  # noqa: E402
from run_tpch_duckdb import connect, create_views  # noqa: E402


def main():
    parquet, cols, out_csv = pc.args()
    con, threads = connect()
    create_views(con, parquet)
    import duckdb
    print(f"duckdb {duckdb.__version__} threads={threads}", flush=True)

    def probe(table, column):
        t0 = time.time()
        con.execute(pc.probe_sql(table, column)).fetchall()
        return "OK", time.time() - t0, ""

    pc.run(cols, out_csv, probe)


if __name__ == "__main__":
    main()
