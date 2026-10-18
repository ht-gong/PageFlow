"""Shared loop of the column-probe workers (probe_<engine>.py).

A worker is one engine process that probes a list of columns, in the given
order, with `SELECT min(c), max(c) FROM <table>`: the query scans and decodes
that one column and does almost nothing else. Engine startup and warm-up happen
before the loop and are excluded from the timings.

  probe_<engine>.py <parquet_dir> <columns_file> <out_csv>

<columns_file> has one "<table> <column>" per line. First the worker runs the
PRIME probes untimed: the first probe of a process carries one-time reader
setup cost, so it is absorbed by one small-table column per physical type
(int64, string, int32, decimal, date). Then, for each listed column, it prints
"PROBE-START <table>.<column>" (the driver, format_profile.py, times out and
restarts on it), runs the probe and appends one row to <out_csv>:
table,column,status,seconds,detail with status OK | FAIL | FALLBACK (the engine
left its native execution path, e.g. a GPU engine ran the probe on the CPU).
A worker whose engine process died stops with "PROBE-ABORT" and exit code 3;
the driver restarts it on the columns not yet probed. PROBE_NO_PRIME=1 skips
priming (the driver sets it after a priming probe timed out).
"""
import csv
import os
import sys

TABLES = ["customer", "lineitem", "nation", "orders",
          "part", "partsupp", "region", "supplier"]
PRIME = [("region", "r_regionkey"), ("region", "r_name"), ("part", "p_size"),
         ("part", "p_retailprice"), ("orders", "o_orderdate")]


class Stop(Exception):
    """The engine process is gone; the remaining columns need a new worker."""


def args():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    parquet, cols_file, out_csv = sys.argv[1:4]
    with open(cols_file) as f:
        cols = [tuple(line.split()) for line in f if line.strip()]
    return parquet.rstrip("/"), cols, out_csv


def probe_sql(table, column):
    return f"SELECT min({column}), max({column}) FROM {table}"


def _call(probe, table, column):
    try:
        return probe(table, column)
    except Stop:
        raise
    except Exception as e:  # engine error: this format is ineligible
        return "FAIL", None, (str(e).splitlines() or [type(e).__name__])[0][:160]


def run(cols, out_csv, probe):
    """probe(table, column) -> (status, seconds, detail); runs PRIME, then cols."""
    try:
        for table, column in ([] if os.environ.get("PROBE_NO_PRIME") == "1" else PRIME):
            print(f"PRIME-START {table}.{column}", flush=True)
            status, secs, detail = _call(probe, table, column)
            print(f"PRIME-END {table}.{column} {status} {secs} {detail}", flush=True)
    except Stop as e:
        print(f"PROBE-ABORT during priming: {e}", flush=True)
        sys.exit(3)
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        for table, column in cols:
            print(f"PROBE-START {table}.{column}", flush=True)
            try:
                status, secs, detail = _call(probe, table, column)
            except Stop as e:
                print(f"PROBE-ABORT {table}.{column} {e}", flush=True)
                sys.exit(3)
            w.writerow([table, column, status, "" if secs is None else f"{secs:.4f}", detail])
            f.flush()
            print(f"PROBE-END {table}.{column} {status} "
                  f"{'-' if secs is None else f'{secs:.4f}'} {detail}", flush=True)
