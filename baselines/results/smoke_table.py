#!/usr/bin/env python3
"""Print and check the query x engine table of a smoke run (run.sh with
RUN_CSV_DIR: `make smoke`, and `make smoke-all` at the TQP-Vortex root).

  smoke_table.py <run_csv_dir> <engines> <sfs> [queries]

Reads <run_csv_dir>/<engine>_sf<SF>.csv and prints, per SF, each query's row
count on each engine, or its status and error. Exits 1 unless every query is OK
on every engine with one row count, at SF1 the reference count below (a count
that differs from it shows as `<rows>!=<reference>`).
"""
import csv
import os
import sys

# q1-q22 at SF1, q11 with SF1's FRACTION (AGENTS.md lists the same).
SF1_ROWS = dict(enumerate([4, 100, 10, 5, 5, 1, 4, 2, 175, 20, 1048, 2, 42, 1, 1,
                           18314, 1, 57, 1, 186, 100, 7], start=1))


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    d, engines, sfs = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
    queries = (sys.argv[4] if len(sys.argv) > 4 else "").split() or [str(q) for q in range(1, 23)]
    bad = 0
    for sf in sfs:
        cell = {}
        for e in engines:
            p = os.path.join(d, f"{e}_sf{sf}.csv")
            for r in (csv.DictReader(open(p)) if os.path.exists(p) else []):
                v = r["result_rows_or_error"]
                cell[(r["query"], e)] = v if r["status"] == "OK" else f"{r['status']}:{v[:20]}"
        print(f"SF{sf}  " + "".join(f"{e:>14}" for e in engines))
        for q in queries:
            row = [cell.get((f"query{q}", e), "MISSING") for e in engines]
            if sf == "1":
                ref = str(SF1_ROWS[int(q)])
                row = [f"{v}!={ref}" if v.isdigit() and v != ref else v for v in row]
            differ = len({v for v in row if v.isdigit()}) > 1
            bad += differ or not all(v.isdigit() for v in row)
            print(f"query{q:<4}" + "".join(f"{v:>14}" for v in row) + ("  counts differ" if differ else ""))
    print("SMOKE OK" if bad == 0 else f"SMOKE FAILED: {bad} queries not OK everywhere or with wrong row counts")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
