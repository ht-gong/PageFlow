#!/usr/bin/env python
"""Upsert ONE engine-run's per-query CSV into results/all_results.csv.

  merge_results.py <engine> <scale_factor> <run_csv> [results_dir]

<run_csv> is the native output of any run_* script — columns
`query,status,seconds,[gpu_ops,]result_rows_or_error`. Each of its rows replaces
the row of the same (engine, scale_factor, query); every other row is kept, so a
run of a query subset updates only those queries (a query listed twice keeps its
last row). `all_results.csv` is the single canonical results table; runners
write a throwaway temp CSV and call this, so no per-run/per-SF CSVs are kept
around. Idempotent; atomic write.
"""
import csv, os, sys

ENGINE_ORDER = ["duckdb_cpu", "sirius", "polars_gpu", "rapids", "tqp_vortex",
                "duckdb_cpu_resident", "sirius_resident", "polars_gpu_resident", "rapids_resident"]
FIELDS = ["engine", "scale_factor", "query", "status", "seconds", "rows_or_error"]


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: merge_results.py <engine> <scale_factor> <run_csv> [results_dir]")
    engine, sf, run_csv = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    resdir = sys.argv[4] if len(sys.argv) > 4 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results")
    allcsv = os.path.join(resdir, "all_results.csv")

    run = {}
    for r in csv.DictReader(open(run_csv)):
        run[r["query"]] = {"engine": engine, "scale_factor": sf, "query": r["query"],
                           "status": r["status"], "seconds": r["seconds"],
                           "rows_or_error": r.get("result_rows_or_error", "")}

    # keep every existing row except the (engine, sf, query) rows of this run
    rows = []
    if os.path.exists(allcsv):
        for r in csv.DictReader(open(allcsv)):
            if r["engine"] == engine and int(r["scale_factor"]) == sf and r["query"] in run:
                continue
            rows.append(r)
    rows += run.values()

    def key(r):
        e = ENGINE_ORDER.index(r["engine"]) if r["engine"] in ENGINE_ORDER else len(ENGINE_ORDER)
        q = int("".join(c for c in r["query"] if c.isdigit()) or 0)
        return (e, int(r["scale_factor"]), q)
    rows.sort(key=key)

    os.makedirs(resdir, exist_ok=True)
    tmp = allcsv + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in FIELDS})
    os.replace(tmp, allcsv)
    print(f"merged {engine} sf{sf}: {len(run)} rows -> {allcsv} ({len(rows)} total)")


if __name__ == "__main__":
    main()
