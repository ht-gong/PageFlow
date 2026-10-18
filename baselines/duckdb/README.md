# DuckDB CPU Baseline

`duckdb_cpu` — DuckDB as the CPU reference point against the GPU engines
(Sirius, Polars-GPU, Spark-RAPIDS). Same TPC-H parquet dataset,
same `results/queries/stream_qualification.sql`, same results contract as every
other engine in this repo (see `AGENTS.md`).

## Running

```bash
# all 22 queries at SF500, merged into results/all_results.csv
TPCH_SF=500 TPCH_PARQUET=/root/tpc-h/sf500_parquet ./duckdb/run_duckdb.sh

# a query subset
TPCH_SF=100 TPCH_PARQUET=/dev/shm/tpch_sf100/parquet ./duckdb/run_duckdb.sh "1 6 9"

# keep the temp CSV instead of merging
TPCH_MERGE=0 OUT_CSV=/tmp/duckdb.csv ./duckdb/run_duckdb.sh
```

The runner self-merges into `results/all_results.csv` via `merge_results.py` and
deletes its temp CSV — no per-run/per-SF CSVs, same as the other engines.

## Measurement protocol

- **Data:** every table is a view over `read_parquet()` of the ramdisk copy, and
  DuckDB's external file cache is off, so every run reads and decodes its parquet
  from the ramdisk, like the GPU engines. Nothing is loaded into memory ahead of
  a query (no in-memory tables under this key).
- **Resident mode** (key `duckdb_cpu_resident`, `LOAD_MODE=resident`, set by
  `run.sh`): before each pass, untimed, every table is loaded into a native
  in-memory DuckDB table, so the timed run reads no parquet (`AGENTS.md`).
- **Timing:** the protocol of every engine (`AGENTS.md`). `run_duckdb.sh` starts
  one process per query; it first runs the query on the SF1 ramdisk copy
  (`WARM_PARQUET`), untimed, then once on the target dataset, timed.

Rows in `all_results.csv` from before these changes were measured on in-memory
tables (`tables` mode) after three warm runs in one process, and are not
comparable to new runs.

Other env knobs: `DUCKDB_THREADS` (defaults to the process CPU affinity, not the
whole box), `DUCKDB_MEMORY_LIMIT`, `DUCKDB_TEMP_DIR` (spill location), `STREAM`
(another query stream; its q11 FRACTION must be the placeholder `:FRACTION`,
which `results/queries/stream.py` fills in for the scale factor).

## Output

One row per query, folded into `results/all_results.csv`:

```csv
engine,scale_factor,query,status,seconds,rows_or_error
duckdb_cpu,500,query1,OK,10.226,4
```

## Getting parquet

Only from the repo's generator (`datagen/`, see `results/GENERATOR.md`);
`run.sh` generates and validates missing datasets.
