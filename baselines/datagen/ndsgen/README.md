# ndsgen — TPC-H straight to NDS-H parquet

`ndsgen` is the repo's TPC-H generator (`datagen/dataset.sh`, `$NDSGEN` in the
image). It writes the dataset the retired dbgen → Spark pipeline
(`datagen/gen_tpch.sh`) wrote (same rows, schema, file and row-group layout,
encodings and row shuffle) without dbgen text or Spark: SF1000 in 12.6 min
instead of 4.2 h on the 92-core box.

- **Rows**: [tpchgen-rs](https://github.com/clflushopt/tpchgen-rs) 3.0.0, the
  dbgen 2.17.3 port; chunk *i* of PARALLEL is `dbgen -C PARALLEL -S i`.
- **Batches**: BATCH chunks per batch, 200 part-files per table per batch,
  nation/region once (the batch holding chunk 1).
- **Row order**: Spark's CSV read of each batch is replayed from the .tbl line
  lengths: splits of `min(128 MB, max(4 MB, bytes / cores))`, a line in the
  split its first byte follows, splits sorted largest first and packed into
  map tasks, each task's rows scrambled and dealt round-robin from a random
  start, a part-file = its share of every task in task order. The task order
  is Spark's exactly; within a task the scramble is a seeded shuffle where
  Spark sorts by row hash.
- **Encoding**: parquet-mr 1.13 defaults (1 MB pages and dictionaries,
  20,000-row pages, 128 MB row groups, zstd level 3, OPTIONAL columns, the
  all-null `ignore` column) and parquet-mr's first-page dictionary check, which
  parquet-rs lacks: a dictionary that does not beat PLAIN on the first page
  is dropped for the chunk.
- **Formats**: uniform (`--compression`, `--dictionary`, `--writer-version`,
  the transcode knobs) or per column from the profiling pass's
  `format_map.json` (`--format-map … --engine …`).

```bash
make data SF=100                            # dataset.sh -> $NDSGEN --sf 100 ...
$NDSGEN --sf 100 --out <dir> --compression zstd                    # dict + zstd
$NDSGEN --sf 5 --out <dir> --format-map results/format_map.json --engine rapids
python compare.py <reference parquet dir> <ndsgen parquet dir>     # equivalence report
cargo build --release --locked             # outside the image (the image pins Rust in versions.env)
```

Output: `<out>/parquet/<table>/part-NNNNN-bBBBBB-c000.<codec>.parquet` (built in
`<out>/_parquet.partial`, renamed when complete) and `<out>/NDSGEN.json` (the
parameters; `results/validate_dataset.py` accepts a parquet-rs writer only next
to it). The output is deterministic for a given seed.

## Equivalence with the retired dbgen → Spark pipeline (compare.py, dict + zstd)

| SF | rows (count + row-hash sum) | files / row groups | dictionary decisions | bytes |
|---:|---|---|---|---|
| 1 | equal, all 8 tables | equal | equal, every column | ±1% per table |
| 10 | equal | equal | equal | ±0.6% |
| 100 | equal | equal (1,600 per table) | equal | ±0.9% (lineitem -0.03%) |
| 1000 | equal (6.0 B lineitem rows) | equal (16,000 per table) | equal | ±0.9% (lineitem -0.02%) |

Differences: Spark writes one extra empty file for nation and region (run.sh
drops 0-row files when it stages); `c_name` is ~5% larger and the all-null
`ignore` column ~15% smaller (both tiny); the writer string is parquet-rs.
Spark's map-task order is reproduced exactly (a supplier batch's 25 tasks run in
the same chunk order in both); within a task Spark's row-hash order becomes a
seeded shuffle.

Engines: `make smoke` (SF1, rapids, polars_gpu, duckdb_cpu, sirius) passes 22/22
with the reference row counts; DuckDB's TPC-H q1-22 at SF100 took 37.43 s on
ndsgen data vs 37.10 s on the old data (two runs each, alternating; run-to-run
spread 0.5-1.5%).

## Speed (dict + zstd, this box: 92-core cgroup quota, /dev/shm)

| SF | dbgen → Spark (retired) | ndsgen |
|---:|---:|---:|
| 1 | | 2.8 s |
| 100 | | 79 s |
| 1000 | 4 h 12 min (dbgen 2 h 06, Spark 2 h 03) | 12 min 37 s |

ndsgen is CPU-bound (93% of the quota at SF1000, ~2.4x stock tpchgen-cli's
CPU for the same SF: the row scramble's gathers and 200 files per batch).
