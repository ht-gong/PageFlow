# datagen — the TPC-H generator

The parquet every engine in this repo benchmarks is produced here, by
`ndsgen/` (Rust): TPC-H rows from tpchgen-rs (the dbgen 2.17.3 port), written
in the NDS-H layout of NVIDIA `spark-rapids-benchmarks` (the dbgen → Spark
transcode pipeline this directory used to run): same schema, same files and row
groups, same encodings, same row shuffle. SF1000 takes ~13 minutes on the
92-core box (the dbgen + Spark pipeline took 4.2 hours).

| file | purpose |
|------|---------|
| `dataset.sh <SF>` | `make data SF=…`: generate `$DATA_DIR/sf<SF>` with ndsgen if missing (PARALLEL = 2·SF, at least 20; BATCH 25), then validate. `run.sh` calls it too |
| `ndsgen/` | the generator (`$NDSGEN` in the image); its README has the design, the knobs and the equivalence evidence; `compare.py` checks a dataset against another build |
| `env.sh` | `source` it for the Spark toolchain (`$PY`, `$JAVA_HOME`, `$SPARK_HOME` from the image; the RAPIDS runner uses it) |

The image builds ndsgen with the pinned Rust toolchain (`versions.env`) from
`ndsgen/Cargo.lock`; its sources are part of the image tag, so an edit to them
needs `make image`.

```bash
make data SF=100            # -> $DATA_DIR/sf100/parquet/<table>/, $DATA_DIR/sf100/NDSGEN.json
make validate SF=100
GEN_OPTS="--compression zstd" make data SF=1000     # another uniform format
```

## What a dataset is

`ndsgen --sf SF --parallel PARALLEL --batch BATCH` produces what the NDS-H
pipeline produced from `dbgen -s SF -C PARALLEL -S i` chunks transcoded BATCH at
a time by Spark (`repartition(200)`, parquet-mr defaults):

1. **rows** — chunk *i* of PARALLEL, as dbgen's `-C/-S`; NDS-H types: keys
   `bigint`, money/quantity `DECIMAL(11,2)`, `p_size` / `ps_availqty` /
   `l_linenumber` / `o_shippriority` `int`, dates `date`, text `string`, plus a
   trailing all-null `ignore` column (dbgen's terminating `|`); all columns
   nullable.
2. **files** — 200 part-files per table per batch of BATCH chunks, so files per
   table = 200 × batches; nation/region once. One row group per file at these
   sizes (128 MB row groups).
3. **row order** — Spark's read + round-robin repartition replayed per batch, so
   rows leave dbgen's key order the way they did under Spark (ndsgen's README).
4. **encoding** — snappy, dictionary with parquet-mr's PLAIN fallback, v1 pages
   by default; `GEN_OPTS` (`--compression`, `--dictionary`, `--writer-version`)
   picks another uniform format, and the profiling pass's `format_map.json`
   (`--format-map … --engine …`) a format per column.

`dataset.sh` generates into `<out>/_gen/`, validates it with
`results/validate_dataset.py` (writer, in-spec `p_brand`, row counts) and only
then moves `parquet/` and `NDSGEN.json` into `$DATA_DIR/sf<SF>/`, so a
half-built dataset never looks finished. The output is deterministic for a seed.

## Provenance

On 2026-10-13 ndsgen was checked against the dbgen + Spark pipeline it replaces
(`ndsgen/compare.py`, dict + zstd) at SF1, SF10, SF100 and SF1000: the same rows
(count and an order-independent row-hash sum, every table), schema, physical
types, file and row-group counts and dictionary decisions, bytes within 1% per
table; all four baseline engines pass `make smoke` on it, and DuckDB runs TPC-H
at SF100 within 1% of its time on the old data. That pipeline
(`datagen/gen_tpch.sh`, `gen_tpch.py`: dbgen → Spark CSV read → parquet-mr) was
itself checked against `spark-rapids-benchmarks@efcfa3f` at SF1/10/100 on
2026-09-22/23; it was removed after the ndsgen check, and the datasets it wrote
stay valid.
