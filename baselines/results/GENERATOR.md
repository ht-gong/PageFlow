# Data generator — one source for every engine

**Invariant: every engine runs the same rows, produced by the NDS-H
generator.** A cross-engine number is only meaningful if all engines read the
same data; each baseline engine reads it in the per-column format its
profiling map chose (`AGENTS.md`, "Before every benchmark"), and TQP-Vortex,
which converts columns into its own format, reads any of them. This document defines the one allowed generator, records a
provenance audit that found the invariant had been violated, and gives the guard
(`validate_dataset.py`) that now enforces it.

## The one allowed generator

TPC-H parquet is produced **only** by the repo's `datagen/ndsgen` (`make data`,
`datagen/dataset.sh`): TPC-H rows from tpchgen-rs 3.0.0, the dbgen 2.17.3 port,
written in the NDS-H (NVIDIA `spark-rapids-benchmarks`) format — the schema,
files, row groups, encodings and row shuffle of NDS-H's dbgen → Spark 3.5.8
transcode, one 17-column table per dir with a trailing `ignore` column absorbing
dbgen's dangling `|` (`datagen/README.md`, `datagen/ndsgen/README.md`). It leaves
`NDSGEN.json` next to `parquet/`.

It replaced the repo's re-implementation of that pipeline (`datagen/gen_tpch.sh`:
dbgen → Spark, writer `parquet-mr 1.13.1`) after `datagen/ndsgen/compare.py`
found the same rows, schema and layout at SF1/10/100/1000 (2026-10-13). Datasets
that pipeline wrote hold the same rows and remain valid. Output layout — the
layout every runner reads:

```
<out>/parquet/<table>/part-NNNNN-bBBBBB-c000.<codec>.parquet     (ndsgen)
<out>/parquet/<table>/part-NNNNN-<uuid>-c000.<codec>.parquet     (the retired Spark transcode)
```

**Forbidden:** parquet written by any other tool — in particular DuckDB's own
`CALL dbgen(...)` export (16 columns, no `ignore`, writer `DuckDB …`) and cudf
exports (writer `cudf …`). Self-generated data is not comparable to the external
standard and has produced materially different query results (below). Do not
point a runner at self-written parquet, even as a convenience.

TQP-Vortex (`tqp_vortex`) has no generator of its own: it reads this parquet.

## Provenance audit (2026-07-25)

Auditing the datasets behind `all_results.csv` found the invariant broken: only
**SF500** was clean external NDS-H. The other scale factors had run on
self-generated or corrupt parquet.

| SF | dataset used | writer | verdict |
|---:|--------------|--------|---------|
| 30 | `sf30_parquet` | cudf 24.10 | ✗ wrong generator |
| 50 | `sf50_parquet` | DuckDB v1.5.4 | ✗ self-generated |
| 100 | `sf100_canon_parquet` | DuckDB v1.5.4 | ✗ self-generated |
| 300 | `sf300_canon_parquet` | DuckDB v1.5.4 | ✗ self-generated |
| 500 | `sf500_parquet` | parquet-mr 1.13.1 | ✓ clean NDS-H |
| 700 | `sf700_parquet` | parquet-mr 1.13.1 | ✓ clean NDS-H (DuckDB-only; >int32 for the GPU engines) |

Two distinct defects:

- **Wrong generator (SF30/50/100/300).** These ran on cudf- or DuckDB-written
  parquet, not the Spark transcode.

- **Corrupt legacy NDS-H `part` (SF100/SF300).** The *real* NDS-H builds for
  SF100/SF300 that exist on disk are themselves broken: `part.p_brand` holds 80
  contiguous values `Brand#11..Brand#90` instead of the 25 the TPC-H spec allows
  (`Brand#<1-5><1-5>`). That inflates distinct `(p_brand, p_type, p_size)` from
  187,500 to ~600,000 and changes the answers to q16/q17/q19 (q16 → 91,640 rows
  vs the correct 27,840). This is a stale artifact from an older dbgen; the
  current `tpch-gen` dbgen does **not** reproduce it — verified with
  `dbgen -s 300 -C 600 -S {1,599}`, which yields exactly the 25 spec brands. So
  the pipeline code is fine; the on-disk SF100/SF300 NDS-H parquet must simply be
  regenerated.

Consequence: the SF30/50/100/300 rows left `all_results.csv`, and only the
certified-clean SF500 slice stayed. On 2026-10-07 those rows were removed too:
they predated the timing protocol (`AGENTS.md`) and were measured on another
box. `all_results.csv` is empty until `make bench` fills it; the removed rows
are in the git history (`git log -- baselines/results`).

## Guard — `results/validate_dataset.py`

```
make validate SF=<scale_factor>          # or: $PY results/validate_dataset.py <parquet_dir> <SF>
```

Exits non-zero unless the dataset is clean external NDS-H. It checks:

- **writer** is ndsgen (`parquet-rs` with `NDSGEN.json` beside `parquet/`) or the
  retired Spark transcode (`parquet-mr`) (rejects DuckDB/cudf self-writes);
- **`part.p_brand`** is within the 25-value spec set, and
  `distinct(p_brand,p_type,p_size) ≤ 187,500` (rejects the SF100/SF300 corruption);
- **row counts** match the TPC-H per-SF formula.

`datagen/dataset.sh` runs this automatically after generation and publishes
`parquet/` only if the fresh dataset passes. Run it by hand before trusting any
dataset you did not just generate.

## Generating and running a scale factor

```bash
# generate + validate (PARALLEL≈2·SF, BATCH 25), then run every engine at that SF;
# each engine self-merges into all_results.csv
make bench SF=300       # generates $DATA_DIR/sf300 if missing, validates, runs
```

(GPU engines: int32 caps ≈ SF300.)
