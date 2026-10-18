# `profiling/` — the offline codec sweep

This directory is exactly one tool, **the sweep**, plus its cost function. It
is the offline step every real run depends on: it measures the seven nvCOMP
codecs the engine ships at six manager chunk sizes on real TPC-H columns, and
writes **`profiling/results/codec_map.json`**, the file
`compression/format_policy.py` loads once per process to decide which codec
each `(type, sorted)` column class gets and which nvCOMP chunk size each codec
runs with. Nothing here runs during a query, and nothing here compiles.

```bash
make data SF=5                      # the SF5 parquet is required by the sweep
make profiling TQP_GPU=<idle>       # python3 -m profiling.sweep  ->  profiling/results/
make q SF=1 TQP_GPU=<idle>          # the two [codec_map] startup lines show the map
```

`make profiling` runs the image copy; after editing this directory either
`make build` first or run through `dev`, which writes the same host directory:
`TQP_GPU=<idle> TQP_DATA_DIR=<DATA_DIR> docker compose run --rm -T dev python3 -m profiling.sweep`.
It is a GPU job — AGENTS.md rule 1 applies (one job at a time, on an idle
card).

## What the sweep measures

* **Inputs go through engine code.** The scan set and the byte weights
  (scan frequency × in-memory bytes) come from the 22 cached plans
  (`queries/pickles/planobj-q*.pkl` → `disk_read_cols`). Columns are read
  whole from the SF5 parquet under `load_config()["tables path"]`
  (`TQP_TABLES_PATH` in containers) with the scan's reader
  (`IO.parquet_reader.ParquetTable`), cast as the scan
  casts them (`IO.data_scanner.cast`) and typed with `TYPE_MAPPING`. Every column is measured twice, as a seeded random
  permutation (**unsorted**) and in ascending / row-wise lexicographic order
  (**sorted**), because delta, RLE and bit-packing codecs behave very
  differently on ordered keys. At run time `scan_to_ram` marks no column
  sorted, so a scanned column takes its type's unsorted cell.
* **The grid.** codec ∈ {ANS, CASCADED, BITCOMP, LZ4, SNAPPY, GDEFLATE,
  DEFLATE} × nvCOMP internal chunk ∈ {16, 32, 64, 128, 256, 512} KiB × run
  ∈ {unsorted, sorted}, every scanned column, minimum over `--reps` (default
  3). Codec options are fixed (`compression/nvcomp_manager.cu`); only the
  codec and the chunk vary.
* **Per cell, timed with CUDA events, through the engine's own write and
  read paths:** compress (a `IO.codec.region_producer.RegionProducer` with
  `chunk_b=...`: the column as one tick, its records compressed and packed
  into one region; the records are the producer's, so power-of-two rows
  within 4 MiB and 2^20 rows by default; the shipped map was swept under
  the earlier 4 MiB records cut on bytes), D2H of the region into its
  pinned home (the home
  writer's copy span; the homes are a pinned pool on GPU 0's NUMA node), H2D
  back (an `IO.transfer.in_stage.DirectIn` over the region's record table,
  its runs copied into the staging buffer), and decompress (the same
  `DirectIn`'s batched decode) of the bytes that made the round trip,
  verified equal to the input. **Cost = compress + D2H + H2D +
  decompress** (`profiling/objective.py`). RAW costs the D2H + H2D of a RAW
  region, measured through the same producer and `DirectIn` paths. Every leg is kept in the CSV so the formula can change
  without re-running.
* **Selection.** Per class `(type, run)` the byte-weighted cost picks the
  codec (RAW wins if nothing beats it). Each codec then gets **one** chunk:
  the cheapest, byte-weighted over the classes it won (over all classes for
  a codec that won none, so `--compression <codec>` still has a tuned chunk).
  The class winners are re-picked with every codec at its final chunk, so the
  map and the chunks are consistent; any class whose winner changed is
  printed. `stable` re-derives each winner under four outlier-robust schemes
  (unweighted median, weighted median, 10 % winsorized, per-column vote) and
  is 1 when all agree.
* The run pins every thread to GPU 0's NUMA node (`utility/numa.py`), exactly
  as a timed `main.py` run does, and calls `mgr_release_scratch()` between
  chunk sizes. Its stdout starts with the GPU, the nvCOMP build, the reps and
  the grid; that provenance lives in the run log, not in the JSON.

## Outputs in `profiling/results/`

The directory is untracked; `make dirs` creates it host-owned. Every
engine-image container mounts it (read-only except `profiling`).

| File | Read by | Contents |
|------|---------|----------|
| `codec_map.json` | **`compression/format_policy.py`** (the only engine reader); `make_regret_fig.py` reads `codec` | `codec`: `{type: {unsorted, sorted}}` over the four mapped types, values `RAW` or a codec; `settings`: `{codec: {chunk_b}}` for exactly the seven codecs. Any other shape is rejected at engine startup. |
| `sweep_per_column.csv` | `make_regret_fig.py` | One row per (column, run, codec, chunk): `table, column, type, run, is_pk, scan_freq, raw_bytes, codec, chunk_b, selected, ratio, compress_s, d2h_s, h2d_s, decompress_s, raw_d2h_s, raw_h2d_s`. `selected = 1` on the rows whose `chunk_b` is the codec's shipped chunk; consumers filter on it and never handle chunk sizes. |
| `sweep_by_class.csv` | `make_pareto_fig.py`, humans | Byte-weighted aggregates per (type, run, codec, chunk): ratio, the four legs and the cost in s/GB, plus `selected`, `on_pareto`, `is_best`, `stable`. RAW appears as a row per class. |
| `scan_frequency.csv` | humans | Per scanned column: type, whether it is its table's primary key (`is_pk`), how many queries scan it, workload scanned bytes. |

The figures over these CSVs live in `experiments/paper_scripts/figures/`:
`make_regret_fig.py` (the per-class cost of one codec per class against the
per-column best, in measured time) and `make_pareto_fig.py` (cost against
ratio per class, every codec at its shipped chunk). Both only plot the
shipped map and write to `experiments/paper_scripts/figures/plot/`,
which is gitignored.

## Files

| File | Purpose |
|------|---------|
| `sweep.py` | The sweep: inputs, measurement, selection, outputs, as above. `--reps N` is its only option. Stops with `run: make data SF=5` when an SF5 table is missing. |
| `objective.py` | `cost_s(row)` = the sum of the four measured legs; `raw_cost_s(row)` = raw D2H + H2D. Shared by the sweep and the regret figure. |
| `__init__.py` | Makes `python3 -m profiling.sweep` resolve. |

Timing caveats from `AGENTS.md` apply: treat these microbenchmark costs as the
policy's inputs, not as query-level performance claims. Query timing comes
from `query_times.csv` or a `--perf_report` JSON, never from here.

The sweep's design record and decision log are in git history:
`git show 20052ca:docs/codec_sweep_plan.md`.
