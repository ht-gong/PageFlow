# TPC-H engine comparison — Spark-RAPIDS · Polars-GPU · DuckDB · Sirius

A harness that runs TPC-H q1–q22 across several query engines on the **same**
parquet dataset and collects per-query runtimes into one results file. Every
workflow — data generation, the benchmark, the parquet-format profiling pass,
the reports — runs in one pinned Docker image through one entry point, `make`.

## Setup

```bash
cp ../local.env.example ../local.env   # set DATA_DIR, SCRATCH, SHM and the GPU to use
make image                             # build the image for the pins in versions.env (~10 min, needs network)
make doctor                            # versions, GPU, io_uring, free space
```

Needs Docker with the NVIDIA Container Toolkit and an NVIDIA driver for CUDA 13.
The image holds the environments only; the repo is bind-mounted, so code
changes need no rebuild. The image tag is a hash of `versions.env`,
`docker/*.lock` and `docker/Dockerfile`. A container with a GPU runs on that
GPU's NUMA node only (its CPUs and memory; `NUMA=off` to unbind).

## Targets

| target | does |
|--------|------|
| `make image` / `make shell` | build the image / a shell in the container |
| `make doctor` | check versions against the pins, that the GPU is idle and not exclusive-mode, that the container is bound to the GPU's NUMA node, that io_uring works, free space |
| `make data SF=100` | generate `$DATA_DIR/sf100` if missing, then validate it |
| `make validate SF=100` | validate `$DATA_DIR/sf100` |
| `make bench SF=100 [ENGINES="…"]` | TPC-H q1–22 on the four baseline engines → `results/all_results.csv` (the TQP-Vortex root's `make bench` adds `tqp_vortex`) |
| `make smoke` | SF1, the four baseline engines, nothing merged (the root's `make smoke-all` adds `tqp_vortex`); prints a query × engine table, fails unless every query is OK with the SF1 reference row counts (`AGENTS.md`) |
| `make ablation [SF=5 ENCODINGS=… COMPRESSIONS=… ROUNDS=3 ENGINES=… TOL=0.05 OUT=results]` | parquet-format profiling pass → `format_profile.csv` (times) + `format_map.json` (decision); prints the results table |
| `make summary` | pivots of `results/all_results.csv`, and per SF the queries whose OK engines disagree on the row count |
| `make spill SF=1000 [ENGINES=…]` | spill pass (`rapids`, `polars_gpu`, `sirius`) → `results/spill_profile.csv` + `spill_settings.json` (the TQP-Vortex root's `make spill` adds `tqp_vortex`) |
| `make spill-smoke` | SF1 under forced spill, nothing merged; fails unless every record and identity holds (the root's `make spill-smoke` adds `tqp_vortex`) |
| `make spill-test` | the spill record builders' unit tests (CPU only) |

`SF` for `bench` is a list of bare numbers (`100 300`) or ranges over the
canonical set `{30,50,100,300,500,700}` (`30-300`). Runner settings pass through
the environment: `QUERIES`, `KEEP_RAMDISK=1`, `GEN_PARALLEL`/`GEN_BATCH`/`GEN_OPTS`,
`DRIVER_MEM`, `GPU_PART_MB`, `MIN_FREE_GB`, `QUERY_TIMEOUT`, `RAPIDS_TIMEOUT` (default
600 s per RAPIDS query process; the smaller of it and `QUERY_TIMEOUT` applies), `SIRIUS_TIMEOUT`,
`PROBE_TIMEOUT`, … (the list is in `docker/run.sh`).

`make bench` (`run.sh`), for each scale factor: ensures the dataset exists and
validates, stages it disk → ramdisk, runs each baseline engine against the
ramdisk copy, then frees the ramdisk. Each engine folds its 22 rows into
`results/all_results.csv`.

## Engines

| key | engine | dir |
|-----|--------|-----|
| `rapids` | Spark 3.5.8 + RAPIDS Accelerator 26.04.2 (GPU) | `rapids/` |
| `polars_gpu` | cudf-polars 26.6 / Polars 1.39.3 (GPU) | `polars/` |
| `duckdb_cpu` | DuckDB 1.5.5 (CPU) | `duckdb/` |
| `sirius` | Sirius (GPU-native SQL, DuckDB 1.5.5 extension, libcudf 26.08) | `sirius/` |
| `tqp_vortex` | TQP-Vortex (GPU), in its own image | `..`, the TQP-Vortex root |
| `sirius_resident` · `polars_gpu_resident` · `duckdb_cpu_resident` · `rapids_resident` | the same engines in resident mode: the query's data in GPU memory (DuckDB: native tables; RAPIDS, best effort: Spark's cache in host memory) before the timed run | as above |

Every baseline engine reads the ramdisk parquet in every query; none loads
tables into memory first, except under the `*_resident` keys, which time
execution alone (what each loads, and how it is checked: `AGENTS.md`).
`tqp_vortex` reads the same parquet from disk; the TQP-Vortex root's `make
bench` runs it after the others, through
`../bench/run_tqp.sh`. Every engine runs each query in its own process: an
untimed warm pass of the same query on SF1, then one timed run on the target
scale factor (`AGENTS.md`).

## Data — one generator for every engine

All engines read the same rows from the repo's **NDS-H-equivalent**
generator `datagen/ndsgen` (TPC-H rows from tpchgen-rs, the dbgen 2.17.3 port,
written in NVIDIA's NDS-H format; it replaced the dbgen → Spark 3.5.8 transcode,
whose datasets hold the same rows). Before a benchmark each engine gets those
rows in the per-column format its profiling map chose (`AGENTS.md`, "Before
every benchmark"); TQP-Vortex converts columns into its own format and reads any
of them. Never point a runner at self-written parquet (DuckDB's own `CALL dbgen` export, cudf exports);
it is not comparable to the external standard. `make data` / `make bench`
generate and validate; full rules and rationale in **`results/GENERATOR.md`**.

The same data is TQP-Vortex's, which has no generator of its own: it scans
the parquet in `$DATA_DIR`, where it looks by default (`TQP_DATA_DIR`).

## Results — `results/all_results.csv`

The single canonical TPC-H results file (the only TPC-H results CSV kept —
everything else is a throwaway; the format profiling pass keeps its own two
files, below). One row per `(engine, scale_factor, query)`. Each runner writes a
temp CSV and upserts its rows via `merge_results.py <engine> <sf> <run_csv>`,
each replacing the row of the same `(engine, scale_factor, query)`;
every other view (summary, matrices, per-engine scaling) is a pivot of this file
(`make summary`). Do **not** add parallel summary CSVs; pivot this instead.

| column | type | values / meaning |
|--------|------|------------------|
| `engine` | string | `rapids` · `polars_gpu` · `duckdb_cpu` · `sirius` · `tqp_vortex`; resident mode: `sirius_resident` · `polars_gpu_resident` · `duckdb_cpu_resident` · `rapids_resident` |
| `scale_factor` | int | TPC-H scale factor (≈ GB of raw data) |
| `query` | string | `query1` … `query22` |
| `status` | string | `OK` · `FAIL` (engine error; GPU out-of-memory shows here with the engine's message in `rows_or_error`, e.g. Sirius's "OOM retry limit") · `KILLED_DISK` (disk-watchdog kill) · `TIMEOUT` (per-query timeout) |
| `seconds` | float | wall-clock of the query's one timed run on the target scale factor, in its own process after an untimed SF1 warm pass (engine startup excluded; `tqp_vortex`: its `approx_time`, whose bounds `AGENTS.md` gives); time-to-failure on error; `NA` on a watchdog kill or where no time was taken |
| `rows_or_error` | int / string | result **row count** when `OK`, else a short error message |

Result row counts must **match across engines at the same scale factor** — a
mismatch means a correctness bug.

```csv
engine,scale_factor,query,status,seconds,rows_or_error
rapids,100,query1,OK,2.308,4
```

## Parquet-format profiling pass

`make ablation [SF=5 ENCODINGS=… COMPRESSIONS=…]` answers which parquet format
each engine scans and decodes fastest, per column: one uniform variant dataset
per encoding × compression (same rows,
`$DATA_DIR/fmt_sf<SF>/shuffle-<encoding>-<compression>`), one
`SELECT min(c), max(c)` probe per (engine, column, format, round), no TPC-H
queries. It writes two files to `OUT`: `format_profile.csv` (per engine, column
and encoding × compression: status, median seconds over rounds, compressed
bytes) and `format_map.json` (per engine and column: the chosen encoding and
compression). Schemas in `AGENTS.md` and `ablation/format_profile.py`. `SF`
defaults to 5.

## Spill pass

`make spill SF=…` measures how much intermediate data each GPU engine's timed
run moves off the GPU, by design and under memory pressure, per tier (host,
disk), what it brings back and the peak it holds per tier. It runs the bench
protocol with each engine's own instrumentation on: RAPIDS's Spark event log,
rapidsmpf's statistics for cudf-polars, Sirius's quent telemetry and DEBUG log,
and, from the TQP-Vortex root (`make spill`, `../bench/run_tqp.sh` with
`SPILL=1`), TQP-Vortex's movement ledger and residency peaks.
The pass's own times never reach `all_results.csv`. Output:
`results/spill_profile.csv`, one row per (engine, SF, query, kind), and
`results/spill_settings.json`; the raw evidence stays under
`$SCRATCH/spill/sf<SF>/<engine>/`. Schema, kinds and categories in `AGENTS.md`
and `spill_results.py`; the design in `../docs/engine_spill_plan.md`.

## Versions

Every pin is in `versions.env` (base image digest, apt snapshot, JDK, Rust,
RAPIDS jar, Sirius commit and build arch, pixi), the hashed Python locks
`docker/py.lock` and `docker/polars.lock`, and ndsgen's `datagen/ndsgen/Cargo.lock`. A version bump is its own change and
means re-running the affected results.

## Layout

```
Makefile       the entry point; every target runs through docker/run.sh
docker/        Dockerfile, run.sh (launcher), seccomp profile, doctor, locks
versions.env   every toolchain pin
run.sh         the benchmark driver (stage -> run engines -> free, per SF)
datagen/       the one TPC-H generator (ndsgen: tpchgen-rs rows -> NDS-H parquet)
rapids/ polars/ duckdb/ sirius/   one runner per engine (+ each GPU engine's spill record builder)
ablation/      parquet-format profiling pass
spill_results.py  the spill pass's contract and merge -> results/spill_profile.csv
tests/         the spill record builders' unit tests and fixtures (CPU; tqp_vortex's builder is
               ../bench/tqp_spill_stats.py)
results/       all_results.csv, spill_profile.csv, query stream (queries/), validator, write-ups
dpfproto/      GOLAP / DPFProto baseline notes (separate; not wired into make)
```

Not committed (regenerate): the parquet datasets and the image.
