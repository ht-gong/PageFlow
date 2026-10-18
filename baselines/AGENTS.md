# AGENTS.md — TPC-H engine comparison

Harness that runs TPC-H q1–q22 across GPU/CPU query engines on the **same**
parquet dataset and collects per-query runtimes into `results/all_results.csv`.
Start with `README.md`; this file is the quick agent orientation and the
results-data contract.

## Running — one entry point: `make`

Every workflow runs in one pinned Docker image through `make`; each target calls
`docker/run.sh`, which mounts the repo, `DATA_DIR` and `SCRATCH` at their host
paths and runs the target's script inside. Box settings (`DATA_DIR`, `SCRATCH`,
`SHM`, `GPU`) come from the environment or `local.env` at the TQP-Vortex root,
`../local.env` (copy `../local.env.example`).

```bash
make image && make doctor            # once per pin change; doctor checks GPU, io_uring, versions
make data SF=100                     # generate (if missing) + validate $DATA_DIR/sf100
make bench SF="100 300" [ENGINES="rapids sirius"] [QUERIES="1 6"]
make smoke                           # SF1, the four baseline engines, nothing merged; fails unless 22/22 everywhere, with the SF1 reference's row counts
make ablation [SF=5] [ENCODINGS="plain dict delta"] [COMPRESSIONS="snappy zstd lz4raw"] [ROUNDS=3] [TOL=0.05] [OUT=dir]
make summary                         # pivots of all_results.csv
make spill SF=1000 [ENGINES=...]    # spill pass -> results/spill_profile.csv (below)
make spill-smoke                     # SF1 under forced spill; fails unless every record and identity holds
make spill-test                      # the spill record builders' unit tests (CPU)
make shell                           # interactive shell in the container
```

`make bench` runs `run.sh`: per scale factor it ensures the dataset
(`datagen/dataset.sh`), stages it disk → ramdisk, runs each engine against the
ramdisk copy, then frees the ramdisk. Engines self-merge into
`results/all_results.csv`. `SF` takes bare numbers or ranges over the canonical
set `{30,50,100,300,500,700}` (`30-300`); `ENGINES` defaults to
`rapids polars_gpu duckdb_cpu sirius`. Pass-through settings (`QUERIES`,
`KEEP_RAMDISK`, `GEN_PARALLEL`, `GEN_BATCH`, `GEN_OPTS`, `DRIVER_MEM`, `GPU_PART_MB`,
`MIN_FREE_GB`, timeouts, …) are listed in `docker/run.sh`.

At the TQP-Vortex root, `make bench` adds the fifth engine, `tqp_vortex`, and
runs all five by default: per SF, `run.sh` for the others, then
`../bench/run_tqp.sh` on the host. It takes `SF` as bare numbers, and `ENGINES`
and `QUERIES` as here; a `RUN_CSV_DIR` there must be under `$SCRATCH`, and each
SF's `results/run.log` is kept as `results/run_sf<SF>.log`. `make smoke-all`
there is SF1 on all five, nothing merged.

Runners read the tool paths the image sets — `PY`, `POLARS_PY`, `JAVA_HOME`,
`SPARK_HOME`, `RAPIDS_JAR`, `SIRIUS_DUCKDB`, `SIRIUS_ENVLIB`, `NDSGEN` — and
`SCRATCH` (engine scratch: `$SCRATCH/rapids`, `$SCRATCH/polars`, `$SCRATCH/probe`).
Nothing in the scripts names a box-specific path.

## Engines & runners

| engine key | dir | runner |
|------------|-----|--------|
| `rapids` (Spark + RAPIDS, GPU) | `rapids/` | `run_tpch_safe.sh` |
| `polars_gpu` (cudf-polars, GPU) | `polars/` | `run_polars_gpu.sh` |
| `duckdb_cpu` (DuckDB, CPU) | `duckdb/` | `run_duckdb.sh` |
| `sirius` (Sirius GPU, DuckDB ext.) | `sirius/` | `run_sirius.sh` |
| `tqp_vortex` (TQP-Vortex, GPU) | `..` (the TQP-Vortex root) | `../bench/run_tqp.sh` (host) |

Engine settings shared by a runner and the format probes live in
`rapids/activate.sh` (`rapids_run_args`), `polars/env.sh`, `sirius/env.sh`.

**Every baseline engine reads the ramdisk parquet in every query; none loads
tables into memory.** DuckDB and Sirius use views over `read_parquet()` (DuckDB's
external file cache off, Sirius's prefetch cache off), RAPIDS temp views over
`spark.read.parquet`, Polars `scan_parquet`. Sirius must not use `pin_table`: a
pin decodes the parquet into cuDF columns in pinned memory, so the parquet format
would no longer reach query time (`sirius/README.md`). DuckDB's in-memory
`tables` mode was removed. Resident mode, below, is the one exception, under
its own engine keys.

`tqp_vortex` is TQP-Vortex itself (the repo root), in its own image. It reads
the same parquet from `$DATA_DIR` on disk and runs its own SQL and plan
lowering of each query, semantically the stream's. The root `make bench` makes
the SF's and SF1's datasets (`make data`) first where they are missing, and
runs `tqp_vortex` after `run.sh` has freed the SF's ramdisk copy.
`../bench/run_tqp.sh` starts one `main.py` per query; each
process keeps its perf report, with the query's load time, in
`../out/bench/<run>/q<N>/`.

**One protocol for every engine: one process per query, an SF1 warm pass, one
timed run.** In each query's process an untimed warm pass first runs the same
query on SF1 (`run.sh` stages `$DATA_DIR/sf1` to the ramdisk once per run and
passes it as `WARM_PARQUET`), so GPU init, kernel JIT and reader setup stay out
of the timing. Then the query runs once on the target SF's ramdisk copy, timed.
No query runs twice on the target data in one process; repeating a measurement
means a new process. A query that fails its warm pass is recorded `FAIL` and not
timed.

`tqp_vortex` keeps TQP-Vortex's own warmup: its SF1 pass runs Q1, then the
query, and runs q11 with SF1's FRACTION, where the baselines' warm pass runs the
target SF's text. Its timer (`approx_time`) has other bounds than the baselines'.
It starts after the query's columns are read from the parquet on disk, not the
ramdisk, decoded on the GPU into pinned memory and compressed, and after the query's
planning and a forced garbage collection; and it leaves out the output step,
which builds the result frame. The baselines' timed run includes reading and
decoding the parquet and fetching the result. The SF1 warmup does not cover
every first use at the target SF: alone in its process, a query can take much
longer than after other queries at that SF (q8 at SF10: 1.0 s, against 0.25 s
in one process that runs all 22).

**Resident mode — data loaded before the timed run.** Four more engine keys
time execution alone, with the query's data already in memory: `sirius_resident`,
`polars_gpu_resident`, `duckdb_cpu_resident`, `rapids_resident`
(`ENGINES="… sirius_resident"`). `run.sh` runs the same runners with
`LOAD_MODE=resident ENGINE_KEY=<key>`; the protocol is otherwise the one above,
and each pass (SF1 warm, target) loads its own data first, untimed. Their rows
never replace the parquet rows.

- `sirius_resident`: the query's columns (`sirius/pin_columns.py`) are pinned in
  GPU memory, `CALL pin_table(..., tier='gpu')` (decoded cuDF columns; Sirius's
  defaults: decimal64 narrowed to decimal32 where it fits, no pin compression),
  and DuckDB CPU fallback is off. A scan that reads an unpinned column falls
  back to the parquet reader *silently*, so a timed run whose Sirius log lacks an
  `assigned pinned entry` line for every pinned table is `FAIL`
  (`pin_not_served:<tables>`). That check is per table, so the pinned columns
  must cover every scan of a table; `pin_columns.py` records how that was
  checked.
- `polars_gpu_resident`: cudf-polars has no API that keeps a frame on the GPU (a
  host frame is copied host→device on every collect), so `polars/device_cache.py`
  patches `DataFrameScan.do_evaluate` from the runner to keep each uploaded column
  on the GPU. That path is the **in-memory executor**'s (the streaming executor
  slices host frames on its own path), so this key runs it, without spilling, with
  `raise_on_fail` and an explicit `CudaAsyncMemoryResource` (left unset, that
  executor allocates from a managed-memory pool, which would page the "resident"
  columns out to host): unlike `polars_gpu`, a query whose working set exceeds the
  GPU fails with an out-of-memory error. The SF1 warm pass records the columns
  the query scans; exactly those target columns are uploaded before the timed
  run, and a timed run that copies any column host→device is `FAIL`
  (`device_cache_miss:<cols>`). The patch checks the cudf-polars version (26.06)
  on import.
- `duckdb_cpu_resident`: every table loaded into a native DuckDB table
  (`CREATE TABLE … AS SELECT * FROM read_parquet(…)`), so the timed run reads no
  parquet.
- `rapids_resident`, best effort: Spark-RAPIDS 26.04 has no GPU-resident cache,
  so its data is resident in **host** memory. The query's columns
  (`sirius/pin_columns.py`) of each table it reads are persisted `MEMORY_ONLY` in
  Spark's cache and materialized, and the views point at the cached frames. The
  cache serializer is RAPIDS's `ParquetCachedBatchSerializer`
  (`RAPIDS_CACHE_SERIALIZER`): parquet-encoded batches in the JVM heap that the
  GPU decodes in the timed run, so that run still pays the host→device copy and
  a parquet decode, but no file read. The cache is built through hard links to
  the dataset's files that are deleted once it is built, so a timed run that
  would recompute a block from the parquet (evicted, or never cached) fails
  instead: `FAIL` (`cache_miss:<tables>`); a cache not wholly in memory after
  the build fails every query (`resident cache: …`).

The GPU keys hold only the query's columns: decoded TPC-H is about 1.2 GB per SF,
so whole datasets fit on a 32 GB GPU at no benchmark SF.

The baseline runners read the same `results/queries/stream_qualification.sql`
and take `TPCH_PARQUET` / `TPCH_SF`; `../bench/run_tqp.sh` converts each
process's `query_results.csv`. Each runner writes a throwaway temp CSV
(`query,status,seconds,result_rows_or_error`; rapids adds a `gpu_ops` column,
which the merge drops), then folds it into `results/all_results.csv` via
`merge_results.py <engine> <sf> <run_csv>`: an idempotent upsert in which each
of its rows replaces the row of the same `(engine, scale_factor, query)`, so a
`QUERIES` subset updates only those queries. A query whose process ends without
a row gets one from its runner, `FAIL` with `no_result_row_exit_code_<rc>`, so a
re-run never leaves its earlier row in place. No per-run CSVs are kept;
`TPCH_MERGE=0` keeps the temp instead of merging.

The stream is qgen's with its default substitutions. Its one parameter that
depends on the scale factor, q11's FRACTION, is the placeholder `:FRACTION`:
`results/queries/stream.py` fills in 0.0001/SF as qgen prints it (`%.10f`, so
SF500 gives 0.0000002000 and SF700 0.0000001429). The rapids, duckdb_cpu and
sirius runners read the stream through it, polars_gpu's native q11 takes the
same value, and the SF1 warm pass runs the target SF's text.

## Data — one generator for every engine

Every engine must run the same rows, from the in-repo generator
`datagen/ndsgen` (TPC-H rows from tpchgen-rs, the dbgen 2.17.3 port, in NVIDIA's
NDS-H format: the schema, layout, encodings and row shuffle of its dbgen → Spark
transcode; provenance: `datagen/README.md`). Datasets from the retired dbgen →
Spark pipeline (`gen_tpch.sh`, writer `parquet-mr`) hold the same rows and stay valid.
ndsgen is compiled into the image: an edit to `datagen/ndsgen/` needs `make image`.
For the benchmark each engine reads those rows in its own per-column format
("Before every benchmark", below).
**Never** point a runner at self-written parquet — DuckDB's own `CALL dbgen`
export (16-col, no trailing `ignore`) or a cudf export — it is not comparable to
the external standard and gives different query answers.

`make data` / `make bench` generate missing datasets and validate
(`results/validate_dataset.py`: writer, `part.p_brand`, row counts); the pipeline
publishes `parquet/` only after it validates. Full rules and rationale:
`results/GENERATOR.md`.

The one-generator rule covers TQP-Vortex too, which has no generator of its
own: it scans the same parquet (`$DATA_DIR`, the root Makefile's default
`TQP_DATA_DIR`).

## Results data contract — `results/all_results.csv`

**The single source of truth.** One row per `(engine, scale_factor, query)`.
Summary / matrices / per-engine scaling are all pivots of this — do NOT add
parallel summary CSVs; pivot this instead (`make summary`).

| column | type | values / meaning |
|--------|------|------------------|
| `engine` | string | `rapids` · `polars_gpu` · `duckdb_cpu` · `sirius` · `tqp_vortex`; resident mode (above): `sirius_resident` · `polars_gpu_resident` · `duckdb_cpu_resident` · `rapids_resident` |
| `scale_factor` | int | TPC-H scale factor (≈ GB raw) |
| `query` | string | `query1` … `query22` |
| `status` | string | `OK` · `FAIL` (engine error; **GPU OOM** shows here with the engine's message in `rows_or_error`: Sirius's "OOM retry limit", TQP-Vortex's "Pool out of memory", or "PinnedPoolExhausted" for its pinned host pool; resident mode: `pin_not_served:…` / `device_cache_miss:…` / `cache_miss:…` when the timed run read data that was not resident) · `KILLED_DISK` (disk-watchdog kill) · `TIMEOUT` (per-query timeout; `rapids`, `rapids_resident`: its process outlived `RAPIDS_TIMEOUT`, default 600 s, or a smaller `QUERY_TIMEOUT`; `tqp_vortex`: its process outlived `TQP_TIMEOUT`) |
| `seconds` | float | wall-clock of the query's one timed run on the target SF, in its own process after an untimed SF1 warm pass (engine startup excluded; `*_resident`: the data load excluded too); `tqp_vortex`: `approx_time`, within the bounds above; time-to-failure on error; `NA` on a disk-watchdog kill or where no time was taken; on `rapids`' and `polars_gpu`'s `TIMEOUT`, the process's elapsed seconds |
| `rows_or_error` | int / string | result **row count** when `OK` (`tqp_vortex` Q16 at SF ≥ 100, whose output step is skipped: its final Sort's rows), else a short error string |

```csv
engine,scale_factor,query,status,seconds,rows_or_error
rapids,100,query1,OK,2.308,4
sirius,500,query9,FAIL,259.488,INTERNAL Error: ... GPU pipeline task exceeded maximum OOM retry limit (100) for
```

**Formerly wrong `tqp_vortex` values.** Until 2026-10-14 TQP-Vortex's
aggregate added up per-chunk ratios where it should have merged partial sums
(`../docs/bug_backlog.md`, "Query results"): from about SF88 up, at the
default 256 MB chunk, q14's `promo_revenue` was wrong while its row count was
right, so `make bench` merged those rows as `OK`; q8's `mkt_share` broke the
same way once its aggregate's input spanned two chunks. The rows record no
values, so their times stand.

## Parquet-format profiling pass

A second experiment, separate from the engine comparison: which parquet format
(encoding × compression) each engine scans and decodes fastest, per column.
`make ablation` (`ablation/format_profile.py`) builds one uniform variant
dataset per format (`$DATA_DIR/fmt_sf<SF>/shuffle-<encoding>-<compression>/`,
written by ndsgen with one seed: the same rows in the same order), then times
one `SELECT min(c), max(c)` probe per (engine, column, format, round), no TPC-H: one worker per (engine, variant, round) probes
every column in shuffled order after untimed priming, engine caches off, CPU
fallback recorded as `FALLBACK`, hangs as `TIMEOUT` (worker restarted). `SF`
defaults to 5. It writes two files to `OUT` (default `results/`):

- `format_profile.csv` — one row per `(engine, scale_factor, table, column,
  encoding, compression)`: `status` (`OK` if OK in every round, else the non-OK
  status), `seconds` (median over rounds; empty unless OK), `bytes` (the
  column's compressed size in that format). A run replaces the rows of the
  engines it ran at that SF.
- `format_map.json` — the decision: `{"sf<SF>": {"scale_factor", "tolerance",
  "engines": {engine: {table: {column: {"encoding", "compression"} | null}}}}}`.
  Fastest OK format per engine and column; formats within `TOL` of it are tied
  and the fewest bytes wins.

It ends by printing the results: per engine, the probe seconds summed over all
columns for each format, and the map's.
Per-round, per-worker probe logs stay in `OUT/format_profile_logs/` (gitignored).

### Before every benchmark: each engine's data in its format map

**Always** produce each baseline engine's parquet from the format map before
running the benchmark; never time a baseline engine on one shared uniform
format. For every baseline engine (`rapids`, `polars_gpu`, `duckdb_cpu`,
`sirius`) and SF of the run:

1. `results/format_map.json` must hold the engine, from a profiling pass
   (`make ablation`) run on the current engine, engine version and ndsgen. Re-run
   the pass after a change to any of them.
2. ndsgen writes the engine's dataset, in a data root of its own, with every
   column in the format the map gives that engine, at the SF and at SF1 (the
   run's warm pass):
   `$NDSGEN --sf <SF> --out $DATA_DIR/<engine>/sf<SF> --format-map results/format_map.json --engine <engine>`
   (`--map-key sf<N>` when the map holds more than one profiling SF). The map's
   format for a column is the profiling metric's winner: among the formats OK in
   every round, the one with the lowest median probe time, with every format
   within `TOL` of it counted as tied and the fewest compressed bytes winning.
   A column the map leaves `null` (no OK format) and the `ignore` column get
   parquet-mr's default, dictionary + snappy. `NDSGEN.json` beside the dataset
   records the map and the engine.
3. Run that engine alone on its root: `DATA_DIR=$DATA_DIR/<engine>
   ENGINES=<engine> make bench SF=<SF>`. `run.sh` validates the datasets it finds
   there and generates only missing ones, in the default format, so step 2 must
   come first.

The engines still read the same rows in the same order (one generator, one
seed); only each column's encoding and codec differ between them.

**Disk.** Each engine's dataset is about as big as the shared one (the SF5 map's
picks: 0.24-0.31 GB per SF, so 240-310 GB at SF1000), and a box may hold only
one at a time. So work one (engine, SF) at a time:

1. **Guard first.** Before ndsgen writes `$DATA_DIR/<engine>/sf<SF>`, estimate
   its size (the map's per-column `bytes` in `format_profile.csv`, scaled by
   SF / the profiling SF) and stop unless the disk holding `$DATA_DIR` has that
   plus 25%, and then still `MIN_FREE_GB` (the runners' floor, default 30 GB,
   below which they kill a query), free. Never let ndsgen fill the disk: a
   full disk mid-write corrupts the run, and the engines' scratch needs it too.
2. Generate it, then run **every** experiment of that engine at that SF on it
   (the benchmark, its resident-mode keys, the spill pass, anything else that
   reads it) before moving on, so it is generated once.
3. Delete `$DATA_DIR/<engine>/sf<SF>` when the next (engine, SF) needs the
   space. Never delete the shared `$DATA_DIR/sf<SF>` this way.

TQP-Vortex is exempt: its scan converts every column into its own format
(`IO/parquet_reader.py`, then its codecs), so any parquet with the baselines'
rows serves it equally, `$DATA_DIR/sf<SF>` or any engine's dataset. Its runs
(`make bench`'s `tqp_vortex`, the `experiments/` scripts, which read
`TQP_DATA_DIR`) can therefore share whichever (engine, SF) cycle has a dataset of
their SF on disk, with `TQP_DATA_DIR=$DATA_DIR/<engine>`, instead of needing one
of their own.

## Spill pass — `results/spill_profile.csv`

What each GPU engine's timed run moves off the GPU, brings back and holds, per
(scale factor, query). `make spill SF=…` (`SPILL=1 ./run.sh`) runs the bench
protocol with each engine's instrumentation on, for `rapids`, `polars_gpu` and
`sirius` (`duckdb_cpu` and the `*_resident` engines are skipped). The
TQP-Vortex root's `make spill` adds `tqp_vortex` (`../bench/run_tqp.sh`, with
the same variables). Spill is not deterministic, so the bytes describe the
pass's own runs. Its times are kept as the `pass_time` kind and never reach
`all_results.csv`. The pass needs the GPU and the box to itself: memory
pressure is what it measures.

- `SPILL_ROOT` (default `$SCRATCH/spill`) holds, per SF and engine,
  `sf<SF>/<engine>/`: `pass.csv`, the raw evidence in `q<N>/`,
  `records.jsonl` (one record per query) and `settings.json`. The log is
  `results/spill_run.log`.
- After each engine, `spill_results.py merge` validates the records and
  upserts them as `merge_results.py` does. `SPILL_MERGE=0` keeps them
  unmerged.
- `SPILL_SMOKE=1` (`make spill-smoke`) is below.

**Columns** (long format, one row per `(engine, scale_factor, query, kind)`):

| column | meaning |
|--------|---------|
| `engine`, `scale_factor`, `query` | as in `all_results.csv` |
| `kind` | from the engine's closed list, `KINDS` in `spill_results.py`; every kind is written for every query, zeros included |
| `run_status` | the query's outcome in the spill pass: `OK` · `FAIL` · `TIMEOUT` · `KILLED_DISK` |
| `quality` | `OK` · `NA` (not measured: the instrumentation cannot, or the builder failed) · `LOWER_BOUND` · `LIFETIME` (a process-lifetime peak that may include the warm pass) · `PARTIAL` (a killed process or a truncated log) |
| `logical_b` | uncompressed bytes; peaks go here |
| `wire_b` | bytes as moved or stored (Spark shuffle files and disk spill are lz4; equal to `logical_b` otherwise; empty when not known) |
| `count` | batches, tasks, copies or retries |
| `seconds` | the engine's own copy/spill time where it reports one; context only, never compared across engines |
| `source` | provenance, e.g. `eventlog:gpuSpillToHostBytes`, `quent:DataBatch InTransit` |

Readers aggregate by **category**, never by kind name: `KINDS` maps each
kind to one of these.

| category | what |
|----------|------|
| `scan_in` / `scan_uva` | base-table reads, copied / zero-copy |
| `design_d2h` / `design_h2d` / `design_uva` | intermediates the design moves off the GPU, and their read-back, copied / zero-copy |
| `design_disk_w` / `design_disk_r` | Spark's shuffle files |
| `design_d2d` | on-GPU copies: repartition, fanout, cache |
| `pressure_d2h` / `pressure_d2disk` / `pressure_h2disk` | spill forced by memory pressure, per tier |
| `restore_h2d` / `restore_disk` | spill read back |
| `result_d2h` | the result's copy to the host |
| `materialized` | intermediates produced, by each engine's own definition |
| `peak_gpu` / `peak_host` / `peak_disk` | the most held per tier |
| `retries` | OOM retries or reschedules (a count) |
| `context` | engine-specific extras no headline reads |

The headline numbers (`spill_results.headline()`):

| headline | sum of |
|----------|--------|
| off-GPU traffic | `design_d2h` + `pressure_d2h` + `pressure_d2disk` |
| pressure spill | `pressure_d2h` + `pressure_d2disk` |
| disk spill | `pressure_h2disk` + `pressure_d2disk` |
| by-design disk | `design_disk_w` |
| read-back | `design_h2d` + `design_uva` + `restore_h2d` + `restore_disk` |

An engine with no kind in a headline's categories has no such tier: "—", not 0.

`spill_settings.json` holds `{"sf<SF>": {engine: {"budgets": {"gpu_b",
"host_b", "disk_b"}, "spill_dirs", "configs" (sha256), "versions", "gpu",
"box", "date", ...}}}`, the budgets as the engine resolved them where it
reports them.

**Reading the numbers across engines.** Each builder's docstring names its
sources (`rapids/eventlog_stats.py`, `polars/spill_stats.py`,
`sirius/spill_stats.py`, `../bench/tqp_spill_stats.py`); the collectors'
design is in `../docs/engine_spill_plan.md` §3. What changes the reading:

- **`rapids`:**
  - Restores are timed, not counted in bytes.
  - A GPU → disk spill counts as GPU → host.
  - Disk-spill bytes are a `LOWER_BOUND`: the larger of the host → disk
    counter and the sampled peak of the spill files.
  - Its disk use at large SFs is mostly shuffle files (`design_disk_w`), not
    spill.
- **`polars_gpu`:** no disk tier. Its pinned pool's peak covers the
  process's lifetime (`LIFETIME`).
- **`sirius`:** peaks are batch bytes, without scratch or hash tables.
- **`tqp_vortex`:**
  - **By design:** every intermediate goes D2H into the pinned host pool and
    is read back, copied or zero-copy (`design_d2h`, `design_h2d`, `design_uva`).
  - **Pressure:** under `--buffer_strategy cpu` (the bench's) there is no
    pressure path, so pressure spill is 0. The granule cache's modes add
    pressure traffic: evictions, rejections and cleaning count as
    `pressure_d2h`, and their read-back as `restore_h2d`.
  - **Read-back:** counted once per consumer, so it can exceed what was
    written.
  - **`peak_host`:** the pinned pool's whole 0.5 MiB blocks handed out for
    intermediates, the unwritten rest of open chunks included, not the bytes
    written.
  - **`scan_h2d`:** the parquet is loaded into the pinned pool before the
    timer, so these are reads from that pool.
  - **No disk tier.**

**Smoke.** `make spill-smoke` runs SF1 with each runner's forced-spill
settings (`SPILL_SMOKE=1`). Its evidence goes to `$SCRATCH/spill_smoke` and is
never merged. It fails unless:
- every query is OK;
- every record keeps the contract;
- every engine spilled;
- every identity its builder checks holds (the checks are named in each
  record).

`tqp_vortex`'s only pressure path is the granule cache, so its smoke runs the
dataflow cache over a 4 MiB slot pool.

## Notes for agents

- Result row counts (`rows_or_error` when `OK`) vary with scale factor but must
  **match across engines at the same SF** — a mismatch means a correctness bug.
  SF1 reference: q1–q22 = 4, 100, 10, 5, 5, 1, 4, 2, 175, 20, 1048, 2, 42, 1, 1,
  18314, 1, 57, 1, 186, 100, 7. `results/smoke_table.py` holds the same list
  and checks every SF1 smoke run against it; at other SFs it checks that the
  engines agree. `make summary` lists, per SF, the queries in
  `all_results.csv` whose OK engines disagree, with each engine's count.
- Validate any dataset you did not just generate: `make validate SF=…`.
- Pins: `versions.env` + `docker/*.lock`. A bump is its own change and triggers
  re-runs; the image tag changes with it.
- Not committed (regenerate): parquet datasets, the image, `../local.env`.
- GPUs are shared with other users: set `GPU` to an idle one; `make doctor`
  warns on a busy GPU and fails on an exclusive-mode one (Sirius fails at
  startup if one is visible). Inside the container the GPU is device 0.
- Container quirks, all handled by `docker/run.sh`: Docker's default seccomp
  profile blocks io_uring, and without it Sirius silently falls back to kvikio
  (`docker/seccomp-iouring.json`, `--ulimit memlock=-1`; `sirius/env.sh` probes
  io_uring and falls back to kvikio only where it is blocked). The container user
  gets a passwd entry (Java/Hadoop need a user name). `--ipc=host` makes
  `/dev/shm` the host's ramdisk. `KVIKIO_COMPAT_MODE=ON` stays set for
  polars_gpu/sirius since cuFile/GDS is unavailable. `run.sh` drops empty 0-row
  parquet part-files from the ramdisk copy (Sirius's GPU reader errors on them).
- NUMA: with a GPU, `docker/run.sh` binds the whole container to the GPU's NUMA
  node (`--cpuset-cpus`/`--cpuset-mems`): every engine thread, every pinned host
  pool and the ramdisk copy (tmpfs pages go to the writer's node) are local to
  the GPU, and Spark `local[*]`, Polars and DuckDB size their thread pools from
  the bound CPUs (104 here). `--cap-add SYS_NICE` lets Sirius `mbind` its pinned
  pool (Docker's seccomp profile blocks `mbind` without it). `NUMA=off` unbinds;
  `make doctor` checks both. On this box GPUs 0-3 are on node 0, 4-7 on node 1.
- TQP-Vortex's rules (its `AGENTS.md`) hold for `tqp_vortex`: one `main.py` at a
  time on an idle GPU, as its default GPU pool takes the whole card, so
  `../bench/run_tqp.sh` stops if another process holds `TQP_GPU` (default
  `GPU`), and the root `make bench` checks that before any engine runs. That
  bench runs every engine on `GPU` and refuses a `TQP_GPU` that differs.
  TQP-Vortex's container is not bound to a NUMA node: `main.py` pins its threads
  to the GPU's node itself and binds its pinned host pool, about 0.69 GB per SF,
  to that node. A ramdisk copy there would take the pool's memory, so the root
  `make bench` runs `tqp_vortex` after `run.sh` has freed it and refuses
  `KEEP_RAMDISK=1` with `tqp_vortex`. `TQP_TIMEOUT` (default 600 + 3 × SF
  seconds; 0: none) covers its whole process: pinning, the warmup, the load and
  the query.
- On this box: `/` is nearly full, so datasets and scratch live on the NVMe
  (`DATA_DIR=/path/to/tpch-data`); Docker's data root is on the NVMe too.
- `dpfproto/` (GOLAP/DPFProto notes; DPFProto is cloned from upstream and
  patched, `dpfproto/docker/README.md`) is separate and not wired into `make`.
