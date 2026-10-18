# AGENTS.md — running TQP-Vortex & operating rules

Operational guide for agents (and humans) working in this repo: **how to run the
engine** and the **rules that keep runs correct, reproducible, and safe on a shared
multi-GPU host**. For architecture and a per-directory tour of the engine, see
`README.md`.

**Everything runs in Docker.** The host has no torch, cmake or nvcc: `python3
main.py`, `test.py` and `pytest` fail on the host. Run them through `make` or
`docker compose run` as shown below. The exception is a box without Docker,
where `bench/native_env.sh` builds a native environment (see "Native runs").

## TL;DR

```bash
nvidia-smi                                  # pick an idle GPU (3 below is an example)
make build                                  # the engine image
make data SF=1                              # SF1 (warmup/tests) into $DATA_DIR/sf1, $DATA_DIR the default $TQP_DATA_DIR
make data SF=5                              # SF5 (the sweep); make data runs baselines/ (make -C baselines image once)
make profiling TQP_GPU=3                    # codec sweep -> profiling/results/codec_map.json (~5 min on an H100)

make q SF=1 TQP_GPU=3                       # all 22 TPC-H queries at SF1
make q SF=20 Q="1 6 14" TQP_GPU=3           # specific queries at SF20
make q SF=1 Q="1 1 1 6 6 6" TQP_GPU=3       # repeated runs; a median is printed per qid
make smoke TQP_GPU=3                        # SF1 end-to-end correctness
make test TQP_GPU=3                         # every pytest suite
make bench SF=10 GPU=3                      # the engine comparison on GPU 3 (GPU, not TQP_GPU; baselines/AGENTS.md)

# Your working tree, without rebuilding the image (the dev service mounts the source).
# A bare `docker compose` reads neither local.env nor the Makefile's defaults:
. ./local.env && export TQP_DATA_DIR=$DATA_DIR
TQP_GPU=3 docker compose run --rm -T dev python3 main.py --SF 1 --q 6
TQP_GPU=3 docker compose run --rm -T dev python3 main.py --smoke
```

**`make q`, `make smoke` and `make test` run the source baked into the image at the
last `make build`, not your working tree.** After editing code, either `make build`
first or run through the `dev` service.

## Correctness and performance: two kinds of check

- **Correctness is pytest.** `tests/` holds every correctness suite: unit tests per
  component, the SF1 smoke (`main.py --smoke`, all 22 queries against
  `tests/verification/SF1_reference.out`) and the e2e ladder (`tests/e2e`, the modes
  and cache policies end to end). Run them on your working tree with
  `TQP_GPU=<idle> TQP_DATA_DIR=<DATA_DIR> docker compose run --rm -T dev python3 test.py`
  (see "Other entry points"). A change is correct when they pass.
- **Performance is the self-verification experiments.** `experiments/self-verification/`
  holds the drivers that measure a change: `e2_speedup.py` (how much faster, per query
  and suite: an in-process A/B over engine switches) and `e5_time_breakdown.py` (where
  the time goes: an Nsight trace split by pipe and host step). The paper's ablation
  and figures are in `experiments/paper_scripts/` (see "Other entry points",
  "Experiments"). Tests never assert timings; experiments never assert results.

## ⛔ TIMING — WHICH NUMBER TO TRUST (READ THIS BEFORE ANY PERF CLAIM) ⛔

> **NEVER draw a performance conclusion from wall-clock / total elapsed time.**
> **DO NOT USE WALL TIME.** Not `time.time()` deltas, not a hand-rolled residual,
> not the elapsed time of a traced or profiled run. Those numbers are dominated by
> things that have nothing to do with the change under test and swing 10s of seconds
> run-to-run: lazy host **memory pinning** (*"can take minutes"*), **disk I/O**, host
> orchestration, and — worst of all — **`--torch_profile`** or **nsys** overhead,
> which inflates elapsed time massively and *non-uniformly*, so it can fake a
> "+60s regression" or a "speedup" that does not exist in the actual work.

**The one timing number is `query_times.csv`**: one `qid,approx_time` row per
non-warmup query occurrence. `approx_time` runs from the start of the query's operator
execution (after its scan into pinned memory) to its end. The output step's result
columns brought onto the GPU and copied to the host (the result's D2H) are timed; only
their conversion into the pandas result frame and its log are left out
(`operators/output.py`, `result_to_host` vs `host_frame`). The perf report's
`wall.approx_time` is the same number.
Two copies are written: `out/query_times.csv` (append-only across every run, see
Containers) and, with `--perf_report <dir>`, `<dir>/query_times.csv` (this invocation
only — prefer it).

**Rules of thumb:**
- Repeat qids (`Q="21 21 21"`) and take the **median** (see Repeated runs).
- This shared host drifts by up to about 4% between runs, so a smaller change needs an
  **in-process A/B**: `experiments/self-verification/e2_speedup.py` alternates the
  arms per query occurrence inside one `main.py`.
- To attribute time (kernels vs copies vs idle, per pipe and host step), take a trace
  with `experiments/self-verification/e5_time_breakdown.py`. A trace inflates host
  time: use it for attribution, never as a timing claim.
- Run timed passes **without** `--torch_profile` and **without**
  `TQP_EXPERIMENT_TELEMETRY` (its hooks cost host time); take the perf report's byte
  counts from a separate run.
- One `main.py` process at a time (rule 1) — a co-running job poisons every timing.
- `make bench` numbers (`baselines/results/all_results.csv`) come from one timed run
  per `main.py` process, after the SF1 warmup only (`baselines/AGENTS.md`); don't
  compare them with medians of repeated qids.

## Operating rules (read before running)

1. **Do NOT parallelize `main.py` runs — not even on different GPUs.** Run one
   `main.py` process (and one test-suite invocation) at a time. Each run pins a large
   GPU pool (`--cmp_pool_gb`, default = capacity − 2 GB) and about 0.69 GB of host RAM
   per scale factor (686 GB at SF1000),
   and runs share the host's RAM, PCIe, NUMA-local CPUs and disk, so concurrent runs
   OOM or slow each other down. Run query sets and sweeps **sequentially**.
   The host has several GPUs that **other users share**. Check `nvidia-smi` and pick
   an idle one with `TQP_GPU=<index>`. The default `--cmp_pool_gb` assumes the card is
   empty: on a card with another tenant every query fails (typically `CUDA error:
   invalid argument`). Either move to an idle GPU or pass a pool that fits
   (`ARGS="--cmp_pool_gb 22"`).

2. **Threads are pinned to the GPU's NUMA node.** After warmup, `main.py` pins every
   thread to the CPUs local to GPU 0's NUMA node (GPU 0 inside the container is the
   physical GPU `TQP_GPU` selects; in a native run, `CUDA_VISIBLE_DEVICES` selects
   it), so the host work that feeds the GPU stays on its socket and timings are
   reproducible; there is no opt-out. It prints `[variance]
   pinned all threads to NUMA node<k> …`, or `NUMA pin failed` and continues unpinned.
   The pin intersects its target with the container's CPU affinity, so never run the
   engine under a narrowed cpuset. A host NUMA daemon (`numad`) can still move the
   threads after the pin; nothing re-asserts it, so check `Cpus_allowed_list` in
   `/proc/<pid>/status` when a timing looks off. The pinned host pool is bound to the
   node separately (see "Host: the pinned pool").

3. **Initial host (CPU) pinning is slow — it is not hung.** Before the first query,
   `main.py` pins a large host pool on GPU 0's NUMA node, sized to the scale factor
   (see "Host: the pinned pool"). At large SF this can take **minutes**; the process is working.
   You'll see `Pinning <N> GB of CPU memory on NUMA node <k> (can take minutes at
   large SF)...` then `done!`.

4. **Use the smoke test to sanity-check changes.** `main.py --smoke` runs the full
   engine on real SF1 data and asserts per-query correctness. Run it after any change
   to operators, the pipeline, or `main.py` — on your working tree:
   `TQP_GPU=<idle> TQP_DATA_DIR=<DATA_DIR> docker compose run --rm -T dev python3 main.py --smoke`.
   (`make smoke` tests the image, so it only covers your change after `make build`.)

## Containers

One `Dockerfile` builds one image, `tqp-vortex:latest` (target `engine`: the built
engine plus pytest, `dataflow_gen`, profiling kernels, experiment scripts and Spark
parsing) on Python 3.11, with cuDF (`pylibcudf-cu12`, pinned in
`requirements.txt`). Its one nvCOMP is the `nvidia-libnvcomp-cu12` wheel cuDF
pins (5.2.0.13): libcudf loads it by path and `compression/` links the same
file. RMM's device resource is the engine's GPU arena (`utility/rmm_arena.py`),
so cuDF allocates there like torch. The data comes from `baselines/`, in its own
image (`make data`, see "The scan"). `docker-compose.yml` defines one-shot
services, all behind profiles, so a bare `docker compose up` starts nothing.

| Service | Started by | Code it runs | Data (`$TQP_DATA_DIR` at `/data/tables`) | Other mounts |
|---|---|---|---|---|
| `engine` | `make q`, `make smoke` | image copy | read-only | `out/`, `profiling/results/` (ro) |
| `test` | `make test` | image copy | read-only | `out/`, `profiling/results/` (ro) |
| `dev` | `make shell`, `docker compose run dev …` | **working tree** (bind mount) | read-only | `build/` on the `tqp-build` volume |
| `plans` | `make plans` | **working tree** (writes `queries/plan-q*.txt`, `queries/pickles/`) | none | `build/` on `tqp-build` (no GPU) |
| `profiling` | `make profiling` | image copy | read-only | `out/`, `profiling/results/` (**read-write**: writes the codec map) |
| `experiments` | `make experiments` | image copy of the engine; host `experiments/` | read-only | `out/`, `experiments/`, `baselines/results/` (ro), `profiling/results/` (ro) |
| `spark` | `make spark` | image copy | none | `out/`, `queries/`, `profiling/results/` (ro) (no GPU) |

`profiling/results/` holds **the codec map**, `codec_map.json`: the
`(type, sorted) → codec` table and one nvCOMP chunk size per codec that
`compression/format_policy.py` loads once per process (see "Adaptive codec
table"). The `profiling` service writes it; every other engine-image service
sees the host directory read-only (`dev` through its working-tree mount). It
is never baked into the image. `make dirs` (run by every
target) creates the directory host-owned.

Every engine-image service except `spark` sees **exactly one GPU**, the physical
device `TQP_GPU` names, as GPU 0. `main.py` uses CUDA device 0, and asks
`nvidia-smi` about it by its PCI bus id (`utility.numa.nvidia_smi_id`), so a
`CUDA_VISIBLE_DEVICES` mask (the native runs') names the same GPU to both. The
services also set `memlock: -1` (the pinned pool counts against it), `ipc: host`,
`cap_add: SYS_NICE` (Docker's default seccomp profile allows `mbind`, which binds the
pinned pool to the GPU's NUMA node, only with it), no
cpuset, `OMP_NUM_THREADS=32`, `TQP_TABLES_PATH=/data/tables` and
`TQP_OUT_DIR=/opt/TQP-Vortex/out`. The working directory is `/opt/TQP-Vortex`.

**Entrypoint** (`docker/entrypoint.sh`, every engine-image service), before the
command runs:
1. restores `TQPlib/<pkg>/*.so` from the image when a package directory in the
   mounted tree has none (`dev`);
2. symlinks `query_times.csv` at the repo root into `out/`;
3. runs `docker/preflight.py` (skipped with `TQP_SKIP_PREFLIGHT=1`). Preflight exits 1
   when no GPU is visible, `RLIMIT_MEMLOCK` is not unlimited, NUMA memory-policy calls
   are refused (no `CAP_SYS_NICE`), or GPU 0's NUMA node cannot be resolved, and warns
   when more than one GPU is visible, the CPU affinity is narrowed, GPU 0's NUMA node is
   too small for the default pinned pool, no dataset is found, the codec map is
   unreadable, or `nsys` / `JAVA_HOME` is missing. It prints the codec map's compact contents (or `no codec
   map`) as an info line. `make doctor` runs it alone.

**Outputs.** Containers run as root, so everything they write is root-owned on the
host; delete it through a container (`docker compose run --rm -T dev rm -rf
out/<dir>`).

| Host path | Contents |
|---|---|
| `out/query_times.csv` | `qid,approx_time`, appended by every run ever made |
| `<perf_report dir>/` | see `--perf_report` |
| `results.out` | each query's whole result (pandas text: a header line, then every row; Q16 at SF ≥ 100 writes `Q16 Output Skipped` instead), written to the working directory after the timed region: the host repo root under `dev`; lost with the container under `engine` unless `--perf_report` copies it |
| `out/bench/<run>/` | `make bench`'s TQP-Vortex runs, one `<run>` per SF: each query's perf dir `q<q>/` (its `query_results.csv` keeps the load time) and console log `q<q>.log`, and the runner's `run.log`; nothing removes them |

Paths passed to `--perf_report`, `--log`, `--torch_profile`, `--sizing_config` and
`--fixed_search_json` resolve against `/opt/TQP-Vortex`. Under `engine` and `test`
only paths under `out/` reach the host (`ARGS="--perf_report out/perf_q1"`); under
`dev` every repo-relative path is in your working tree.

### The scan

The scan (`IO/data_scanner.py`) reads each dataset's datagen parquet,
`$TQP_DATA_DIR/sf<sf>/parquet/<table>/`, through `IO/parquet_reader.py`, which
decodes it with cuDF on the GPU, in the arena and before the query's timed
region, into the engine's encoding (the contract in its module docstring).
Each query occurrence reads each table once, the columns it needs together, and
writes each column's reps through region producers ("Host: the pinned pool",
`IO/README.md`). It marks no column sorted, so every scanned column takes its
type's unsorted codec class ("Adaptive codec table"). While it reads, cuDF's
buffers of at most 1 MiB share arena blocks (`utility/rmm_arena.small_buffers`):
the reader holds one per part file for a whole pass, 16,000 at SF1000. When it
returns, cuDF has given back every buffer and the scan releases its producers'
nvCOMP scratch, so the query starts with the arena the scan found.

## `make` targets and variables

A target that runs a compose service is one `docker compose run --rm` (with `-T`,
except `shell` and `spark`); `data`, `validate` and `summary` run
`make -C baselines` instead, and `bench` and `smoke-all` run both.

| Target | Runs |
|---|---|
| `make help` | prints the Makefile header |
| `make build` | builds `engine` (profile `run`) |
| `make doctor` | `docker/preflight.py` alone: GPU / memlock / NUMA / host RAM / data readiness |
| `make data SF=<sf>` | generates the standard TPC-H parquet of one SF into `$DATA_DIR/sf<sf>/parquet` if it is missing, then validates it (the baselines' image: build it once with `make -C baselines image`) |
| `make validate SF=<sf>` | validates `$DATA_DIR/sf<sf>/parquet` |
| `make smoke` | `python3 main.py --smoke $(ARGS)` in `engine` |
| `make test` | `python3 test.py $(ARGS)` in `test` |
| `make q` | `python3 main.py --SF $(SF) [--q $(Q)] $(ARGS)` in `engine` |
| `make bench SF="<sf> ..."` | the engine comparison (see "The engine comparison" below). With `tqp_vortex` among the engines (the default), it first stops if another process holds the GPU, runs `make build`, and makes each SF's and SF1's dataset where it is missing (`make data`). Then per SF: the four baseline engines in their image (`baselines/run.sh`), then `bench/run_tqp.sh` on the host, one `main.py --SF <sf> --q <q> --no_debug --perf_report out/bench/<run>/q<q>` per query in `engine`, each under `TQP_TIMEOUT`. Every engine's rows are merged into `baselines/results/all_results.csv`, TQP-Vortex's as `tqp_vortex`. With `RUN_CSV_DIR=<dir under $SCRATCH>` nothing is merged and a query × engine table ends the run |
| `make smoke-all` | `make bench` at SF1 on all five engines with nothing merged (`RUN_CSV_DIR=$SCRATCH/smoke`); fails unless every query is OK with the SF1 reference row counts |
| `make spill SF="<sf> ..."` | the spill pass, shaped like `make bench` (its checks, `make build` and data first): per SF the GPU baselines through `baselines/run.sh` with `SPILL=1` (`duckdb_cpu` has no spill instrumentation), then `bench/run_tqp.sh` with `SPILL=1`: the bench protocol under `TQP_EXPERIMENT_TELEMETRY=1`, each query's perf report turned into a record of `baselines/results/spill_profile.csv` (`baselines/AGENTS.md`, "Spill pass"). Its times never reach `all_results.csv`. `SPILL_MERGE=0` keeps the records unmerged; the evidence goes to `$SPILL_ROOT` (default `$SCRATCH/spill`). |
| `make spill-smoke` | `make spill` at SF1 under each engine's forced-pressure settings (`SPILL_SMOKE=1`; TQP-Vortex: the dataflow granule cache over a 4 MiB slot pool, its only pressure path), nothing merged (evidence in `$SCRATCH/spill_smoke`); fails unless every record keeps the contract, every query is OK, every identity its builder asserts holds and every engine spilled |
| `make summary` | pivots of `baselines/results/all_results.csv`, and per SF the queries whose OK engines disagree on the row count |
| `make shell` | interactive bash in `dev` |
| `make plans` | `python3 -m parsing.regenerate [--q $(Q)] $(ARGS)` in `plans`: captures `queries/plan-q*.txt` with the pinned pyspark and lowers it to `queries/pickles/` (`ARGS=--lower-only` skips Spark) |
| `make profiling` | `python3 -m profiling.sweep $(ARGS)` in `profiling`: the codec sweep, writes `profiling/results/codec_map.json` (needs the SF5 parquet) |
| `make experiments SCRIPT=<path>` | `python3 $(SCRIPT) $(ARGS)` in `experiments`; fails without `SCRIPT` |
| `make spark` | interactive bash in `spark` (JDK + pyspark, no GPU) |
| `make clean` | `docker compose down --remove-orphans`, then removes the `tqp-vortex_tqp-build` volume |
| `make clean-all` | `clean`, then removes the engine image (`tqp-vortex:latest`) |

| Variable | Default | Meaning |
|---|---|---|
| `TQP_GPU` | `$GPU`, else `0` | physical GPU index the container sees (as its GPU 0) |
| `TQP_DATA_DIR` | `$DATA_DIR` | host data root (`sf<sf>/parquet`), mounted read-only at `/data/tables` in every engine service but `spark` and `plans`; with neither it nor `DATA_DIR` set, the targets that mount it stop |
| `SF` | `1` for `make q` | scale factor; `make data` and `validate` take one and require it, `make bench` one or more |
| `Q` | empty (all 22) | query IDs for `make q` and `make plans` |
| `ENGINES`, `QUERIES` | all five, all 22 | engines and query IDs for `make bench` and `make spill` |
| `SPILL_ROOT`, `SPILL_MERGE` | `$SCRATCH/spill`, `1` | the spill pass's evidence root, and whether its records are merged |
| `TQP_TIMEOUT` | 600 + 3 × SF | seconds per `main.py` process of `make bench`, read by `bench/run_tqp.sh`: pinning, the warmup, the load and the query; `0`: none. A process that outlives it is removed and its query recorded `TIMEOUT` |
| `ARGS` | empty | extra flags for `smoke`, `test`, `q`, `profiling` and `experiments`. Expanded by make and re-split by the shell, so quoted values survive (`ARGS="--log 'out/my runs'"`); avoid `$` |
| `SCRIPT` | empty | experiment script path for `make experiments` |
| `DOCKER` | `docker` | docker command, e.g. `"sudo docker"` without docker-group access |
| `COMPOSE` | `$(DOCKER) compose` | compose command |

`TQP_GPU` and `TQP_DATA_DIR` work as make variables or exported environment
variables. The Makefile also reads `local.env` at the repo root (copy
`local.env.example`), the box settings it shares with `baselines/`; its `DATA_DIR`
and `GPU` give the two defaults above. The environment and the command line win
over it. A bare `docker compose` reads neither it nor the Makefile's defaults (its
own are `./data` and GPU 0). Build-time settings read from the environment by
`docker-compose.yml`: `CUDA_ARCHITECTURES` (`80;86;89;90`); at run time,
`OMP_NUM_THREADS` (`32`). The nvCOMP version is not a setting: it is the one
`pylibcudf-cu12`'s libcudf pins.

## The engine comparison: `baselines/`

`baselines/` is the engine comparison. In its own image it makes the TPC-H data
(`make data`) and runs the 22 queries on four other engines
(Spark-RAPIDS, cudf-polars, DuckDB, Sirius); `make bench` runs TQP-Vortex on the
same data as the fifth. `baselines/AGENTS.md` is its guide: the data generator,
the four baseline engines and their runners, the protocol (one process per
query, an untimed SF1 warm pass, one timed run) and the results contract of
`baselines/results/all_results.csv`, where TQP-Vortex is the engine `tqp_vortex`,
with its timer's bounds and its known-wrong values.

Its one-generator rule (`baselines/results/GENERATOR.md`: every engine reads the
same validated parquet, never data it wrote itself) covers TQP-Vortex too: its scan
reads that parquet and nothing else.

## Native runs (no Docker)

On a box without Docker (an unprivileged container such as the Vast box of
`docs/engine_spill_plan.md`) the engine runs on the host, from a checkout:

- **The environment.** `bench/native_env.sh` makes a venv at `/opt/envs/tqp`
  (`TQP_ENV`) from the system Python 3.12 with exactly `bench/native.lock`: the
  image's packages (`requirements.txt`, `requirements-dev.txt`, cmake, ninja) but
  torch 2.8.0+cu128, since the image's 2.6.0+cu126 has no sm_120 kernels. It then
  builds this checkout's modules as the image's builder stage does, with the box's
  CUDA toolkit (`CUDA_HOME`, 12.8 or later), for the visible GPUs' architectures
  (`CUDA_ARCHITECTURES`, e.g. `120`) and against the nvCOMP wheel cuDF pins,
  stages them into `TQPlib/<pkg>/` (gitignored), and checks that a torch and a cuDF
  kernel run and that the process holds one nvCOMP. After a C++/CUDA edit, run
  `bench/native_env.sh modules`; after a requirements change, `bench/native_env.sh
  lock`.
- **The bench.** `TQP_NATIVE=1 bench/run_tqp.sh <SF> [Q]` is `make bench`'s
  tqp_vortex half, with the same protocol, timeout and run CSV: `main.py` with the
  venv's Python (`TQP_PY`), `CUDA_VISIBLE_DEVICES=$TQP_GPU`,
  `TQP_TABLES_PATH=$TQP_DATA_DIR`, `OMP_NUM_THREADS=32`, a 64 MiB stack (the compose
  service's), its CPUs bound to the GPU's NUMA node with `taskset` (`NUMA=off`:
  unbound) and `TQP_NUMA_BIND=0` unless set. A timeout or an interrupt kills the
  process's own tree. The run CSV is merged with `$PY` (default
  `/opt/envs/py/bin/python`, the baselines' native Python). For example:
  `TQP_NATIVE=1 TQP_GPU=1 DATA_DIR=/workspace/tpch-data RUN_CSV_DIR=$SCRATCH/smoke
  bench/run_tqp.sh 1`.
- **The spill pass.** `SPILL=1` on the same command is `make spill`'s tqp_vortex
  half (`baselines/AGENTS.md`, "Spill pass"; `SPILL_SMOKE=1` for the forced
  pressure); its record builder, check and merge run with `$PY`.
- **By hand**, with the same environment, e.g. `CUDA_VISIBLE_DEVICES=1
  TQP_NUMA_BIND=0 TQP_TABLES_PATH=/workspace/tpch-data taskset -c <node CPUs>
  /opt/envs/tqp/bin/python main.py --SF 1 --q 6` (or `… python test.py
  tests/allocator`). No preflight runs, and `query_times.csv` and `results.out`
  are written at the checkout root (the image's entrypoint links the first into
  `out/`).

What differs from the image: Python 3.12 (3.11) and torch 2.8.0+cu128
(2.6.0+cu126); the pinned pool is placed by first touch where `mbind` is refused;
the box's memlock limit applies (`cudaHostRegister` pins under a 64 KB one).

## Memory: pools and sizing

### GPU: the cmp pool, the cache region and the operator space

All GPU memory the engine uses comes from one arena, the cmp pool. With a cache, the
arena's top blocks are the granule cache's region, behind a boundary; everything
below the boundary is the operator space, the headroom every operator computes in:

```
GPU 0 capacity (nvidia-smi)
├── outside the arena: capacity − cmp pool (2 GB with the default)
│     CUDA context, and anything allocated outside torch
│     (e.g. nvCOMP scratch under TQP_MGR_SCRATCH=cuda)
└── cmp pool (--cmp_pool_gb): the arena, reserved with one cudaMalloc at startup;
    │ every torch CUDA allocation comes from it (1 MiB blocks)
    ├── cache region (--slot_pool_gb): the arena's top blocks, behind the
    │     boundary, out of the arena's free list; one granule of the granule
    │     cache per 1 MiB block, paged in from the top; a run without a cache
    │     (and an occurrence under TQP_CACHE_CORE=cpu) has none
    └── operator space = cmp pool − cache region: every torch and cuDF
          allocation, contiguous, first fit from the bottom, e.g. per-tick
          H2D/D2H buffers, decompressed columns, hash tables, sort and
          argsort scratch, rearrange gather windows, aggregation runs and
          slabs (pass 2 holds one round's run and its stage slab), nvCOMP
          scratch; an allocation that misses borrows the region's lowest
          blocks (lending, below)
```

- **Headroom is what runs out.** Out-of-memory failures at large SF come from the
  operator space, whose per-tick peak grows about linearly with `--chunk_size_mb`. A
  larger region buys cache capacity at the cost of headroom, which lending gives
  back on demand (below) as far as the cache can empty its lowest blocks; a larger
  chunk needs a larger `--cmp_pool_gb`, a smaller `--slot_pool_gb`, or a region
  the operators can borrow from. Two holdings do not shrink
  with the chunk: a join core tick holds at least one whole bucket pair, and a keyed
  aggregation's pass 2 one round's run (the whole run at B = 1), each predicted
  to fit `--partition_budget_mb` unless no B up to the operator's cap does (a dense
  aggregation state is its whole key domain, whatever the budget).
- **The region is the cache's, and only the operators'.** Only a run with a cache
  (`--buffer_strategy` `lru`, `mru` or `dataflow`) sets one. With `--buffer_strategy
  cpu` (the default) there is none: the whole cmp pool is operator space, and
  `--slot_pool_gb` and `--slot_pool_gb_list` change nothing. With a cache, each query
  occurrence sizes the region after its scan, before its timer starts, and empties it
  at its end: the scan never uses the cache and the cache is empty between queries,
  so the scan has the whole arena (a 14 GiB region of a 30 GiB arena failed Q13's
  SF100 scan otherwise). An occurrence under `TQP_CACHE_CORE=cpu` sizes none.
- **The arena's stats, window peaks and the OOM diagnostic describe the operator
  space** (`memory/README.md`): `get_arena_stats` reports its capacity (lent blocks
  included), allocated, free and largest free run, the free run that ends at the
  boundary (`top_free_run_b`), plus `region_b`, `lendable_b` (what a miss could
  borrow from the region now) and `lent_b`. The sizing search's transients (window
  peak − allocated at entry) read them. Its **budget** is the whole arena's room:
  operator capacity + `lendable_b` − allocated − 512 MiB, with the contiguity belt
  capped at the larger of the largest free run and the run at the boundary plus
  `lendable_b`. A lend does not change it, so it decides every size and is every
  level the search records or compares; a size within it runs even when the
  operator space alone could not hold it, its allocations borrowing from the
  region. Without a cache (or under `TQP_CACHE_CORE=cpu`) `lendable_b` is 0.
  (The search counts `lendable_b` at its decision and the lend happens at the
  allocation, which a pin may refuse.)
- **Lending, at every allocation, in every `--sizing_search` mode.** Once an
  occurrence's cache is attached to its region (after the scan), an operator
  allocation that still misses after the arena has folded in every
  pending free borrows the region's lowest blocks, inside the allocator and all in
  C++: no Python, no GIL, no slot governor (there is none). Every operator
  allocation is torch's; cuDF allocates from the arena only during the scan, before
  any cache is attached, so its allocations never lend. The arena takes the
  fewest blocks that, with the free run ending at the boundary, hold the request,
  synchronizes the device, and the granule cache's core empties them: its surplus
  evicted in victim order (a dirty granule copied home first), the granules left in
  those blocks moved to free slots higher up, both by copies on the home writer's
  stream that end before the blocks are handed out; then the boundary moves up and
  the allocation is tried again. A pinned granule (a tick's reads, or a read outside
  the ticks) or one admitted this tick is neither evicted nor moved, so a lend that
  needs one gives nothing and the allocation is the usual `Pool out of memory`
  (all or nothing). The lent blocks go back to the region when they are free again,
  at every pipeline's start and only there (an admission short of slots evicts as it
  would with nothing lent): the
  boundary moves back down over the free blocks just
  below it, up to those still lent; the region empties at the query's end anyway,
  and its lent count with it. Each query with a cache
  prints one `[lend] Q<n>:` line (even under `--no_debug`, not for the warmup):
  the misses the cache lent for (and those it could not, `short`, or that still
  failed), the blocks lent and returned and those still lent at the end, the
  granules evicted per class (and the dirty bytes copied home), those relocated and
  in how many copies, and the host time of the lend path; the perf report gets a
  `lend` section, and under experiment telemetry the movement ledger files the
  evictions as a reclaim's (`spill_evict`, `avoided_clean`, `avoided_dead`).
- The arena is reserved whole at startup, so on a card with another tenant the
  default size cannot be reserved (rule 1).

Flags:

- **`--cmp_pool_gb <int>`** (default GPU 0's total memory in whole GiB, from
  `nvidia-smi`, minus 2; minimum 1) — arena size in GiB. Startup fails with `could not
  detect GPU memory` if `nvidia-smi` gives no answer and no value is passed.
- **`--slot_pool_gb <float>`** (default `4.0`) — the cache region's size in GiB, with
  a cache: whole 1 MiB blocks at the arena's top, set for each query occurrence's
  operators, after its scan (the warmup's SF1 pass uses 4 GB).
- **`--slot_pool_gb_list <float> [<float> ...]`** (default none) — one region size
  per `--q` entry (the lengths must match); takes precedence over `--slot_pool_gb`.
  Each occurrence's region, set after its scan, takes the list's size (the cache,
  empty between queries, takes the region's blocks); nothing is allocated. A size
  other than the previous occurrence's prints `[resize] Q<n> cache region -> <gb> GB:
  region <b> B, operator space <b> B (<b> B free)`, a `TQP_CACHE_CORE=cpu`
  occurrence's too. The cmp pool stays fixed while the operator space varies — for
  ablation sweeps. A grow the arena refuses (a block the scan left allocated below the
  boundary) fails the occurrence with `the arena refused to grow the cache region`.
  Under `--sizing_search`, a size change from the list drops the search's
  converged budgets, and those budgets are kept per region size, so occurrences
  with and without the cache (`TQP_CACHE_CORE`) each reuse only their own.

### Host: the pinned pool

Scanned columns and spilled intermediates live in one pinned host pool, freed and
re-used per query. Each scanned column is held as RAW plus at most one compressed rep
(the codec the active compression mode assigns), one region per scan batch of a whole
number of 2^20-row quanta, at most 256 MiB, or one quantum when that is larger (only
rows over 256 B), so a batch boundary is a record boundary of every column (see
`TQP_MGR_RECORD_MB`):
every rep is written by a region producer, one tick per batch, its batch regions
carved from per-column pinned chunks (16 MiB; a larger region gets its own) that the
table frees (`RecordTable.host_blocks`). The scan decodes on the GPU and stages
nothing in pageable memory.

**Every page of the pool is on GPU 0's NUMA node.** Each 128 GiB segment is an
anonymous mapping bound to the node with `mbind(MPOL_BIND)` and pinned with
`cudaHostRegister`, so the placement does not depend on which CPUs the threads run on
or on how much memory is free at that moment, and pinned pages never migrate. A bound
pool cannot spill to the other node: the kernel evicts the node's page cache to make
room, and past that would run the OOM killer, possibly on another user's process. So
startup stops with `NUMA node <k> cannot hold a <n> GiB pinned pool` unless the node's
free memory plus page cache, minus 16 GiB of headroom, covers the pool. The node
comes from GPU 0's PCI address (`nvidia-smi`, then sysfs); if it cannot be resolved,
startup stops rather than guessing. The node has roughly half the host's RAM on a
dual-socket machine; that, not the host total, bounds `--pinned_pool_gb`.

- **`--pinned_pool_gb <int>`** (default: the formula below) — pinned pool size in GiB
  for the real run; must be positive. The warmup's SF1 pool uses the formula at SF1
  whatever this says, so a query that exhausts it fails its SF1 warm pass and is not
  timed; raise `main._PINNED_NEED_PER_SF` for that.

Without `--pinned_pool_gb` the pool is `max(1, ceil(1.05 × 0.6524 × SF))` GiB
(`_pinned_capacity_gb` in `main.py`): 7 at SF10, 69 at SF100, 686 at SF1000. It must
fit on GPU 0's NUMA node, which bounds the SF a host can run: on a 921 GB node, up to
SF1321 if nothing else held memory there (preflight prints this bound). A query that
needs more than the pool stops with `PinnedPoolExhausted: pinned host pool exhausted:
…`; pass a larger `--pinned_pool_gb`.

A box that refuses `mbind` (an unprivileged container without `CAP_SYS_NICE`: `mbind
to NUMA node <k> failed: Operation not permitted`) runs with `TQP_NUMA_BIND=0`, as
the native runs do. The pool is then pinned unbound (`cudaHostRegister` alone), with
no node capacity check, and each page lands on the node of the thread that pins it.
The real run's pool is pinned after `main.py` has pinned its threads to the GPU's
node (rule 2), so it lands there while the node has free memory; the warmup's SF1
pool is pinned before that. Startup prints `(not bound: TQP_NUMA_BIND off)`.

### UVM mode: the managed arena

Under `--uvm` (see "UVM mode" in the flag reference) there is no pinned pool
and no slot pool: the arena (`--uvm_pool_gb`) is one `cudaMallocManaged`
range, every torch and cuDF buffer is in it, and the driver migrates its
pages between the GPU and host memory. `--cmp_pool_gb` is the GPU budget, the
most the managed pages may hold on the GPU: a balloon of plain device memory
holds the rest of the card for the life of the process. The balloon is sized
after the warmup (under `--no_warmup`, before the first query), from the
card's free memory then, so the modules the queries load are not taken from
the budget; it prints `[uvm] managed arena <a> GiB, GPU budget <b> GiB
(balloon <c> GiB)`. A card with less free memory than the budget (another
tenant) stops the run there, `[uvm] GPU 0 has <f> GiB free, less than the GPU
budget …`: use an idle GPU or a smaller `--cmp_pool_gb`. Before each
occurrence's scan, and once after the warmup, the arena is freed and made
again, so no page an earlier query touched is resident anywhere; a live
allocation fails the occurrence (after the warmup, the run). Managed pages
are pageable and not bound to a node; the arena's default size is the budget
plus GPU 0's node's free memory and page cache, less 16 GiB. The calls
(managed `init`, `reserve_device`, `remake`, `prefetch`) are described in
`memory/README.md`.

## Repeated runs & medians

To get repeated-run stability, **repeat the query numbers** in `--q` (e.g. `--q 1 1 1`,
or `Q="1 1 1"` with `make q`). The pools are warm and reused across occurrences. At the
end of the run, any query that appears more than once gets a **median** line (with
min/max and the raw per-run times), e.g.:

```
====================== Repeated-run medians ======================
[median] Q1 over 3 runs: 0.1234s (min 0.1200, max 0.1280; runs: 0.1280, 0.1234, 0.1200)
```

Per-occurrence times are also appended to `query_times.csv`. With `--perf_report`,
each occurrence overwrites `SF<sf>_<qid>.json` unless `TQP_PERF_PER_OCCURRENCE` is set
(then `SF<sf>_<qid>_occ<i>.json`, `i` = position in `--q`). Repeating qids reuses the
warm pools; it does **not** re-pin the host pool per occurrence — that cold-start
variance is not what this measures.

## `main.py` flags — complete reference

Invoke as `make q SF=<sf> Q="<ids>" ARGS="<flags>"`, or inside `dev` as `python3
main.py <flags>`. Every flag, with its exact type and default:

### Workload

- **`--SF <int>`** (default `1`) — scale factor. Scans `sf<sf>/parquet/` under
  `TQP_TABLES_PATH` and sizes the pinned pool. Unless `--no_warmup` is given, the
  SF1 dataset must exist as well (the warmup runs at SF1).
- **`--q <int> [<int> ...]`** (default `1 … 22`) — TPC-H query IDs, run in the order
  given. Repeat an ID to repeat the query (see Repeated runs).
- **`-h`, `--help`** — print the flag list and exit.

### Run control

- **`--no_warmup`** (flag) — skip the warmup. By default, before the real run, an SF1
  pass runs Q1 and then each distinct requested query, with debug output off,
  a 4 GB cache region (with a cache) and the SF1 pinned pool; it writes no timings or perf reports. The real run skips a requested query that
  fails in it and prints `[QFAIL] Q<n>: SF1 warm pass: <error> (not timed)` instead. A
  failing Q1 counts against Q1 only; every other query is warmed by its own SF1 run.
  Skipping the warmup leaves CUDA context, kernel and pool priming to the real run: use
  it for correctness runs (e.g. `tests/e2e`), never for timing runs.
- **`--smoke`** (flag) — run `tests/smoke` in-process under pytest and exit with its
  status. The suite runs SF1, all 22 queries, `--buffer_strategy cpu`, adaptive
  compression, and compares every row of each result, in any order, with
  `tests/verification/SF1_reference.out` (DuckDB's answers on the standard SF1 data,
  see `tests/README.md`) at 10 significant figures. `--SF`, `--q`,
  `--buffer_strategy` and `--sizing_search` do not apply (sizing is disarmed); the
  arena is sized by `--cmp_pool_gb`. No warmup, no NUMA pin.

### Memory

`--cmp_pool_gb`, `--slot_pool_gb`, `--slot_pool_gb_list` and `--pinned_pool_gb` are
documented with the pools they size in "Memory: pools and sizing".

### UVM mode

- **`--uvm`** (flag) — the UVM baseline: the same plans, leaf kernels,
  partitioning and chunk budget, with every column and intermediate in CUDA
  managed memory and the driver migrating pages ("UVM mode: the managed
  arena"). The UVM executor (`uvm/`) runs each operator as a loop over
  morsels of plain tensors, a morsel being `--chunk_size_mb` over the
  operator's input bytes per row, at least one row and at most 2^30; there
  is no pinned pool, record table, tick pipeline, explicit copy,
  compression, cache or search. It runs every operator of the 22 queries.
  `--cmp_pool_gb` is the GPU budget; `--SF`, `--q`, `--no_warmup`,
  `--perf_report`, `--log`, `--no_debug`, `--torch_profile`,
  `--cuda_profile`, `--chunk_size_mb` and `--partition_budget_mb` work as
  without it. It rejects the flags of the streaming machinery (see "Invalid
  combinations") and ignores `--compression` (every column is RAW),
  `--mask_selectivity` and `--slot_pool_gb`, which it sets to RAW and None
  at parse time with one startup line.
  The host start: the scan (`uvm/scan.py`) decodes each column into one
  managed tensor and prefetches it to GPU 0's NUMA node before the timer,
  the mode's only hint to the driver, so the query's first GPU access to
  each page faults it in. The timer, `query_times.csv`, the perf report's
  `wall`, `results.out`, the medians, the warmup and the NUMA pin are the
  streaming engine's; under `--uvm` the timed region holds every migration
  the query causes (host to GPU, eviction and back). Every kernel call is
  bounded in rows (a morsel, `uvm.morsel.MORSEL_ROWS_MAX`; a sort and a
  residual condition's pairs, `operators.operator_utils.INT_ROWS_MAX`; a
  join core call, `join_core.MAX_ROWS`): a call over its bound fails the
  query with an error naming the operator. The morsels, the
  aggregation's and join's partitioning and the gathers are described in
  the module docstrings of `uvm/` (`uvm/__init__.py` lists them); the
  memory calls in `memory/README.md`.
- **`--uvm_pool_gb <int>`** (default: the GPU budget plus GPU 0's NUMA node's
  free memory and page cache, minus 16 GiB) — the managed arena in GiB, only
  with `--uvm`.

### Execution

- **`--buffer_strategy {cpu,lru,mru,dataflow,freelist}`** (default `cpu`; case-insensitive) —
  the GPU cache. `cpu`: no cache. `lru` / `mru`: a cache in the cache region with that
  eviction policy. `dataflow`: a cache that evicts against the query's dataflow graph
  (data with no remaining consumer is dead). The graph lists exactly the pipelines
  that run, in order; the cache's `advance` asserts that each pipeline's graph
  position is the counter's next value. The cache is the **granule cache**
  (`IO/cache/native_cache.py`, its bookkeeping in the C++ core `TQPlib.tqpcache`;
  `TQP_CACHE_CORE=cpu` runs an occurrence without it): cache-on runs take the paths a run without a cache
  takes (the region producer, the table scan, the region readers), and the cache
  holds granules of the producer's regions in the cache region, one per 1 MiB arena
  block (slot `i` is arena block `i`, handed out from the top), admitted at pack
  time and written into their slots on the home writer's stream, behind the
  copies home of the victims they replace (a region under one granule goes home
  instead, `TQP_CACHE_FLOOR`), copied home when evicted (cleaning ahead of
  eviction is off, `TQP_CACHE_CLEAN`) and read piece by piece (slots by D2D, the
  home by H2D; one `pieces` call per tick for the stages that read only cached
  regions). A gather's outputs are never cached. The region is lent to the operators on demand ("Memory: pools and sizing",
  "Lending"); each query prints a `[lend]` line.
  `freelist`: the **free-list baseline** (`IO/pages/device_regions.py`), no
  granule cache and no cache region. Intermediates follow `--compression` like any
  strategy's (`TQP_RAW_INTERMEDIATES` makes them RAW), and each region the region
  producer writes stays on the GPU as one arena allocation, first fit beside the
  operators' temporaries, with its pinned home reserved but not written: a RAW
  output's region is its output tensor, a compressed output's its packed records,
  compacted from the compress slabs into a buffer of their own; an eviction copies
  either home as it is. When an allocation misses after the arena
  has folded in every pending free, the arena synchronizes the device and the
  allocation stalls while spillable regions are evicted **one at a time, least
  recently used first** (registration and each read make a region the most
  recent; regions made recent together are ordered arbitrarily), skipping those
  pinned by a read in flight: each is copied to its home, the copy is waited for,
  the region becomes a host region (read H2D from then on), its blocks coalesce,
  and the allocation is tried again. It stops at the first fit; when no unpinned
  region is left the allocation is the query's `Pool out of memory`. Readers move GPU
  regions D2D; a rearrange gather reads them in place. Each query prints a
  `[freelist] Q<n>:` line (misses, of them fragmented — the free space held the
  request but no run did — and failed; evictions, regions and bytes moved and the
  time; the regions' peak in data bytes and in whole arena blocks; the arena peak)
  and the perf report gets a `freelist` section. Checks that read the arena's free
  space (a gather's streamed-or-UVA choice) count the spillable regions as free.
  Not with `--compression_search`, `--sizing_search` or `--slot_pool_gb_list`.
- **`--freelist_reserve_gb <float>`** (default `0`, none; `--buffer_strategy freelist`
  only, below `--cmp_pool_gb`) — the free list's **reserve**: that many GiB at the
  arena's top kept free through each query's timed run, for the allocations
  eviction cannot serve (every region pinned by a read in flight, or the free space
  split by pinned regions and live temporaries). After the scan, before the timer,
  the arena's region boundary moves down over them (the free list has no cache
  region), and nothing allocates there. An allocation that still misses once no
  unpinned region is left draws on it: the boundary moves up by what the request
  lacks beyond the free run just below it (all of the reserve when that is not
  enough), and the arena tries again. While short, the reserve refills from the free
  run below the boundary at each region registration and unpin; it is given back at
  the query's end. The operator space is the arena less the reserve, and the
  admission checks see only it. The `[freelist]` line adds the reserve's draws, the
  bytes they gave and took back, and the least it held (the perf report's
  `reserve_*` fields).
- **`--chunk_size_mb <int>`** (default `256`, MiB) — **the one chunk budget.** It
  budgets every chunked pipeline: scan/filter/project, aggregate, join core, the join's
  sort pipe, and the rearrange gather window; and the final sort's decode of a column
  larger than it that only follows the order (window by window). Tick pipes charge it as *input*
  bytes per tick; rearrange charges *output row bytes + argsort scratch* (24 B/row), so
  the same number buys a smaller window there — by design. **It rounds down to whole
  quanta**: a tick ends on a record boundary of every column it reads (its rows a
  whole number of the pipe's quantum, 2^20 rows for a pipe that reads or writes a
  mask or a column of up to 4 bytes, 2^19 with 8-byte columns only), so at 56 B a row with a mask a
  256 MiB tick holds 224 MiB; the join core packs whole radix partitions, a gather
  window whole records of its output. A pipe whose quantum costs more than the budget
  (wide rows, small budgets) is cut by rows instead, and its reads trim the records
  at each tick's ends (`IO/pipeline/README.md`). Raising it cuts per-chunk
  barriers and deepens in-flight PCIe at a ~linear cost in compute headroom (the
  rearrange leg dominates; see "Memory: pools and sizing"). With `--sizing_search`
  on, the searched budgets start from the sizing config instead (see below).
- **`--partition_budget_mb <int>`** (default `512`, MiB) — the partition budget:
  the bytes of the largest unit an operator holds whole, a join's bucket pair or a
  keyed aggregation's pass-2 round. Each join and each keyed aggregation takes its
  own partition count B, a power of two (at 1 every row is in one bucket), as the
  smallest B whose largest unit is predicted to fit (`operators/partitions.py`),
  then packs whole measured buckets: a join's core into its ticks, an
  aggregation's pass 2 into rounds within the budget. The prediction spreads an
  input's rows over an estimate of its distinct keys, with every key's rows in one
  bucket and the fullest bucket's keys at about 90% (Raab–Steger's
  `mean + sqrt(2 mean ln B)` when loaded, else the binomial tail
  `B·P(X ≥ k) ≥ 0.1`). When no B fits, the smallest B with the least predicted
  unit. Skew among the keys (one key much heavier than the rest) is not predicted.
  B is not searched by `--sizing_search`. At SF1 every join and keyed aggregation
  takes B = 1 at the default.
  - **A join** takes B in [1, 512] (`partitions.JOIN_B_MAX`, the partition
    kernel's limit). Its unit is a bucket pair: its left and right rows at the
    bytes per row the join core reads (the sorted key and condition columns, each
    its element size times its width, a string's full width, plus the 8-byte
    gidx). The prediction, per side: the rows its sort keeps (its rows times the
    selectivity the filter recorded on the mask the sort folds, so exact), spread
    over an estimate of its distinct keys (a side proven unique on the partition
    key has one row per key; the other side of such a join is taken to draw its
    keys from the unique side's rows before any mask: a scanned table's key
    domain under referential integrity, or for a join output only its matched
    rows, an under-count that only raises the prediction; otherwise every row is
    a key). When no B fits: 512 for a join of many keys, less for one of few (a
    nation-key join of 800M rows at SF1000 takes 256: up to two keys may share a
    bucket at 256 or 512, and two are already over the budget). A bucket is
    un-splittable (the core packs whole buckets into ticks, at least one per
    tick), so this is the floor of the core's per-tick input, whatever
    `--chunk_size_mb` says. Few distinct keys that no uniqueness proof bounds are
    not predicted either. B is the same for both sides of a join. With debug on,
    each join logs `[join_b]` (B, each side's rows, bytes per row and keys, and
    the largest pair predicted against the budget, ` OVER` when over), then
    `partitioned join: B=… num_ticks=…` with the estimated and partitioned bytes
    and the largest bucket pair, then its per-bucket rows.
  - **A keyed aggregation** takes B in [1, 256] (`partitions.AGG_B_MAX`: pass 1
    writes B streams per output). Its unit is a pass-2 round, adjacent buckets
    merged, emitted and reset together: the partial rows pass 2's run holds,
    times the run's bytes per row. B is fixed before pass 1
    (`partitions.agg_partitions`) from:
    - P, its input rows, times the selectivity the filter recorded on the mask
      pass 1 reads, if it reads one;
    - T, pass 1's ticks: the bytes of every column it reads (its mask
      included), at its rows before the mask, over pass 1's chunk;
    - G, a static bound on its groups: the product, over the GROUP BY keys, of
      each key's dtype's values (int8 and int16, per element of its width, so a
      string's 256 per byte); unknown when a key is wider, and then every row
      is taken as its own group;
    - w, pass 2's run bytes per row: on the key route (one scalar grouping key,
      emitted, every aggregate mergeable) the key and the values, else the 8-byte
      hash, the emitted keys and the values; a partial state (an AVG, or a
      result over two or more aggregate calls) counts its fields, one per SUM
      or COUNT call and two per AVG;
    - on the key route, a bound on the key's range (its largest minus its
      smallest value plus one), at which pass 2 caps the run: an integer key's
      dtype's values (int8, int16); none for a wider key.

    A tick emits at most one partial row per group, so pass 2 holds
    R1 = min(P, T·G) partial rows. B is 1 when that run fits the budget (on the
    key route counted at no more than the key range's bound), else the smallest
    B whose largest bucket, of R1 rows over G keys, is predicted to fit. A dense
    shape (one scalar integer grouping key, every aggregate a partial state: the
    direct-addressed state of "Aggregation pass 2") takes 1 by choice: an
    allocated dense state spans the key domain whatever the buckets, so
    bucketing pass 1 would only cost. The risk is on the other side: a declined
    dense state falls back to the key route unpartitioned, a run of the whole
    observed key range, and a key whose range is over its partial rows never
    gets the dense state and runs the key route on all of them; either run can
    be over the budget. G and the key range only choose B: pass 2 sizes every
    buffer from measured rows, so a loose or missing bound costs speed, never a
    result. With B > 1, pass 1 still groups on the grouping hash; it buckets
    each tick's groups by the top log2(B) bits of the hash's mix64 (the join's
    partition function, computed in torch's own ops, so no kernel compiles
    inside the first partitioned query's timed region), reads the tick's
    bucket rows to the host once, right after `torch.unique`,
    and writes every output in the sort pipe's padded layout, one record-table
    stream per bucket. Pass 2 packs the measured buckets into rounds
    ("Aggregation pass 2"). The module constant
    `operators.aggregate.AGGREGATION_PARTITIONS` (default `0`: from the budget)
    forces B when positive, a power of two, dense shapes included. With debug on,
    each keyed aggregation logs `[agg_b]` before pass 1 (B and its reason,
    `forced`, `dense`, `fits`, `budget` or `at cap`; the predicted round, but for
    `forced` and `dense`; P, T, G with the bound each key takes; w, the route, on
    the key route the key range's bound and what gives it, and the budget;
    ` OVER` when the prediction is over), and with B > 1
    `partitioned pass 2: B=… rounds=…` (the route, the run's bytes per row, the
    largest round's rows and bytes against the prediction, the run's capacity,
    the stage slab; ` OVER` when the largest round is over the budget).
- **`--mask_selectivity <float>`** (default `0.1`; `0` disables) — a column whose rep is
  RAW and entirely in pinned host memory, filtered by a mask that keeps fewer than
  this fraction of rows, is read zero-copy over UVA instead of being staged H2D.

### Compression

- **`--compression {adaptive,RAW,ANS,CASCADED,BITCOMP,LZ4,SNAPPY,GDEFLATE,DEFLATE}`**
  (default `adaptive`) — `adaptive`: each column gets the codec its `(mapped type,
  sorted)` class maps to in the codec map (see "Adaptive codec table"), for scans
  **and** intermediates; intermediate column types come from the static `col_types`
  annotation of the in-memory dataflow graph. Naming a codec pins it for every read and
  write; `RAW` means no compression. The seven codecs are the nvCOMP manager codecs
  built into `compression/nvcomp_manager.cu`, each with fixed options; their chunk
  sizes come from the codec map. The scan materializes RAW plus at most that one
  compressed rep per column. Some columns are always RAW (see "Columns that stay
  RAW").
- **`--compression_search {off,read-only,read-and-write,fixed}`** (default `off`) —
  search over which columns of a pipe to compress, composed with `--compression` (the
  policy supplies each column's single compressed candidate).
  - `off`: no search.
  - `read-only`: per-pipe search over the read leg across a pipe's ticks; it ranks
    columns by benefit and ternary-searches how many to compress.
  - `read-and-write`: the read-leg search plus write-leg probe windows. Write picks
    take effect at tick t+1 (each tick writes one region per column); columns
    whose regions mix codecs are accepted downstream.
  - `fixed`: no search; applies `--fixed_search_json`'s per-column pins.

  A pipe with too few ticks gets no search, so at the default 256 MB chunk few SF1
  pipes search at all. With `--compression RAW`, `read-only` and `read-and-write`
  degrade to `off` (printed). Search outcomes land in the perf report's
  `compression` section (`TQP_EXPERIMENT_TELEMETRY`). The searches compare per-tick
  lanes, whatever the telemetry switch says, all spans with host
  gaps: per transfer stream, the first copy batch's start to the last one's end
  (`IO/telemetry/copy_events.py`), and the compute stream's spans (`compute`,
  and inside it `compress_out`, the tick's output-compress step). They read
  no timing of their own.
- **`--fixed_search_json <path>`** (default none) — required by, and only valid with,
  `--compression_search fixed`. Format: `{"<qid>": {"<column tag>": "<codec>"}}` over the
  dataflow graph's instance names (`<pipe name>:<column id or label>`, e.g. `scan 5:71`,
  `join 7 core pipeline:out_index_left`, `deferred gather 6 g0:54`) and the codec names.
  Validated at load: an unknown codec, a tag not in the query's dataflow, a non-RAW pin
  on an untyped tag, or a non-RAW pin on a scan tag naming a rep the scan does not
  materialize is rejected before the run. Unlisted tags keep the
  policy's choice. Needs a non-RAW `--compression`. `tests/verification/SF1_fixed_search.json`
  is a complete example; `tests/dataflow/test_fixed_search_json.py` covers every rejection.

### Sizing search

- **`--sizing_search {off,per_pipe,mid_stream}`** (default `off`) — adaptive chunk
  sizing (`docs/adaptive_intermediate_sizing.md`).
  - `off`: every pipe runs at `--chunk_size_mb`.
  - `per_pipe`: every tick-loop pipe (sort / join core / aggregate / filter / project)
    starts at `c0` (`--chunk_size_mb` unless the config sets `c0_mb`). One that can
    be re-cut mid-run (sort, filter, project, aggregation pass 1; uniform geometry)
    learns its size on its event-ordered ticks (`TickJumpSession`): it measures 6
    chunks at a size (its tick time, the gap between consecutive compute ends plus
    its outputs' readback, and its transient per input byte), doubles once (the
    walk), then fits the tick time as `F + t·x` and moves to the size that
    minimizes the pipe's time `(F + t·x)·(D/x + 1)`, `sqrt(D·F/t)` for its `D`
    input bytes, capped by the size its measured transient per byte fits on the
    budget without the cache region's lendable bytes, by `c_max_mb` and by what
    can still be measured. A step is kept only
    when the pipe's measured time at it, `q·(D/b + 1)`, is `epsilon` lower, else
    the pipe is re-cut back and fixed. Join cores, aggregation
    pass 2 and the scalar aggregation run `c0`. A repeated qid starts each pipe
    where its last occurrence stopped. Each query prints one `[sizing] Q<n>:` line
    (even under `--no_debug`): every session's sizes, measurements and steps.
  - `mid_stream`: `per_pipe` plus a search over the rearrange gather window
    (sense, then march with escalating steps; each gather iteration is
    synchronized), from a 1 GB base window.

  The gather leg's budget counts what the cache region could lend (`lendable_b`),
  which the allocator lends on a miss ("Memory: pools and sizing"); the tick
  leg plans only into the operator space's own headroom. The startup
  lines `[sizing_search] …` print the
  effective parameters. Disarmed under `--smoke`. Out-of-range start budgets
  (`c0_mb` outside `[c_min_mb, c_max_mb]`) fail an assertion at startup.
- **`--sizing_config <path>`** (default none = built-in defaults) — JSON, all keys
  optional, unknown keys ignored. The tick leg's `c0` is `--chunk_size_mb` and the
  gather leg's **1 GB** unless `c0_mb` is set (the `--help` text says the start
  defaults to `--chunk_size_mb`; for the gather leg the code does not). Built-in
  defaults:
  ```json
  {"rearrange": {"c0_mb": 1024, "static_mb": null, "c_min_mb": 64, "c_max_mb": 8192,
                 "epsilon": 0.02, "samples_per_step": 1, "objective": "iter"},
   "march":     {"delta_mb": 256, "escalation": 2.0, "step_max_mb": 2048,
                 "patience": 2, "sense_delta_mb": 64, "up_probes": 2},
   "safety":    {"contiguity_gate": true, "grow_budget_frac": 0.9, "M_abs_mb": 512,
                 "pad_min_mb": 256, "seed_slope": 2.0},
   "tick":      {"c0_mb": null, "c_min_mb": 64, "c_max_mb": 4096, "samples": 6}}
  ```
  `epsilon` and `safety` serve both legs, `march` the gather leg; `tick.samples` is
  the chunks a pipe measures per size.
  An older config's `governor` section (the retired slot governor's floor) is
  ignored. `adaptive/sizing_controller.py::load_config` is the schema of record.

### Outputs & profiling

- **`--perf_report <dir>`** (default none) — creates `<dir>` and writes
  `cmdline.json` (argv, cwd, UTC timestamp) at startup, `query_times.csv` (truncated
  once the warmup and the real run's pool setup are done, then one row per query
  occurrence: a run that stops before that leaves an earlier run's rows in place),
  `query_results.csv`, one `SF<sf>_<qid>.json` per query, and a copy of `results.out`
  at the end. `query_results.csv` gets its header,
  `qid,status,approx_time,rows_or_error,load_time`, at startup and then one row per
  query occurrence: `OK` with the `approx_time` of `query_times.csv`, the result's row
  count (Q16 at SF ≥ 100 copies its result to the host, skips the frame and counts its final Sort's rows) and the
  load time (the scan, which runs before the timer); or `FAIL` with the time to failure
  (empty if the timer had not started), the exception's first line (commas become `;`)
  and the load time if the scan finished. A query whose SF1 warm pass failed is not
  run: `FAIL` with that pass's time to failure and `SF1 warm pass: <error>`. A
  report holds `wall` (`approx_time`, `occurrence`), with a cache `lend` (the
  `[lend]` line's counters: `misses`, `lent`, `short`, `failed`, `blocks_lent`,
  `blocks_returned`, `lent_now`, `ns`, `evicted_dead` / `_clean` / `_dirty`,
  `dirty_b`, `relocated`, `copies`, `block_b`) and under the free-list baseline
  `freelist`; with
  `TQP_EXPERIMENT_TELEMETRY=1` (see "Environment variables") it also holds:
  - `movement`, the movement ledger: per kind (`scan_h2d`, `spill_bypass`,
    `restore_h2d`, `cache_hit`, …), the pages, logical (decompressed) bytes and
    wire (stored) bytes of every host↔GPU copy the query made and every copy the
    granule cache avoided (`IO/telemetry/README.md`): `kinds` without the output
    step, `output` the output step's reads and its copy of the result
    (`result_d2h`);
  - `residency`, the most the timed run held in each memory tier: the arena's
    peak, the pinned pool's per tag (the intermediates, the scanned tables) and
    the granule cache's slots (`IO/telemetry/residency.py`);
  - `sizing_model` (the sizing search's predicted vs observed memory) and
    `compression` (the compression searches' outcomes per searched pipe);
  - each pipe's `peak_gpu_bytes`, the operator space's high-water mark over the
    pipe (the cache region not included), under its perf tag.

  A query that fails then still writes its report, partial: `wall` with `status`
  `FAIL`, its time to failure and error, and the `movement` and `residency` as far
  as it got. `cmdline.json` then also holds the settings as resolved (`effective`:
  the pool sizes, the cache, the codecs, the partition budget, the NUMA binding),
  the `TQP_*` and thread environment, the codec map, the versions, the GPU and
  `cpu_pin` (the node the threads were pinned to after warmup, None when the pin
  failed: with `TQP_NUMA_BIND` off also where the pinned pool landed).

  Under `--uvm`, `cmdline.json` holds `effective` whether or not the
  telemetry is on, with the settings the run uses (`compression` `RAW`;
  `pinned_pool_gb`, `slot_pool_gb` and `mask_selectivity` None),
  and `effective.uvm` holds the managed arena, the budget and the balloon in
  bytes (`arena_b`, `budget_b`, `balloon_b`), the driver version and every
  readable `/sys/module/nvidia_uvm/parameters` value (None without `--uvm`).
  With the telemetry, the `movement` ledger stays empty
  (the driver's migrations are not copies the engine makes) and `residency`
  holds the managed arena's peak (`gpu_arena`), with no `host_pool` or
  `slot_pool`.
- **`--log <dir>`** (default none) — besides stderr, write the engine's log streams to
  timestamped files in `<dir>`: `debug_<YYYYMMDD_HHMM>.log` (not under `--no_debug`),
  `perf_<…>.log` and `profile_<…>.log`. The debug log carries messages. A few
  quote numbers, such as each query's `Approx Time` (also in `query_times.csv`)
  and the sizing behind each aggregation-slab decision; results come from the
  perf report and `query_times.csv`, not from the log.
- **`--no_debug`** (flag) — silence the engine's message log (every level) and the
  per-query plan dump on stdout. Pool, pinning, search and median lines still
  print.
- **`--torch_profile <dir>`** (default none) — wrap every filter, aggregation and join
  operator in `torch.profiler` and export Chrome traces to `<dir>/<qid>/<operator
  tag>_trace.json` (e.g. `filter 3_trace.json`); the top CUDA ops by memory go to the
  profile log. **⚠️ Inflates elapsed time — never compare its
  `query_times.csv` to a non-profiled run's** (see the TIMING banner). `TQP_PROF_SHAPES`,
  `TQP_PROF_MEMORY`, `TQP_PROF_STACK` control its collectors.
- **`--cuda_profile`** (flag) — push NVTX ranges (per query and per operator) and call
  `cudaProfilerStart`/`Stop` around the real run only, so `nsys profile
  --capture-range=cudaProfilerApi python3 main.py … --cuda_profile` traces the timed
  run without warmup or the host pinning pass. `nsys` is in the engine image.

### Invalid combinations

These stop at startup with an argparse error:
- `--compression_search fixed` without `--fixed_search_json`, with a path that does not
  exist, or with `--compression RAW`;
- `--fixed_search_json` without `--compression_search fixed`;
- `--pinned_pool_gb` ≤ 0;
- `--partition_budget_mb` ≤ 0;
- `TQP_EXPERIMENT_TELEMETRY` on without `--perf_report`;
- with `--uvm`: `--smoke` (the smoke runs the streaming engine only),
  `--buffer_strategy` other than `cpu`, `--compression_search` other than `off`,
  `--sizing_search` other than `off`, `--sizing_config`, `--fixed_search_json`,
  `--slot_pool_gb_list`, `--pinned_pool_gb`, or `--uvm_pool_gb` ≤ 0;
- `--uvm_pool_gb` without `--uvm`.

A `--slot_pool_gb_list` whose length differs from `--q` fails an assertion when the
run starts.

## Engine settings that are not flags

- **The join core's path** follows from the join, per core tick
  (`operators/hashjoin.py`; the kernels are `operators/kernels/join_core.cu`,
  in `TQPlib/tqpkernels`, through `operators/join_core.py`). An inner join
  whose 1–2 int64 equality keys (the partition key among them) are statically
  proven unique on a side builds that side into a hash table (the smaller one
  per tick when both are), unless it has more than 8× the probe's rows or the
  table and outputs would pass 2 GiB. A semi or anti join on one int64 key
  builds a hash set of the build keys; a residual that compares one int64
  build column with one int64 probe column (`<>`, `<`, `<=`, `>`, `>=`, or
  `NOT` of a comparison, as Q21's `NOT (l_suppkey = l_suppkey)`) is answered
  by the column's min and/or max per key. A build key equal to int64 min (the
  set's empty-slot value) sends the tick to the sort/search core. Every other
  join sorts the build per radix bucket and expands each probe row's run of
  equal keys into pairs. A hash table has twice the build rows in slots. Each
  path waits for the GPU once per tick, for its output (or pair) count; on
  the sort/search core, a residual on the pairs, an outer join's unmatched
  rows and a semi or anti join's kept rows cost one more wait each.
- **Tick ordering** is per pipe; there is no switch. A pipe orders its ticks
  by events, with or without the granule cache: the host issues tick i+1
  while the GPU still computes tick i, and waits only on the tick event two
  ticks old (`IO/pipeline/README.md`, "Event-ordered ticks"). Only while a
  compression-search observer (`read-only`, `read-and-write`) of the pipe is
  live does every tick end with the tick barrier, a host sync of the compute
  and exchange streams, because its samples assume a quiet tick; once it has
  settled (it emits no more picks), the pipe's remaining ticks run on events.
  A pipe the compression search does not reach (too few ticks) runs on events
  throughout, as does every pipe of a sizing search alone (the tick sizing
  session measures on events). `--compression_search fixed` runs events.
- **Adaptive codec table and chunk sizes: the codec map.** Under `--compression
  adaptive` the codec for each `(mapped type, sorted)` class — every column the
  engine compresses, scanned or intermediate, is in its type's unsorted class (in
  parquet order only `O_SHIPPRIORITY`, a constant column, is non-decreasing) —
  and, under every mode, the nvCOMP manager internal chunk size of each codec
  come from **`profiling/results/codec_map.json`**,
  written by `make profiling` (`python3 -m profiling.sweep`, see "Microbenchmarks")
  and read only by `compression/format_policy.py`. The file is loaded **once per
  process, on first use** (`format_policy.codec_map()`), which `main.py` forces at
  startup and prints as two `[codec_map]` lines (source path and mtime or
  `defaults`, the eight codec cells, and each codec's chunk read back through
  `mgr_chunk_size`). Its shape:

  ```json
  {"codec":    {"int":    {"unsorted": "BITCOMP", "sorted": "CASCADED"},
                "float":  {"unsorted": "ANS",     "sorted": "CASCADED"},
                "date":   {"unsorted": "BITCOMP", "sorted": "CASCADED"},
                "string": {"unsorted": "ANS",     "sorted": "ANS"}},
   "settings": {"ANS": {"chunk_b": 131072}, "CASCADED": {"chunk_b": 32768},
                "BITCOMP": {"chunk_b": 131072}, "LZ4": {"chunk_b": 65536},
                "SNAPPY": {"chunk_b": 65536}, "GDEFLATE": {"chunk_b": 65536},
                "DEFLATE": {"chunk_b": 65536}}}
  ```

  The values above are the built-in defaults (`_DEFAULT_BEST` and
  `_DEFAULT_SETTINGS` in `format_policy.py`), used when the file is **missing** —
  then one warning prints to stderr, even under `--no_debug`: `[format_policy]
  WARNING: no codec map at profiling/results/codec_map.json; using built-in
  defaults. Run: make profiling TQP_GPU=<idle>`. A file that is **present but
  wrong** (bad JSON, a missing or extra key, an unknown codec, a non-positive
  chunk) raises at startup and stops the run. Every engine-image service mounts
  the host's `profiling/results/`, so `engine`, `test`, `experiments` and `dev` all
  run the same map; move the file away to compare against the defaults. The
  pytest suites pin the defaults through an autouse fixture in
  `tests/conftest.py`, whatever the file says. The chunk sizes are the only
  chunk setting: there is no per-codec environment override.
- **Gathers read their sources one of two ways**, per source column, whichever
  moves fewer bytes (`IO/pipeline/rearrange_pipeline.py`): over UVA, the indexed
  rows from a RAW rep, 32-byte sectors at a time, or streamed, the source loaded
  whole onto the GPU through a tick pipe (its records in their own codec,
  overlapped) and gathered there. A scan keeps a RAW rep, so a scanned source
  takes either; a producer's column (a join's output, a lineage) is in its policy
  codec like any intermediate, so it is streamed; a source too large for the GPU
  is read over UVA, first decoded home RAW if it has no RAW rep. Lineages and
  deferred-gather outputs are typed, so they are compressed too. Debug lines
  `[gather] <node> <source>: uva|streamed|raw_home <codec>` name each choice.
- **Aggregation pass 2** resolves as its own `agg` instance under the sizing search,
  but its chunk is only the input feed quantum. Persistent state uses one fixed
  aggregation-owned representation: dense typed state plus occupancy for compact
  partial-state domains, an in-place scalar or losslessly packed-composite sorted run, or a hash run
  carrying the full wide-key payload. There is no aggregation chunk flag. Pass 2 holds
  one round of its partials at a time. It runs one round, every row, at B = 1 (see
  `--partition_budget_mb`) or when the whole run fits the partition budget at the
  run's bytes per row (counted, for one scalar integer key, at no more than its
  observed key range), else it packs the measured buckets, in order, into rounds of
  adjacent whole buckets within the budget (`tick_cutter.pack`; a bucket over the
  budget is a round of its own); a dense state is one round, whatever the
  buckets. The fixed run is sized for the largest round, and at each round's end the
  round's groups are emitted (cloned unless it is the last round) and the run, or the
  dynamic accumulator, reset. Rows stage in a fixed slab of min(2^26, the largest
  round's) rows, freed before the last round emits, and a full slab merges in views
  of at most 2^26 rows (`operators/aggregate.py`,
  `docs/adaptive_intermediate_sizing.md`). The dense state, the fixed run and the
  stage slab are allocated as they are, borrowing from the cache region on a miss
  like every other allocation; when one still does not fit, pass 2 takes the next
  route (the dense state the sorted run's, the fixed run a dynamic accumulator:
  concatenate, `torch.unique`, reduce per feed; without the stage slab every feed
  merges directly). Any other allocation that does not fit fails the query there.

## Environment variables

Containers do not inherit your shell's environment: set engine variables with
`docker compose run --rm -T -e NAME=value <service> …` (`make` targets cannot pass
them). Three parsing rules recur below:

- **non-empty** — any non-empty string turns it on, **including `0` and `false`**;
  unset it to turn it off.
- **flag** — unset, `0`, `false`, `False` or empty is off; anything else is on.
- **on-word** — `1`, `true`, `yes` or `on` (case-insensitive) is on; anything else is
  off.

**Container and paths**

| Variable | Default | Effect |
|---|---|---|
| `TQP_GPU` | `0` (make: `$GPU`) | host-side (make / compose): the physical GPU the container sees |
| `TQP_DATA_DIR` | `./data` (make: `$DATA_DIR`) | host-side (make / compose): the data root the engine services mount at `/data/tables` |
| `TQP_TABLES_PATH` | `/data/tables` in the image | non-empty overrides `config.json`'s `"tables path"`: the data root, which holds each dataset's parquet, `sf<sf>/parquet/` |
| `TQP_OUT_DIR` | `/opt/TQP-Vortex/out` | entrypoint only: where the repo-root CSVs are symlinked |
| `TQP_SKIP_PREFLIGHT` | unset | exactly `1` skips `docker/preflight.py` in the entrypoint |

**Engine behaviour and performance** (these change results or timings — record any
you set next to the numbers)

| Variable | Default | Parsing | Effect |
|---|---|---|---|
| `TQP_WRITE_UNGATED` | off | non-empty | `read-and-write` search: write policy `ungated` (no benefit floor, blind greedy switch, encode gate off). Wins over `TQP_WRITE_PREDICTIVE`. |
| `TQP_WRITE_PREDICTIVE` | off | non-empty | `read-and-write` search: write policy `predictive` (P2 benefit floor plus greedy benefit threshold). With neither set the policy is `measured`: the greedy fallback switches to RAW only if a measured k=0-vs-incumbent probe lowers the bottleneck. |
| `TQP_WRITE_NO_ENCODE_GATE` | off | non-empty | `read-and-write` search: drops the encode gate, which otherwise stops the write search and keeps the incumbent codec when median(compute − encode) > median(transfer). |
| `TQP_MGR_RECORD_MB` | `4` | number, MiB (fractions allowed) | The raw-byte cap of a record. Every record of the scan (RAW and compressed) and of the region producer (every output, RAW included) holds a power-of-two number of rows, the largest within this cap and within 2^20 rows (the row cap, the module global `IO.codec.region_producer.RECORD_ROWS_MAX`, `None` for none; it keeps a mask's records, and so the quantum of a pipe with a mask, at 2^20 rows), raised to the fewest that keep it a 512-byte multiple (only for rows over 8 KiB), cut from the start of the scan's row space (row 0) or of the producer's (tick, bucket) slice, so record boundaries nest across a table's columns (1-byte mask and 4 B 2^20 rows, 8 B 2^19, a 25-byte string 2^17), and scan batches are whole 2^20-row quanta; ticks are cut in whole quanta and read whole records (`--chunk_size_mb`). Read at import into `IO.codec.region_producer.RECORD_MB` (`None` when unset: 4), which `switch_ab.py` can set per occurrence. |
| `TQP_MGR_SCRATCH` | torch allocator | exactly `cuda` changes it | By default nvCOMP manager scratch comes from the engine's arena; `cuda` leaves it in nvCOMP's own pool. |
| `KVIKIO_NTHREADS` | `32` | int | libkvikio's file-read threads, which cuDF's parquet reader (`IO/parquet_reader.py`) reads through; libkvikio reads it once, at the first read. Importing the reader sets it to 32 unless the environment sets it (1 thread decodes the SF100 parquet at about 9 GB/s, 8 to 32 at about 16). |
| `TQP_NUMA_BIND` | on | off only for `0`, `false`, `False`, empty | Bind the pinned host pool to GPU 0's NUMA node with `mbind` ("Host: the pinned pool"). Off pins it unbound, for a box that refuses `mbind`; the native runner sets `0` unless it is set. Read at import (`utility.numa.NUMA_BIND`). |
| `TQP_RAW_INTERMEDIATES` | off | on-word | Every intermediate (non-scan column instance) resolves RAW; the scan's reps still follow `--compression`. For any strategy, the free-list baseline's included. Refused with a `--compression_search` other than `off` (it would override the fixed search's pins and leave the searches' menus no compressed candidate). |
| `TQP_ARENA_BLOCK_B` | 1 MiB | int, bytes | The GPU arena's block size (a positive multiple of 512; not with `--uvm`): every allocation is rounded up to whole blocks and starts on a block boundary, and a granule of the granule cache is one block. `512` makes the arena byte-granular (512-B aligned), for the free-list baseline; below 1 MiB, cuDF's small buffers no longer share blocks during the scan (`utility/rmm_arena.small_buffers`). |
| `TQP_SERIAL_TICKS` | off | on-word | An experiment switch: every pipe runs under the tick barrier with nothing overlapped (a tick's input copies end before its compute starts; the pack's and the home writer's copies end before the next copy or compute). Read at pipe construction (`IO.pipeline.vortex_pipeline.SERIAL_TICKS`). Rearrange gathers are serial either way. |
| `TQP_FREELIST_SQUEEZE_GB` | unset | number, GiB | `--buffer_strategy freelist` only, a correctness stress (never a timing setting): hold that much of the arena through each query's timed run, so its operators and regions run in a smaller arena while its scan, before the timer, has the whole one (`tests/e2e/test_freelist.py`); the reported arena peaks include it. |
| `TQP_HASH_SEED` | `2029` | int | Base seed of the grouping-hash coefficients. 2029 is the seed validated collision-free (`tests/operators/test_hash_seed_injective.py`); another value can produce wrong group-by results. |
| `TQP_CACHE_CORE` | `native` | `native` or `cpu` | With a cache (`--buffer_strategy` other than `cpu`), whether each query occurrence runs it: the granule cache over the C++ core `TQPlib.tqpcache` (`native`, `IO/cache/native_cache.py`), or none for this occurrence (`cpu`: the occurrence sizes no cache region and has the whole cmp pool as operator space). Any other value fails each occurrence with a `ValueError`. Read into `IO.cache.config.CORE` at import; `main.py` reads it at the top of each occurrence, so `switch_ab.py` can alternate it (`docs/cache-host-overhead-investigation.md`). |
| `TQP_CACHE_POLICY` | unset | `lru`, `mru` or `dataflow` | The policy of each occurrence's cache instead of `--buffer_strategy`'s (`IO.cache.config.POLICY`); the cache is empty between queries, so the policy swaps. |
| `TQP_CACHE_FLOOR` | `1` | int, granules | A cacheable output region of fewer bytes than this many granules is not offered to the cache and goes home as a plain region (`IO.cache.config.FLOOR_GRANULES`, `IO/codec/region_producer.py`); `0` offers every region. |
| `TQP_CACHE_PLAN` | on | on-word | The reader derives each tick's cached spans, settle spans and granule ids when the tick's layout is compiled (`IO.cache.config.PLAN_AT_COMPILE`, `IO/transfer/tick_layout.py`), and binds the stages of a tick that read only cached regions with one `pieces` call, instead of per column per tick from the record selection; `0` (any non-on-word) turns it off. |
| `TQP_CACHE_CLEAN` | off | on-word | Cleaning: after a tick's admissions the producer copies the dirty granules at the front of the victim order (25% of the live residents) home, so their eviction needs no copy (`IO.cache.config.CLEANING`, `RegionProducer._clean`). Off, a dirty granule is copied home when it is evicted. At SF100 with a 2 GiB cache, cleaning wrote 22 GB home to spare 14.5 GB of eviction copies and the pass was 1.2% slower with it; not measured at larger SF. |

**Telemetry and debug switches** (`IO/telemetry/gates.py`). Each is read once, at
import, so setting it inside a running process changes nothing. Every one is on-word
(`TQP_VARCHK=0` is off) and off by default; a timing run sets none of them.

| Variable | Effect |
|---|---|
| `TQP_EXPERIMENT_TELEMETRY` | Writes the perf report's `movement`, `residency`, `sizing_model` and `compression` sections and each pipe's `peak_gpu_bytes`, a failed query's partial report, and the resolved settings in `cmdline.json` (see `--perf_report`). Its hooks run per run, region, granule or pool allocation, so they cost host time: an experiment takes these numbers from a separate run (the spill pass, `bench/run_tqp.sh` with `SPILL=1`, is one). Without `--perf_report` the run stops at startup. What engine decisions read (each tick's copy and compute spans, the per-slot output byte counters, the allocator window peak, a pipe's host time for the sizing search) is computed whatever it says. |
| `TQP_VARCHK` | Assert each compressed read's record index as its decode template compiles: positive original sizes that sum to the decoded size, records that do not overlap. |
| `TQP_PICK_DEBUG` | Record the expected codec pick state at each tick's rebuild horizon and assert the in-stages match it at launch. |
| `TQP_TIMED_GC_VALIDATE` | Raise if a cyclic garbage collection ran inside a query's timed region. |

**Diagnostics and debug**

| Variable | Default | Parsing | Effect |
|---|---|---|---|
| `TQP_PERF_PER_OCCURRENCE` | off | non-empty | Perf JSON per occurrence: `SF<sf>_<qid>_occ<i>.json` instead of overwriting `SF<sf>_<qid>.json`. |
| `TQP_QFAIL_TRACE` | off | non-empty | Print the full traceback after a `[QFAIL] Q<n>: …` line (the run continues either way). |
| `TQP_PROF_SHAPES` / `TQP_PROF_MEMORY` / `TQP_PROF_STACK` | **on** | off only for `0`, `false`, `False`, empty | `torch.profiler` `record_shapes` / `profile_memory` / `with_stack` under `--torch_profile`. Memory profiling forces shapes and stack on, so turning those off also needs `TQP_PROF_MEMORY=0`. |

## Other entry points

- **Plan text and pickles** (build-time tooling; the runtime loads the checked-in
  `queries/pickles/planobj-q<id>.pkl`): `make plans Q="20"` runs `python3 -m
  parsing.regenerate --q <ids>`. It captures Spark's formatted plan for
  `queries/tpch-q<id>.sql` into `queries/plan-q<id>.txt`, then lowers that text to
  the pickle with `ParsingLayer.lower` (pure Python). Plan column ids (`#NNN`) are
  Spark expression ids, drawn from a JVM-wide counter, so the tool refuses any
  pyspark but the one `requirements.txt` pins (`pyspark==…`) and captures each
  query in its own process (a fresh JVM). Regenerating any subset, in any order,
  reproduces the same ids. `--lower-only` re-lowers the checked-in text without
  Spark (after a lowering change). `tests/parsing/test_lowering.py` checks that
  every pickle is the lowering of its text and that the text was captured under
  the pinned pyspark. `main.py` never starts Spark: a missing pickle is lowered
  from the text. Changing the pyspark pin renumbers every plan: `make build`,
  `make plans`, then update the tests and fixtures that name
  column ids (`tests/verification/SF1_fixed_search.json`,
  `tests/parsing/test_uniqueness.py`, `tests/dataflow/`).
- **The dataflow graph** is built in memory by `main.py` from the prepared
  plans on every query occurrence (`dataflow.dataflow_gen.build_query_dataflow`);
  nothing is checked in. To inspect one, dump it locally: `docker compose run
  --rm -T dev python3 -m dataflow.dataflow_gen --q 1 2 --out /tmp/dataflow`.
  Never check the dump in.
- **Tests (correctness)**: `make test` runs `python3 test.py $(ARGS)` on the image copy; for your
  working tree use `TQP_GPU=<idle> TQP_DATA_DIR=<DATA_DIR> docker compose run --rm -T dev python3 test.py …`.
  `test.py` forwards its arguments to pytest (`-q`) and adds the `tests/` target
  when every argument is a flag: `ARGS="tests/operators"`, `ARGS="-k lend"`,
  `ARGS='-m "not e2e"'`. The SF1 suites need the SF1 dataset in `$TQP_DATA_DIR`
  (`make data SF=1`); the `tests/conftest.py::sf1_parquet` fixture fails without
  it. The shared test arena is fixed at 22 GB, so on a shared card pick
  an idle GPU. `tests/e2e` (marker `e2e`, ~3 min, 10 `main.py` runs) drives `main.py`
  as a subprocess (with `TQP_EXPERIMENT_TELEMETRY=1`) up a ladder at
  `--chunk_size_mb 4` and `--partition_budget_mb 2` (so most of its joins and twelve
  of its keyed aggregations take B > 1), all under `--compression adaptive`: everything off → `+ --compression_search read-and-write` → `+
  --sizing_search mid_stream` → then the `lru`, `dataflow` and `mru` granule caches
  and `cpu` (no cache) on top; the
  `lru` and `dataflow` caches over the C++ core
  with the compile-time read plan (`TQP_CACHE_CORE=native`, `TQP_CACHE_PLAN=1`) at
  the first rung's geometry, matching it and admitting pages; plus one
  `--compression_search fixed` replay of
  `tests/verification/SF1_fixed_search.json` that must match the first rung digit
  for digit. Every rung's results, each row in any order, must also match
  `tests/verification/SF1_reference.out` at 6 significant figures
  (`tests/sigfig_compare.py`). `tests/e2e/test_uvm.py` (marker `e2e`, ~40 s,
  3 `main.py --uvm` runs at SF1 with no warmup and a 22 GiB budget unless the
  cell sets one) runs the UVM mode: the defaults at 10 significant figures,
  the ladder's geometry (`--chunk_size_mb 4 --partition_budget_mb 2`) and a
  1 GiB GPU budget (`--cmp_pool_gb 1`) at 6, each 22/22 OK.
  `tests/e2e/test_cache_lend.py` (marker `e2e`, ~40 s, 3 runs) runs the
  `lru`, `dataflow` and `mru` caches (sizing `off`, `per_pipe`, `mid_stream`) on a
  2 GiB arena with a 1.75 GiB region, so the operators borrow from it: 22/22 OK at
  6 significant figures, misses lent for and blocks returned.
  Deselect them with `-m "not e2e"` for a quick pass. Run one pytest invocation
  at a time (rule 1).
- **Data**: `make data SF=<sf>` generates the standard TPC-H parquet of one scale
  factor, which the scan reads, into `$DATA_DIR/sf<sf>/parquet` (if missing) and
  validates it; it runs `baselines/` in its own image (`make -C baselines image`;
  see `baselines/AGENTS.md`).
- **Microbenchmarks — the codec sweep**: `make profiling TQP_GPU=<idle>` runs
  `python3 -m profiling.sweep` (`ARGS="--reps 3"`; `--out <dir>` writes the
  results elsewhere) on the image copy. It measures the seven codecs at nvCOMP
  chunks of 16–512 KiB on every SF5 column the 22 queries scan, unsorted and
  sorted, through the engine's own write and read paths (a region producer
  into a pinned home, a `DirectIn` back), timing compress, D2H, H2D and decompress with CUDA
  events (cost = the sum of the four), and writes `profiling/results/`:
  `codec_map.json` (the engine's input, see "Adaptive codec table"),
  `sweep_per_column.csv`, `sweep_by_class.csv` and `scan_frequency.csv`. **The SF5
  parquet is required** (`make data SF=5`; it reads the columns with the scan's
  reader); it stops otherwise. It pins to GPU 0's
  NUMA node like a timed run and is a GPU job: rule 1 applies (one at a time,
  idle card). After editing `profiling/`, `make build` first or run it through
  `dev` (`TQP_GPU=<idle> TQP_DATA_DIR=<DATA_DIR> docker compose run --rm -T dev python3 -m profiling.sweep`),
  which writes the same host directory. The figures over its CSVs live in
  `experiments/paper_scripts/figures/make_regret_fig.py` and
  `make_pareto_fig.py` (`make experiments SCRIPT=…`); they only plot the
  shipped map.
- **Experiments (performance)** (`experiments/`). The drivers run on the host
  (`python3 <driver>.py --help`); each runs its engine passes one at a time in
  the working tree's `dev` container on `--gpu`, pinned to that GPU's NUMA
  node, and writes under `out/<driver>/<timestamp>/`. They take their defaults
  as the Makefile does (`paper_scripts/helpers/common.py`): `--gpu` is
  `TQP_GPU`, else `GPU`, else 0, and the data root is `TQP_DATA_DIR`, else
  `$DATA_DIR`, with `DATA_DIR` and `GPU` from the environment or
  `local.env`. Without a data root a driver stops rather than mount
  compose's `./data`.

  | Driver | Question | Reads |
  |---|---|---|
  | `self-verification/e2_speedup.py` | How much faster, per query and suite? An in-process A/B over engine switches, so the host's drift cancels | `query_times.csv` |
  | `self-verification/e5_time_breakdown.py` | Where the time goes: an Nsight trace with NVTX ranges, kernel/copy/idle per pipe class and host step (attribution, not timing) | the trace |
  | `paper_scripts/e6_ablation.py` | What each mechanism buys: the four-arm ladder and the paper's figures 1-5 | `query_times.csv`, `sizing_model`, `compression`, the ledger |

  `self-verification/helpers/` holds the in-process A/B (`switch_ab.py`),
  `nvtx_ranges.py` and `trace_analyze.py`;
  `paper_scripts/helpers/` the ablation and its figures plus what both folders
  share (`common.py`, `perf_json.py`). `paper_scripts/figures/` holds the
  paper figures drawn from hand-copied or external numbers
  (`scaling_study.py`, `make_motivation_fig.py`, `make_tpch_comparison_fig.py`)
  and the codec-sweep figures; `scaling_study.py --refresh` checks its inline
  numbers against `baselines/results/all_results.csv` (`TQP_BASELINES_CSV`
  overrides it).
- **Spark plan parsing**: `make spark` opens an interactive shell with JDK +
  pyspark (image copy of the code, `queries/` mounted) for `parsing/layer.py`.
  Regenerate checked-in plans with `make plans`, not from this shell: a shared
  JVM shifts the column ids.

## Notes for agents editing the engine

- **Where your edit runs.** `dev` mounts the working tree, so Python edits
  take effect immediately there. `engine`, `test`, `profiling` and `experiments`
  run the image copy — `make build` before `make q` / `make smoke` / `make test`.
- **C++/CUDA edits** (`memory/*.cc`, `IO/*.cu`, `IO/cache/*.cc`, `compression/*.cu`,
  `operators/kernels/*.cu`): rebuild inside `dev` and stage the modules into
  `TQPlib/`:
  ```bash
  docker compose run --rm -T dev bash -c '
    cmake --build build -j &&
    cp build/memory/*.so            TQPlib/tqpmemory/ &&
    cp build/IO/*.so                TQPlib/zero_copy_reader/ &&
    cp build/IO/cache/*.so          TQPlib/tqpcache/ &&
    cp build/compression/*.so       TQPlib/tqpcompress/ &&
    cp build/operators/kernels/*.so TQPlib/tqpkernels/'
  ```
  The `dev` service's `build/` volume is seeded from the image on first use, so
  its CMake cache names the image's Python and nvCOMP wheel: after a `make build`
  that changes either (the Python version or the cuDF pin), remove the volume
  (`make clean`) and the next `dev` run seeds it again.
  The entrypoint copies the image's modules into `TQPlib/` only when a package
  directory has none, so the working tree's copies are never refreshed
  automatically.
- After touching operators / the pipeline / `main.py`, validate with the smoke test
  on your working tree (rule 4) before reporting done.
- **Codec changes.** A new nvCOMP codec is one `kCodec*` id and `make_manager` case
  in `compression/nvcomp_manager.cu`, one name (same index) in `_ALGO_NAMES` in
  `compression/compression_algorithms.py`, one entry in `format_policy.CODECS` /
  `_DEFAULT_SETTINGS`, and one line in `tests/compression/test_nvcomp_manager.py`'s
  `CODECS`; `--compression`, the applier and the sweep pick it up from the registry.
  The codec must exist in the nvCOMP cuDF pins (5.2.0.13); another nvCOMP means
  another cuDF pin, then `make build` and `make profiling`.
  If a run with it stops with `PinnedPoolExhausted` at the default pool size, raise
  `main._PINNED_NEED_PER_SF`. Re-run `make profiling` afterwards so the map covers it.
- The **naive H2D/D2H exchange is the only transfer path**; `main.py` installs it at
  startup (`set_exchange_to_naive()`).
- **A new host↔GPU copy or pool allocation** must reach the experiment numbers: a
  copy records its movement-ledger kind at its site, under `gates.EXPERIMENT`
  (`IO/telemetry/README.md`; a new kind also goes into `perf_json.KINDS` and
  `KINDS["tqp_vortex"]` of `baselines/spill_results.py`, which tests check), and a
  pinned-pool allocation names its `tag` when it holds something other than an
  intermediate.
- Operators execute on CUDA only; there is no CPU execution path.
