# TQP-Vortex

Tensor Query Processing on a single GPU. TPC-H queries are compiled from Spark
plans into dataflow graphs and executed as PyTorch tensor programs; the
compressed CPU→GPU transfer path — paging, caching, nvCOMP codecs, and the
runtime controllers that tune them — is what the system optimizes.

The repo has two halves: the engine, at the root, and `baselines/`, the engine
comparison. In its own Docker image, `baselines/` makes the TPC-H data and runs
the 22 queries on four other engines (Spark-RAPIDS, cudf-polars, DuckDB,
Sirius). The root's `make bench` adds TQP-Vortex as the fifth, on the same data,
through `bench/run_tqp.sh`. Every engine's per-query times go into one table,
`baselines/results/all_results.csv`. `baselines/README.md` and
`baselines/AGENTS.md` describe the comparison.

## Structure

```
TQP-Vortex/
├── main.py               engine entry point
├── test.py               pytest runner for tests/
├── config.json           the data root (tables path)
│
├── queries/              TPC-H SQL, Spark plans, and the compiled plan
│                         pickles the engine loads
├── parsing/              Spark plan ─► typed IR
├── dataflow/             IR ─► the in-memory column dataflow graph
│
├── IO/                   the CPU→GPU pipeline: paging, caching, (de)compression
├── compression/          nvCOMP 5.2 bindings + per-type format policy (the codec map)
├── memory/               custom PyTorch CUDA and pinned-host allocator
├── operators/            join / filter / aggregate / sort as tensor programs
├── adaptive/             runtime controllers for compression and sizing
├── uvm/                  the UVM mode (main.py --uvm): managed memory, the
│                         driver moves the data
├── TQPlib/               compiled C++/CUDA extensions install here
│
├── profiling/            the offline codec sweep -> profiling/results/codec_map.json
├── tests/  experiments/  docs/  utility/
├── Makefile  Dockerfile  docker-compose.yml  docker/
├── local.env.example     box settings for both halves; copy to local.env
├── bench/run_tqp.sh      TQP-Vortex's runner in the engine comparison
├── baselines/            the engine comparison and the TPC-H data (own image)
└── AGENTS.md             operating rules — read before any perf claim
```

A query flows `tpch-q*.sql` →(Spark)→ `plan-q*.txt` →(lowering)→
`pickles/planobj-q*.pkl` → `main.py`, which builds the column dataflow graph
in memory from the prepared plan on every run. **Both intermediates are checked
in**, so no generation step is needed to run the engine.

## Running with Docker (recommended)

Needs the NVIDIA Container Toolkit. Nothing builds on the host — cmake runs
inside the image.

```bash
cp local.env.example local.env   # box settings (DATA_DIR, GPU, ...); edit it
make build                  # the engine image
make -C baselines image     # the baselines' image, which makes the data
make data SF=1              # TPC-H SF1 parquet in $DATA_DIR/sf1 (generated, validated)
make q SF=1 Q="1 6 14"      # run queries; omit Q for all 22
make smoke                  # SF1 end-to-end correctness
make test                   # full pytest suites (need the SF1 dataset)
```

One `local.env` serves both halves: the Makefile, `bench/run_tqp.sh` and
`baselines/docker/run.sh` read it, and the environment and the command line win
over it. Three variables cover normal use; each works as a make variable or an
exported environment variable:

| | |
|---|---|
| `TQP_DATA_DIR` | host data root (default `$DATA_DIR`): each dataset's `sf<sf>/parquet`. |
| `TQP_GPU` | which physical GPU the container sees (default `$GPU`, else `0`). |
| `ARGS` | extra flags forwarded to the entry point, e.g. `ARGS="--cmp_pool_gb 64"`. |

```bash
make TQP_DATA_DIR=/mnt/scratch/tpch TQP_GPU=6 test
```

Outputs land in `./out`, including `query_times.csv`. `make help` lists the
remaining targets (`doctor`, `validate`, `shell`, `profiling`, `experiments`,
`clean`).

`make test` needs no such flag — `tests/conftest.py` sets its own  pool.

**Regenerating plan artifacts** (optional; only after a plan or rewrite change):

```bash
make plans Q="1 2"          # capture queries/plan-q*.txt (pinned pyspark) + lower to pickles
```

`main.py` lowers a missing pickle from the checked-in plan text; it never starts
Spark. See `parsing/README.md` for why capture is pinned to one pyspark version.

## The data and the engine comparison

Every dataset comes from `baselines/`, in its image. TQP-Vortex's scan reads
the same validated parquet the other engines read, decoding it on the GPU with
cuDF (`IO/parquet_reader.py`).

```
make data SF=N    [baselines image]  ndsgen (TPC-H, NDS-H format) -> $DATA_DIR/sfN/parquet (validated)
make bench SF=N   per SF, on GPU $GPU:
                  [baselines image]  rapids, polars_gpu, duckdb_cpu, sirius on a ramdisk copy
                                     of the parquet, freed before TQP-Vortex runs
                  [engine image]     bench/run_tqp.sh: one main.py --SF N --q <q> per query
                                     on $DATA_DIR/sfN/parquet
                  -> baselines/results/all_results.csv, one row per (engine, SF, query)
```

* `make bench SF="10 100" [ENGINES="…"] [QUERIES="…"]` runs all five engines by
  default, after `make build` and any missing dataset. With
  `RUN_CSV_DIR=<dir under $SCRATCH>` it merges nothing and prints a query ×
  engine table instead.
* `make smoke-all` runs all five engines at SF1 and merges nothing; it fails
  unless every query is OK with the SF1 reference's row counts.
* `make summary` prints pivots of `all_results.csv`, and per SF the queries
  whose engines disagree on the row count.

`baselines/AGENTS.md` holds the protocol and the results contract, and
`AGENTS.md` the bench's settings.

## Running without Docker

**1. Environment.** Python 3.11 or later (the cuDF pin needs it). Any Python
environment manager works; Conda as an example.

```bash
conda create -y -n torch126 python=3.12 cmake ninja gcc_linux-64=11 gxx_linux-64=11 cuda-toolkit=12.6 -c conda-forge
conda activate torch126
pip3 install torch==2.6.0+cu126 torchvision==0.21.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
```

Keep torch pinned: 2.9 and later require a C++20 compiler, and the extensions
here build at C++17.

**2. cuDF and nvCOMP.** `requirements.txt` pins cuDF (`pylibcudf-cu12`), whose
libcudf pins and loads the `nvidia-libnvcomp-cu12` wheel (5.2.0.13). The
compression modules build against that wheel and link the same file
(`compression/CMakeLists.txt`), so a process holds one nvCOMP; do not install
another.

**3. Build the extensions.** From the repo root:

```bash
NVCOMP_ROOT=$(python3 -c 'import os, nvidia.libnvcomp as m; print(os.path.dirname(m.__file__))')
cmake -S . -B build/ -DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux -Dnvcomp_DIR=$NVCOMP_ROOT/lib64/cmake/nvcomp
cmake --build build -j
pip3 install --no-cache-dir .
```

**4. Data.** The parquet comes from `baselines/`, which runs in its own image
even here: `make data SF=1` writes it to `$DATA_DIR/sf1/parquet` (see "Running
with Docker").

**5. Point the engine at the data root** it scans, either in `config.json`
(`"tables path"`) or, without touching the tracked file:

```bash
. ./local.env && export TQP_TABLES_PATH=$DATA_DIR
```

**6. Run.**

```bash
python3 main.py --SF 1 --q 1 6 14 [--no_debug]
python3 main.py --smoke           # SF1 correctness
python3 test.py                   # full pytest suites
```

Two host settings the Docker images already encode and a bare host may not:
`ulimit -l unlimited` (`main.py` pins about 0.69 GB of host RAM per scale
factor, every page of it on GPU 0's NUMA node), and one GPU — `main.py` uses
CUDA device 0, so pick it with `CUDA_VISIBLE_DEVICES` (`nvidia-smi` is asked
about the same GPU, by PCI bus id). The NUMA binding uses
`mbind`, which a container allows only with `CAP_SYS_NICE` (the compose file
grants it; `TQP_NUMA_BIND=0` pins the pool unbound where it is refused). After
warmup `main.py` also pins its threads to that node,
intersected with the process's CPU affinity, so leave it an unrestricted cpuset.
`bench/native_env.sh` builds such a host environment, and `TQP_NATIVE=1
bench/run_tqp.sh` runs the bench in it (`AGENTS.md`, "Native runs").

## Correctness, performance and reproducing results

**Correctness is pytest; performance is the self-verification experiments.**
`tests/` holds the correctness suites: unit tests per component, the SF1 smoke
(`python3 main.py --smoke`) and the e2e ladder; a change is correct when they
pass (`tests/README.md`). Performance is measured by
`experiments/self-verification/`: `e2_speedup.py` (an in-process A/B over
engine switches) and `e5_time_breakdown.py` (an Nsight trace split by pipe and
host step).

> Wall-clock time is **not** the number of record: it is dominated by host
> pinning, disk I/O, and profiler overhead. Use `query_times.csv` (per-query
> `approx_time`).

```bash
python3 main.py --SF 20 --q 1 1 1 6 6 6      # repeat qids; take the median
TQP_EXPERIMENT_TELEMETRY=1 python3 main.py --SF 20 --q 1 --perf_report out/perf_q1
```

Run one `main.py` at a time — a co-running job poisons every timing.

**Telemetry is off unless its switch turns it on.** Without it, a perf report
holds only each query's `approx_time`: a timing run leaves it off.
`TQP_EXPERIMENT_TELEMETRY=1` adds what experiments read: the movement ledger,
the residency peaks of each memory tier, each pipe's `peak_gpu_bytes`, the
sizing-model samples and the compression-search outcomes (`AGENTS.md`,
`--perf_report`). The spill pass (`make spill`, `baselines/AGENTS.md`) turns
the first two into the engine comparison's spill records.

**The pinned host pool** is `max(1, ceil(1.05 × 0.6524 × SF))` GiB by default (686
GiB at SF1000), every page of it on the GPU's NUMA node. A query that needs more
stops with `pinned host pool exhausted`; pass a larger `--pinned_pool_gb`
(`AGENTS.md`, "Host: the pinned pool").

**The codec sweep.** `make profiling TQP_GPU=<idle>` runs
`python3 -m profiling.sweep`: the seven nvCOMP codecs at six chunk sizes on
every SF5 column the queries scan, timed through the engine's own compress
path (compress + D2H + H2D + decompress). It writes
`profiling/results/codec_map.json`, the per-`(type, sorted)` codec table and
per-codec chunk size that `compression/format_policy.py` loads at startup.
Needs the SF5 parquet (`make data SF=5`); without the file the engine runs on
its built-in defaults and says so.

**Experiments and figures.** `experiments/paper_scripts/` holds the paper's
ablation (`e6_ablation.py`) and figures (`figures/`);
`experiments/self-verification/` holds the drivers that check a change:
end-to-end speedup and a time breakdown (AGENTS.md, "Experiments"). The
drivers run on the host and take their GPU and data root as the Makefile does
(`TQP_GPU`/`GPU`, `TQP_DATA_DIR`/`$DATA_DIR`, from `local.env`):

```bash
python3 experiments/self-verification/e2_speedup.py --help
make experiments SCRIPT=experiments/paper_scripts/figures/scaling_study.py
```

Under Docker these need `requirements-dev.txt` (matplotlib, numpy), which the
engine image already installs. Design notes and findings live in `docs/`.

**Older numbers are stale.** TQP-Vortex numbers from a tree without
`baselines/`, such as the experiment outputs and the findings in `docs/`
measured there, do not compare with runs on this tree, whatever data they
used. On those trees q11 ran without its HAVING filter, q16 counted duplicate
suppliers (since July 2025), q20 ran another plan, and primary keys were typed
sorted, which can change their codecs. Most of those runs also used the
retired generator's data and a codec map swept on it. Re-measure before
comparing with them.

## Components

### `adaptive/` — runtime feedback planes

Two independent control planes retune a running query from measured per-tick
timings. Both register through `adaptive_driver.py`, which owns epochs and fans
the resource monitor's last-tick view out to observers.

*Compression plane.* `decision_menu.py` is the actuation interface: per pipe the
pipeline publishes each column's legal codec range, and the adaptive side
returns picks positionally. `compression_search.py` searches *k*, how many of
the benefit-ranked columns to compress — `f(k)` is unimodal, so it ternary-
searches rather than sweeping. `compression_controller.py` makes the per-column
RAW-vs-candidate call; it owns no bytes and launches no kernels.

*Sizing plane.* `sizing_controller.py` sets chunk budgets on two legs:
per-gather-iteration inside a rearrange stream, and per-pipe for the tick loop.
`sizing_memory_model.py` predicts the arena transient at a candidate budget from
a conservative per-key linear fit, so a step that would OOM is gated before it
runs. Its budget counts what the granule cache's region could lend: the arena
itself lends the region's lowest blocks to any allocation that misses (the cache
evicts and relocates granules out of them, in C++) and takes them back once
free (`memory/README.md`, "Lending").

`resource_monitor.py` produces the per-tick transfer/compute split both planes
consume; `run_stats.py` folds predicted-vs-observed into the perf report (under
`TQP_EXPERIMENT_TELEMETRY`).
Design notes: `docs/adaptive_compression_switching.md` and
`docs/adaptive_intermediate_sizing.md`; `adaptive/README.md` has the per-file
tour, and open issues are in `docs/bug_backlog.md`.

### `compression/` — nvCOMP codecs and format policy

`nvcomp_manager.cu` wraps the nvCOMP **5.2 high-level manager API**: one
`compress()` emits one self-describing `NVCOMP_NATIVE` object and nvCOMP owns
internal chunking and scratch. This is the engine's only non-RAW compress path;
`compression.cu` holds the copy and pack kernels the IO layer stages records with.

`compression_algorithms.py` is the registry of the seven codecs (ANS, CASCADED,
BITCOMP, LZ4, SNAPPY, GDEFLATE, DEFLATE) and their indices; a compressed
record's index lives in its record table (`IO/pages/record_table.py`). It
imports no CUDA, so plan-time and solver code can use it. `format_policy.py` keys the
choice of "what to materialize on the scan path, what to select at runtime" on
`(mapped_type, is_sorted)`.

Builds into `TQPlib/tqpcompress`.

### `dataflow/` — static column dataflow

`dataflow_gen.py` builds, in memory and on every run, the column dataflow
graph of the prepared plans: an operator-centric view (what each operator
consumes and produces) and a column-centric one (each column instance's single
producer and *N* consumers). Nothing is checked in, so the graph cannot drift
from the plan the runtime executes. The cache's Dataflow policy evicts
against it — data with no remaining consumers is dead, which LRU has no way
to know.

`compression_applier.py` resolves each column's codec from that graph, the
gathers' sources included: `rearrange_pipeline` reads a source over UVA from a
RAW rep (a scan keeps one) or streams its compressed records onto the GPU,
whichever moves fewer bytes.

### `IO/` — the CPU→GPU pipeline

The core of the system. `pipeline/vortex_pipeline.py::VortexPipeline` runs one operator
over host-resident columns in **ticks**: a chunk of each input column moves H2D,
the operator runs, its outputs go home as regions, repeat. Everything else here is either
a stage in that loop or a resource the loop contends for. The files are grouped
by layer into `pages/`, `codec/`, `cache/`, `transfer/`, `pipeline/` and
`telemetry/`; `IO/README.md` has the per-file tour.

Per tick: `in_stage` (H2D, under a compiled layout that binds every compressed
stage so each codec costs one nvCOMP call) → `batched_decompress` →
`pre_comp_stage` → **operator** → `codec/region_producer.py` (the outputs compressed
at tick *t*, packed into one region per column at *t+1* and copied to their pinned
homes by `codec/home_writer.py` in the background; the buckets of a *B*-way
partitioned output, the join's sort pipe's or a partitioned aggregation's pass
1's, are streams of one region).

Memory and reuse:

* `cache/cache_region.py` — the granule cache's GPU memory: the arena's top blocks
  (`--slot_pool_gb`) behind a boundary `main.py` sets after each query's scan
  and empties at its end; set only with a cache.
* `cache/native_cache.py` — the GPU cache (`--buffer_strategy`): granules (1 MiB
  pieces) of the region producer's regions, one per arena block of the cache
  region, filled from its top, admitted as a prefix of each
  region at pack time, cleaned ahead of eviction, and read piece by piece (slots by
  D2D, the pinned home by H2D), with LRU / MRU / dataflow policies. Its bookkeeping
  is the C++ core `cache/granule_core.cc` (`TQPlib.tqpcache`); `TQP_CACHE_CORE=cpu`
  runs an occurrence without it.
* `pages/pinned_mem.py` — the host side, a variable-block first-fit pool. Scan
  batches and region homes are heterogeneously sized, so uniform slotting does not
  apply. Every
  page of it is bound to the GPU's NUMA node.

`zero_copy_reader.cu` reads highly selective columns straight over UVA instead
of staging its records (`--mask_selectivity`), and builds into
`TQPlib/zero_copy_reader`. `telemetry/copy_events.py` times every copy batch with
one CUDA event pair around its copies: the compression searches read them as one
span per transfer stream. `telemetry/movement.py` is the movement ledger: under
`TQP_EXPERIMENT_TELEMETRY` it counts every host↔GPU copy (and every copy the
cache avoided) once, at its site, by kind, for the perf report's `movement`
section; `telemetry/residency.py` reads the most each memory tier held for its
`residency` section. `telemetry/gates.py` holds that switch and the debug
validators' switches, each read once at import.

### `memory/` — the GPU arena

A pooled, block-based CUDA allocator installed as PyTorch's allocator
(`CUDAPluggableAllocator`, `main.py:1405`), so every torch op inside an operator
draws from one pre-reserved slab rather than calling `cudaMalloc`. Sized by
`--cmp_pool_gb`; it must be installed before any CUDA tensor exists.

`cuda_memory_allocator.cc` is CUDA-only. Past alloc/free it exposes what the adaptive plane needs — `get_arena_stats`
(allocated bytes, largest free run) for sizing decisions with no device sync, and
`get_window_peak_bytes` for per-pipe transient brackets, both over the operator
space below the cache region (`region_init`, `region_move`: the arena's top
blocks behind a movable boundary, the granule cache's) — plus
`pinned_host_alloc_tensor`, which pins host memory bound to one NUMA node, for
the host pool. Under `main.py --uvm` the arena is one `cudaMallocManaged`
range instead; `memory/README.md` has the calls.

Two artifacts come out of this directory: `libcustom_allocator.so`, which torch
loads *by filename* and so must sit on the library path, and the `tqpmemory`
pybind module that links against it and installs into `TQPlib/tqpmemory`.

### `operators/` — relational operators as tensor programs

One module per operator, each a function over the current tick's tensors: `scan`,
`filter`, `project`, `hashjoin`, `sort`, `aggregate`,
`output`.

* `hashjoin.py` — the join: per side a partition sort pipe into B buckets
  (from the partition budget, `partitions.py`), then the core pipe, which packs
  whole bucket pairs into its ticks; a tick takes the unique-build hash core,
  the semi/anti hash set or the sort/search pairs.
* `join_core.py` — the join core's kernels (`kernels/join_core.cu`, built into
  `TQPlib/tqpkernels`): the unique-build hash join, the semi/anti hash set with
  its per-key min/max summary for one residual comparison, and the sort/search
  pair expansion; one wait per tick.
* `partition_sort.py` — the join sort pipe's hash partition on one fused CUDA
  kernel (`kernels/partition_sort.cu`, built into `TQPlib/tqpkernels`), which
  applies the pipe's mask itself; a tick outside its limits raises.
* `partitions.py` — the partition count B of a join or a keyed aggregation
  (`--partition_budget_mb`): the smallest power of two whose largest unit, a
  join's bucket pair or an aggregation's pass-2 round, is predicted to fit the
  budget. Stdlib only.
* `aggregate.py` — two-pass hash aggregation. `aggregation_slab.py` gives pass 2
  fixed operator-owned byte slabs, so changing the chunk size changes how a batch
  fills without making the accumulated state reallocate per tick. A keyed
  aggregation whose pass-2 run is predicted over the partition budget writes
  pass 1's partials in B hash buckets, and pass 2 aggregates one round of
  adjacent buckets at a time (at B = 1, one round of every row).
* `sort.py` — the plan's sort (ORDER BY, LIMIT, DISTINCT): its key and
  distinct columns read to the GPU whole and sorted there; a column that only
  follows the order and is larger than the chunk budget is decoded window by
  window, reading only the windows that hold output rows (at most 20 under a
  LIMIT 20). Its core over tensors (`sort_tensors`), like
  the output's (`output.output_frame`) and the global aggregation's
  (`aggregate.GlobalAggregate`) and the keyed aggregation's pass 1 per chunk
  (`aggregate.KeyedAggregate`) and the join's arguments and B
  (`hashjoin.join_call_args`, `hashjoin.join_partition_count`), is shared with
  the UVM executor.

### `uvm/` — the UVM mode

A baseline of the engine with the CUDA driver moving the data: `main.py
--uvm` runs the same plans and leaf kernels with every column and
intermediate in CUDA managed memory and each operator a loop over morsels of
plain tensors, with no pinned pool, tick pipeline, explicit copy,
compression, cache or search (`make q ARGS="--uvm"`, or `python3 main.py
--uvm …` in `dev`). AGENTS.md has its flags and the GPU budget;
`uvm/__init__.py` lists its modules, whose docstrings hold the details.

### `parsing/` — Spark plan → typed IR

`layer.py::ParsingLayer` starts a Spark session and captures the physical plan
of a TPC-H query as text (`queries/plan-q*.txt`, via `make plans`);
`ParsingLayer.lower` turns that text into the `QueryPlanRep` of `models.py`.
`expression_parser.py` is a PyParsing grammar for Spark's expression syntax;
`expressions.py` walks what it returns.

`pipeline.py::prepare_query_plans` reruns on **every** load, cached or fresh,
because the pickles deliberately store only the stable physical plan:

* `uniqueness.py` propagates relational unique-key proofs. It carries its own
  TPC-H key schema on purpose — the plans' own uniqueness metadata is not
  trustworthy, and `queries/schema.json`'s `PRIMARY_KEY` field cannot express a
  composite key.
* `materialization.py` computes late materialization: which payload columns a
  streaming plan can defer past a join.
