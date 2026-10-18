# Sirius GPU SQL engine baseline

[Sirius](https://github.com/sirius-db/sirius) is a **GPU-native SQL engine** from
NVIDIA + UW-Madison. It loads as a **DuckDB extension** and *transparently*
intercepts plain SQL, running supported operators on the GPU via NVIDIA CUDA-X
(cuDF / RMM / cuCascade). Its recommended path, **`gpu_execution`** ("Super
Sirius"), is **out-of-core**: it streams Parquet through a tiered memory manager
(GPU → pinned host → disk) with automatic partitioning and spilling, so it can
run datasets far larger than VRAM. Consumes query plans via the Substrait format;
unsupported operators fall back to DuckDB CPU.

This baseline runs the **same TPC-H q1-22 stream** as the other engines
(`results/queries/stream_qualification.sql`) on the same parquet.

## Paper notes

- Reports ~7× speedup over DuckDB on TPC-H **SF100** at equal $/hour, and 5× on
  1 TB (SF1000) on a DGX Station GB300 (repo README; arXiv 2508.04701).
- Gains come from GPU-resident columnar execution (cuDF kernels) + out-of-core
  tiering, avoiding the JVM/host bottlenecks of CPU engines.
- Standard TPC-H uses no window functions, so every query is GPU-eligible; a plan
  that DuckDB lowers to a cross-product (or that hits an unsupported type such as
  128-bit decimal / nested struct) silently falls back to CPU.

## Build (in the image)

The `sirius-build` stage of `docker/Dockerfile` clones `sirius-db/sirius` at the
commit pinned in `versions.env` (`SIRIUS_REF`; its DuckDB submodule is the
`v1.5.5-patches` branch, reporting `v1.5.5`), installs the toolchain from
Sirius's own `pixi.lock` with `pixi install --frozen` (libcudf 26.08.01, CUDA
13.3, clang 21), and runs `make` for H100 only (`CUDAARCHS=90a-real;90`). The
final image keeps only `build/release/duckdb` (extension statically linked and
auto-loading) and the pixi env's shared libraries, at the path the binary's
RPATH names: `$SIRIUS_DUCKDB` and `$SIRIUS_ENVLIB`.

## Layout

```
env.sh               engine environment shared by the runner and the format probes:
                     config with the IO backend resolved, spill dir
sirius.yaml          gpu_execution config: 1 GPU, 95% VRAM, 256Gi host tier, disk spill
run_tpch_sirius.py   parses the shared query stream, runs each query through the Sirius
                     duckdb binary (transparent GPU): SF1 warm pass, one timed run, result rows
run_sirius.sh        safe per-query driver: one duckdb process per query + disk watchdog
```

## Requirements

| need | this setup |
|------|------------|
| GPU compute capability ≥ 7.5 | H100 = 9.0 (build targets `90a`) |
| CUDA 13.x driver | driver 595.71.05; CUDA 13.3 user space from the pixi env |
| glibc ≥ 2.28, `io_uring` | Ubuntu 24.04 (glibc 2.39). Docker's default seccomp profile blocks io_uring, so `docker/run.sh` uses `docker/seccomp-iouring.json` and `--ulimit memlock=-1`; `sirius/env.sh` probes `io_uring_setup` and falls back to the kvikio POSIX backend where it is blocked (`SIRIUS_IO=kvikio` forces it). `make doctor` checks |
| `O_DIRECT`-capable parquet storage | `/dev/shm` (the runs read the ramdisk copy) |

## Run

```bash
make bench SF=100 ENGINES=sirius        # stage -> run 22 queries -> merge into all_results.csv
```

Output: rows are upserted into `results/all_results.csv` (`seconds` = the one
timed run after the SF1 warm pass, startup excluded — see the schema in
`README.md` / `AGENTS.md`). Per-query
and engine logs go to `results/*.log` / `results/*_logs/` (gitignored).

### Methodology (matches rapids/polars)

- **One duckdb process per query** so a heavy query's GPU/spill state can't poison
  the rest, guarded by a **disk watchdog** that kills a query if free `/` drops
  below `MIN_FREE_GB` (Sirius's disk spill tier is also capacity-bounded in
  `sirius.yaml`, a second safety net).
- Each process creates DuckDB **views** over the SF1 ramdisk copy and runs the
  query there, untimed, with its output discarded (the **SF1 warm pass**: GPU
  init, cuDF kernel JIT, reader setup), then re-points the views at the target
  dataset and runs the query **once**, timed — the protocol of every engine
  (`AGENTS.md`). An error in either run fails the query (the CLI stops at the
  first error).
- Single GPU (`topology.num_gpus: 1`) to match `rapids` (`local[*]`, one GPU) and
  `polars` (`device 0`).
- **No `pin_table`** (key `sirius`). Sirius can pin tables in pinned host or GPU
  memory, but a pin decodes the parquet (cuDF's GPU reader) into plain cuDF
  columns, so the parquet format would no longer reach query time. The runner and
  the format probes therefore read the ramdisk parquet in every query, as the
  other engines do. Sirius's pinned host pools are still used for its own
  transfers and spill.
- **Resident mode** (key `sirius_resident`, `LOAD_MODE=resident`) is the
  exception: before each pass, untimed, the query's columns (`pin_columns.py`)
  are pinned with `CALL pin_table('<dir>/<table>/*.parquet', tier='gpu', …)`,
  with CPU fallback off, and the timed run must be served by every pin (checked
  in the Sirius log, `AGENTS.md`).

## Tuning knobs

- `sirius.yaml` — `memory.gpu.usage_limit_fraction`, `memory.host.capacity_bytes`
  (pinned; one host space for the box's NUMA nodes, so a total: 256Gi),
  `memory.disk.downgrade_root_dirs` (spill dir).
- Env: `SIRIUS_TIMEOUT` (default 2400s, covers the warm pass too), `MIN_FREE_GB`
  (default 25), `SIRIUS_CONFIG_FILE`, `SIRIUS_IO`.
- To *prove* GPU execution (surface fallbacks as errors instead of silent CPU):
  add `SET enable_duckdb_fallback=false;` (the format probes do) — the runner
  also greps the Sirius log and records a `gpu`/`fallback` flag per query in the
  detail CSV.
