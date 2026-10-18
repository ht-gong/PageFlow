#!/bin/bash
# Source this for the polars_gpu engine environment, shared by the TPC-H runner
# (run_polars_gpu.sh) and the format probes (ablation/): the GPU memory / IO
# settings the benchmark runs use. The cudf-polars env's python is $POLARS_PY
# (set by the image).
: "${POLARS_PY:?POLARS_PY is not set}"
export GPU_MR="${GPU_MR:-async}"
# kvikio POSIX/compat mode: cuFile/GDS is unavailable on these boxes; without it
# the parquet reader aborts in CUFileThreadPoolWorker.
export KVIKIO_COMPAT_MODE="${KVIKIO_COMPAT_MODE:-ON}"
export GPU_PART_MB="${GPU_PART_MB:-128}"
export RAPIDSMPF_SPILL_DEVICE_LIMIT="${RAPIDSMPF_SPILL_DEVICE_LIMIT:-$((22*1024*1024*1024))}"
# Spill device -> pinned host memory (3-6x pageable's bandwidth). The pool is
# unbounded and on the GPU's NUMA node (polars/spill_stats.py build_engine);
# RAPIDSMPF_PINNED_MAX_POOL_SIZE must stay unset: a capped pool aborts the query
# instead of falling back to pageable memory.
export RAPIDSMPF_PINNED_MEMORY="${RAPIDSMPF_PINNED_MEMORY:-1}"
