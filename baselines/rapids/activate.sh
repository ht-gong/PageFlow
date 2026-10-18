#!/bin/bash
# Source this to get a ready-to-use spark-rapids shell:
#   source rapids/activate.sh
# It puts the Spark toolchain on PATH (datagen/env.sh: PY, JAVA_HOME,
# SPARK_HOME) and exports $RAPIDS_JAR (set by the image) and $RAPIDS_CONF, plus a
# `rapids-submit` helper and `rapids_run_args` (the flags of the benchmark runs).

RAPIDS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${RAPIDS_DIR}/../datagen/env.sh"
: "${RAPIDS_JAR:?RAPIDS_JAR is not set}"
export RAPIDS_CONF="${RAPIDS_CONF:-${RAPIDS_DIR}/conf/spark-rapids.conf}"

# Convenience: spark-submit with the GPU plugin already wired in.
# `env -u CONTAINER_ID` stops Spark from mistaking this Vast container for a
# YARN container (which would fail with "Yarn Local dirs can't be empty").
rapids-submit() {
  env -u CONTAINER_ID \
  spark-submit \
    --master "local[*]" \
    --jars "${RAPIDS_JAR}" \
    --properties-file "${RAPIDS_CONF}" \
    "$@"
}
export -f rapids-submit

# spark-submit flags of every benchmark run (TPC-H runner and format probes):
# local mode on one GPU, host spill, RAPIDS shuffle. $1 = spark.local.dir.
# Sets the RAPIDS_RUN_ARGS array.
rapids_run_args() {
  RAPIDS_RUN_ARGS=(
    --master "local[*]" --driver-memory "${DRIVER_MEM:-96g}"
    --jars "${RAPIDS_JAR}"
    --conf spark.local.dir="$1"
    --conf spark.plugins=com.nvidia.spark.SQLPlugin
    --conf spark.rapids.sql.enabled=true
    --conf spark.rapids.sql.concurrentGpuTasks=2
    --conf spark.rapids.memory.pinnedPool.size=8G
    # The host spill store (off-heap, not pinned). 512G (200G until
    # 2026-10-12) is headroom only: at SF1000 the store peaked at 36 GB (q9),
    # far below either, and nothing spilled to disk. RAPIDS's disk use is the
    # shuffle files under spark.local.dir, up to ~200 GB per SF1000 query.
    --conf spark.rapids.memory.host.spillStorageSize=512G
    --conf spark.shuffle.manager=com.nvidia.spark.rapids.spark358.RapidsShuffleManager
    # RAPIDS_SHUFFLE_MODE=CACHE_ONLY keeps shuffle blocks in the spill store
    # (GPU -> host -> disk) instead of files under spark.local.dir; RAPIDS
    # documents it for a single executor, for testing.
    --conf spark.rapids.shuffle.mode="${RAPIDS_SHUFFLE_MODE:-MULTITHREADED}"
    --conf spark.sql.files.maxPartitionBytes=1g
    --conf spark.sql.shuffle.partitions=1024
    --conf spark.sql.adaptive.enabled=true
  )
}

echo "spark-rapids env ready:"
echo "  JAVA_HOME = ${JAVA_HOME}"
echo "  SPARK_HOME= ${SPARK_HOME}"
echo "  RAPIDS_JAR= ${RAPIDS_JAR}"
echo "  helper    : rapids-submit <script.py>   (local[*] + plugin + conf)"
