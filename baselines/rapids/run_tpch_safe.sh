#!/bin/bash
# Run NDS-H (TPC-H) queries 1-22 on the ramdisk dataset with RAPIDS, SAFELY:
#   * one spark-submit per query  -> scratch is reclaimed when each JVM exits,
#     so a heavy query can't poison the rest;
#   * a disk watchdog kills a query (and moves on) if free disk drops below a
#     threshold -> the box can never wedge on a full disk again;
#   * every query process is killed after RAPIDS_TIMEOUT seconds (default
#     600: 10 minutes, the JVM, the warm pass and the timed run; a smaller
#     QUERY_TIMEOUT wins) -> a query that stalls (at SF1000 under CACHE_ONLY
#     shuffle, q5 and q18 retried device allocations for 27 and 19 minutes) is
#     recorded TIMEOUT, its seconds the process's elapsed time, and the run
#     moves on;
#   * each process first runs its query on the SF1 copy in WARM_PARQUET,
#     untimed (GPU init, JIT), then times it on TPCH_PARQUET (run.sh protocol).
#
#   run_tpch_safe.sh [QUERY_LIST]      e.g. "1 2 3"  (default: 1..22)
#
# Env: TPCH_PARQUET (dataset), TPCH_SF, WARM_PARQUET (SF1 warm copy), SCRATCH
# (Spark local dirs go to $SCRATCH/rapids), plus the image's PY / JAVA_HOME /
# SPARK_HOME / RAPIDS_JAR. SPILL_STATS_DIR (run.sh, SPILL=1): the spill pass;
# each query writes a zstd event log, the driver log (spill loggers at INFO) and
# run_tpch_queries.py's runner.json / monitor.csv to $SPILL_STATS_DIR/q<N>/, and
# eventlog_stats.py appends its record to $SPILL_STATS_DIR/records.jsonl, kills
# and timeouts included. SPILL_SMOKE=1 adds forced-spill settings.
# RAPIDS_TIMEOUT and QUERY_TIMEOUT (whole seconds): the limit above. RAPIDS_SHUFFLE_MODE: the
# shuffle mode (activate.sh). LOAD_MODE=resident + ENGINE_KEY=rapids_resident
# (run.sh sets both) run over Spark's cache instead (run_tpch_queries.py,
# AGENTS.md), with RAPIDS_CACHE_SERIALIZER as spark.sql.cache.serializer.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUERIES="${1:-$(seq 1 22)}"
RAPIDS_DIR="${ROOT}/rapids"
RAM_PQ="${TPCH_PARQUET:?TPCH_PARQUET is not set}"
: "${WARM_PARQUET:?WARM_PARQUET is not set (the SF1 warm copy)}"
STREAM="${ROOT}/results/queries/stream_qualification.sql"
TPCH_SF="${TPCH_SF:?TPCH_SF is not set}"
# throwaway temp; folded into results/all_results.csv at the end (no per-run CSV kept)
ENGINE_KEY="${ENGINE_KEY:-rapids}"
SUFFIX="${ENGINE_KEY#rapids}"       # "" or "_resident": names the logs
OUT_CSV="${OUT_CSV:-/tmp/tpch_${ENGINE_KEY}_sf${TPCH_SF}.csv}"
SPARK_DIR="${SCRATCH:?SCRATCH is not set}/rapids"
LOG="${ROOT}/results/safe${SUFFIX}_run.log"
DRIVER_MEM="${DRIVER_MEM:-96g}"      # JVM heap (host spill store is off-heap, separate)
MIN_FREE_GB="${MIN_FREE_GB:-30}"     # kill a query if free disk drops below this
QUERY_TIMEOUT="${QUERY_TIMEOUT:-0}"  # kill a query after this many seconds (0 = never); row -> TIMEOUT
# RAPIDS's own limit, whatever QUERY_TIMEOUT (shared by the engines) says: a
# query process outliving it is killed and recorded TIMEOUT (0 = none). The
# limit is the smaller of the two. Sized on the RTX 5090 box: the slowest OK
# RAPIDS query on record, SF1000 q8, takes ~380 s timed, ~440 s in all. SF1000
# q9 has never finished (killed for disk by its shuffle files); scaled from
# SF700 it would take ~450-545 s timed, close to the limit, so a q9 TIMEOUT
# there may be the limit, not the engine. Raise it for larger SFs or slower
# GPUs.
RAPIDS_TIMEOUT="${RAPIDS_TIMEOUT:-600}"
for v in QUERY_TIMEOUT RAPIDS_TIMEOUT; do
  [[ "${!v}" =~ ^[0-9]+$ ]] || { echo "run_tpch_safe.sh: ${v} must be whole seconds, got '${!v}'" >&2; exit 2; }
done
LIMIT="${RAPIDS_TIMEOUT}"
if [ "${QUERY_TIMEOUT}" -gt 0 ] && { [ "${LIMIT}" -eq 0 ] || [ "${QUERY_TIMEOUT}" -lt "${LIMIT}" ]; }; then
  LIMIT="${QUERY_TIMEOUT}"
fi

source "${RAPIDS_DIR}/activate.sh" >/dev/null 2>&1
rapids_run_args "${SPARK_DIR}"
# Resident: a static conf, so set at submit. The GPU's own serializer keeps the
# cache as parquet-encoded batches in host memory and decodes them on the GPU.
[ "${LOAD_MODE:-parquet}" != resident ] || RAPIDS_RUN_ARGS+=(--conf \
  spark.sql.cache.serializer="${RAPIDS_CACHE_SERIALIZER:-com.nvidia.spark.ParquetCachedBatchSerializer}")
mkdir -p "${SPARK_DIR}"
# The header first, so a row this script adds itself (a kill, a timeout, a
# process that left none) lands under it.
echo "query,status,seconds,gpu_ops,result_rows_or_error" > "${OUT_CSV}"; : > "${LOG}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }

free_gb(){ df -k --output=avail / | tail -1 | awk '{print int($1/1024/1024)}'; }
# A process and everything below it: this query's JVM and python, nobody else's.
kill_tree(){ local c; kill -STOP "$1" 2>/dev/null; for c in $(pgrep -P "$1"); do kill_tree "${c}"; done; kill -9 "$1" 2>/dev/null; }

# The spill pass: a zstd event log per query (outside SPARK_DIR, which is wiped),
# SQL metrics at MODERATE ("data read size"), a large listener queue so no event
# is dropped, and the RAPIDS spill loggers at INFO.
SPILL_CONF=(--conf spark.eventLog.enabled=true --conf spark.eventLog.compress=true
            --conf spark.eventLog.compression.codec=zstd
            --conf spark.scheduler.listenerbus.eventqueue.eventLog.capacity=100000
            --conf spark.rapids.sql.metrics.level=MODERATE
            --driver-java-options "-Dlog4j2.configurationFile=file:${RAPIDS_DIR}/conf/log4j2-spill.properties")
# Forced spill for `make spill-smoke` (SPILL_SMOKE=1): a 256 MiB RMM pool, 64 MiB
# batches, a 16 MiB host spill store, 8 shuffle partitions that AQE keeps apart.
SMOKE_CONF=(--conf spark.rapids.memory.gpu.allocSize=256m --conf spark.rapids.sql.batchSizeBytes=64m
            --conf spark.rapids.memory.host.spillStorageSize=16m --conf spark.sql.shuffle.partitions=8
            --conf spark.sql.adaptive.coalescePartitions.enabled=false)

log "Safe per-query RAPIDS run (${ENGINE_KEY}, load=${LOAD_MODE:-parquet}). queries=[${QUERIES}] min_free=${MIN_FREE_GB}GB driver=${DRIVER_MEM} limit=${LIMIT}s"
log "free disk at start: $(free_gb)GB ; ramdisk: $(du -sh ${RAM_PQ%/parquet} 2>/dev/null | cut -f1)"

for q in ${QUERIES}; do
  rm -rf "${SPARK_DIR:?}"/* 2>/dev/null
  qlog="${ROOT}/results/q${q}${SUFFIX}.log"
  extra=() stats_env=()
  if [ -n "${SPILL_STATS_DIR:-}" ]; then
    sdir="${SPILL_STATS_DIR}/q${q}"; rm -rf "${sdir:?}"; mkdir -p "${sdir}"
    qlog="${sdir}/driver.log"
    extra=("${SPILL_CONF[@]}" --conf "spark.eventLog.dir=file://${sdir}")
    [ "${SPILL_SMOKE:-0}" != 1 ] || extra+=("${SMOKE_CONF[@]}")
    stats_env=(RAPIDS_STATS_DIR="${sdir}")
  fi
  log "=== query ${q} starting (free $(free_gb)GB) ==="
  env -u CONTAINER_ID "${stats_env[@]}" spark-submit "${RAPIDS_RUN_ARGS[@]}" "${extra[@]}" \
    "${RAPIDS_DIR}/run_tpch_queries.py" "${RAM_PQ}" "${STREAM}" "${OUT_CSV}" "${q}" append \
    >"${qlog}" 2>&1 &
  pid=$!

  # watchdog: kill the query if free disk gets dangerously low, or if it
  # outlives LIMIT (only this query's JVM/python are killed).
  killed=0; timedout=0; tstart=$(date +%s)
  while kill -0 "${pid}" 2>/dev/null; do
    if [ "$(free_gb)" -lt "${MIN_FREE_GB}" ]; then
      log "    !! free disk < ${MIN_FREE_GB}GB during query ${q} -> killing to protect the box"
      kill_tree "${pid}"
      killed=1; break
    fi
    if [ "${LIMIT}" -gt 0 ] && [ $(( $(date +%s) - tstart )) -ge "${LIMIT}" ]; then
      log "    !! query ${q} exceeded ${LIMIT}s -> killing it"
      kill_tree "${pid}"
      timedout=1; break
    fi
    sleep 3
  done
  wait "${pid}" 2>/dev/null; rc=$?

  if [ "${killed}" = 1 ]; then
    echo "query${q},KILLED_DISK,NA,0,exceeded_disk_scratch_on_single_GPU" >> "${OUT_CSV}"
    log "    query ${q}: KILLED (disk). result line recorded."
  elif [ "${timedout}" = 1 ]; then
    res=$(grep -E "^query${q}[, ]" "${OUT_CSV}" | tail -1)
    if [ -n "${res}" ]; then
      # The query wrote its row and the limit came during the JVM's shutdown:
      # the row stands.
      log "    query ${q}: killed during shutdown, its row stands -> ${res}"
    else
      echo "query${q},TIMEOUT,$(( $(date +%s) - tstart )),0,exceeded_query_timeout_${LIMIT}s" >> "${OUT_CSV}"
      log "    query ${q}: TIMEOUT. result line recorded."
    fi
  else
    res=$(grep -E "^query${q}[, ]" "${OUT_CSV}" | tail -1)
    # A process that died without a row is FAIL: the merge replaces rows by
    # query, so the query's earlier row must not stay.
    [ -n "${res}" ] || { res="query${q},FAIL,NA,0,no_result_row_exit_code_${rc}"; echo "${res}" >> "${OUT_CSV}"; }
    log "    query ${q}: done rc=${rc} -> ${res}"
  fi
  if [ -n "${SPILL_STATS_DIR:-}" ]; then
    "${PY}" "${RAPIDS_DIR}/eventlog_stats.py" record "${sdir}" --sf "${TPCH_SF}" --query "${q}" \
        --evidence-dir "${SPILL_STATS_DIR}" >>"${LOG}" 2>&1 \
      || "${PY}" "${ROOT}/spill_results.py" stub rapids "${TPCH_SF}" "${q}" "${SPILL_STATS_DIR}" \
           --error "eventlog_stats.py record failed" >>"${LOG}" 2>&1
  fi
  rm -rf "${SPARK_DIR:?}"/* 2>/dev/null
done
if [ -n "${SPILL_STATS_DIR:-}" ]; then
  "${PY}" "${RAPIDS_DIR}/eventlog_stats.py" settings "${SPILL_STATS_DIR}" --sf "${TPCH_SF}" \
      --spark-dir "${SPARK_DIR}" >>"${LOG}" 2>&1 || log "    !! spill settings.json failed"
fi

log "ALL DONE. free disk: $(free_gb)GB"
log "results:"; cat "${OUT_CSV}" | tee -a "${LOG}"

if [ "${TPCH_MERGE:-1}" = 1 ]; then
  "${PY}" "${ROOT}/merge_results.py" "${ENGINE_KEY}" "${TPCH_SF}" "${OUT_CSV}" | tee -a "${LOG}" \
    && rm -f "${OUT_CSV}"
fi
