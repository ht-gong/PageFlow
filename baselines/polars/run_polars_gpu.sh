#!/bin/bash
# Run native-Polars TPC-H queries 1-22 on the ramdisk dataset on the GPU
# (cudf-polars / RAPIDS), one python process per query so each gets a clean GPU
# context. The streaming executor spills device->host via rapidsmpf (async
# pool). Each process first runs its query on the SF1 copy in WARM_PARQUET,
# untimed, then times it on TPCH_PARQUET (run.sh protocol).
#
#   run_polars_gpu.sh [QUERY_LIST]   e.g. "1 2 3"   (default: 1..22)
#
# Env: TPCH_PARQUET (dataset), TPCH_SF, WARM_PARQUET (SF1 warm copy), SCRATCH
# (spill goes to $SCRATCH/polars), POLARS_PY (the image's cudf-polars env).
# SPILL_STATS_DIR (run.sh, SPILL=1): the spill pass; rapidsmpf statistics on,
# each pass metered into $SPILL_STATS_DIR/q<N>/meter.json, and spill_stats.py
# appends the query's record to $SPILL_STATS_DIR/records.jsonl (an all-NA one
# for a kill, a timeout or a process that left no timed meter). SPILL_SMOKE=1
# lowers the device spill limit to 128 MiB.
# LOAD_MODE=resident + ENGINE_KEY=polars_gpu_resident (run.sh sets both) run over
# GPU-resident columns instead (run_tpch_polars.py, device_cache.py, AGENTS.md).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUERIES="${1:-$(seq 1 22)}"
DIR="${ROOT}/polars"
RAM_PQ="${TPCH_PARQUET:?TPCH_PARQUET is not set}"
: "${WARM_PARQUET:?WARM_PARQUET is not set (the SF1 warm copy)}"
TPCH_SF="${TPCH_SF:?TPCH_SF is not set}"
# throwaway temp; folded into results/all_results.csv at the end (no per-run CSV kept)
ENGINE_KEY="${ENGINE_KEY:-polars_gpu}"
OUT_CSV="${OUT_CSV:-/tmp/tpch_${ENGINE_KEY}_sf${TPCH_SF}.csv}"
SPILL_DIR="${SCRATCH:?SCRATCH is not set}/polars"
LOG="${ROOT}/results/${ENGINE_KEY}_run.log"
MIN_FREE_GB="${MIN_FREE_GB:-30}"
QUERY_TIMEOUT="${QUERY_TIMEOUT:-0}"  # kill a query after this many seconds (0 = never); row -> TIMEOUT

source "${DIR}/env.sh"
export POLARS_TEMP_DIR="${SPILL_DIR}"
if [ -n "${SPILL_STATS_DIR:-}" ]; then
  export RAPIDSMPF_STATISTICS=1
  [ "${SPILL_SMOKE:-0}" != 1 ] || export RAPIDSMPF_SPILL_DEVICE_LIMIT=128MiB
fi
mkdir -p "${SPILL_DIR}"
# The header first, so a row this script adds itself (a kill, a timeout, a
# process that left none) lands under it.
echo "query,status,seconds,result_rows_or_error" > "${OUT_CSV}"; : > "${LOG}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }
free_gb(){ df -k --output=avail / | tail -1 | awk '{print int($1/1024/1024)}'; }
# A process and everything below it: this query's processes, nobody else's.
kill_tree(){ local c; kill -STOP "$1" 2>/dev/null; for c in $(pgrep -P "$1"); do kill_tree "${c}"; done; kill -9 "$1" 2>/dev/null; }

log "Polars GPU per-query run (${ENGINE_KEY}, load=${LOAD_MODE:-parquet}). queries=[${QUERIES}] mr=${GPU_MR} part=${GPU_PART_MB}MB pinned=${RAPIDSMPF_PINNED_MEMORY} min_free=${MIN_FREE_GB}GB"
log "free disk at start: $(free_gb)GB"

for q in ${QUERIES}; do
  rm -rf "${SPILL_DIR:?}"/* 2>/dev/null
  qlog="${ROOT}/results/${ENGINE_KEY}_q${q}.log"
  if [ -n "${SPILL_STATS_DIR:-}" ]; then
    sdir="${SPILL_STATS_DIR}/q${q}"; rm -rf "${sdir:?}"; mkdir -p "${sdir}"
    qlog="${sdir}/run.log"
  fi
  log "=== query ${q} starting (free $(free_gb)GB) ==="
  "${POLARS_PY}" "${DIR}/run_tpch_polars.py" "${RAM_PQ}" "${OUT_CSV}" "${q}" append \
    >"${qlog}" 2>&1 &
  pid=$!
  killed=0; timedout=0; tstart=$(date +%s)
  while kill -0 "${pid}" 2>/dev/null; do
    if [ "$(free_gb)" -lt "${MIN_FREE_GB}" ]; then
      log "    !! free disk < ${MIN_FREE_GB}GB during query ${q} -> killing"
      kill_tree "${pid}"; killed=1; break
    fi
    if [ "${QUERY_TIMEOUT}" -gt 0 ] && [ $(( $(date +%s) - tstart )) -ge "${QUERY_TIMEOUT}" ]; then
      log "    !! query ${q} exceeded ${QUERY_TIMEOUT}s -> killing it"
      kill_tree "${pid}"; timedout=1; break
    fi
    sleep 3
  done
  wait "${pid}" 2>/dev/null; rc=$?
  if [ "${killed}" = 1 ]; then
    echo "query${q},KILLED_DISK,NA,exceeded_disk_scratch" >> "${OUT_CSV}"
    log "    query ${q}: KILLED (disk)."
  elif [ "${timedout}" = 1 ]; then
    echo "query${q},TIMEOUT,$(( $(date +%s) - tstart )),exceeded_query_timeout_${QUERY_TIMEOUT}s" >> "${OUT_CSV}"
    log "    query ${q}: TIMEOUT."
  else
    res=$(grep -E "^query${q}[, ]" "${OUT_CSV}" | tail -1)
    # A process that died without a row is FAIL: the merge replaces rows by
    # query, so the query's earlier row must not stay.
    [ -n "${res}" ] || { res="query${q},FAIL,NA,no_result_row_exit_code_${rc}"; echo "${res}" >> "${OUT_CSV}"; }
    log "    query ${q}: done rc=${rc} -> ${res}"
  fi
  if [ -n "${SPILL_STATS_DIR:-}" ]; then
    "${POLARS_PY}" "${DIR}/spill_stats.py" record "${sdir}" --sf "${TPCH_SF}" --query "${q}" \
        --evidence-dir "${SPILL_STATS_DIR}" >>"${LOG}" 2>&1 \
      || "${POLARS_PY}" "${ROOT}/spill_results.py" stub polars_gpu "${TPCH_SF}" "${q}" "${SPILL_STATS_DIR}" \
           --error "spill_stats.py record failed" >>"${LOG}" 2>&1
  fi
  rm -rf "${SPILL_DIR:?}"/* 2>/dev/null
done
if [ -n "${SPILL_STATS_DIR:-}" ]; then
  "${POLARS_PY}" "${DIR}/spill_stats.py" settings "${SPILL_STATS_DIR}" --sf "${TPCH_SF}" >>"${LOG}" 2>&1 \
    || log "    !! spill settings.json failed"
fi

log "ALL DONE. free disk: $(free_gb)GB"
log "results:"; cat "${OUT_CSV}" | tee -a "${LOG}"

if [ "${TPCH_MERGE:-1}" = 1 ]; then
  "${POLARS_PY}" "${ROOT}/merge_results.py" "${ENGINE_KEY}" "${TPCH_SF}" "${OUT_CSV}" | tee -a "${LOG}" \
    && rm -f "${OUT_CSV}"
fi
