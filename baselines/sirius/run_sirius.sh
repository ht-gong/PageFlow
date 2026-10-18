#!/bin/bash
# Run TPC-H queries 1-22 through Sirius (GPU, gpu_execution) on the ramdisk
# parquet, SAFELY -- one duckdb process per query so each gets a clean GPU
# context and any spill scratch is reclaimed on exit, plus a disk watchdog that
# kills a query (and moves on) if free disk drops below a threshold so heavy
# spill can never wedge the box. Mirrors polars/run_polars_gpu.sh and
# rapids/run_tpch_safe.sh.
#
# One GPU; Sirius spills GPU -> pinned host RAM -> disk per sirius.yaml.
# Each process first runs its query on the SF1 copy in WARM_PARQUET, untimed
# (GPU init, JIT), then times it once on SIRIUS_PARQUET (run.sh protocol);
# results are appended incrementally.
#
#   run_sirius.sh [QUERY_LIST]   e.g. "1 2 3"   (default: 1..22)
#
# Env: SIRIUS_PARQUET (dataset), TPCH_SF, WARM_PARQUET (SF1 warm copy), PY, and
# the image's SIRIUS_DUCKDB / SIRIUS_ENVLIB (sirius/env.sh). SPILL_STATS_DIR
# (run.sh, SPILL=1): the spill pass; each query's evidence goes to
# $SPILL_STATS_DIR/q<N>/ and spill_stats.py appends its record to
# $SPILL_STATS_DIR/records.jsonl, kills and timeouts included. LOAD_MODE=resident
# + ENGINE_KEY=sirius_resident (run.sh sets both) pin the query's columns on the
# GPU before each pass (run_tpch_sirius.py, AGENTS.md).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUERIES="${1:-$(seq 1 22)}"
DIR="${ROOT}/sirius"
RAM_PQ="${SIRIUS_PARQUET:?SIRIUS_PARQUET is not set}"
: "${WARM_PARQUET:?WARM_PARQUET is not set (the SF1 warm copy)}"
STREAM="${ROOT}/results/queries/stream_qualification.sql"
LOGDIR="${ROOT}/results"
TPCH_SF="${TPCH_SF:?TPCH_SF is not set}"
# Per-query rows go to a throwaway temp CSV and are folded into the single
# canonical results/all_results.csv at the end (merge_results.py); no per-run
# CSV is kept. TAG only names the gitignored per-query log files.
ENGINE_KEY="${ENGINE_KEY:-sirius}"
TAG="${SIRIUS_TAG:-${ENGINE_KEY}_sf${TPCH_SF}}"
OUT_CSV="${OUT_CSV:-/tmp/tpch_${ENGINE_KEY}_sf${TPCH_SF}.csv}"
DETAIL_CSV="/tmp/tpch_${ENGINE_KEY}_sf${TPCH_SF}_detail.csv"
LOG="${LOGDIR}/${TAG}_run.log"
MIN_FREE_GB="${MIN_FREE_GB:-25}"

# shellcheck disable=SC1091
source "${DIR}/env.sh"
SPILL="${SIRIUS_SPILL}"
export SIRIUS_TIMEOUT="${SIRIUS_TIMEOUT:-2400}"
export SIRIUS_DETAIL_CSV="${DETAIL_CSV}"
export SIRIUS_LOG_DIR="${SIRIUS_LOG_DIR:-${LOGDIR}/${TAG}_logs}"
mkdir -p "${SPILL}" "${SIRIUS_LOG_DIR}"
# The headers first, so a row this script adds itself (a kill, a process that
# left none) lands under it.
echo "query,status,seconds,result_rows_or_error" > "${OUT_CSV}"
echo "query,status,seconds,rows,gpu" > "${DETAIL_CSV}"; : > "${LOG}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }
free_gb(){ df -k --output=avail / | tail -1 | awk '{print int($1/1024/1024)}'; }
# A process and everything below it: this query's processes, nobody else's.
kill_tree(){ local c; kill -STOP "$1" 2>/dev/null; for c in $(pgrep -P "$1"); do kill_tree "${c}"; done; kill -9 "$1" 2>/dev/null; }

if [ ! -x "${SIRIUS_DUCKDB}" ]; then
  log "ERROR: no Sirius duckdb binary at ${SIRIUS_DUCKDB}"; exit 1
fi

log "Sirius per-query run (${ENGINE_KEY}, load=${LOAD_MODE:-parquet}). queries=[${QUERIES}] warm=${WARM_PARQUET} timeout=${SIRIUS_TIMEOUT}s min_free=${MIN_FREE_GB}GB"
log "duckdb=${SIRIUS_DUCKDB}"
log "config=${SIRIUS_CONFIG_FILE} ; io=${SIRIUS_IO} ; parquet=${RAM_PQ}"
log "free disk at start: $(free_gb)GB ; ramdisk: $(du -sh ${RAM_PQ%/parquet} 2>/dev/null | cut -f1)"

for q in ${QUERIES}; do
  rm -rf "${SPILL:?}"/* 2>/dev/null
  qlog="${LOGDIR}/${TAG}_q${q}.log"
  log "=== query ${q} starting (free $(free_gb)GB) ==="
  "${PY:?PY is not set}" "${DIR}/run_tpch_sirius.py" "${RAM_PQ}" "${STREAM}" "${OUT_CSV}" "${q}" append \
      >"${qlog}" 2>&1 &
  pid=$!

  killed=0
  while kill -0 "${pid}" 2>/dev/null; do
    if [ "$(free_gb)" -lt "${MIN_FREE_GB}" ]; then
      log "    !! free disk < ${MIN_FREE_GB}GB during query ${q} -> killing to protect the box"
      kill_tree "${pid}"
      killed=1; break
    fi
    sleep 3
  done
  wait "${pid}" 2>/dev/null; rc=$?

  if [ "${killed}" = 1 ]; then
    echo "query${q},KILLED_DISK,NA,exceeded_disk_scratch" >> "${OUT_CSV}"
    log "    query ${q}: KILLED (disk)."
  else
    res=$(grep -E "^query${q}[, ]" "${OUT_CSV}" | tail -1)
    # A process that died without a row is FAIL: the merge replaces rows by
    # query, so the query's earlier row must not stay.
    [ -n "${res}" ] || { res="query${q},FAIL,NA,no_result_row_exit_code_${rc}"; echo "${res}" >> "${OUT_CSV}"; }
    log "    query ${q}: done rc=${rc} -> ${res}"
  fi
  if [ -n "${SPILL_STATS_DIR:-}" ]; then
    "${PY}" "${DIR}/spill_stats.py" record "${SPILL_STATS_DIR}/q${q}" --sf "${TPCH_SF}" --query "${q}" \
        --evidence-dir "${SPILL_STATS_DIR}" >>"${LOG}" 2>&1 \
      || "${PY}" "${ROOT}/spill_results.py" stub sirius "${TPCH_SF}" "${q}" "${SPILL_STATS_DIR}" \
           --error "spill_stats.py record failed" >>"${LOG}" 2>&1
  fi
  rm -rf "${SPILL:?}"/* 2>/dev/null
done
if [ -n "${SPILL_STATS_DIR:-}" ]; then
  cfg="$(ls "${SPILL_STATS_DIR}"/q*/sirius.yaml 2>/dev/null | head -1)"
  "${PY}" "${DIR}/spill_stats.py" settings "${SPILL_STATS_DIR}" --sf "${TPCH_SF}" \
      --config "${cfg:-${SIRIUS_CONFIG_FILE}}" >>"${LOG}" 2>&1 || log "    !! spill settings.json failed"
fi

log "ALL DONE. free disk: $(free_gb)GB"
log "results:"; cat "${OUT_CSV}" | tee -a "${LOG}"

# Fold this run into the single canonical results table, then drop the temp CSVs.
if [ "${TPCH_MERGE:-1}" = 1 ]; then
  "${PY}" "${ROOT}/merge_results.py" "${ENGINE_KEY}" "${TPCH_SF}" "${OUT_CSV}" | tee -a "${LOG}" \
    && rm -f "${OUT_CSV}" "${DETAIL_CSV}"
fi
