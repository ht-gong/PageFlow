#!/bin/bash
# DuckDB CPU baseline: run TPC-H q1-22 over the shared parquet dataset and fold
# the result into results/all_results.csv (like every other engine's runner).
#
#   run_duckdb.sh [QUERY_LIST]        e.g. "1 6 9"   (default: 1..22)
#
# Env: TPCH_PARQUET (dataset), TPCH_SF, WARM_PARQUET (SF1 warm copy), PY (the
# image's py env, which has duckdb 1.5.5), and optionally DUCKDB_THREADS,
# DUCKDB_MEMORY_LIMIT, DUCKDB_TEMP_DIR, TPCH_MERGE=0 (keep temp CSV).
# Tables are views over the ramdisk parquet: every run reads it. One process per
# query, which first runs the query on the SF1 copy, untimed (run.sh protocol).
# LOAD_MODE=resident + ENGINE_KEY=duckdb_cpu_resident (run.sh sets both) load
# native tables before each pass instead (AGENTS.md).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

QUERIES="${1:-$(seq 1 22)}"
TPCH_SF="${TPCH_SF:?TPCH_SF is not set}"
TPCH_PARQUET="${TPCH_PARQUET:?TPCH_PARQUET is not set}"
STREAM="${STREAM:-${ROOT}/results/queries/stream_qualification.sql}"
# throwaway temp; folded into results/all_results.csv at the end (no per-run CSV kept)
ENGINE_KEY="${ENGINE_KEY:-duckdb_cpu}"
OUT_CSV="${OUT_CSV:-/tmp/tpch_${ENGINE_KEY}_sf${TPCH_SF}.csv}"
LOG="${LOG:-${ROOT}/results/${ENGINE_KEY/duckdb_cpu/duckdb}_run.log}"
: "${WARM_PARQUET:?WARM_PARQUET is not set (the SF1 warm copy)}"

[ -d "${TPCH_PARQUET}" ] || { echo "ERROR: no parquet dir at ${TPCH_PARQUET}" >&2; exit 1; }

mkdir -p "$(dirname "${OUT_CSV}")" "$(dirname "${LOG}")"
# The header first, so a row this script adds itself lands under it.
echo "query,status,seconds,result_rows_or_error" > "${OUT_CSV}"

PY="${PY:?PY is not set}"

echo "[$(date +%H:%M:%S)] ${ENGINE_KEY} SF${TPCH_SF} queries=[${QUERIES}] parquet=${TPCH_PARQUET} python=${PY}" | tee "${LOG}"
for q in ${QUERIES}; do
  echo "[$(date +%H:%M:%S)] === query ${q} ===" | tee -a "${LOG}"
  "${PY}" "${ROOT}/duckdb/run_tpch_duckdb.py" \
    "${TPCH_PARQUET}" "${STREAM}" "${OUT_CSV}" "${q}" append 2>&1 | tee -a "${LOG}"
  rc=${PIPESTATUS[0]}
  # A process that died without a row is FAIL: the merge replaces rows by
  # query, so the query's earlier row must not stay.
  grep -q "^query${q}," "${OUT_CSV}" \
    || echo "query${q},FAIL,NA,no_result_row_exit_code_${rc}" | tee -a "${OUT_CSV}" "${LOG}"
done

if [ "${TPCH_MERGE:-1}" = 1 ]; then
  "${PY}" "${ROOT}/merge_results.py" "${ENGINE_KEY}" "${TPCH_SF}" "${OUT_CSV}" "${ROOT}/results" \
    | tee -a "${LOG}" && rm -f "${OUT_CSV}"
fi
