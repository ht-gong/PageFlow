#!/bin/bash
# ==========================================================================
# run.sh — the single entry point for the TPC-H engine comparison.
#
# For each requested scale factor: ensure the dataset exists on disk (generate
# it with datagen/ndsgen if missing), validate it, stage it disk ->
# ramdisk, run each requested engine against the ramdisk copy, then free the
# ramdisk before the next scale factor. Each engine self-merges its 22 rows into
# results/all_results.csv.
#
# Protocol, the same for every engine: one process per query. In that process an
# untimed warm pass first runs the same query on SF1 (a ramdisk copy staged once
# per run and passed as WARM_PARQUET), then the timed run reads the target SF's
# ramdisk parquet. No query runs twice on the target data within a process.
#
#   ./run.sh [SF_SPEC ...]
#
#   SF_SPEC   scale factors to run. A bare number (e.g. 100) or an inclusive
#             range over the canonical set {30,50,100,300,500,700} (e.g. 30-300).
#             Default: 30 50 100 300 500 700
#
# Env:
#   ENGINES   space-separated engines, in run order.
#             Default: "rapids polars_gpu duckdb_cpu sirius"
#             Resident mode (data loaded before the timed run; AGENTS.md):
#             duckdb_cpu_resident polars_gpu_resident sirius_resident rapids_resident
#   DATA_DIR  where datasets live on disk, as $DATA_DIR/sf<SF>/parquet/<table>/
#             (required). Missing datasets are generated here.
#   SCRATCH   engine scratch root (required; runners use $SCRATCH/<engine>).
#   SHM       ramdisk root (default /dev/shm).
#   QUERIES   query subset (default "1 2 ... 22").
#   KEEP_RAMDISK=1   keep the staged ramdisk copies, incl. the SF1 warm copy (default: delete).
#   GEN_PARALLEL / GEN_BATCH / GEN_OPTS   generation chunking and format (datagen/dataset.sh).
#   DRIVER_MEM, GPU_PART_MB, MIN_FREE_GB ... passed through to runners.
#   RUN_CSV_DIR=<dir>  smoke mode: nothing is merged; each engine's rows stay in
#             <dir>/<engine>_sf<SF>.csv, a query x engine table of status and row
#             counts is printed at the end (results/smoke_table.py), and the exit
#             code is 1 unless every query is OK with one row count, at SF1 the
#             reference's (`make smoke`).
#   SPILL=1   the spill pass instead (AGENTS.md, "Spill pass"): the same
#             protocol with each engine's data-movement instrumentation on, for
#             rapids, polars_gpu and sirius (other engines are skipped). Nothing
#             reaches all_results.csv. Per SF and engine, the pass's rows, raw
#             evidence and per-query records go to $SPILL_ROOT/sf<SF>/<engine>/
#             (default $SCRATCH/spill) and are merged into
#             results/spill_profile.csv and results/spill_settings.json
#             (spill_results.py), unless SPILL_MERGE=0. Log:
#             results/spill_run.log. SPILL_SMOKE=1: each runner's forced-spill settings,
#             evidence in $SCRATCH/spill_smoke, never merged, and the exit code
#             is 1 unless every record and identity holds (`make spill-smoke`).
#             RUN_CSV_DIR is ignored.
# The tool paths (PY, POLARS_PY, JAVA_HOME, SPARK_HOME, RAPIDS_JAR,
# SIRIUS_DUCKDB, SIRIUS_ENVLIB, NDSGEN) come from the image.
#
# Examples:
#   ./run.sh                       # all four engines, SF30..SF700 (normally: make bench)
#   ./run.sh 30-300                # SF30,50,100,300
#   ENGINES="duckdb_cpu" ./run.sh 500 700
# ==========================================================================
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHM="${SHM:-/dev/shm}"
DATA_DIR="${DATA_DIR:?DATA_DIR is not set}"
: "${SCRATCH:?SCRATCH is not set}" "${PY:?PY is not set}"
ENGINES="${ENGINES:-rapids polars_gpu duckdb_cpu sirius}"
QUERIES="${QUERIES:-$(seq 1 22)}"
CANON_SFS="30 50 100 300 500 700"

RESULTS="${REPO}/results"
LOG="${RESULTS}/run.log"
RUN_CSV_DIR="${RUN_CSV_DIR:-}"
SPILL="${SPILL:-0}"
SPILL_MERGE="${SPILL_MERGE:-1}"
SPILL_ROOT="${SPILL_ROOT:-${SCRATCH}/spill}"
if [ "${SPILL_SMOKE:-0}" = 1 ]; then
  # Forced-spill settings: checked, never merged.
  SPILL_MERGE=0; SPILL_ROOT="${SPILL_ROOT%/spill}/spill_smoke"
fi
[ "${SPILL}" != 1 ] || LOG="${RESULTS}/spill_run.log"
SPILL_DIRS=()

: > "${LOG}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "${LOG}"; }
# Free ramdisk in GB. When the container is bound to one NUMA node
# (docker/run.sh), the copy can only use that node's memory: its free memory plus
# reclaimable page cache, if that is less than /dev/shm's own free space.
shm_free_gb(){
  local shm mems node
  shm="$(df -k --output=avail "${SHM}" | tail -1 | awk '{print int($1/1024/1024)}')"
  mems="$(sed -n 's/^Mems_allowed_list:\s*//p' /proc/self/status)"
  case "${mems}" in *[,-]*) echo "${shm}"; return ;; esac
  node="$(awk '/MemFree:/{f=$4} /FilePages:/{p=$4} /Shmem:/{s=$4} END{print int((f+p-s)/1024/1024)}' \
            "/sys/devices/system/node/node${mems}/meminfo")"
  echo $(( node < shm ? node : shm ))
}

# Expand SF_SPEC tokens: a bare number is itself; "A-B" expands to the canonical
# scale factors within [A,B].
expand_sfs(){
  local tok lo hi sf out=""
  for tok in "$@"; do
    if [[ "${tok}" == *-* ]]; then
      lo="${tok%-*}"; hi="${tok#*-}"
      for sf in ${CANON_SFS}; do
        [ "${sf}" -ge "${lo}" ] && [ "${sf}" -le "${hi}" ] && out+=" ${sf}"
      done
    else
      out+=" ${tok}"
    fi
  done
  echo ${out}
}

# Drop empty (0-row) parquet part-files from the ramdisk copy so every engine
# reads identical files (Sirius' GPU reader errors on a file with no row groups;
# harmless to others).
remove_empty_parquets(){
  "${PY}" - "$1" <<'PY'
import glob, os, sys
import pyarrow.parquet as pq
for f in glob.glob(f"{sys.argv[1]}/*/*.parquet"):
    if pq.ParquetFile(f).metadata.num_rows == 0:
        os.remove(f)
PY
}

# Copy a parquet dataset disk -> <ramdisk dir>/parquet and drop its empty
# part-files. $1=disk parquet dir  $2=ramdisk dir
stage(){
  rm -rf "$2"; mkdir -p "$2"
  cp -r "$1" "$2/parquet" || { rm -rf "$2"; return 1; }
  remove_empty_parquets "$2/parquet"
}

run_engine(){   # $1=engine  $2=ramdisk_parquet  $3=SF
  local e="$1" pq="$2" sf="$3" smoke=()
  [ -z "${RUN_CSV_DIR}" ] || smoke=(TPCH_MERGE=0 OUT_CSV="${RUN_CSV_DIR}/${e}_sf${sf}.csv")
  # Spill pass: the pass's own rows stay in its evidence dir, never merged.
  [ "${SPILL}" != 1 ] || smoke=(TPCH_MERGE=0 OUT_CSV="${SPILL_ROOT}/sf${sf}/${e}/pass.csv"
                                SPILL_STATS_DIR="${SPILL_ROOT}/sf${sf}/${e}")
  case "${e}" in
    rapids)     env "${smoke[@]}" TPCH_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/rapids/run_tpch_safe.sh"  "${QUERIES}" ;;
    polars_gpu) env "${smoke[@]}" TPCH_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/polars/run_polars_gpu.sh" "${QUERIES}" ;;
    duckdb_cpu) env "${smoke[@]}" TPCH_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/duckdb/run_duckdb.sh"      "${QUERIES}" ;;
    sirius)     env "${smoke[@]}" SIRIUS_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/sirius/run_sirius.sh"    "${QUERIES}" ;;
    # Resident mode (AGENTS.md): the same runners, the query's data loaded
    # before the timed run, rows under their own engine key.
    duckdb_cpu_resident)  env "${smoke[@]}" LOAD_MODE=resident ENGINE_KEY="${e}" TPCH_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/duckdb/run_duckdb.sh" "${QUERIES}" ;;
    polars_gpu_resident)  env "${smoke[@]}" LOAD_MODE=resident ENGINE_KEY="${e}" TPCH_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/polars/run_polars_gpu.sh" "${QUERIES}" ;;
    sirius_resident)      env "${smoke[@]}" LOAD_MODE=resident ENGINE_KEY="${e}" SIRIUS_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/sirius/run_sirius.sh" "${QUERIES}" ;;
    rapids_resident)      env "${smoke[@]}" LOAD_MODE=resident ENGINE_KEY="${e}" TPCH_PARQUET="${pq}" TPCH_SF="${sf}" bash "${REPO}/rapids/run_tpch_safe.sh" "${QUERIES}" ;;
    *) log "    unknown engine '${e}'"; return 1 ;;
  esac
}

# Ensure $DATA_DIR/sf<SF>/parquet exists and validates (datagen/dataset.sh
# generates it if missing). Echoes the parquet dir on success, empty on failure.
ensure_dataset(){
  local sf="$1"
  if ! "${REPO}/datagen/dataset.sh" "${sf}" >>"${LOG}" 2>&1; then
    log "    !! SF${sf} dataset could not be generated or failed validation (see ${LOG}, results/GENERATOR.md)"; return 1
  fi
  echo "${DATA_DIR}/sf${sf}/parquet"
}

SFS="$(expand_sfs "${@:-30 50 100 300 500 700}")"
log "TPC-H run. SFs=[${SFS}] engines=[${ENGINES}] data=${DATA_DIR} ramdisk=${SHM}${RUN_CSV_DIR:+ smoke -> ${RUN_CSV_DIR}}"
# Smoke mode drops an earlier run's CSVs of these engines and SFs only: the
# TQP-Vortex root's `make bench` fills one dir an SF at a time.
if [ -n "${RUN_CSV_DIR}" ]; then
  mkdir -p "${RUN_CSV_DIR}"
  for sf in ${SFS}; do for e in ${ENGINES}; do rm -f "${RUN_CSV_DIR}/${e}_sf${sf}.csv"; done; done
fi

# The SF1 copy every engine process warms up on (see the protocol above).
WARM_DIR="${SHM}/tpch_warm_sf1"
WARM_DISK="$(ensure_dataset 1)" && [ -d "${WARM_DISK}" ] \
  || { log "!! no valid SF1 dataset for the warm pass"; exit 1; }
log "staging ${WARM_DISK} -> ${WARM_DIR}/parquet (SF1 warm pass) ..."
stage "${WARM_DISK}" "${WARM_DIR}" || { log "!! staging the SF1 warm copy failed"; exit 1; }
export WARM_PARQUET="${WARM_DIR}/parquet"

for SF in ${SFS}; do
  log "=== SF${SF} ==="

  DISK_PQ="$(ensure_dataset "${SF}")" && [ -d "${DISK_PQ}" ] || { log "    skipping SF${SF}"; continue; }

  # Stage disk -> ramdisk (need ~1x parquet size + headroom).
  need=$(( SF * 36 * 13 / 1000 + 20 ))
  RAM_DIR="${SHM}/tpch_sf${SF}"; RAM_PQ="${RAM_DIR}/parquet"
  rm -rf "${RAM_DIR}"
  if [ "$(shm_free_gb)" -lt "${need}" ]; then
    log "    !! not enough ramdisk on the GPU's NUMA node (need ~${need}GB, free $(shm_free_gb)GB) -> skipping SF${SF}"; continue
  fi
  log "    staging ${DISK_PQ} -> ${RAM_PQ} ..."
  stage "${DISK_PQ}" "${RAM_DIR}" || { log "    !! stage failed"; continue; }
  log "    staged: $(du -sh "${RAM_PQ}" 2>/dev/null | cut -f1)"

  for E in ${ENGINES}; do
    log "    --- SF${SF} / ${E} ---"
    if [ "${SPILL}" = 1 ]; then
      case "${E}" in
        rapids|polars_gpu|sirius) ;;
        *) log "        ${E}: no spill instrumentation, skipped"; continue ;;
      esac
      sdir="${SPILL_ROOT}/sf${SF}/${E}"
      mkdir -p "${sdir}"; : > "${sdir}/records.jsonl"; SPILL_DIRS+=("${sdir}")
    fi
    run_engine "${E}" "${RAM_PQ}" "${SF}" >>"${LOG}" 2>&1 || log "        ${E} returned nonzero"
    if [ "${SPILL}" = 1 ]; then
      log "        ${E}: $(grep -c . "${sdir}/records.jsonl") spill records in ${sdir}"
      if [ "${SPILL_MERGE}" = 1 ]; then
        if "${PY}" "${REPO}/spill_results.py" merge "${E}" "${SF}" "${sdir}" >>"${LOG}" 2>&1; then
          log "        ${E}: merged -> results/spill_profile.csv"
        else
          log "        !! ${E}: spill merge refused (see ${LOG})"
        fi
      fi
    elif [ -n "${RUN_CSV_DIR}" ]; then
      ok=$(awk -F, '$2=="OK"' "${RUN_CSV_DIR}/${E}_sf${SF}.csv" 2>/dev/null | wc -l)
      log "        ${E}: ${ok}/$(echo ${QUERIES} | wc -w) OK (kept in ${RUN_CSV_DIR})"
    else
      ok=$(awk -F, -v e="${E}" -v s="${SF}" '$1==e && $2==s && $4=="OK"' \
             "${RESULTS}/all_results.csv" 2>/dev/null | wc -l)
      log "        ${E}: ${ok}/22 OK (merged -> all_results.csv)"
    fi
  done

  [ "${KEEP_RAMDISK:-0}" = 1 ] || rm -rf "${RAM_DIR}"
done
[ "${KEEP_RAMDISK:-0}" = 1 ] || rm -rf "${WARM_DIR}"

if [ "${SPILL}" = 1 ]; then
  if [ "${SPILL_SMOKE:-0}" = 1 ]; then
    # Fail unless every record keeps the contract, every query is OK, every
    # identity its builder asserts holds, and every engine spilled somewhere.
    "${PY}" "${REPO}/spill_results.py" check --expect-spill "${SPILL_DIRS[@]}" | tee -a "${LOG}"
    exit "${PIPESTATUS[0]}"
  fi
  [ "${SPILL_MERGE}" = 1 ] && into="${RESULTS}/spill_profile.csv" || into="nothing merged"
  log "=== spill pass done -> ${into} (evidence in ${SPILL_ROOT}) ==="
  exit 0
fi
if [ -z "${RUN_CSV_DIR}" ]; then
  log "=== done -> ${RESULTS}/all_results.csv ==="
  exit 0
fi
# Smoke summary: status:rows per query x engine; fail unless every query is OK
# with one row count, at SF1 the reference's.
"${PY}" "${REPO}/results/smoke_table.py" "${RUN_CSV_DIR}" "${ENGINES}" "${SFS}" "$(echo ${QUERIES})" | tee -a "${LOG}"
