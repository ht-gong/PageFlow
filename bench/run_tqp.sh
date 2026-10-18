#!/bin/bash
# The tqp_vortex runner of the engine comparison (`make bench`; the protocol and
# the results contract are in baselines/AGENTS.md). Runs on the host.
#
#   bench/run_tqp.sh <SF> [QUERY_LIST]     e.g. 10 "1 6"   (default: 1..22)
#
# One main.py process per query, in the engine image (so `make build` first),
# with the default SF1 warmup (Q1, then the query) and one timed run at <SF> on
# the parquet in $TQP_DATA_DIR. A query listed twice runs once. Each process
# writes its perf report to out/bench/<run>/q<N>/ (its query_results.csv keeps
# the load time) and its console to q<N>.log beside it. Its row becomes the
# query's row of the run CSV (query,status,seconds,result_rows_or_error):
# seconds is approx_time (or the time to failure), NA where none was taken; an
# error is cut to 120 characters; a process that ends without a row is FAIL
# with its exit code; one that outlives TQP_TIMEOUT is removed and recorded
# TIMEOUT. The run CSV is then merged into baselines/results/all_results.csv
# as tqp_vortex.
#
# Natively (TQP_NATIVE=1, for a box without Docker) each process is the same
# main.py on the host, from this checkout, with the environment the image and
# its compose service give it: $TQP_PY (bench/native_env.sh builds it and this
# checkout's modules), CUDA_VISIBLE_DEVICES=$TQP_GPU (its CUDA device 0),
# TQP_TABLES_PATH=$TQP_DATA_DIR, OMP_NUM_THREADS (default 32) and a 64 MiB
# stack. Its CPUs are bound to the GPU's NUMA node with taskset, as
# /workspace/native-run.sh binds the baselines (NUMA=off: unbound), and
# TQP_NUMA_BIND defaults to 0: an unprivileged container refuses mbind, so the
# pinned pool is placed on the node by first touch instead (AGENTS.md, "Host:
# the pinned pool"). A timeout or an interrupt kills the process's own tree
# only. The run CSV is merged with $PY.
#
# Env, filled in from local.env at the repo root where unset:
#   DATA_DIR, GPU, SCRATCH   box settings (the run CSV goes to $SCRATCH, where
#                  the baselines container that merges it can read it)
#   TQP_DATA_DIR   the data root (default $DATA_DIR): the datasets of SF1 and
#                  <SF> (make data)
#   TQP_GPU        the GPU (default $GPU, else 0). The runner stops if another
#                  process holds it: one main.py at a time, on an idle card.
#   TQP_TIMEOUT    seconds per process (default 600 + 3 x SF; 0: none):
#                  pinning, the warmup, the load and the query
#   TPCH_MERGE=0   keep the run CSV ($SCRATCH/tqp_vortex_sf<SF>.csv) unmerged
#   RUN_CSV_DIR    smoke mode: the run CSV is <dir>/tqp_vortex_sf<SF>.csv and
#                  nothing is merged
#   DOCKER         e.g. "sudo docker"; COMPOSE (default "$DOCKER compose") as
#                  in the Makefile
#   TQP_NATIVE=1   run natively, as above; then also
#     TQP_PY       the engine's Python (default /opt/envs/tqp/bin/python)
#     PY           the Python that merges (default /opt/envs/py/bin/python, the
#                  baselines')
#     NUMA=off     no CPU binding
#     TQP_NUMA_BIND, OMP_NUM_THREADS   passed on (defaults 0 and 32)
#
# The spill pass (SPILL=1; baselines/AGENTS.md, "Spill pass") runs the same
# protocol with TQP_EXPERIMENT_TELEMETRY=1 instead of the bench: a process per
# query, its perf report (the movement ledger and the residency peaks)
# copied with its log into $SPILL_ROOT/sf<SF>/tqp_vortex/q<N>/, the run CSV
# as pass.csv beside it (the pass's own times, never merged into
# all_results.csv), and bench/tqp_spill_stats.py's record of each query in
# records.jsonl (an all-NA stub when there is no usable report: a timeout, a
# kill), then settings.json. The records are merged into
# baselines/results/spill_profile.csv (spill_results.py merge) unless
# SPILL_MERGE=0. Natively the builder and the merge run with $PY; in Docker
# mode with the host's python3 (stdlib only). The variables are run.sh's:
#   SPILL=1        the spill pass instead of the bench
#   SPILL_ROOT     evidence root (default $SCRATCH/spill)
#   SPILL_MERGE=0  keep the records unmerged
#   SPILL_SMOKE=1  forced pressure: TQP's only pressure path is the granule
#                  cache's (under --buffer_strategy cpu, the bench's, a full
#                  arena or pinned pool fails the query), so each query runs
#                  with the dataflow cache over a 4 MiB slot pool
#                  (SMOKE_ARGS); evidence in $SCRATCH/spill_smoke, never
#                  merged, and the exit code is 1 unless every record keeps
#                  the contract, every query is OK, every identity holds and
#                  some query spilled (spill_results.py check --expect-spill)
set -uo pipefail
# A '.' decimal point, and the error cut counts characters, not bytes.
export LC_ALL=C.UTF-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}" || exit 1
# local.env fills in what the environment does not set, as in
# baselines/docker/run.sh.
if [ -f local.env ]; then
  while IFS='=' read -r key value || [ -n "${key}" ]; do
    case "${key}" in ''|'#'*) continue ;; esac
    [ -n "${!key+x}" ] || export "${key}=${value}"
  done < local.env
fi

usage(){ echo "usage: bench/run_tqp.sh <SF> [QUERY_LIST]   (query ids 1-22, space-separated)" >&2; exit 2; }
SF="${1:-}"
[[ "${SF}" =~ ^[0-9]+$ ]] && (( 10#${SF} > 0 )) || usage
SF=$(( 10#${SF} ))
QUERIES=""
for q in ${2:-$(seq 1 22)}; do
  [[ "${q}" =~ ^[0-9]+$ ]] && (( 10#${q} >= 1 && 10#${q} <= 22 )) || usage
  q=$(( 10#${q} ))
  [[ " ${QUERIES} " == *" ${q} "* ]] || QUERIES+="${QUERIES:+ }${q}"
done
export TQP_DATA_DIR="${TQP_DATA_DIR:-${DATA_DIR:?set DATA_DIR in local.env (or TQP_DATA_DIR)}}"
export TQP_GPU="${TQP_GPU:-${GPU:-0}}"
TIMEOUT="${TQP_TIMEOUT:-$(( 600 + 3 * SF ))}"
[[ "${TIMEOUT}" =~ ^[0-9]+$ ]] || { echo "TQP_TIMEOUT must be whole seconds (0: none)" >&2; exit 2; }
NATIVE="${TQP_NATIVE:-0}"
DOCKER="${DOCKER:-docker}"
COMPOSE="${COMPOSE:-${DOCKER} compose}"
SPILL="${SPILL:-0}"
SMOKE_ARGS=()
if [ "${SPILL}" = 1 ]; then
  SPILL_MERGE="${SPILL_MERGE:-1}"
  SPILL_ROOT="${SPILL_ROOT:-${SCRATCH:?set SCRATCH in local.env (or SPILL_ROOT)}/spill}"
  if [ "${SPILL_SMOKE:-0}" = 1 ]; then
    # Forced pressure (the header): checked, never merged.
    SPILL_MERGE=0; SPILL_ROOT="${SPILL_ROOT%/spill}/spill_smoke"
    SMOKE_ARGS=(--buffer_strategy dataflow --slot_pool_gb 0.00390625)
  fi
  SPILL_DIR="${SPILL_ROOT}/sf${SF}/tqp_vortex"
  OUT_CSV="${SPILL_DIR}/pass.csv"
  # The builder, the check and the merge: stdlib Python.
  if [ "${NATIVE}" = 1 ]; then SPY="${PY:-/opt/envs/py/bin/python}"; else SPY="${PY:-python3}"; fi
elif [ -n "${RUN_CSV_DIR:-}" ]; then
  OUT_CSV="${RUN_CSV_DIR}/tqp_vortex_sf${SF}.csv"
else
  OUT_CSV="${SCRATCH:?set SCRATCH in local.env}/tqp_vortex_sf${SF}.csv"
fi
rm -f "${OUT_CSV}"
RUN="sf${SF}_$(date +%Y%m%d_%H%M%S)_$$"
RUN_DIR="out/bench/${RUN}"
log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "${RUN_DIR}/run.log"; }

for s in 1 "${SF}"; do
  [ -d "${TQP_DATA_DIR}/sf${s}/parquet" ] \
    || { echo "no SF${s} dataset in ${TQP_DATA_DIR} (make data SF=${s})" >&2; exit 1; }
done
# One main.py at a time, on an idle card: the default pools take the whole GPU.
busy="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i "${TQP_GPU}" 2>&1)" \
  || { echo "nvidia-smi found no GPU ${TQP_GPU}: ${busy}" >&2; exit 1; }
[ -z "${busy}" ] || { printf 'GPU %s is in use; stopping:\n%s\n' "${TQP_GPU}" "${busy}" >&2; exit 1; }

# The native process: the engine image's environment (Dockerfile, the compose
# service), its one GPU as CUDA device 0, and its CPUs on the GPU's NUMA node.
mode="engine image"
if [ "${NATIVE}" = 1 ]; then
  TQP_PY="${TQP_PY:-/opt/envs/tqp/bin/python}"
  [ -x "${TQP_PY}" ] || { echo "no ${TQP_PY}: build it with bench/native_env.sh" >&2; exit 1; }
  export CUDA_VISIBLE_DEVICES="${TQP_GPU}" TQP_TABLES_PATH="${TQP_DATA_DIR}" \
         OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}" TQP_NUMA_BIND="${TQP_NUMA_BIND:-0}" \
         PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
  ulimit -s 65536 || exit 1
  native=("${TQP_PY}")
  mode="native, TQP_NUMA_BIND=${TQP_NUMA_BIND}"
  if [ "${NUMA:-auto}" != off ]; then
    # The GPU's NUMA node from its PCI device (nvidia-smi: 00000000:81:00.0,
    # sysfs: 0000:81:00.0); -1 when the box reports none.
    bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${TQP_GPU}")"; bus="${bus: -12}"
    node="$(cat "/sys/bus/pci/devices/${bus,,}/numa_node")"
    if [ "${node}" -ge 0 ]; then
      native=(taskset -c "$(cat "/sys/devices/system/node/node${node}/cpulist")" "${native[@]}")
      mode+=", CPUs of NUMA node ${node}"
    fi
  fi
fi

mkdir -p "${RUN_DIR}" "$(dirname "${OUT_CSV}")" || exit 1
echo "query,status,seconds,result_rows_or_error" > "${OUT_CSV}"
if [ "${SPILL}" = 1 ]; then
  : > "${SPILL_DIR}/records.jsonl"
  mode+=", spill pass${SPILL_SMOKE:+ smoke=${SPILL_SMOKE}} -> ${SPILL_DIR}"
  [ "${NATIVE}" != 1 ] || export TQP_EXPERIMENT_TELEMETRY=1
fi
log "tqp_vortex SF${SF} queries=[${QUERIES}] data=${TQP_DATA_DIR} GPU ${TQP_GPU} timeout=${TIMEOUT}s (${mode}) -> ${RUN_DIR}"

# An interrupted run takes its main.py off the GPU too. timeout runs in its own
# process group, which a Ctrl-C does not reach: kill the group first (the
# compose client, or natively main.py itself), so that a client cannot start
# the container after the removal.
name=""; pid=""
stop(){
  [ -z "${pid}" ] || { kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null; wait "${pid}" 2>/dev/null; }
  [ -z "${name}" ] || ${DOCKER} rm -f "${name}" >/dev/null 2>&1
}
trap 'stop; exit 130' INT TERM

# run_main <q> <log> <main.py args...>: one main.py under TQP_TIMEOUT; sets rc
# (124, or 137 after the grace period, on a timeout) and secs. The spill pass
# turns the experiment telemetry on.
run_main(){
  local q="$1" out="$2"; shift 2
  local t0 tel=(); t0=$(date +%s)
  [ "${SPILL}" != 1 ] || tel=(-e TQP_EXPERIMENT_TELEMETRY=1)
  if [ "${NATIVE}" = 1 ]; then
    timeout -k 30 "${TIMEOUT}" "${native[@]}" main.py "$@" </dev/null >"${out}" 2>&1 &
  else
    name="tqp_bench_${RUN}_q${q}"
    timeout -k 30 "${TIMEOUT}" ${COMPOSE} run --rm -T "${tel[@]}" --name "${name}" engine \
      python3 main.py "$@" </dev/null >"${out}" 2>&1 &
  fi
  pid=$!; wait "${pid}"; rc=$?; pid=""
  secs=$(( $(date +%s) - t0 ))
  # Killing only the compose client (as timeout does) can leave the container
  # on the GPU, so remove it whatever the exit; after --rm this does nothing.
  [ -z "${name}" ] || ${DOCKER} rm -f "${name}" >/dev/null 2>&1
  name=""
}

# The spill pass's evidence of query <q>: the perf report and the log, and its
# record (a stub when the builder fails).
spill_record(){
  local q="$1" sdir="${SPILL_DIR}/q$1"
  [ ! -d "${RUN_DIR}/q${q}" ] || cp -r "${RUN_DIR}/q${q}" "${sdir}/perf"
  cp "${RUN_DIR}/q${q}.log" "${sdir}/run.log" 2>/dev/null
  "${SPY}" "${ROOT}/bench/tqp_spill_stats.py" record "${sdir}" --sf "${SF}" --query "${q}" \
      --evidence-dir "${SPILL_DIR}" >>"${RUN_DIR}/run.log" 2>&1 \
    || "${SPY}" "${ROOT}/baselines/spill_results.py" stub tqp_vortex "${SF}" "${q}" "${SPILL_DIR}" \
         --error "tqp_spill_stats.py record failed" >>"${RUN_DIR}/run.log" 2>&1
}

for q in ${QUERIES}; do
  [ "${SPILL}" != 1 ] || { rm -rf "${SPILL_DIR:?}/q${q}"; mkdir -p "${SPILL_DIR}/q${q}"; }
  run_main "${q}" "${RUN_DIR}/q${q}.log" --SF "${SF}" --q "${q}" --no_debug --perf_report "${RUN_DIR}/q${q}" \
    "${SMOKE_ARGS[@]}"
  row="$(awk -F, -v q="${q}" '$1 == q' "${RUN_DIR}/q${q}/query_results.csv" 2>/dev/null | tail -1)"
  if [ "${TIMEOUT}" -gt 0 ] && { [ "${rc}" = 124 ] || { [ "${rc}" = 137 ] && [ "${secs}" -ge "${TIMEOUT}" ]; }; }; then
    line="query${q},TIMEOUT,${secs},exceeded_query_timeout_${TIMEOUT}s"
  elif [ -z "${row}" ]; then
    line="query${q},FAIL,NA,no_result_row_exit_code_${rc}"
  else
    IFS=, read -r _ status t res _ <<< "${row}"
    [ -n "${t}" ] && t="$(printf '%.3f' "${t}")" || t=NA
    line="query${q},${status},${t},${res:0:120}"
  fi
  echo "${line}" >> "${OUT_CSV}"
  log "    ${line} (${secs}s process)"
  [ "${SPILL}" != 1 ] || spill_record "${q}"
done

ok=$(grep -c '^query[0-9]*,OK,' "${OUT_CSV}")
log "tqp_vortex SF${SF}: ${ok}/$(echo ${QUERIES} | wc -w) OK"
if [ "${SPILL}" = 1 ]; then
  log "tqp_vortex: $(grep -c . "${SPILL_DIR}/records.jsonl") spill records in ${SPILL_DIR}"
  CUDA_VISIBLE_DEVICES="${TQP_GPU}" "${SPY}" "${ROOT}/bench/tqp_spill_stats.py" settings "${SPILL_DIR}" \
    --sf "${SF}" >>"${RUN_DIR}/run.log" 2>&1 || log "!! spill settings.json failed"
  if [ "${SPILL_SMOKE:-0}" = 1 ]; then
    # Every record keeps the contract, every query is OK, every identity
    # holds, and some query spilled under the forced pressure.
    "${SPY}" "${ROOT}/baselines/spill_results.py" check --expect-spill "${SPILL_DIR}" | tee -a "${RUN_DIR}/run.log"
    exit "${PIPESTATUS[0]}"
  fi
  if [ "${SPILL_MERGE}" != 1 ]; then
    log "kept ${SPILL_DIR} (nothing merged)"
  elif "${SPY}" "${ROOT}/baselines/spill_results.py" merge tqp_vortex "${SF}" "${SPILL_DIR}" >>"${RUN_DIR}/run.log" 2>&1; then
    log "tqp_vortex: merged -> baselines/results/spill_profile.csv"
  else
    log "!! spill merge refused (see ${RUN_DIR}/run.log)"; exit 1
  fi
  exit 0
fi
merge(){
  if [ "${NATIVE}" = 1 ]; then
    "${PY:-/opt/envs/py/bin/python}" "${ROOT}/baselines/merge_results.py" tqp_vortex "${SF}" "${OUT_CSV}"
  else
    "${ROOT}/baselines/docker/run.sh" bash -c '"$PY" merge_results.py tqp_vortex "$1" "$2"' _ "${SF}" "${OUT_CSV}"
  fi
}
if [ -n "${RUN_CSV_DIR:-}" ] || [ "${TPCH_MERGE:-1}" != 1 ]; then
  log "kept ${OUT_CSV} (nothing merged)"
elif merge; then
  rm -f "${OUT_CSV}"
else
  log "!! merge failed; ${OUT_CSV} kept"; exit 1
fi
