#!/bin/bash
# Start one column-probe worker (ablation/probe_<engine>.py) in the same engine
# environment its TPC-H runner uses. Called by ablation/format_profile.py.
#
#   probe.sh <engine> <parquet_dir> <columns_file> <out_csv>
#
# Env: PROBE_SCRATCH (engine scratch dir), PROBE_LOG_DIR (Sirius log dir), and
#      the image's tool variables (PY, POLARS_PY, JAVA_HOME, SPARK_HOME,
#      RAPIDS_JAR, SIRIUS_DUCKDB, SIRIUS_ENVLIB).
set -euo pipefail
ENGINE="${1:?engine}"; shift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
SCRATCH="${PROBE_SCRATCH:?PROBE_SCRATCH}"
# Workers run one at a time; empty the scratch like the runners do between
# queries (a killed Spark worker leaves its local dir, incl. an 880 MB jar copy).
mkdir -p "${SCRATCH}"
rm -rf "${SCRATCH:?}"/* 2>/dev/null || true

case "${ENGINE}" in
  rapids)
    # shellcheck disable=SC1091
    source "${ROOT}/rapids/activate.sh" >/dev/null
    rapids_run_args "${SCRATCH}"
    exec env -u CONTAINER_ID spark-submit "${RAPIDS_RUN_ARGS[@]}" "${HERE}/probe_rapids.py" "$@" ;;
  polars_gpu)
    # shellcheck disable=SC1091
    source "${ROOT}/polars/env.sh"
    export POLARS_TEMP_DIR="${SCRATCH}"
    exec "${POLARS_PY}" "${HERE}/probe_polars.py" "$@" ;;
  duckdb_cpu)
    exec "${PY:?PY is not set}" "${HERE}/probe_duckdb.py" "$@" ;;
  sirius)
    # shellcheck disable=SC1091
    source "${ROOT}/sirius/env.sh"
    export SIRIUS_LOG_DIR="${PROBE_LOG_DIR:-${SCRATCH}}"
    rm -rf "${SIRIUS_SPILL:?}"/* 2>/dev/null || true
    "${PY:?PY is not set}" "${HERE}/probe_sirius.py" "$@" ;;
  *) echo "probe.sh: unknown engine '${ENGINE}'" >&2; exit 2 ;;
esac
