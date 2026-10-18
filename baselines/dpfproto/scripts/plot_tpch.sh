#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="$ROOT/logs/golap_ramdisk/20260718_214537_pruning_on"

if [[ ! -d "$LOG_DIR" ]]; then
  LOG_DIR="$(find "$ROOT/logs/golap_filedev" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)"
fi

python3 "${SCRIPT_DIR}/plot_tpch.py" "${LOG_DIR}" --out "${LOG_DIR}/tpch_runtime.png"
echo "Wrote ${LOG_DIR}/tpch_runtime.png"
