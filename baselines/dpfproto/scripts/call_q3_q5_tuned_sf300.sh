#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF="${SF:-300}"
DEVICE_SIZE="${DEVICE_SIZE:-128G}"
RUN_DIR="${RUN_DIR:-$ROOT/logs/golap_ramdisk/$(date +%Y%m%d_%H%M%S)_sf${SF}_q3_q5_tuned}"

GOLAP_GPU_MEM_GB="${GOLAP_GPU_MEM_GB:-20}" \
Q3_TILE_PAGES="${Q3_TILE_PAGES:-256}" \
Q5_TILE_PAGES="${Q5_TILE_PAGES:-256}" \
SF="$SF" DEVICE_SIZE="$DEVICE_SIZE" QUERIES="q3 q5" RUN_DIR="$RUN_DIR" \
  "$ROOT/scripts/run_golap_pruning.sh"
