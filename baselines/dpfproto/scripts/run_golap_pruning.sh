#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF="${SF:-300}"
QUERIES="${QUERIES:-q1 q3 q5 q6 q13 q16}"
DEVICE_SIZE="${DEVICE_SIZE:-64G}"
RUN_DIR="${RUN_DIR:-$ROOT/logs/golap_ramdisk/$(date +%Y%m%d_%H%M%S)_sf${SF}_pruning}"
FAILED=0

run() {
  local zonemap="$1"
  local name="$2"
  if ! SF="$SF" QUERIES="$QUERIES" DEVICE_SIZE="$DEVICE_SIZE" \
    ZONEMAP="$zonemap" LOG_DIR="$RUN_DIR/$name" \
    "$ROOT/scripts/run_golap_ramdisk.sh"; then
    FAILED=1
  fi
}

# run both modes
run 0 pruning_off
run 1 pruning_on

# do plotting
python3 "$ROOT/scripts/plot_pruning.py" \
  "$RUN_DIR/pruning_off" "$RUN_DIR/pruning_on" \
  --sf "$SF" --out "$RUN_DIR/pruning_runtime.png"

exit "$FAILED"
