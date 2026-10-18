#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPF_ROOT="$ROOT/DPFProto"

# settings
SF="${SF:-300}"
THREADS="${THREADS:-$(nproc)}"
DATA_BASE="$ROOT/data/tpch"
export PATH="$HOME/.local/bin:$PATH"
DBGEN_DIR="${TPCH_DBGEN_DIR:-}"

# find dbgen
if [[ -x "$DBGEN_DIR/dbgen/dbgen" ]]; then
  DBGEN_DIR="$DBGEN_DIR/dbgen"
fi

if [[ ! -f "$DBGEN_DIR/dbgen" || ! -x "$DBGEN_DIR/dbgen" ]]; then
  echo "Set TPCH_DBGEN_DIR to the official TPC-H dbgen directory" >&2
  exit 1
fi
export TPCH_DBGEN_DIR="$DBGEN_DIR"

# check duckdb
if ! command -v duckdb >/dev/null; then
  echo "Install the DuckDB CLI first" >&2
  exit 1
fi

# clear empty failed run
for dir in "$DATA_BASE/input$SF" "$DATA_BASE/sideways/sf$SF"; do
  if [[ -d "$dir" && -z "$(find "$dir" -type f -print -quit)" ]]; then
    find "$dir" -depth -type d -empty -delete
  fi
done

# keep existing data
if [[ -e "$DATA_BASE/input$SF" || -e "$DATA_BASE/sideways/sf$SF" ]]; then
  echo "SF$SF data already exists under $DATA_BASE" >&2
  exit 1
fi

# make raw tables
"$DPF_ROOT/scripts/tpch/run_dbgen.sh" "$SF" "$THREADS" "$DATA_BASE"

# make sideways data
DATA_BASE="$DATA_BASE" bash "$DPF_ROOT/scripts/golap/01_sideways_pruning_data_base.sh" \
  -s "$SF" -n "$THREADS"

echo "Done: $DATA_BASE/sideways/sf$SF"
