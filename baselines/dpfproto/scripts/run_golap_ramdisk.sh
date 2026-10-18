#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPF_ROOT="$ROOT/DPFProto"

# settings
SF="${SF:-100}"
THREADS="${THREADS:-1}"
TRIALS="${TRIALS:-3}"
QUERIES="${QUERIES:-q1 q3 q5 q6 q13 q16}"
ZONEMAP="${ZONEMAP:-0}"
TAG="${TAG:-}"
DEVICE_SIZE="${DEVICE_SIZE:-64G}"
DATA_BASE="$ROOT/data/tpch"
DEV_BASE="/dev/shm/dpfproto_golap_filedev/sf${SF}"
if [[ -z "${LOG_DIR:-}" ]]; then
  LOG_DIR="$ROOT/logs/golap_ramdisk/$(date +%Y%m%d_%H%M%S)"
  [[ -n "$TAG" ]] && LOG_DIR+="_$TAG"
fi

cd "$DPF_ROOT"

# build binaries
export DPF_DEPS=$HOME/.local/dpfproto-deps
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$CUDA_HOME/gds/tools:$DPF_DEPS/bin:$PATH
export PKG_CONFIG_PATH=$DPF_DEPS/lib/pkgconfig:$DPF_DEPS/lib64/pkgconfig:${PKG_CONFIG_PATH:-}
export LD_LIBRARY_PATH=$DPF_DEPS/lib:$DPF_DEPS/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$DPF_DEPS/lib:$DPF_DEPS/lib64:${LIBRARY_PATH:-}
export CMAKE_PREFIX_PATH=$DPF_DEPS:${CMAKE_PREFIX_PATH:-}

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
make -C build -j"$(nproc)" tpchdb tpchloader

mkdir -p "$DEV_BASE" "$LOG_DIR"

# use cufile compat mode
CUFILE_COMPAT_JSON="$LOG_DIR/cufile.compat.json"
sed 's/"allow_compat_mode": false/"allow_compat_mode": true/' \
  "$DPF_ROOT/config/cufile.json" > "$CUFILE_COMPAT_JSON"
export CUFILE_ENV_PATH_JSON="$CUFILE_COMPAT_JSON"

# make ramdisk devices
for i in 0 1 2 3; do
  truncate -s "$DEVICE_SIZE" "$DEV_BASE/dev${i}.img"
done
DEVICES_NVME="$DEV_BASE/dev0.img,$DEV_BASE/dev1.img,$DEV_BASE/dev2.img,$DEV_BASE/dev3.img"
TPCH_ARGS=()
[[ "$ZONEMAP" == 1 ]] && TPCH_ARGS=(-Z)

echo "DATA_BASE=$DATA_BASE"
echo "DEVICES_NVME=$DEVICES_NVME"
echo "ZONEMAP=$ZONEMAP"
echo "LOG_DIR=$LOG_DIR"

# load pages
./build/tpchloader \
  -i "$DATA_BASE/sideways/sf${SF}" \
  -d "$DEVICES_NVME" \
  -x gidp \
  -A \
  -c | tee "$LOG_DIR/load.log"

# run queries
FAILED=0
for q in $QUERIES; do
  if ! scripts/bench.sh -n "$TRIALS" -w -t 60 \
    -o "$LOG_DIR/${q}.txt" \
    -- ./build/tpchdb \
      -q "$q" \
      -x gidp \
      -w "$THREADS" \
      "${TPCH_ARGS[@]}" \
      "$DEVICES_NVME"; then
    echo "failed $q" | tee -a "$LOG_DIR/failed.txt"
    FAILED=1
  fi
done

echo "Done: $LOG_DIR"
exit "$FAILED"
