#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPF_ROOT="$ROOT/DPFProto"
ARCH_PATCHES=(
  "$ROOT/patches/0002-q5-sm86-cuda-arch.patch"
  "$ROOT/patches/0003-global-sm86-cuda-arch.patch"
)
cd "$DPF_ROOT"

# Fixed experiment settings.
# generate things here:
# dpfproto/data/tpch/input100
# dpfproto/data/tpch/sideways/sf100
SF=100
THREADS=1
TRIALS=3
QUERIES="q1 q3 q5 q6 q13 q16"
DEVICE_SIZE=64G
DATA_BASE="$ROOT/data/tpch"
DEV_BASE="$ROOT/data/golap_filedev/sf${SF}"
LOG_DIR="$ROOT/logs/golap_filedev/$(date +%Y%m%d_%H%M%S)"

# Build for the RTX 3090 sm86 target.
NEED_CLEAN=0
for patch in "${ARCH_PATCHES[@]}"; do
  if git apply --reverse --check "$patch" >/dev/null 2>&1; then
    echo "Patch already applied: $patch"
  else
    git apply "$patch"
    NEED_CLEAN=1
    echo "Applied patch: $patch"
  fi
done

export DPF_DEPS=$HOME/.local/dpfproto-deps
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$CUDA_HOME/gds/tools:$DPF_DEPS/bin:$PATH
export PKG_CONFIG_PATH=$DPF_DEPS/lib/pkgconfig:$DPF_DEPS/lib64/pkgconfig:${PKG_CONFIG_PATH:-}
export LD_LIBRARY_PATH=$DPF_DEPS/lib:$DPF_DEPS/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$DPF_DEPS/lib:$DPF_DEPS/lib64:${LIBRARY_PATH:-}
export CMAKE_PREFIX_PATH=$DPF_DEPS:${CMAKE_PREFIX_PATH:-}
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
CLEAN_MARKER="build/.sm86_clean_done"
if [[ "$NEED_CLEAN" == 1 || ! -f "$CLEAN_MARKER" ]]; then
  cmake --build build --target clean
  touch "$CLEAN_MARKER"
fi
make -C build -j"$(nproc)" tpchdb tpchloader

mkdir -p "$DEV_BASE" "$LOG_DIR"

# cuFile compat mode is needed when nvidia-fs is not loaded.
CUFILE_COMPAT_JSON="$LOG_DIR/cufile.compat.json"
sed 's/"allow_compat_mode": false/"allow_compat_mode": true/' \
  "$DPF_ROOT/config/cufile.json" > "$CUFILE_COMPAT_JSON"
export CUFILE_ENV_PATH_JSON="$CUFILE_COMPAT_JSON"

# Regular files stand in for NVMe device images.
for i in 0 1 2 3; do
  truncate -s "$DEVICE_SIZE" "$DEV_BASE/dev${i}.img"
done
DEVICES_NVME="$DEV_BASE/dev0.img,$DEV_BASE/dev1.img,$DEV_BASE/dev2.img,$DEV_BASE/dev3.img"

echo "DATA_BASE=$DATA_BASE"
echo "DEVICES_NVME=$DEVICES_NVME"
echo "LOG_DIR=$LOG_DIR"

# Load GOLAP-style compressed pages.
./build/tpchloader \
  -i "$DATA_BASE/sideways/sf${SF}" \
  -d "$DEVICES_NVME" \
  -x gidp \
  -A \
  -c | tee "$LOG_DIR/load.log"

# Run the partial TPC-H suite.
FAILED=0
for q in $QUERIES; do
  if ! scripts/bench.sh -n "$TRIALS" -w -t 60 \
    -o "$LOG_DIR/${q}.txt" \
    -- ./build/tpchdb \
      -q "$q" \
      -x gidp \
      -w "$THREADS" \
      "$DEVICES_NVME"; then
    echo "FAILED: $q" | tee -a "$LOG_DIR/failed.txt"
    FAILED=1
  fi
done

echo "Done: $LOG_DIR"
exit "$FAILED"
