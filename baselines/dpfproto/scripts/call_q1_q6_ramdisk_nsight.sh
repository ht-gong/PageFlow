#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DPF_ROOT="$ROOT/DPFProto"
SF="${SF:-100}"
THREADS="${THREADS:-1}"
DEVICE_SIZE="${DEVICE_SIZE:-64G}"
QUERIES="${QUERIES:-q1 q6}"
DEV_BASE="/dev/shm/dpfproto_golap_filedev/sf${SF}"
LOG_DIR="$ROOT/logs/golap_ramdisk_nsight/$(date +%Y%m%d_%H%M%S)"

# load ramdisk once
SF="$SF" QUERIES="q1" TRIALS=1 THREADS="$THREADS" DEVICE_SIZE="$DEVICE_SIZE" \
  "$ROOT/scripts/run_golap_ramdisk.sh"

mkdir -p "$LOG_DIR"
DEVICES="$DEV_BASE/dev0.img,$DEV_BASE/dev1.img,$DEV_BASE/dev2.img,$DEV_BASE/dev3.img"

cd "$DPF_ROOT"

# cufile compat
CUFILE_COMPAT_JSON="$LOG_DIR/cufile.compat.json"
sed 's/"allow_compat_mode": false/"allow_compat_mode": true/' \
  "$DPF_ROOT/config/cufile.json" > "$CUFILE_COMPAT_JSON"
export CUFILE_ENV_PATH_JSON="$CUFILE_COMPAT_JSON"
export LD_LIBRARY_PATH="$HOME/.local/dpfproto-deps/lib:$HOME/.local/dpfproto-deps/lib64:/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-}"

echo "LOG_DIR=$LOG_DIR"
echo "DEVICES=$DEVICES"

# profile queries
FAILED=0
for q in $QUERIES; do
  if ! nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cpuctxsw=process-tree \
    --gpu-metrics-devices=all \
    --stats=true \
    --force-overwrite=true \
    -o "$LOG_DIR/${q}_nsys" \
    ./build/tpchdb -q "$q" -x gidp -w "$THREADS" "$DEVICES" \
    2>&1 | tee "$LOG_DIR/${q}.txt"; then
    echo "failed $q" | tee -a "$LOG_DIR/failed.txt"
    FAILED=1
  fi
done

echo "Done: $LOG_DIR"
exit "$FAILED"


# running nsight after
# nsys stats dpfproto/logs/golap_ramdisk_nsight/20260714_143630/q1_nsys.nsys-rep > dpfproto/logs/golap_ramdisk_nsight/20260714_143630/q1_nsys_stats.txt
# nsys stats dpfproto/logs/golap_ramdisk_nsight/20260714_143630/q6_nsys.nsys-rep > dpfproto/logs/golap_ramdisk_nsight/20260714_143630/q6_nsys_stats.txt

# grep "time:\\|throughput:\\|io_throughput_gbs\\|effective_throughput_gbs\\|gpu_mem_mb" dpfproto/logs/golap_ramdisk_nsight/20260714_143630/q1.txt
# grep "time:\\|throughput:\\|io_throughput_gbs\\|effective_throughput_gbs\\|gpu_mem_mb" dpfproto/logs/golap_ramdisk_nsight/20260714_143630/q6.txt

# dd if=/dev/shm/dpfproto_golap_filedev/sf100/dev0.img of=/dev/null bs=1M count=8192 status=progress

# we want to see:
# Q1 runtime and GOLAP throughput
# Q6 runtime and GOLAP throughput
# Nsight kernel time summary
# ramdisk dd bandwidth
# whether GOLAP IO throughput is below ramdisk bandwidth