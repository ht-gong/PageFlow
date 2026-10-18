#!/bin/bash
# The one container launcher. Every `make` target runs through it.
#
#   docker/run.sh build          build the image for the current pins
#   docker/run.sh tag            print the image tag
#   docker/run.sh [CMD ARGS...]  run CMD (default: bash) in the container
#
# The image tag is a hash of versions.env, docker/*.lock, the Dockerfile and the
# ndsgen sources (compiled into the image), so a pin change always gets a new image. Box-specific settings come from the
# environment or else local.env at the TQP-Vortex root (see local.env.example
# there): DATA_DIR, SCRATCH, SHM, GPU, NUMA.
#
# Run flags:
#   --gpus device=$GPU     one GPU (GPUs are shared; Sirius fails at startup if
#                          an exclusive-mode GPU is visible). Inside, it is GPU 0.
#   --cpuset-cpus/-mems    the container runs on the GPU's NUMA node only: its
#                          CPUs and its memory. Every engine thread, every pinned
#                          host pool and the ramdisk copy (tmpfs pages are
#                          allocated by the writer) are local to the GPU, and the
#                          engines size their thread pools from the bound CPUs.
#                          NUMA=off (with a GPU) or no GPU: unbound.
#   --cap-add SYS_NICE     lets NUMA-aware engines bind their own memory (Sirius
#                          mbinds its pinned pool); Docker's seccomp profile
#                          allows mbind/set_mempolicy only with this capability
#   --ipc=host             /dev/shm is the host's, so ramdisk staging (SF500 is
#                          ~180 GB) behaves as on the host
#   --security-opt seccomp=docker/seccomp-iouring.json
#                          Docker's default profile blocks io_uring; without it
#                          Sirius silently falls back to kvikio
#   --ulimit memlock=-1    io_uring registered buffers and pinned host memory
#   same-path -v           repo, DATA_DIR and SCRATCH are mounted at their host
#                          paths, so scripts see the same paths inside and out
#   --user + HOME + passwd no root-owned files; the user gets a passwd entry
#                          (Java/Hadoop need a user name, Spark a writable home)
#   --init                 the per-query timeouts kill whole process trees
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# local.env fills in what the environment does not already set, so a one-off
# `DATA_DIR=/elsewhere make data SF=1` wins over it.
LOCAL_ENV="${REPO}/../local.env"
if [ -f "${LOCAL_ENV}" ]; then
  while IFS='=' read -r key value || [ -n "${key}" ]; do
    case "${key}" in ''|'#'*) continue ;; esac
    [ -n "${!key+x}" ] || export "${key}=${value}"
  done < "${LOCAL_ENV}"
fi
TAG="tqp-baselines:$(cat "${REPO}/versions.env" "${REPO}/docker/py.lock" "${REPO}/docker/polars.lock" \
  "${REPO}/docker/Dockerfile" "${REPO}"/datagen/ndsgen/Cargo.{toml,lock} "${REPO}"/datagen/ndsgen/src/*.rs \
  | sha256sum | cut -c1-12)"

case "${1:-}" in
  tag) echo "${TAG}"; exit 0 ;;
  build)
    args=()
    while IFS= read -r line; do
      case "${line}" in ''|'#'*) continue ;; esac
      args+=(--build-arg "${line}")
    done < "${REPO}/versions.env"
    exec docker build -f "${REPO}/docker/Dockerfile" "${args[@]}" -t "${TAG}" "${REPO}" ;;
esac

docker image inspect "${TAG}" >/dev/null 2>&1 \
  || { echo "image ${TAG} not built for the current pins; run: make -C ${REPO} image" >&2; exit 1; }
: "${DATA_DIR:?set DATA_DIR in ${LOCAL_ENV}}" "${SCRATCH:?set SCRATCH in ${LOCAL_ENV}}"
SHM="${SHM:-/dev/shm}"
mkdir -p "${DATA_DIR}" "${SCRATCH}/home"
passwd="${SCRATCH}/home/.passwd"
printf 'root:x:0:0:root:/root:/bin/bash\nnobody:x:65534:65534:nobody:/nonexistent:/usr/sbin/nologin\n%s:x:%s:%s::%s:/bin/bash\n' \
  "$(id -un)" "$(id -u)" "$(id -g)" "${SCRATCH}/home" > "${passwd}"

opts=(--rm --init --ipc=host --ulimit memlock=-1 --cap-add SYS_NICE
      --security-opt "seccomp=${REPO}/docker/seccomp-iouring.json"
      --user "$(id -u):$(id -g)" -e HOME="${SCRATCH}/home" -v "${passwd}:/etc/passwd:ro"
      -v "${REPO}:${REPO}" -v "${DATA_DIR}:${DATA_DIR}" -v "${SCRATCH}:${SCRATCH}" -w "${REPO}"
      -e DATA_DIR -e SCRATCH -e SHM="${SHM}")
if [ -n "${GPU:-}" ]; then
  opts+=(--gpus "device=${GPU}" -e CUDA_VISIBLE_DEVICES=0)
  if [ "${NUMA:-auto}" != off ]; then
    # The GPU's NUMA node from its PCI device (nvidia-smi: 00000000:85:00.0,
    # sysfs: 0000:85:00.0); -1 when the box reports none.
    bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${GPU}")"
    bus="${bus: -12}"
    node="$(cat "/sys/bus/pci/devices/${bus,,}/numa_node")"
    if [ "${node}" -ge 0 ]; then
      opts+=(--cpuset-cpus "$(cat "/sys/devices/system/node/node${node}/cpulist")" --cpuset-mems "${node}")
    fi
  fi
fi
[ -t 0 ] && [ -t 1 ] && opts+=(-it)
# Settings the targets and runners read, forwarded when set.
for v in SF ENGINES QUERIES KEEP_RAMDISK GEN_PARALLEL GEN_BATCH DRIVER_MEM GPU_PART_MB MIN_FREE_GB \
         QUERY_TIMEOUT RAPIDS_TIMEOUT RAPIDS_SHUFFLE_MODE SIRIUS_TIMEOUT SIRIUS_IO PROBE_TIMEOUT TPCH_MERGE \
         DUCKDB_THREADS DUCKDB_MEMORY_LIMIT DUCKDB_TEMP_DIR RUN_CSV_DIR NUMA \
         SPILL SPILL_ROOT SPILL_MERGE SPILL_SMOKE; do
  [ -z "${!v+x}" ] || opts+=(-e "${v}")
done
[ $# -gt 0 ] || set -- bash
exec docker run "${opts[@]}" "${TAG}" "$@"
