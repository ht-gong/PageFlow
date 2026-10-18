#!/bin/bash
# The native TQP-Vortex environment, for a box without Docker (the Vast box of
# docs/engine_spill_plan.md, §4 F4): the Dockerfile's base, deps and builder
# stages as a venv, plus this checkout's engine modules in build/ and TQPlib/
# (both gitignored). bench/run_tqp.sh runs main.py with it under TQP_NATIVE=1.
#
#   bench/native_env.sh [all]    venv, modules, check
#   bench/native_env.sh venv     the venv at $TQP_ENV, exactly bench/native.lock
#   bench/native_env.sh modules  cmake build/ and stage its .so files into
#                                TQPlib/<pkg>/, as `pip3 install .` does in the
#                                image (again after any C++/CUDA edit)
#   bench/native_env.sh check    versions, and a torch and a cuDF kernel on
#                                $TQP_GPU (default 0)
#   bench/native_env.sh lock     re-resolve bench/native.lock (with hashes) from
#                                requirements.txt, requirements-dev.txt and the
#                                pins below; then rebuild the venv
#
# Env:
#   TQP_ENV             the venv (default /opt/envs/tqp)
#   PYTHON              the interpreter it is made from (default
#                       /usr/bin/python3.12, the system's, as for /opt/envs/py);
#                       the build needs its headers and libpython. A venv made
#                       from another one needs a fresh build/ (rm -rf build).
#   CUDA_HOME           the CUDA toolkit nvcc comes from (default
#                       /usr/local/cuda): 12.8 or later, to match torch's cu128
#   CUDA_ARCHITECTURES  CMAKE_CUDA_ARCHITECTURES (default: the compute
#                       capabilities nvidia-smi lists, e.g. 120 on an RTX 5090)
#
# Where it differs from the image: Python 3.12 (the image's is 3.11) and
# torch 2.8.0+cu128 (the image's 2.6.0+cu126 has no sm_120 kernels; torch
# >= 2.9 needs C++20, see the Dockerfile). cuDF, and with it the one nvCOMP
# (the nvidia-libnvcomp-cu12 wheel libcudf pins), are requirements.txt's.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
ENV="${TQP_ENV:-/opt/envs/tqp}"
LOCK="bench/native.lock"
PY="${ENV}/bin/python"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# The pins the image's base stage installs, with torch moved to a release
# that has sm_120 kernels (and without torchvision, which nothing imports).
PINS=("torch==2.8.0+cu128" "cmake>=3.28,<4" "ninja" "setuptools>=61" "wheel")
TORCH_INDEX=https://download.pytorch.org/whl/cu128
# torch comes from its index, everything else from PyPI.
INDEX=(--extra-index-url "${TORCH_INDEX}" --index-strategy unsafe-best-match)

lock(){
  printf '%s\n' "${PINS[@]}" | uv pip compile requirements.txt requirements-dev.txt - \
    --python-version 3.12 --python-platform x86_64-manylinux_2_28 "${INDEX[@]}" \
    --generate-hashes --custom-compile-command "bench/native_env.sh lock" -o "${LOCK}"
}

venv(){
  [ -x "${PY}" ] || uv venv --python "${PYTHON:-/usr/bin/python3.12}" "${ENV}"
  uv pip sync --python "${PY}" "${INDEX[@]}" --require-hashes "${LOCK}"
}

modules(){
  [ -x "${PY}" ] || { echo "no venv at ${ENV}: bench/native_env.sh venv" >&2; exit 1; }
  local archs="${CUDA_ARCHITECTURES:-}"
  if [ -z "${archs}" ]; then
    archs="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d . | sort -u | paste -sd';')"
    [ -n "${archs}" ] || { echo "no GPU to take the architectures from: set CUDA_ARCHITECTURES" >&2; exit 1; }
  fi
  export VIRTUAL_ENV="${ENV}" PATH="${ENV}/bin:${CUDA_HOME}/bin:${PATH}"
  # As in the image: compression/ builds against the nvCOMP wheel cuDF pins and
  # links the libnvcomp.so.5 libcudf loads (one nvCOMP per process).
  local nvcomp
  nvcomp="$("${PY}" -c 'import os, nvidia.libnvcomp as m; print(os.path.dirname(m.__file__))')"
  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="${archs}" \
    -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
    -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
    -DPython3_EXECUTABLE="${PY}" \
    -Dnvcomp_DIR="${nvcomp}/lib64/cmake/nvcomp"
  cmake --build build -j"$(nproc)"
  # setup.py's staging (build/<module>/*.so -> TQPlib/<pkg>/); main.py imports
  # the in-tree copy, with the repo root on sys.path.
  local m p
  for m in memory:tqpmemory IO:zero_copy_reader IO/cache:tqpcache \
           compression:tqpcompress \
           operators/kernels:tqpkernels; do
    p="${m#*:}"
    cp build/"${m%%:*}"/*.so "TQPlib/${p}/"
    ls TQPlib/"${p}"/*.so >/dev/null
  done
  echo "modules built for sm ${archs} and staged in TQPlib/"
}

check(){
  CUDA_VISIBLE_DEVICES="${TQP_GPU:-0}" "${PY}" - <<'EOF'
import sys
import torch
import pylibcudf as plc
from importlib.metadata import version
sys.path.insert(0, ".")
import TQPlib.tqpmemory, TQPlib.zero_copy_reader, TQPlib.tqpcompress  # noqa: F401
import TQPlib.tqpkernels  # noqa: F401

cc = torch.cuda.get_device_capability(0)
print(f"python {sys.version.split()[0]}, torch {torch.__version__} (CUDA "
      f"{torch.version.cuda}), pylibcudf {version('pylibcudf-cu12')}, GPU "
      f"{torch.cuda.get_device_name(0)} sm_{cc[0]}{cc[1]}")
assert f"sm_{cc[0]}{cc[1]}" in torch.cuda.get_arch_list(), torch.cuda.get_arch_list()
x = torch.arange(1 << 20, device="cuda", dtype=torch.int64)
want = (1 << 20) * ((1 << 20) - 1) // 2
assert x.sum().item() == want, "torch kernel"
got = plc.reduce.reduce(plc.Column.from_cuda_array_interface(x),
                        plc.aggregation.sum(), plc.DataType(plc.TypeId.INT64))
assert got.to_py() == want, "cuDF kernel"
# One nvCOMP per process: the wheel's, which libcudf loads at import and
# tqpcompress links.
nvcomp = {l.split()[-1] for l in open("/proc/self/maps") if "libnvcomp.so" in l}
assert len(nvcomp) == 1, nvcomp
print(f"torch and cuDF kernels ran; nvCOMP {nvcomp.pop()}")
EOF
}

case "${1:-all}" in
  lock) lock ;;
  venv) venv ;;
  modules) modules ;;
  check) check ;;
  all) venv; modules; check ;;
  *) awk 'NR > 1 { if (/^#/) { sub(/^# ?/, ""); print } else exit }' "${BASH_SOURCE[0]}" >&2; exit 2 ;;
esac
