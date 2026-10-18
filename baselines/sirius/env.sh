#!/bin/bash
# Source this for the Sirius engine environment, shared by the TPC-H runner
# (run_sirius.sh) and the format probes (ablation/): the Sirius duckdb binary
# ($SIRIUS_DUCKDB) and its pixi env's lib dir ($SIRIUS_ENVLIB), both set by the
# image, its gpu_execution config with the IO backend resolved, and the spill
# dir. Sets SIRIUS_CONFIG_FILE, SIRIUS_IO (uring|kvikio), SIRIUS_SPILL.
_SI_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SIRIUS_DUCKDB:?SIRIUS_DUCKDB is not set}" "${SIRIUS_ENVLIB:?SIRIUS_ENVLIB is not set}"
export SIRIUS_DUCKDB SIRIUS_ENVLIB
export SIRIUS_CONFIG_FILE="${SIRIUS_CONFIG_FILE:-${_SI_DIR}/sirius.yaml}"
SIRIUS_SPILL="/dev/shm/_sirius_spill"   # matches downgrade_root_dirs in sirius.yaml
# kvikio compat (POSIX) mode for anything still routed through kvikio: cuFile/GDS
# needs nvidia-fs, which these boxes lack, and kvikio's auto-detection has been
# seen to abort instead of falling back (polars hit it). Harmless for the
# io_uring datasource, which does not use kvikio.
export KVIKIO_COMPAT_MODE="${KVIKIO_COMPAT_MODE:-ON}"

# IO backend: SIRIUS_IO=auto (default) probes io_uring_setup and uses Sirius'
# native io_uring datasource when the kernel/container allows it (sirius.yaml as
# committed); otherwise -- or with SIRIUS_IO=kvikio -- the config is rewritten
# to the kvikio POSIX fallback (use_sirius_datasource:false), which is what the
# old seccomp-restricted Vast container needed.
SIRIUS_IO="${SIRIUS_IO:-auto}"
_sirius_io_uring_ok(){
  [ "$(cat /proc/sys/kernel/io_uring_disabled 2>/dev/null || echo 0)" != 2 ] || return 1
  python3 - <<'PY' >/dev/null 2>&1
import ctypes, os, sys
libc = ctypes.CDLL(None, use_errno=True)
params = ctypes.create_string_buffer(120)           # struct io_uring_params
fd = libc.syscall(425, 8, params)                    # __NR_io_uring_setup (x86_64)
sys.exit(0 if fd >= 0 else 1)
PY
}
if [ "${SIRIUS_IO}" = auto ]; then
  if _sirius_io_uring_ok; then SIRIUS_IO=uring; else SIRIUS_IO=kvikio; fi
fi
if [ "${SIRIUS_IO}" = kvikio ]; then
  _sirius_cfg="/tmp/sirius_config_kvikio_$$.yaml"
  sed 's/use_sirius_datasource: *true/use_sirius_datasource: false/' "${SIRIUS_CONFIG_FILE}" > "${_sirius_cfg}"
  export SIRIUS_CONFIG_FILE="${_sirius_cfg}"
  trap 'rm -f "${_sirius_cfg}"' EXIT
fi
export SIRIUS_IO
mkdir -p "${SIRIUS_SPILL}"
