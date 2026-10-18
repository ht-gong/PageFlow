#!/bin/bash
# `make doctor`: is this container ready to produce comparable numbers?
# Checks the installed versions against the pins, the GPU, io_uring (Sirius
# silently falls back to kvikio without it) and free space. Exits non-zero if a
# check fails.
set -uo pipefail
PINS=/opt/envs/pins
fail=0
ok(){ printf '  %-10s ok    %s\n' "$1" "$2"; }
bad(){ printf '  %-10s FAIL  %s\n' "$1" "$2"; fail=1; }
warn(){ printf '  %-10s warn  %s\n' "$1" "$2"; }
pin(){ sed -n "s/^$1=//p" "${PINS}/versions.env"; }

echo "versions"
for env in py polars; do
  py="$([ "${env}" = py ] && echo "${PY}" || echo "${POLARS_PY}")"
  want="$(grep -E '^[A-Za-z0-9_.-]+==' "${PINS}/${env}.lock" | sed 's/ .*//' | tr 'A-Z_' 'a-z-' | sort)"
  have="$("${py}" -m pip freeze 2>/dev/null | tr 'A-Z_' 'a-z-' | sort)"
  if [ "${want}" = "${have}" ]; then ok "${env}" "$(echo "${want}" | wc -l) packages match ${env}.lock"
  else bad "${env}" "differs from ${env}.lock: $(diff <(echo "${want}") <(echo "${have}") | grep '^[<>]' | tr '\n' ' ')"; fi
done
jv="$("${JAVA_HOME}/bin/java" -version 2>&1 | head -1)"
[[ "${jv}" == *17.0.20.1* ]] && ok java "${jv}" || bad java "${jv}"
sha="$(sha1sum "${RAPIDS_JAR}" | cut -c1-40)"
[ "${sha}" = "$(pin RAPIDS_JAR_SHA1)" ] && ok rapids "$(basename "${RAPIDS_JAR}")" || bad rapids "jar sha1 ${sha}"
sv="$(SIRIUS_DISABLE=1 "${SIRIUS_DUCKDB}" -noheader -list -c 'select version()' 2>/dev/null | grep -v -e mbind -e '^$' | tail -1)"
[ -n "${sv}" ] && ok sirius "duckdb ${sv}, sirius $(pin SIRIUS_REF | cut -c1-8)" || bad sirius "binary does not start"
nv="$("${NDSGEN}" --version 2>/dev/null)"
[ -n "${nv}" ] && ok ndsgen "${nv}" || bad ndsgen "missing (${NDSGEN})"

echo "gpu"
if ! command -v nvidia-smi >/dev/null; then bad gpu "no GPU in the container (set GPU in local.env at the TQP-Vortex root)"
else
  while IFS=, read -r idx name mode util mem; do
    [ "$(echo ${mode})" = Default ] || bad gpu "${name} is in ${mode} compute mode (Sirius fails at startup)"
    # an idle H100 shows a few MiB used by the driver
    if [ "${util// /}" != 0 ] || [ "${mem// /}" -gt 64 ]; then
      warn gpu "${name}: ${util// /}% busy, ${mem// /} MiB used by others -> timing noise"
    else ok gpu "${name} idle, $(echo ${mode}) mode"; fi
  done < <(nvidia-smi --query-gpu=index,name,compute_mode,utilization.gpu,memory.used \
             --format=csv,noheader,nounits)
fi

echo "numa"
cpus="$(sed -n 's/^Cpus_allowed_list:\s*//p' /proc/self/status)"
mems="$(sed -n 's/^Mems_allowed_list:\s*//p' /proc/self/status)"
bus="$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i 0 2>/dev/null)"
node="$(cat "/sys/bus/pci/devices/$(echo "${bus: -12}" | tr A-Z a-z)/numa_node" 2>/dev/null || echo -1)"
if [ -z "${bus}" ] || [ "${node}" -lt 0 ]; then warn numa "GPU NUMA node unknown; CPUs ${cpus}, memory nodes ${mems}"
elif [ "${NUMA:-auto}" = off ]; then warn numa "unbound (NUMA=off): CPUs ${cpus}, memory nodes ${mems}; the GPU is on node ${node}"
elif [ "${mems}" = "${node}" ] && [ "${cpus}" = "$(cat "/sys/devices/system/node/node${node}/cpulist")" ]; then
  ok numa "bound to the GPU's node ${node}: CPUs ${cpus}, memory node ${mems}"
else bad numa "the GPU is on node ${node}, but the container has CPUs ${cpus}, memory nodes ${mems}"; fi
if python3 - "$((node < 0 ? 0 : node))" <<'PY'
import ctypes, sys
libc = ctypes.CDLL(None, use_errno=True)
mask = ctypes.c_ulong(1 << int(sys.argv[1]))
sys.exit(0 if libc.syscall(238, 2, ctypes.byref(mask), 64) == 0 else 1)   # set_mempolicy(MPOL_BIND)
PY
then ok mbind "allowed: NUMA-aware engines (Sirius) can bind their pinned memory"
else bad mbind "blocked: Sirius cannot bind its pinned pool to the GPU's node (needs --cap-add SYS_NICE)"; fi

echo "io"
if python3 - <<'PY'
import ctypes, sys
libc = ctypes.CDLL(None, use_errno=True)
sys.exit(0 if libc.syscall(425, 8, ctypes.create_string_buffer(120)) >= 0 else 1)   # io_uring_setup
PY
then ok io_uring "allowed (Sirius uses its native io_uring datasource)"
else bad io_uring "blocked: Sirius would fall back to kvikio (check the seccomp profile)"; fi
[ "$(ulimit -l)" = unlimited ] && ok memlock unlimited || bad memlock "$(ulimit -l) KiB"

echo "space"
for d in / "${DATA_DIR}" "${SCRATCH}" "${SHM:-/dev/shm}"; do
  printf '  %-10s %s free  (%s)\n' "" "$(df -h --output=avail "${d}" | tail -1 | tr -d ' ')" "${d}"
done
exit "${fail}"
