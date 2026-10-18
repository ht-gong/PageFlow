#!/usr/bin/env python3
"""Container preflight for TQP-Vortex.

The engine leans on host settings that a default ``docker run`` does not
provide, and every one of them fails *late* — after a multi-minute host
pinning pass, or as a silently wrong benchmark number.  This checks them up
front.  Run from ``docker/entrypoint.sh``; set ``TQP_SKIP_PREFLIGHT=1`` to
bypass, or invoke directly (``python3 docker/preflight.py``) as a doctor.

Hard failures (exit 1):
  * no GPU visible — the whole engine is CUDA-only
  * RLIMIT_MEMLOCK not unlimited — ``main.py`` pins about 0.69 GB of host
    RAM per scale factor (``_pinned_capacity_gb``), which counts against it
  * NUMA memory-policy calls refused — ``main.py`` binds that pool to GPU 0's
    NUMA node with ``mbind(2)``, which Docker's default seccomp profile
    allows only with ``CAP_SYS_NICE``
  * GPU 0's NUMA node cannot be resolved — the same binding needs it

Warnings (exit 0): everything else, since some entry points (the telemetry
suite, figure scripts) legitimately run without data.

Stdlib only, and deliberately no ``import torch`` — this must stay fast.
"""
from __future__ import annotations

import ctypes
import glob
import json
import os
import platform
import resource
import shutil
import subprocess
import sys

_ERRORS: list[str] = []
_WARNINGS: list[str] = []


def _err(msg: str) -> None:
    _ERRORS.append(msg)


def _warn(msg: str) -> None:
    _WARNINGS.append(msg)


def _info(msg: str) -> None:
    print(f"[preflight] {msg}")


def _parse_cpulist(s: str) -> set[int]:
    """Expand a Linux cpulist like '0-63,128-191' (mirrors utility.numa.parse_cpulist)."""
    cpus: set[int] = set()
    for part in s.strip().split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus


# ── 1. GPU visibility ───────────────────────────────────────────────────

def check_gpu() -> list[str]:
    """Return the list of GPU names the container can see."""
    if shutil.which("nvidia-smi") is None:
        _err("nvidia-smi is not on PATH — the container has no GPU runtime.\n"
             "         Run with `--gpus all` (or the compose `deploy.resources`\n"
             "         devices block); main.py shells out to nvidia-smi for pool\n"
             "         capacity, SM clocks and the NUMA node of GPU 0.")
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
        _err(f"nvidia-smi failed ({e}) — no usable GPU in this container.")
        return []

    gpus = [line.strip() for line in out.splitlines() if line.strip()]
    if not gpus:
        _err("nvidia-smi reports no GPUs.  Check the device reservation.")
        return gpus

    for i, g in enumerate(gpus):
        _info(f"GPU {i}: {g}")
    if len(gpus) > 1:
        _warn(f"{len(gpus)} GPUs are visible.  main.py addresses GPU 0 only "
              "(nvidia-smi -i 0 for capacity/clocks/NUMA, torch device 0), and "
              "AGENTS.md rule 1 forbids concurrent runs.  Expose exactly one "
              "device (TQP_GPU=<id>) so the two indices cannot diverge.")
    return gpus


# ── 2. Locked memory ────────────────────────────────────────────────────

def check_memlock() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY:
        _info("RLIMIT_MEMLOCK: unlimited")
        return
    _err(f"RLIMIT_MEMLOCK is {soft // (1024 ** 2)} MiB (hard "
         f"{'unlimited' if hard == resource.RLIM_INFINITY else hard // (1024 ** 2)}"
         " MiB), not unlimited.\n"
         "         main.py pins about 0.69 GB of host RAM per scale "
         "factor (_pinned_capacity_gb),\n"
         "         which counts against this limit.  Run with "
         "`--ulimit memlock=-1`\n"
         "         (compose: `ulimits: {memlock: -1}`).")


# ── 3. NUMA memory policy ───────────────────────────────────────────────

# get_mempolicy(2) sits in the same seccomp group as mbind(2), and has no side
# effects, so it stands in for the call main.py needs.
_SYS_GET_MEMPOLICY = {"x86_64": 239, "aarch64": 236}


def check_mempolicy() -> None:
    nr = _SYS_GET_MEMPOLICY.get(platform.machine())
    if nr is None:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    mode = ctypes.c_int()
    if libc.syscall(nr, ctypes.byref(mode), None, 0, None, 0) == 0:
        _info("NUMA memory policy calls: allowed")
        return
    err = ctypes.get_errno()
    _err(f"get_mempolicy(2) fails: {os.strerror(err)}.\n"
         "         main.py binds the pinned host pool to GPU 0's NUMA node "
         "with mbind(2),\n"
         "         which Docker's default seccomp profile allows only with "
         "CAP_SYS_NICE.\n"
         "         Run with `--cap-add SYS_NICE` (compose: "
         "`cap_add: [SYS_NICE]`).")


# ── 4. The GPU node's RAM vs the scale-factor pinning budget ────────────

# Mirror main._PINNED_NEED_PER_SF, main._PINNED_HEADROOM_PCT and
# utility.numa.NUMA_BIND_HEADROOM_B; kept local so preflight stays import-free.
_PINNED_NEED_PER_SF = 6524               # 1e-4 GiB per SF
_PINNED_HEADROOM_PCT = 105
_NUMA_BIND_HEADROOM_GIB = 16


def check_host_ram(have_gpu: bool) -> None:
    # The pinned pool is bound to GPU 0's NUMA node, so it must fit there.
    where, path = "host RAM", "/proc/meminfo"
    if have_gpu:
        try:
            node = _gpu0_numa_node()
            where = f"GPU 0's NUMA node {node}"
            path = f"/sys/devices/system/node/node{node}/meminfo"
        except Exception:  # fall back to the whole host
            pass
    try:
        with open(path, encoding="utf-8") as f:
            # '/proc/meminfo: MemTotal: N kB'; node meminfo: 'Node 1 MemTotal: N kB'
            total_kb = next(int(l.split()[l.split().index("MemTotal:") + 1])
                            for l in f if "MemTotal:" in l)
    except (OSError, StopIteration, ValueError):
        return
    total_gb = total_kb / (1024 ** 2)
    # Largest SF whose default pool (ceil(1.05 x need x SF)) fits, with
    # main.py's bind headroom, if nothing else held memory on the node.
    ceiling = int((total_gb - _NUMA_BIND_HEADROOM_GIB) * 1_000_000
                  // (_PINNED_NEED_PER_SF * _PINNED_HEADROOM_PCT))
    if ceiling < 1:
        _warn(f"{where} has {total_gb:.0f} GB; even SF1 wants a 1 GB pinned "
              f"pool plus {_NUMA_BIND_HEADROOM_GIB} GB of headroom.  Expect "
              "the host pinning pass to fail.")
    else:
        _info(f"{where} has {total_gb:.0f} GB -> the default pinned pool "
              f"fits up to SF{ceiling}")


# ── 5. GPU 0's NUMA node and CPU affinity ───────────────────────────────

def _gpu0_numa_node() -> int:
    """GPU 0's NUMA node, resolved as utility.numa.gpu_numa_node does: sysfs
    reports -1 for a device with no NUMA affinity, accepted only on a host
    with a single node."""
    bdf = subprocess.run(
        ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader",
         "-i", "0"], capture_output=True, text=True, timeout=30,
        check=True).stdout.strip().lower()
    dom, rest = bdf.split(":", 1)
    sysfs = f"/sys/bus/pci/devices/{dom[-4:]}:{rest}"
    node = int(open(f"{sysfs}/numa_node", encoding="utf-8").read().strip())
    if node >= 0:
        return node
    if len(glob.glob("/sys/devices/system/node/node[0-9]*")) == 1:
        return 0
    raise RuntimeError("GPU 0 reports no NUMA node (-1) on a multi-node host")


def check_numa(have_gpu: bool) -> None:
    if have_gpu:
        try:
            node = _gpu0_numa_node()
            node_cpus = _parse_cpulist(
                open(f"/sys/devices/system/node/node{node}/cpulist",
                     encoding="utf-8").read())
        except Exception as e:
            _err(f"cannot resolve GPU 0's NUMA node ({e}).\n"
                 "         main.py binds the pinned host pool to that node "
                 "and refuses to start\n"
                 "         without it.")
            return
    affinity = os.sched_getaffinity(0)
    online = os.cpu_count() or len(affinity)
    if len(affinity) < online:
        _warn(f"the container can use {len(affinity)} of {online} CPUs.  "
              "main.py pins its threads to GPU 0's NUMA node after warmup "
              "(AGENTS.md rule 2), intersected with this set, so a narrowed "
              "cpuset silently changes which cores a timed run lands on.  "
              "Drop --cpuset-cpus for benchmark runs.")
        return
    if not have_gpu:
        return
    missing = node_cpus - affinity
    if missing:
        _warn(f"GPU 0 is on NUMA node {node} ({len(node_cpus)} CPUs) but "
              f"{len(missing)} of them are outside the container's affinity "
              "mask; main.py cannot pin its threads to the full node.")
    else:
        _info(f"GPU 0 on NUMA node {node}; all {len(node_cpus)} local CPUs "
              "reachable")


# ── 6. The datasets ─────────────────────────────────────────────────────

def check_data() -> None:
    path = os.environ.get("TQP_TABLES_PATH", "")
    if not path:
        _info("TQP_TABLES_PATH unset; falling back to config.json")
        return
    if not os.path.isdir(path):
        _warn(f"TQP_TABLES_PATH={path} does not exist.  Make a dataset with "
              "`make data SF=<n>` (or point TQP_DATA_DIR at an existing data "
              "root).")
        return
    sfs = sorted((int(os.path.basename(os.path.dirname(p))[2:])
                  for p in glob.glob(os.path.join(path, "sf*", "parquet"))
                  if os.path.basename(os.path.dirname(p))[2:].isdigit()))
    if not sfs:
        _warn(f"{path} holds no sf<n>/parquet dataset.  Make one with "
              "`make data SF=<n>`.")
        return
    _info(f"datasets at {path}: SF{', SF'.join(map(str, sfs))}")


# ── 7. The codec map ────────────────────────────────────────────────────

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODEC_MAP = os.path.join(_REPO, "profiling", "results", "codec_map.json")
_TYPES = ("int", "float", "date", "string")


def check_codec_map() -> None:
    """Info line: the compact contents of profiling/results/codec_map.json
    (the class table + chunk per codec compression/format_policy.py will
    load), or `no codec map`.  Stdlib JSON only; the strict validation is
    format_policy's job, so a malformed file is only a warning here."""
    rel = os.path.relpath(_CODEC_MAP, _REPO)
    if not os.path.exists(_CODEC_MAP):
        _info(f"no codec map at {rel}: the engine uses its built-in defaults "
              "(run `make profiling TQP_GPU=<idle>` to sweep one)")
        return
    try:
        with open(_CODEC_MAP, encoding="utf-8") as f:
            doc = json.load(f)
        cells = " ".join(
            f"{t}:{doc['codec'][t]['unsorted']}/{doc['codec'][t]['sorted']}"
            for t in _TYPES)
        chunks = " ".join(f"{c}={v['chunk_b']}"
                          for c, v in sorted(doc["settings"].items()))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        _warn(f"{rel} is unreadable ({e}); main.py will refuse to start "
              "until it is fixed or removed.")
        return
    _info(f"codec map {rel}: codec(unsorted/sorted) {cells} | chunk_b {chunks}")


# ── 8. Optional tooling ─────────────────────────────────────────────────

def check_optional() -> None:
    if shutil.which("nsys") is None:
        _warn("nsys is not installed; `main.py --cuda_profile` will not work.  "
              "Rebuild with --build-arg WITH_NSYS=1 on a base image whose CUDA "
              "apt repo carries nsight-systems-cli.")
    if not os.environ.get("JAVA_HOME"):
        _warn("JAVA_HOME is unset; parsing/layer.py (pyspark) needs a JDK.")


def main() -> int:
    gpus = check_gpu()
    check_memlock()
    check_mempolicy()
    check_host_ram(bool(gpus))
    check_numa(bool(gpus))
    check_data()
    check_codec_map()
    check_optional()

    sys.stdout.flush()  # keep the info lines above the warnings
    for w in _WARNINGS:
        print(f"[preflight] WARNING: {w}", file=sys.stderr)
    for e in _ERRORS:
        print(f"[preflight] ERROR: {e}", file=sys.stderr)

    if _ERRORS:
        print("[preflight] refusing to start; set TQP_SKIP_PREFLIGHT=1 to "
              "override.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
