"""NUMA placement: pin threads to a GPU's node, and check a node's memory.

:func:`pin_to_gpu_node` is shared by ``main.py`` (after warmup) and
``profiling/sweep.py`` so the offline codec sweep is timed under exactly the
pinning a timed query run uses.  ``docker/preflight.py`` carries a
stdlib-only mirror of :func:`parse_cpulist` because it must not import torch.

:func:`check_node_capacity` guards the pinned host pool
(``IO/pages/pinned_mem.py``), whose pages are all bound to the GPU's node
unless ``TQP_NUMA_BIND`` is off (:data:`NUMA_BIND`).

The GPU is CUDA's device ``gpu_index`` (main.py's GPU 0): nvidia-smi is asked
about it by PCI bus id (:func:`nvidia_smi_id`), since nvidia-smi numbers every
GPU of the box whatever ``CUDA_VISIBLE_DEVICES`` says.
"""
from __future__ import annotations

import os
import subprocess

# Room left on a node after binding the pinned pool to it: the kernel's
# watermarks plus whatever else on the node cannot be reclaimed quickly.
NUMA_BIND_HEADROOM_B = 16 * 1024**3

# TQP_NUMA_BIND, read once at import: bind the pinned host pool's pages to the
# GPU's node with mbind(2) (the default).  Off only for ``0``, ``false``,
# ``False`` or empty, for a box that refuses mbind: an unprivileged container
# without CAP_SYS_NICE gets EPERM (bench/run_tqp.sh's native mode).  The pool
# is then pinned unbound, so each page lands on the node of the thread that
# first touches it, as cudaHostRegister pins it: the real run's pool is made
# after main.py has pinned every thread to the GPU's node.
NUMA_BIND = os.environ.get('TQP_NUMA_BIND') not in ('0', 'false', 'False', '')


def parse_cpulist(s: str) -> set[int]:
    """Expand a Linux cpulist like ``'0-63,128-191'`` into a set of ints."""
    cpus: set[int] = set()
    for part in s.strip().split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus


def nvidia_smi_id(cuda_index: int = 0) -> str:
    """The ``nvidia-smi -i`` id of CUDA device ``cuda_index``: its PCI bus id.

    CUDA numbers the devices ``CUDA_VISIBLE_DEVICES`` leaves visible and
    nvidia-smi every GPU of the box, so with ``CUDA_VISIBLE_DEVICES=1`` CUDA's
    device 0 is nvidia-smi's GPU 1.  The CUDA driver names the device's bus id
    without creating a context, so this is safe before main.py swaps torch's
    allocator.  Without a CUDA driver (a CPU-only box) the index is returned
    as it is; a driver that finds no such device raises RuntimeError.
    """
    import ctypes
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
    except OSError:
        return str(cuda_index)
    dev = ctypes.c_int()
    bus = ctypes.create_string_buffer(32)
    err = (cuda.cuInit(0) or cuda.cuDeviceGet(ctypes.byref(dev), cuda_index)
           or cuda.cuDeviceGetPCIBusId(bus, len(bus), dev))
    if err:
        raise RuntimeError(f"CUDA device {cuda_index}: CUresult {err}")
    return bus.value.decode()          # '0000:81:00.0'


def gpu_numa_node(gpu_index: int = 0) -> int:
    """NUMA node CUDA device <gpu_index> hangs off, via sysfs (numactl-free).

    Raises RuntimeError when the node cannot be resolved: the pinned host pool
    is bound to this node, so a guess would put it on the wrong socket.  sysfs
    reports -1 for a device with no NUMA affinity, which is unambiguous only
    on a host with a single node.
    """
    try:
        bdf = nvidia_smi_id(gpu_index)
        if ':' not in bdf:                     # no CUDA driver: ask nvidia-smi
            bdf = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=pci.bus_id', '--format=csv,noheader',
                 '-i', bdf]).decode().strip()
        dom, rest = bdf.lower().split(':', 1)  # '00000000' or '0000', '01:00.0'
        bdf = dom[-4:] + ':' + rest            # sysfs form '0000:01:00.0'
        with open(f'/sys/bus/pci/devices/{bdf}/numa_node') as f:
            node = int(f.read().strip())
    except Exception as e:
        raise RuntimeError(
            f"cannot resolve GPU {gpu_index}'s NUMA node: {e}") from e
    if node >= 0:
        return node
    n_nodes = len([d for d in os.listdir('/sys/devices/system/node')
                   if d.startswith('node') and d[4:].isdigit()])
    if n_nodes == 1:
        return 0
    raise RuntimeError(
        f"GPU {gpu_index} ({bdf}) reports no NUMA node (-1) on a host with "
        f"{n_nodes} nodes")


def pin_to_gpu_node(gpu_index: int = 0):
    """Pin EVERY current thread to the CPUs local to GPU<gpu_index>'s NUMA node.

    On a dual-socket host, unpinned threads float across both sockets, so the
    host work that feeds the GPU (copy launches, pageable memory, the CPU side
    of every operator) crosses the inter-socket link nondeterministically.
    Discover the GPU's node rather than assuming the PCIe attachment is the
    same on every machine.  The pinned pool does not depend on this pin when
    its pages are bound to the node by address range (:data:`NUMA_BIND`);
    with TQP_NUMA_BIND off they land where these threads first touch them.

    IMPORTANT — call this AFTER CUDA/allocator/torch init (i.e. after warmup):
    that init resets the process affinity to all CPUs exactly ONCE during startup,
    so an early pin (before init) is silently wiped.  The reset does NOT recur, so
    a single pass here holds for the whole run; threads spawned later inherit it.
    Because warmup already spawned pools (now widened), we re-pin all existing
    /proc/self/task threads, not just the caller.  A host NUMA daemon (numad)
    can still rebind the threads later; nothing here re-asserts the pin.
    Returns (node, first_cpu, n_cpus) on success, or (None, err, 0) if the
    pin itself fails; an unresolvable node raises (:func:`gpu_numa_node`).
    """
    import torch
    node = gpu_numa_node(gpu_index)
    try:
        cpus = parse_cpulist(
            open(f'/sys/devices/system/node/node{node}/cpulist').read())
        for tid in os.listdir('/proc/self/task'):   # one-shot, every thread
            try:
                os.sched_setaffinity(int(tid), cpus)
            except (OSError, ValueError):
                pass                                 # thread may have exited
        # Keep PyTorch's intra-op thread pool within one socket too.  Respect
        # the benchmark's explicit OMP thread control: some hosts expose fewer
        # physical cores than the historical 32-thread default, and forcing 32
        # workers there adds avoidable SMT scheduling variance.
        try:
            requested_threads = int(os.environ.get('OMP_NUM_THREADS', '32'))
        except ValueError:
            requested_threads = 32
        torch.set_num_threads(max(1, min(requested_threads, len(cpus))))
        os.environ.setdefault('OMP_PROC_BIND', 'close')
        os.environ.setdefault('OMP_PLACES', 'cores')
        return node, min(cpus), len(cpus)
    except Exception as e:
        return None, str(e), 0


def node_meminfo(node: int) -> dict[str, int]:
    """``/sys/devices/system/node/node<node>/meminfo`` in bytes, by field."""
    out = {}
    with open(f'/sys/devices/system/node/node{node}/meminfo') as f:
        for line in f:
            # 'Node 1 MemFree:        77673648 kB'
            parts = line.split()
            if len(parts) >= 4 and parts[2].endswith(':'):
                out[parts[2][:-1]] = int(parts[3]) * 1024
    return out


def node_reclaimable_b(node: int,
                       headroom_b: int = NUMA_BIND_HEADROOM_B) -> int:
    """What ``node`` can hand over to a bound allocation without an OOM kill:
    its free pages plus its file-backed pages (what reclaim can take), less
    ``headroom_b``.  Negative when the node holds less than the headroom."""
    mi = node_meminfo(node)
    return (mi['MemFree'] + mi.get('Active(file)', 0)
            + mi.get('Inactive(file)', 0) - headroom_b)


def check_node_capacity(node: int, nbytes: int,
                        headroom_b: int = NUMA_BIND_HEADROOM_B) -> int:
    """Raise unless ``node`` can take ``nbytes`` more, bound, without an OOM kill.

    A bound allocation cannot spill to another node: the kernel reclaims the
    node's page cache and, past that, runs the OOM killer, which on a shared
    host may pick another tenant's process.  Free pages plus file-backed pages
    are what reclaim can hand over (``node_reclaimable_b``).  Returns that
    amount in bytes.
    """
    avail = node_reclaimable_b(node, headroom_b=0)
    if nbytes + headroom_b > avail:
        mi = node_meminfo(node)
        gib = 1024**3
        raise RuntimeError(
            f"NUMA node {node} cannot hold a {nbytes / gib:.0f} GiB pinned pool: "
            f"free {mi['MemFree'] / gib:.0f} GiB + page cache "
            f"{(avail - mi['MemFree']) / gib:.0f} GiB - {headroom_b / gib:.0f} GiB "
            f"headroom = {(avail - headroom_b) / gib:.0f} GiB.  Use a smaller "
            f"--pinned_pool_gb, or free memory on that node.")
    return avail

