"""One managed-arena check, in a process of its own (tests/allocator/test_managed_arena.py).

The allocator can be installed once per process, and the pytest process's
arena is device memory, so each check runs here: install the pluggable
allocator, reserve a managed arena and make it RMM's resource, as main.py
does with its device arena, then run the case named on the command line and
print one JSON line of what it measured.

    python3 tests/allocator/managed_arena_child.py oversubscribe|remake|prefetch|refusals
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from torch.cuda.memory import (CUDAPluggableAllocator,  # noqa: E402
                               change_current_allocator)

import TQPlib.tqpmemory as cmp  # noqa: E402
from utility import rmm_arena  # noqa: E402

GIB = 1024 ** 3
MIB = 1024 ** 2
ARENA_B = 8 * GIB
BLOCK_B = 4 * MIB


def install(arena: bool = True):
    change_current_allocator(CUDAPluggableAllocator(
        path_to_so_file="libcustom_allocator.so",
        alloc_fn_name="pa_cuda_malloc", free_fn_name="pa_cuda_free"))
    if not arena:
        return
    cmp.init(ARENA_B, BLOCK_B, managed=True)
    rmm_arena.install(BLOCK_B)


def filled(n: int, offset: int) -> torch.Tensor:
    t = torch.empty(n, dtype=torch.int64, device="cuda")
    torch.arange(n, out=t)
    return t.add_(offset)


def reads_back(t: torch.Tensor, offset: int) -> bool:
    """``t`` is ``filled(len(t), offset)``: its sum on the GPU and a strided
    sample on the host match."""
    n = t.numel()
    if t.sum().item() != n * (n - 1) // 2 + offset * n:
        return False
    step = max(1, n // 4096)
    sample = t[::step].cpu()
    return torch.equal(sample, torch.arange(0, n, step) + offset)


def oversubscribe():
    """3 GiB of tensors written on the GPU with about 1 GiB of the card left
    to managed pages."""
    reads_back(filled(1024, 0), 0)          # load the kernels before the balloon
    free, _ = torch.cuda.mem_get_info()
    cmp.reserve_device(free - GIB)
    budget_b = torch.cuda.mem_get_info()[0]
    n = GIB // 8
    tensors = [filled(n, i * 1000) for i in range(3)]
    torch.cuda.synchronize()
    ok = all(reads_back(t, i * 1000) for i, t in enumerate(tensors))
    return {"ok": ok, "budget_b": budget_b,
            "written_b": sum(t.nbytes for t in tensors),
            "arena_allocated_b": cmp.get_arena_stats()["allocated_b"]}


def remake():
    """A reset with a live tensor raises; with none it succeeds and the arena
    allocates again."""
    live = filled(1 << 20, 3)
    raised = None
    try:
        cmp.remake()
    except RuntimeError as err:
        raised = str(err)
    still_right = reads_back(live, 3)
    del live
    cmp.remake()
    stats = cmp.get_arena_stats()
    after = filled(1 << 20, 5)
    return {"raised": raised, "live_read_back": still_right,
            "after_reset": {k: stats[k] for k in ("capacity_b", "allocated_b")},
            "managed": cmp.is_managed(), "allocates_again": reads_back(after, 5)}


def last_prefetch(t: torch.Tensor) -> tuple[str, int]:
    """The driver's record of ``t``'s last prefetch: its location type's name
    and id."""
    from cuda.bindings import runtime as rt
    attr = rt.cudaMemRangeAttribute
    out = []
    for a in (attr.cudaMemRangeAttributeLastPrefetchLocationType,
              attr.cudaMemRangeAttributeLastPrefetchLocationId):
        err, value = rt.cudaMemRangeGetAttribute(4, a, t.data_ptr(), t.nbytes)
        assert err == rt.cudaError_t.cudaSuccess, err
        out.append(value)
    return rt.cudaMemLocationType(out[0]).name, out[1]


def prefetch():
    """A tensor prefetched to GPU 0's host NUMA node reads back right on the
    GPU."""
    from utility.numa import gpu_numa_node
    node = gpu_numa_node(0)
    t = filled(GIB // 8, 7)
    stream = torch.cuda.current_stream().cuda_stream
    torch.cuda.synchronize()
    cmp.prefetch(t.data_ptr(), t.nbytes, node, stream=stream)
    torch.cuda.synchronize()
    to_node = last_prefetch(t)
    return {"node": node, "to_node": to_node, "from_node": reads_back(t, 7)}


def refusals():
    """With no arena an allocation raises (no null pointer); a managed arena
    of 5 MiB blocks is refused and one of 4 MiB is then made; a prefetch to
    host node -1 is refused."""
    def error(call, kind):
        try:
            call()
        except kind as err:
            return str(err)
        return None

    out = {"no_arena": error(lambda: torch.empty(1024, device="cuda"),
                             RuntimeError),
           "block_5mib": error(lambda: cmp.init(ARENA_B, 5 * MIB, managed=True),
                               RuntimeError)}
    cmp.init(ARENA_B, BLOCK_B, managed=True)
    rmm_arena.install(BLOCK_B)
    t = filled(1 << 20, 9)
    out["block_4mib"] = cmp.is_managed() and reads_back(t, 9)
    out["host_node_-1"] = error(
        lambda: cmp.prefetch(t.data_ptr(), t.nbytes, -1), ValueError)
    return out


if __name__ == "__main__":
    case = sys.argv[1]
    install(arena=case != "refusals")
    print(json.dumps({"oversubscribe": oversubscribe, "remake": remake,
                      "prefetch": prefetch, "refusals": refusals}[case]()))
