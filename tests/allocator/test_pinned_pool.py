"""
Tests for the pinned host pool (IO/pages/pinned_mem.py): its NUMA binding and
its behaviour when full.

The pool binds its pages with mbind(2), which a container allows only with
CAP_SYS_NICE (docker-compose.yml grants it); with TQP_NUMA_BIND off (a box
that refuses mbind) it is pinned unbound, and the binding test skips.  The
pools here are 1 GiB, so the node capacity check passes on any host that can
run the suite.
"""
import os

import pytest
import torch

from IO.pages.pinned_mem import PinnedMemory, PinnedPoolExhausted
from utility import numa
from utility.numa import check_node_capacity, gpu_numa_node, parse_cpulist

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA (pinned host memory)"
)

GIB = 1024**3
MIB = 1024**2
BLOCK_MB = 0.5


def _numa_maps_fields(addr):
    """This process's /proc/self/numa_maps entry for the mapping at addr."""
    with open("/proc/self/numa_maps") as f:
        for line in f:
            if int(line.split()[0], 16) == addr:
                return line.split()
    raise AssertionError(f"no numa_maps entry at {addr:#x}")


def _node_pages(fields):
    """Pages per node of a numa_maps entry: {"N<k>": pages}."""
    return {k: int(v) for k, v in (f.split("=") for f in fields[1:] if "=" in f)
            if k[0] == "N" and k[1:].isdigit()}


@pytest.mark.skipif(not numa.NUMA_BIND, reason="TQP_NUMA_BIND is off")
def test_every_page_is_bound_to_the_gpu_node():
    pool = PinnedMemory(capacity_gb=1, block_size_mb=BLOCK_MB)
    node = gpu_numa_node(0)
    assert pool.numa_node == node
    fields = _numa_maps_fields(pool._segments[0].data_ptr())
    assert f"bind:{node}" in fields
    pages = _node_pages(fields)
    # The kernel merges the segment with a mapping right above it that has
    # the same binding (a pool an earlier test left mapped), so the entry
    # can hold more than this pool: every page in it is on the node, and
    # at least this pool's.
    assert list(pages) == [f"N{node}"]
    assert pages[f"N{node}"] >= GIB // 4096


def test_unbound_pages_land_where_the_pinning_thread_runs(monkeypatch):
    # TQP_NUMA_BIND off: no mbind, so a box that refuses it still pins the
    # pool, and first touch puts the pages on the node of the pinning thread
    # (main.py has pinned its threads to the GPU's node by then).
    monkeypatch.setattr(numa, "NUMA_BIND", False)
    node = gpu_numa_node(0)
    cpus = parse_cpulist(open(f"/sys/devices/system/node/node{node}/cpulist").read())
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus & before or before)
    try:
        pool = PinnedMemory(capacity_gb=1, block_size_mb=BLOCK_MB)
    finally:
        os.sched_setaffinity(0, before)
    assert not pool.numa_bound and pool.numa_node == node
    fields = _numa_maps_fields(pool._segments[0].data_ptr())
    assert not any(f.startswith("bind:") for f in fields)
    # As above, the entry can hold a neighbouring mapping too.
    if cpus & before:
        assert _node_pages(fields).get(f"N{node}", 0) >= GIB // 4096


def test_a_request_that_does_not_fit_raises():
    pool = PinnedMemory(capacity_gb=1, block_size_mb=BLOCK_MB)
    with pytest.raises(PinnedPoolExhausted, match="--pinned_pool_gb"):
        pool.get_storage(2 * GIB)


def test_try_get_storage_returns_none_when_full():
    pool = PinnedMemory(capacity_gb=1, block_size_mb=BLOCK_MB)
    assert pool.try_get_storage(GIB) is not None
    assert pool.try_get_storage(1) is None


def test_pool_storage_round_trips_through_the_gpu():
    pool = PinnedMemory(capacity_gb=1, block_size_mb=BLOCK_MB)
    _, host = pool.get_storage(4 * MIB)
    host.copy_(torch.arange(4 * MIB, dtype=torch.int64).to(torch.uint8))
    dev = host.to('cuda', non_blocking=True)
    back = torch.empty_like(host)
    back.copy_(dev, non_blocking=True)
    torch.cuda.synchronize()
    assert torch.equal(back, host)


def test_capacity_check_rejects_a_pool_the_node_cannot_hold():
    with pytest.raises(RuntimeError, match="cannot hold"):
        check_node_capacity(gpu_numa_node(0), 1 << 60)
