"""The residency counters the perf report's ``residency`` section reads
(IO/telemetry/residency.py): the pinned pool's blocks per tag and their
peaks (IO/pages/pinned_mem.py), kept only under TQP_EXPERIMENT_TELEMETRY,
the granule cache's used-slot peak (IO/cache/native_cache.py), and the
arena's peak since its last reset (``get_peak_bytes``,
memory/cuda_memory_allocator.cc).

The pinned pool here hands out block ranges and the cache CPU slots, so
only the arena's test and the report's need a GPU.
"""
import numpy as np
import pytest
import torch

from IO.pages.pinned_mem import MemoryPool

MIB = 1024**2
BLOCK = MIB // 2


class BlockPool(MemoryPool):
    """``MemoryPool`` whose storage is the allocation's block range."""

    def _storage_slice(self, start_block, actual_size_bytes):
        blocks = -(-actual_size_bytes // self.block_size_bytes)
        return range(start_block, start_block + blocks)


@pytest.fixture
def on(telemetry):
    telemetry(experiment=True)


def test_the_pinned_pool_counts_blocks_per_tag_only_under_the_switch(telemetry):
    telemetry(experiment=True)
    pool = BlockPool(capacity_gb=1, block_size_mb=0.5)
    scan, _ = pool.get_storage(3 * BLOCK, tag="scan")
    pool.reset_peaks()                       # a query's timer starts
    a, _ = pool.get_storage(4 * BLOCK)       # "intermediate", 4 blocks
    b, _ = pool.try_get_storage(BLOCK + 1)   # rounded up: 2 blocks
    pool.shrink(a, BLOCK)                    # 3 blocks back
    pool.free(b)
    c, _ = pool.get_storage(2 * BLOCK)
    u = pool.usage()
    assert u["in_use_b"] == {"scan": 3 * BLOCK, "intermediate": 3 * BLOCK}
    assert u["peak_b"] == {"scan": 3 * BLOCK, "intermediate": 6 * BLOCK}
    assert (u["in_use_total_b"], u["peak_total_b"]) == (6 * BLOCK, 9 * BLOCK)
    assert (u["capacity_b"], u["block_b"]) == (2048 * BLOCK, BLOCK)
    # free_all empties the tags and keeps the peaks; reset_peaks rebaselines.
    pool.free_all()
    u = pool.usage()
    assert u["in_use_b"] == {"scan": 0, "intermediate": 0}
    assert u["peak_b"]["intermediate"] == 6 * BLOCK
    pool.reset_peaks()
    assert pool.usage()["peak_total_b"] == 0

    # With the switch off nothing is counted, and an allocation made then is
    # not taken off the counts once the switch is on.
    telemetry(experiment=False)
    pool = BlockPool(capacity_gb=1, block_size_mb=0.5)
    a, _ = pool.get_storage(4 * BLOCK)
    c, _ = pool.get_storage(4 * BLOCK)
    pool.shrink(c, BLOCK)
    pool.free(c)
    u = pool.usage()
    assert u["in_use_b"] == u["peak_b"] == {}
    assert u["peak_total_b"] == 0
    assert pool._tags == {}
    telemetry(experiment=True)
    b, _ = pool.get_storage(2 * BLOCK)
    pool.shrink(a, BLOCK)
    pool.free(a)
    assert pool.usage()["in_use_b"] == {"intermediate": 2 * BLOCK}


def _cache(slots, slot, device="cpu"):
    """A granule cache over ``slots`` slots of ``slot`` bytes, and a region
    of ``n`` whole granules admitted into it (no home writer: nothing here
    is dirty when it leaves)."""
    pytest.importorskip("TQPlib.tqpcache")
    from IO.cache.native_cache import NativeGranuleCache
    from tests.page_cache.regions import TensorRegion
    cache = NativeGranuleCache(TensorRegion(slots, slot, device=device), "lru")

    def admitted(n):
        cr = cache.register(n * slot, 1 << 40, [0], [n * slot], [0],
                            next_use_op=0, expected_reads=1)
        assert cache.admit([cr]).admitted == [n]
        return cr
    return cache, admitted


def test_the_granule_cache_keeps_its_most_slots_in_use():
    slot = 4096
    cache, admitted = _cache(16, slot)
    a = admitted(6)
    b = admitted(3)
    cache.drop(a)
    c = admitted(2)
    assert cache.used_bytes() == 5 * slot
    assert cache.peak_used_bytes() == 9 * slot
    cache.reset_peak()
    assert cache.peak_used_bytes() == 5 * slot
    cache.drop(b)
    assert cache.used_bytes() == 2 * slot
    assert cache.peak_used_bytes() == 5 * slot
    # Dead granules leave their slots too, and the peak with them once
    # reset.
    cache.note_read(c, 0, c.nbytes)
    cache.advance_tick()
    assert cache.release_dead() == 2
    assert cache.used_bytes() == 0 == cache.resident_bytes()
    cache.clear()
    cache.reset_peak()
    assert cache.used_bytes() == 0 and cache.peak_used_bytes() == 0


def test_the_arena_peak_and_the_report_over_every_tier(cuda_allocator, on):
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    import TQPlib.tqpmemory as cmp
    from IO.telemetry import residency
    # The arena's peak is the most allocated since its last reset.
    torch.cuda.synchronize()
    cmp.reset_peak()
    base = cmp.get_arena_stats(False)["allocated_b"]
    assert cmp.get_peak_bytes() == base
    x = torch.empty(64 * MIB, dtype=torch.uint8, device="cuda:0")
    y = torch.empty(32 * MIB, dtype=torch.uint8, device="cuda:0")
    top = cmp.get_arena_stats(False)["allocated_b"]
    assert top >= base + 96 * MIB
    del x
    torch.cuda.synchronize()
    assert cmp.get_peak_bytes() == top
    cmp.reset_peak()
    assert cmp.get_peak_bytes() == cmp.get_arena_stats(False)["allocated_b"] < top
    del y
    # The report: each tier from where the timer started.
    host = BlockPool(capacity_gb=1, block_size_mb=0.5)
    host.get_storage(3 * BLOCK, tag="scan")
    cache, admitted = _cache(8, MIB, device="cuda:0")
    residency.reset()
    assert residency.report(host, None)["timed"] is False
    cmp.reset_peak()
    residency.begin(host, cache)
    host.get_storage(4 * BLOCK)
    admitted(5)
    r = residency.report(host, cache)
    assert r["timed"]
    assert r["gpu_arena"]["peak_b"] >= r["gpu_arena"]["at_start_b"] >= 8 * MIB
    assert r["host_pool"]["at_start_b"] == {"scan": 3 * BLOCK}
    assert r["host_pool"]["peak_b"] == {"scan": 3 * BLOCK, "intermediate": 4 * BLOCK}
    assert r["slot_pool"] == {"capacity_b": 8 * MIB, "slab_b": 8 * MIB, "slot_b": MIB,
                              "at_start_used_b": 0, "peak_used_b": 5 * MIB}
    residency.reset()
