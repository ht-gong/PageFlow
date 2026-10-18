"""
Tests for the pinned host pool's free list (``MemoryPool`` in
IO/pages/pinned_mem.py): first fit, and when freed runs are merged.

A free appends its run to the list, and the list is merged (sorted by start
block, touching runs joined) once every six frees, so up to five freed runs
can sit unmerged beside their neighbours.  A request that misses merges them
and looks once more before it reports no fit.

The pool here hands out block ranges instead of pinned memory, so these tests
need no GPU.
"""
import random

import pytest

from IO.pages.pinned_mem import MemoryPool, PinnedPoolExhausted

MIB = 1024**2
BLOCK_MB = 0.5
BLOCK = int(BLOCK_MB * MIB)


class BlockPool(MemoryPool):
    """``MemoryPool`` whose storage is the allocation's block range."""

    def _storage_slice(self, start_block, actual_size_bytes):
        blocks = -(-actual_size_bytes // self.block_size_bytes)
        return range(start_block, start_block + blocks)


def count_coalesces(pool):
    """Wrap ``pool.coalesce``; return the list each call appends to."""
    calls = []
    merge = pool.coalesce

    def counted():
        calls.append(None)
        merge()

    pool.coalesce = counted
    return calls


def fill(pool):
    """Take every block of ``pool``, one per allocation; return the pointers."""
    return [pool.get_storage(BLOCK)[0] for _ in range(pool._num_blocks)]


def test_ten_thousand_frees_merge_at_most_once_per_six_frees():
    pool = BlockPool(capacity_gb=1, block_size_mb=BLOCK_MB)     # 2048 blocks
    coalesces = count_coalesces(pool)
    rng = random.Random(7)
    live, used, frees = {}, set(), 0
    # At most 100 live allocations of at most 8 blocks: some free run always
    # holds 8 blocks, so no request misses and every merge comes from frees.
    while frees < 10_000:
        if live and (len(live) == 100 or rng.random() < 0.5):
            ptr = rng.choice(list(live))
            pool.free(ptr)
            used.difference_update(range(ptr, ptr + live.pop(ptr)))
            frees += 1
        else:
            blocks = rng.randint(1, 8)
            ptr, storage = pool.get_storage(blocks * BLOCK)
            assert storage == range(ptr, ptr + blocks)
            assert used.isdisjoint(storage)
            used.update(storage)
            live[ptr] = blocks
    assert len(coalesces) <= frees // 6
    for ptr in list(live):
        pool.free(ptr)
    assert pool.free_space() == pool.pool_size_bytes
    assert pool._free == [(0, pool._num_blocks)]


def test_free_all_restarts_the_count():
    pool = BlockPool(capacity_gb=1, block_size_mb=BLOCK_MB)
    coalesces = count_coalesces(pool)
    for _ in range(2):
        for ptr in [pool.get_storage(BLOCK)[0] for _ in range(5)]:
            pool.free(ptr)
        pool.free_all()
    assert coalesces == []


def test_a_miss_that_fits_after_merging_succeeds():
    pool = BlockPool(capacity_gb=1, block_size_mb=BLOCK_MB)
    ptrs = fill(pool)
    coalesces = count_coalesces(pool)
    for ptr in ptrs[10:13]:
        pool.free(ptr)                  # three touching one-block runs
    assert coalesces == []              # fewer than six frees: not merged
    assert pool.get_storage(3 * BLOCK) == (10, range(10, 13))
    assert len(coalesces) == 1
    # Full, with nothing freed since the merge: the miss is final.
    assert pool.try_get_storage(BLOCK) is None
    assert len(coalesces) == 1


def test_a_miss_on_a_full_pool_with_a_free_pending_returns_none():
    pool = BlockPool(capacity_gb=1, block_size_mb=BLOCK_MB)
    ptrs = fill(pool)
    pool.free(ptrs[5])
    assert pool.get_storage(BLOCK)[0] == ptrs[5]      # the list is empty again
    assert pool.try_get_storage(BLOCK) is None
    with pytest.raises(PinnedPoolExhausted, match="largest free run 0.000 GiB"):
        pool.get_storage(BLOCK)


def test_free_space_counts_touching_runs_as_one_region():
    pool = BlockPool(capacity_gb=1, block_size_mb=BLOCK_MB)
    ptrs = fill(pool)
    for ptr in ptrs[10:13] + ptrs[20:21]:
        pool.free(ptr)
    assert pool.free_space() == 3 * BLOCK


def test_shrink_returns_the_tail_and_keeps_the_head():
    pool = BlockPool(capacity_gb=8 * BLOCK / 1024**3, block_size_mb=BLOCK_MB)
    ptr, blocks = pool.get_storage(5 * BLOCK)
    pool.shrink(ptr, 2 * BLOCK - 1)            # two blocks kept
    assert pool._allocs[ptr] == 2
    # The freed tail joins the free run after it: the six blocks after the
    # head are one run again, and a full free of the shrunk allocation
    # returns only its head.
    nxt, got = pool.get_storage(6 * BLOCK)
    assert nxt == ptr + 2
    pool.free(ptr)
    pool.free(nxt)
    assert not pool._allocs
    assert pool.free_space() == 8 * BLOCK
    with pytest.raises(ValueError):
        pool.shrink(12345, BLOCK)
    # Shrinking to more than held, or to nothing, keeps at least a block.
    p2, _ = pool.get_storage(BLOCK)
    pool.shrink(p2, 3 * BLOCK)
    pool.shrink(p2, 0)
    assert pool._allocs[p2] == 1
