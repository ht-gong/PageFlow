"""Slot-pool block elasticity of the reference model's pool.

Covers :class:`tests.page_cache.reference_slot_pool.SlotRunPool`'s
block-backed elasticity surface: block geometry, slot addresses, live
counts, releasing and restoring a block, and ``free_all`` with a block on
loan.

CPU-only: pools use ``device='cpu'`` — the bookkeeping under test is
device-agnostic (only ``torch.empty`` sees the device).
"""

import pytest

from tests.page_cache.reference_slot_pool import SlotRunPool

SLOT = 1024


def make_pool(num_slots: int = 10, block_slots: int = 4,
              slot_size: int = SLOT) -> SlotRunPool:
    return SlotRunPool(
        capacity_b=num_slots * slot_size,
        slot_size_b=slot_size,
        device='cpu',
        block_slots=block_slots,
    )


def take(pool: SlotRunPool):
    """One free slot (``None`` when none is left)."""
    grant = pool.acquire_runs(1)
    return None if grant is None else grant.runs[0][0]


def drain(pool: SlotRunPool) -> list:
    """Acquire until exhaustion; return every slot handed out."""
    out = []
    while (slot := take(pool)) is not None:
        out.append(slot)
    return out


# --------------------------------------------------------------------------- #
# Block geometry
# --------------------------------------------------------------------------- #

def test_block_geometry_exact_division():
    pool = make_pool(num_slots=8, block_slots=4)
    assert pool.num_blocks == 2
    assert [pool._slots_in_block(b) for b in range(2)] == [4, 4]
    assert [pool.block_nbytes(b) for b in range(2)] == [4 * SLOT, 4 * SLOT]
    assert sum(t.numel() for t in pool._blocks) == 8 * SLOT
    assert pool.free_count() == 8


def test_block_geometry_last_block_short():
    pool = make_pool(num_slots=10, block_slots=4)
    assert pool.num_blocks == 3
    assert [pool._slots_in_block(b) for b in range(3)] == [4, 4, 2]
    assert pool.block_nbytes(2) == 2 * SLOT
    assert pool._blocks[2].numel() == 2 * SLOT
    assert pool.pool_size_bytes == 10 * SLOT
    assert pool.free_count() == 10


def test_single_block_default():
    pool = SlotRunPool(capacity_b=8 * SLOT, slot_size_b=SLOT, device='cpu')
    assert pool.num_blocks == 1
    assert pool.block_slots == pool.num_slots == 8


def test_slot_ptr_is_block_base_plus_offset_across_blocks():
    pool = make_pool(num_slots=10, block_slots=4)
    slots = drain(pool)
    assert slots == list(range(10))
    want = [pool._blocks[s // 4].data_ptr() + (s % 4) * SLOT for s in slots]
    assert pool.slot_ptrs(slots).tolist() == want


def test_block_live_counts_exact_under_acquire_release():
    pool = make_pool(num_slots=10, block_slots=4)
    slots = [take(pool) for _ in range(5)]            # first fit: 0..4
    assert slots == [0, 1, 2, 3, 4]
    assert [pool.block_live_count(b) for b in range(3)] == [4, 1, 0]
    pool.release_slots([slots[4]])                     # block 1 -> empty
    assert [pool.block_live_count(b) for b in range(3)] == [4, 0, 0]
    pool.release_slots([slots[0]])
    assert pool.block_live_count(0) == 3
    pool.release_slots(slots[1:4])
    assert [pool.block_live_count(b) for b in range(3)] == [0, 0, 0]
    assert pool.empty_block_indices() == [0, 1, 2]


# --------------------------------------------------------------------------- #
# release_block
# --------------------------------------------------------------------------- #

def test_release_block_with_live_slot_raises():
    pool = make_pool(num_slots=10, block_slots=4)
    take(pool)                                         # slot 0 -> block 0 live
    with pytest.raises(ValueError):
        pool.release_block(0)
    # Other (empty) blocks stay releasable.
    assert pool.release_block(1) == 4 * SLOT


def test_release_block_pulls_slots_and_keeps_accounting_exact():
    pool = make_pool(num_slots=10, block_slots=4)
    held = [take(pool) for _ in range(4)]              # 0..3, all of block 0
    assert pool.release_block(2) == 2 * SLOT           # short tail block
    # Freed slots left the free list: 10 - 4 live - 2 released.
    assert pool.free_count() == 4
    assert pool.borrowed_bytes() == 2 * SLOT
    assert pool.released_block_indices() == [2]
    assert not pool.block_is_present(2)
    # Acquire never returns a slot of the released block.
    rest = drain(pool)
    assert rest == [4, 5, 6, 7]
    assert not set(rest) & {8, 9}
    # Allocated slots in other blocks still resolve.
    assert (pool.slot_ptrs(held) != 0).all()
    # Existing API invariant: configured capacity unchanged while on loan.
    assert pool.pool_size_bytes == 10 * SLOT


def test_release_block_double_release_raises():
    pool = make_pool(num_slots=10, block_slots=4)
    pool.release_block(1)
    with pytest.raises(ValueError):
        pool.release_block(1)
    with pytest.raises(ValueError):
        pool.release_block(99)                         # no such block


# --------------------------------------------------------------------------- #
# restore_block
# --------------------------------------------------------------------------- #

def test_restore_block_returns_slots_and_zeroes_borrowed():
    pool = make_pool(num_slots=10, block_slots=4)
    pool.release_block(2)
    assert pool.borrowed_bytes() == 2 * SLOT
    assert pool.restore_block(2) == 2 * SLOT
    assert pool.free_count() == 10
    assert pool.borrowed_bytes() == 0
    assert pool.released_block_indices() == []
    assert pool.block_is_present(2)
    # The restored block's slots are allocatable again and resolve.
    assert set(drain(pool)) == set(range(10))
    assert pool.slot_ptrs([8])[0] == pool._blocks[2].data_ptr()


def test_restore_block_of_present_block_raises():
    pool = make_pool(num_slots=10, block_slots=4)
    with pytest.raises(ValueError):
        pool.restore_block(1)                          # never released


# --------------------------------------------------------------------------- #
# free_all with a block on loan
# --------------------------------------------------------------------------- #

def test_free_all_returns_only_present_blocks_slots():
    pool = make_pool(num_slots=10, block_slots=4)
    [take(pool) for _ in range(6)]                     # 0..5; block 2 empty
    pool.release_block(2)
    pool.free_all()
    assert pool.free_count() == 8                      # blocks 0,1 only
    assert not pool._used.any()
    # Borrow accounting survives the reset.
    assert pool.borrowed_bytes() == 2 * SLOT
    assert pool.released_block_indices() == [2]
    assert set(drain(pool)) == set(range(8))
    assert pool.pool_size_bytes == 10 * SLOT
