"""The reference model's slot pool (``reference_slot_pool.py``): runs of consecutive slots, first fit, and
slots reused only after the copies on them have completed.

CPU only: the slab is built on the CPU and the copies' events are stand-ins
whose ``query`` the test controls.
"""
from __future__ import annotations

import pytest

from tests.page_cache.reference_slot_pool import SlotRunPool

SLOT = 1024


class Event:
    """A copy's event; fires when the test says so."""

    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired

    def synchronize(self):
        self.fired = True


def _pool(slots=16, block_slots=None):
    return SlotRunPool(slots * SLOT, SLOT, device="cpu",
                       block_slots=block_slots)


def _largest_run(pool):
    pool.coalesce()
    return max((n for _, n in pool._runs), default=0)


def test_one_run_when_it_fits():
    pool = _pool(16)
    a = pool.acquire_runs(5)
    b = pool.acquire_runs(3)
    assert a.runs == [(0, 5)] and b.runs == [(5, 3)]
    assert a.events == [] and pool.free_count() == 8
    assert list(a.slots()) == [0, 1, 2, 3, 4]


def test_first_fit_then_scattered_runs():
    pool = _pool(16)
    pool.acquire_runs(16)
    pool.release_slots([1, 2, 6, 7, 8, 12])
    # A run of three fits at 6; first fit takes it.
    assert pool.acquire_runs(3).runs == [(6, 3)]
    # No run of three is left: the grant takes the free runs in order.
    grant = pool.acquire_runs(3)
    assert sorted(grant.runs) == [(1, 2), (12, 1)]
    assert pool.acquire_runs(1) is None


def test_runs_never_span_a_block():
    pool = _pool(16, block_slots=4)
    grant = pool.acquire_runs(6)
    for s, n in grant.runs:
        assert s // 4 == (s + n - 1) // 4
    assert sum(n for _, n in grant.runs) == 6
    pool.release_slots(grant.slots())
    pool.coalesce()
    assert all(s // 4 == (s + n - 1) // 4 for s, n in pool._runs)
    assert _largest_run(pool) == 4


def test_release_behind_an_event_waits_for_it():
    pool = _pool(4)
    pool.acquire_runs(4)
    copy = Event()
    pool.release_slots([1, 2], [copy])
    # Not free, but not in use either: a grant of them lists the copy.
    assert pool.free_count() == 2 and pool._pending_n == 2
    grant = pool.acquire_runs(1)
    assert grant.runs == [(1, 1)] and grant.events == [copy]
    copy.fired = True
    assert pool.reap() == 1
    assert pool.acquire_runs(1).events == []


def test_taking_pending_slots_lists_their_events():
    pool = _pool(4)
    pool.acquire_runs(4)
    first, second = Event(), Event()
    pool.release_slots([0], [first])
    pool.release_slots([3], [second])
    pool.release_slots([2])
    grant = pool.acquire_runs(2)
    # The free slot first, then the oldest release.
    assert grant.runs == [(2, 1), (0, 1)]
    assert grant.events == [first]
    assert not second.fired and pool._pending_n == 1


def test_a_fired_event_releases_at_once():
    pool = _pool(2)
    pool.acquire_runs(2)
    done = Event()
    done.fired = True
    pool.release_slots([0], [done])
    assert pool._pending_n == 0 and pool._free_n == 1


def test_double_release_raises():
    pool = _pool(4)
    pool.acquire_runs(2)
    pool.release_slots([0])
    with pytest.raises(ValueError):
        pool.release_slots([0])


def test_block_with_pending_slots_is_not_empty():
    pool = _pool(8, block_slots=4)
    pool.acquire_runs(4)                     # block 0
    copy = Event()
    pool.release_slots([0, 1, 2, 3], [copy])
    assert pool.empty_block_indices() == [1]
    with pytest.raises(ValueError):
        pool.release_block(0)
    copy.fired = True
    assert pool.empty_block_indices() == [0, 1]
    assert pool.release_block(0) == 4 * SLOT
    assert pool.free_count() == 4
    got = pool.acquire_runs(4)
    assert got.runs == [(4, 4)]
    pool.release_slots(got.slots())
    pool.restore_block(0)
    assert pool.free_count() == 8 and _largest_run(pool) == 4


def test_free_all_waits_for_copies():
    pool = _pool(4)
    pool.acquire_runs(4)
    copy = Event()
    pool.release_slots([0, 1], [copy])
    pool.free_all()
    assert copy.fired
    assert pool._free_n == 4 and pool.acquire_runs(4).runs == [(0, 4)]


def test_slot_pointers_follow_blocks():
    pool = _pool(8, block_slots=4)
    ptrs = pool.slot_ptrs([0, 3, 4, 7])
    b0, b1 = pool._blocks[0].data_ptr(), pool._blocks[1].data_ptr()
    assert ptrs.tolist() == [b0, b0 + 3 * SLOT, b1, b1 + 3 * SLOT]


def test_cuda_event_holds_a_slot_until_the_copy_ends():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    pool = SlotRunPool(4 * SLOT, SLOT, device="cuda")
    pool.acquire_runs(4)
    stream = torch.cuda.Stream()
    event = torch.cuda.Event()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(200_000_000)       # keeps the copy in flight
        event.record(stream)
    pool.release_slots([0], [event])
    assert pool._pending_n == 1
    grant = pool.acquire_runs(1)
    assert grant.runs == [(0, 1)] and grant.events == [event]
    event.synchronize()
