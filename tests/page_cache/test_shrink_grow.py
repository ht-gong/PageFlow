"""Moving the granule cache's region boundary in the C++ core
(``IO/cache/granule_core.cc``): a shrink gives the region's lowest slots to
the arena, evicting the surplus in victim order and relocating the
residents of those slots to free slots higher up; a grow takes slots back
below.  The arena's lend runs one (``tests/allocator/test_arena_lend.py``
covers it on the GPU: relocated granules reading back, dirty victims
reaching home, pinned and same-tick granules stopping a lend, the
producer's copy plan and a read outside the ticks surviving one).

- random sequences of registrations, admissions, reads, pins, cleaning,
  reclaims, drops, shrinks and grows on the core keep its slots consistent:
  no slot held twice, every resident inside the region, the floor kept, a
  pinned or same-tick granule never moved or evicted, and each shrink gives
  ``min(k, lendable_slots(floor))``, read before the copies home were
  marked fired;
- ``lendable_slots`` is what a shrink of every slot gives: up to a pinned
  or same-tick granule, within the floor, slots pending behind a copy home
  counted free;
- a shrink stops below a pinned or same-tick granule, gives nothing when
  the floor leaves nothing, and moves its residents in slot order.

CPU only.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

tqpcache = pytest.importorskip("TQPlib.tqpcache")

_I64 = np.int64


def _a(x):
    return np.asarray(x, dtype=_I64)


# -- the core alone (CPU) ------------------------------------------------------

class Model:
    """A ``GranuleCore`` with the bookkeeping the random test checks it by."""

    def __init__(self, n, lo, G=1024, policy="lru"):
        self.n, self.G = n, G
        self.core = tqpcache.GranuleCore(n, G, policy, 25, 0, 1)
        self.core.set_base(1 << 40)
        self.core.set_region(lo)
        self.regions = []            # (index, first, n granules)
        self.pinned = []             # gids pinned by the test
        self.fresh = set()           # gids admitted this tick
        self.home = 1 << 44

    def register(self, granules, pos=0):
        nb = granules * self.G - (self.G // 3 if granules > 1 else 0)
        idx, first, n, *_ = self.core.register(
            nb, self.home, _a([0]), _a([nb]), _a([0]), None, pos + 1000, 1)
        self.home += granules * self.G
        self.regions.append((idx, first, n))
        return idx

    def admit(self, idx):
        _, counts, _ = self.core.admit(_a([idx]))
        first = self.regions[idx][1]
        self.fresh.update(range(first, first + int(counts[0])))

    def advance_tick(self):
        self.core.advance_tick()
        self.fresh = set()

    def fire(self):
        b = self.core.newest_batch()
        if b > self.core.fired_upto():
            self.core.mark_fired_upto(b)

    def residents(self):
        owner = self.core.owner()
        return {int(g): s for s, g in enumerate(owner.tolist()) if g >= 0}

    def check(self):
        core = self.core
        lo = core.region_lo()
        owner = core.owner()
        res = self.residents()
        assert all(lo <= s < self.n for s in res.values())
        assert (owner[:lo] < 0).all()
        if res:
            gids = _a(sorted(res))
            slots = core.slot_of(gids)
            assert slots.tolist() == [res[g] for g in gids.tolist()]
            assert len(set(slots.tolist())) == len(slots)
        assert core.free_count() + len(res) == self.n - lo


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("policy", ["lru", "dataflow"])
def test_random_sequences_keep_the_slots_consistent(seed, policy):
    rng = random.Random(seed)
    n = 96
    m = Model(n, rng.randrange(0, 48), policy=policy)
    pos = 0
    shrinks = relocations = 0
    for step in range(400):
        op = rng.random()
        core = m.core
        if op < 0.25:
            m.admit(m.register(rng.randrange(1, 12), pos))
        elif op < 0.35:
            m.advance_tick()
        elif op < 0.42 and policy == "dataflow":
            core.advance(pos)
            pos += 1
        elif op < 0.50:
            res = list(m.residents())
            if res:
                g = rng.choice(res)
                core.pin(_a([g]))
                m.pinned.append(g)
        elif op < 0.56 and m.pinned:
            core.unpin(_a([m.pinned.pop(rng.randrange(len(m.pinned)))]))
        elif op < 0.62:
            t = core.cleaning_targets()
            if len(t[0]):
                core.commit_cleaning(t[0])
        elif op < 0.66:
            core.reclaim(rng.randrange(1, 6) * m.G)
        elif op < 0.70 and m.regions:
            idx, first, cnt = rng.choice(m.regions)
            if not any(first <= g < first + cnt for g in m.pinned):
                core.drop(idx)
        elif op < 0.75 and m.regions:
            idx, first, cnt = rng.choice(m.regions)
            core.note_read(_a([idx]), _a([0]), _a([m.G]))
        elif op < 0.90:
            k = rng.randrange(0, 24)
            floor = rng.randrange(0, 8)
            # What a shrink could give, read before the copies home are
            # marked fired (their slots count as free).
            lendable = core.lendable_slots(floor)
            m.fire()
            assert core.plan_shrink(1 << 40, floor)[0] == lendable
            lo = core.region_lo()
            before = m.residents()
            tick = core.tick()
            stay = {g for g in before
                    if core.pins_of(_a([g]))[0] > 0 or g in m.fresh}
            (given, ev, cls, dsrc, ddst, dsize, src, dst,
             size) = core.plan_shrink(k, floor)
            assert 0 <= given <= k
            assert given == min(k, lendable)
            # It stops below the lowest slot of a pinned or same-tick granule.
            fixed = [s for g, s in before.items() if g in stay]
            if fixed:
                assert lo + given <= min(fixed)
            assert set(ev.tolist()).isdisjoint(stay)
            assert len(cls) == len(ev) and set(cls.tolist()) <= {0, 1, 2}
            assert len(dsrc) == int((cls == 2).sum())
            assert (src >= lo).all() and (src < lo + given).all()
            assert (dst >= lo + given).all()
            assert len(set(dst.tolist())) == len(dst)
            core.commit_shrink(given, ev, src, dst)
            assert core.region_lo() == lo + given
            after = m.residents()
            for g in stay:
                assert after[g] == before[g]
            for s, d in zip(src.tolist(), dst.tolist()):
                g = next(x for x, y in before.items() if y == s)
                assert after[g] == d
            assert set(after) == set(before) - set(ev.tolist())
            if given:
                assert core.free_count() >= floor
                shrinks += 1
                relocations += len(src)
            assert core.tick() == tick
        else:
            lo = core.region_lo()
            k = rng.randrange(0, lo + 1)
            core.grow(k)
            assert core.region_lo() == lo - k
        m.check()
    assert shrinks and relocations


def _full(n=16, lo=4, G=1024):
    """A region of ``n - lo`` slots filled by one region, granule ``k`` in
    slot ``lo + k``."""
    m = Model(n, lo, G)
    m.admit(m.register(n - lo))
    assert len(m.residents()) == n - lo
    first = m.regions[0][1]
    assert m.core.slot_of(_a([first])).tolist() == [lo]
    return m, first


def test_a_shrink_stops_below_a_pinned_granule():
    m, first = _full()
    m.core.advance_tick()
    m.core.pin(_a([first + 3]))
    given, ev, cls, *_rest, src, dst, size = m.core.plan_shrink(8, 0)
    assert given == 3
    assert first + 3 not in ev.tolist()
    m.core.commit_shrink(given, ev, src, dst)
    assert m.core.region_lo() == 4 + 3
    assert m.core.slot_of(_a([first + 3])).tolist() == [7]


def test_a_shrink_stops_below_a_granule_admitted_this_tick():
    m = Model(16, 4)
    a = m.register(6)
    m.admit(a)                              # slots 10..15
    m.advance_tick()
    b = m.register(4)
    m.admit(b)                              # slots 6..9
    m.core.drop(a)
    m.advance_tick()
    c = m.register(6)
    m.admit(c)                              # this tick: slots 10..15
    first_c = m.regions[c][1]
    assert m.core.slot_of(_a([first_c])).tolist() == [10]
    given, ev, cls, *_rest, src, dst, size = m.core.plan_shrink(12, 0)
    # Stopped at slot 10; the older region is evicted to make room (nothing
    # is free above the given slots).
    assert given == 6
    assert sorted(ev.tolist()) == list(range(m.regions[b][1],
                                             m.regions[b][1] + 4))
    assert not len(src)


def test_a_shrink_in_front_of_a_same_tick_granule_gives_nothing():
    m, first = _full()                      # admitted this tick
    given, ev, *_ = m.core.plan_shrink(4, 0)
    assert given == 0 and not len(ev)


def test_a_shrink_keeps_the_floor_or_gives_nothing():
    m, first = _full()                      # 12 slots, 12 residents
    m.core.advance_tick()
    m.core.pin(_a([first + 11]))            # one granule cannot leave
    # 11 evictable: giving k leaves 12 - k slots, 1 + floor of them for the
    # pinned granule and the floor.
    assert m.core.plan_shrink(20, 4)[0] == 7
    assert m.core.plan_shrink(20, 11)[0] == 0
    given, ev, cls, *_ = m.core.plan_shrink(5, 2)
    # 12 residents in 7 slots with 2 free: 7 evicted.
    assert given == 5 and len(ev) == 7 and (cls == 2).all()


def test_lendable_slots_is_what_a_shrink_of_every_slot_gives():
    m, first = _full()                      # 12 slots, admitted this tick
    core = m.core
    # A same-tick granule in the lowest slot: nothing.
    assert core.lendable_slots(0) == 0 == core.plan_shrink(1 << 40, 0)[0]
    m.advance_tick()
    assert core.lendable_slots(0) == 12
    assert core.lendable_slots(4) == 8 == core.plan_shrink(1 << 40, 4)[0]
    # A pinned granule in slot lo + 3: the three slots below it, or fewer
    # when the floor binds first (12 slots, 1 pinned, 10 free kept: 1).
    core.pin(_a([first + 3]))
    assert core.lendable_slots(0) == 3 == core.plan_shrink(1 << 40, 0)[0]
    assert core.lendable_slots(10) == 1 == core.plan_shrink(1 << 40, 10)[0]
    assert core.lendable_slots(12) == 0 == core.plan_shrink(1 << 40, 12)[0]
    core.unpin(_a([first + 3]))
    # Dirty victims' slots wait on their copies home: they count as free,
    # though a shrink must wait for the copies first.
    core.reclaim(4 * m.G)
    assert core.lendable_slots(2) == 10
    with pytest.raises(RuntimeError, match="marked fired"):
        core.plan_shrink(1 << 40, 2)
    m.fire()
    assert core.plan_shrink(1 << 40, 2)[0] == 10
    with pytest.raises(ValueError):
        core.lendable_slots(-1)


def test_a_grow_adds_free_slots_below():
    m, first = _full()
    m.core.advance_tick()
    free = m.core.free_count()
    m.core.grow(3)
    assert m.core.region_lo() == 1 and m.core.free_count() == free + 3
    with pytest.raises(ValueError):
        m.core.grow(2)
    # The new slots are handed out like any other.
    idx = m.register(3)
    assert m.core.admit(_a([idx]))[1].tolist() == [3]
    assert sorted(m.core.slot_of(_a(range(m.regions[-1][1],
                                          m.regions[-1][1] + 3)))) == [1, 2, 3]


def test_a_shrink_moves_residents_in_slot_order():
    """The residents of the given slots take the highest free slots above
    them in slot order, so a run of them moves as one run (one copy)."""
    m = Model(16, 4)
    a = m.register(6)
    m.admit(a)                              # slots 10..15
    b = m.register(4)
    m.admit(b)                              # slots 6..9
    m.core.drop(a)
    m.advance_tick()
    given, ev, cls, *_rest, src, dst, size = m.core.plan_shrink(4, 0)
    assert given == 4 and not len(ev)
    assert src.tolist() == [6, 7] and dst.tolist() == [14, 15]
    assert size.tolist() == [m.G, m.G]


def test_a_shrink_refuses_copies_not_marked_fired():
    m, first = _full()
    m.core.advance_tick()
    t = m.core.cleaning_targets()
    assert len(t[0])
    m.core.commit_cleaning(t[0])
    with pytest.raises(RuntimeError, match="marked fired"):
        m.core.plan_shrink(2, 0)
    m.fire()
    assert m.core.plan_shrink(2, 0)[0] == 2
