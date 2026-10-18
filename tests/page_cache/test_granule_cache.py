"""The granule cache (``IO/cache/native_cache.py``): reclaim order, pins,
prefix admission, the policies' victim order (against a reference order
written out here), cleaning, the dead rule, pieces and slot reuse behind
copy events, and the movement ledger's counts of the granules it copies
home or releases.

CPU only: slots (a ``TensorRegion``) and homes are CPU tensors, copies are ``memmove``s, and the
copies' events are stand-ins whose ``query`` the test controls.  Needs
``TQPlib.tqpcache``.
"""
from __future__ import annotations

import ctypes
import random
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.page_cache.regions import TensorRegion

pytest.importorskip("TQPlib.tqpcache")
from IO.cache.native_cache import NativeGranuleCache  # noqa: E402
from IO.telemetry import movement  # noqa: E402

G = 1024


class Event:
    def __init__(self, fired=False):
        self.fired = fired

    def query(self):
        return self.fired

    def synchronize(self):
        self.fired = True


class HomeWriter:
    """Copies at once; the event fires when the test says so."""

    def __init__(self):
        self.calls = []

    def __call__(self, src, dst, size):
        for s, d, n in zip(src.tolist(), dst.tolist(), size.tolist()):
            ctypes.memmove(d, s, n)
        event = Event()
        self.calls.append((np.asarray(src), np.asarray(dst),
                           np.asarray(size), event))
        return event


class Fixture:
    """A cache over a CPU region of ``slots`` slots (above ``below`` slots
    outside it), and regions with real bytes and homes."""

    def __init__(self, policy="lru", slots=8, percent=25, gate=None,
                 below=0):
        self.region_ = TensorRegion(below + slots, G, boundary=below)
        self.writer = HomeWriter()
        self.cache = NativeGranuleCache(self.region_, policy,
                                        copy_home=self.writer,
                                        clean_percent=percent, clean_gate=gate)
        self.data = {}
        self.homes = {}
        self._fill = 0

    def region(self, streams_bytes, nu=10, reads=1):
        """A region holding ``streams_bytes[s]`` bytes of stream ``s``, in
        stream order, one record per stream."""
        sizes = np.asarray(streams_bytes, dtype=np.int64)
        offsets = np.r_[0, np.cumsum(sizes)[:-1]]
        nbytes = int(sizes.sum())
        self._fill += 1
        data = torch.arange(nbytes, dtype=torch.int64).add_(
            self._fill * 7).to(torch.uint8)
        home = torch.zeros(nbytes, dtype=torch.uint8)
        cr = self.cache.register(nbytes, home.data_ptr(), offsets, sizes,
                                 np.arange(len(sizes)), next_use_op=nu,
                                 expected_reads=reads)
        self.data[cr.index] = data
        self.homes[cr.index] = home
        return cr

    def offer(self, *regions):
        """Admit, then pack: admitted bytes into slots, the rest home."""
        adm = self.cache.admit(list(regions))
        for cr in regions:
            data = self.data[cr.index]
            p = self.cache.pieces(cr, 0, cr.nbytes)
            for off, size, src, gpu in zip(p.offset.tolist(), p.size.tolist(),
                                           p.src.tolist(), p.on_gpu.tolist()):
                ctypes.memmove(src, data.data_ptr() + off, size)
        return adm

    def read(self, cr, lo=0, hi=None):
        """The region's bytes as a reader would assemble them."""
        hi = cr.nbytes if hi is None else hi
        out = torch.empty(hi - lo, dtype=torch.uint8)
        p = self.cache.pieces(cr, lo, hi)
        for off, size, src in zip(p.offset.tolist(), p.size.tolist(),
                                  p.src.tolist()):
            ctypes.memmove(out.data_ptr() + off - lo, src, size)
        return out


def _gids(cr):
    return list(range(cr.first, cr.first + cr.n))


def _slot(c, gid):
    """The slot granule ``gid`` sits in, or -1."""
    return int(c.slot_of([gid])[0])


def _free(c):
    """The cache's free slots, those still waiting on a copy included."""
    return c._core.free_count()


def _move_to(c, regions, cur):
    """Advance the Dataflow position to ``cur`` with every region's granule
    pinned, so no granule dead by position is released (the victim order
    then still holds them)."""
    c.replace_consume_pins(gids=np.array([cr.first for cr in regions]))
    while c._core.position() < cur:
        c.advance()
    c.replace_consume_pins()


def _victim_regions(c, regions, k=None):
    """The victim order (of single-granule regions) as region positions."""
    pos = {cr.first: i for i, cr in enumerate(regions)}
    return [pos[int(g)] for g in c.victim_order(k)]


# -- reclaim -------------------------------------------------------------

@pytest.mark.parametrize("policy", ["lru", "mru", "dataflow"])
def test_reclaim_frees_dead_first_then_policy_order(policy):
    f = Fixture(policy, slots=6)
    c = f.cache
    regions = [f.region([G], nu=10 + i) for i in range(6)]
    for cr in regions:
        f.offer(cr)
        c.advance_tick()
    dead, clean = regions[4], regions[1]
    c.note_read(dead, 0, G)                     # its one read: dead
    cl = SimpleNamespace(gids=np.array([clean.first]), src=None, dst=None,
                         size=None)
    c.commit_cleaning(cl, Event(fired=True))
    c.poll()
    order = c.victim_order().tolist()
    assert order[0] == dead.first
    live = [r.first for r in regions if r is not dead]
    # A clean granule keeps its place in the policy order.
    expected = {"lru": live, "mru": live[::-1],
                # every consumer is in the future: the furthest first
                "dataflow": live[::-1]}[policy]
    assert order[1:] == expected

    got = c.reclaim(3 * G)
    n_clean = int(clean.first in expected[:2])
    assert (got.slots, got.dead, got.clean, got.dirty) == (
        3, 1, n_clean, 2 - n_clean)
    # Only the dirty ones went home, through the writer.
    (src, dst, _, _), = f.writer.calls
    homes = {f.homes[r.index].data_ptr(): r.first for r in regions}
    assert sorted(homes[d] for d in dst.tolist()) == sorted(
        g for g in expected[:2] if g != clean.first)


def test_clean_goes_before_dirty_only_on_an_equal_key():
    f = Fixture("dataflow", slots=4)
    c = f.cache
    cr = f.region([4 * G], nu=10)               # one stream: equal keys
    f.offer(cr)
    c.advance_tick()
    cl = SimpleNamespace(gids=np.array([cr.first]), src=None, dst=None,
                         size=None)
    c.commit_cleaning(cl, Event(fired=True))
    c.poll()
    # The clean first granule, then the later granules first.
    assert c.victim_order().tolist() == [cr.first, cr.first + 3,
                                         cr.first + 2, cr.first + 1]


def test_reclaim_skips_pinned_and_same_tick_granules():
    f = Fixture("lru", slots=4)
    c = f.cache
    a, b = f.region([G]), f.region([G])
    f.offer(a, b)
    c.advance_tick()
    later = f.region([G])
    f.offer(later)                              # admitted this tick
    c.replace_consume_pins(a, 0, G)
    assert c.victim_order().tolist() == [b.first]
    got = c.reclaim(4 * G)
    assert got.slots == 1
    assert c.state(_gids(a) + _gids(later)) == ["dirty", "dirty"]
    assert c.state(_gids(b)) == ["evicted"]
    c.replace_consume_pins()
    c.advance_tick()
    assert c.victim_order().tolist() == [a.first, later.first]


def test_dirty_reclaim_copies_home_and_holds_the_slot_until_the_copy():
    f = Fixture("lru", slots=2)
    c = f.cache
    cr = f.region([G, G])
    f.offer(cr)
    c.advance_tick()
    got = c.reclaim(G)
    # LRU, one region: its later granule goes first.
    assert got.dirty == 1 and c.state(_gids(cr)) == ["dirty", "evicted"]
    (src, dst, size, event), = f.writer.calls
    assert size.tolist() == [G]
    assert torch.equal(f.homes[cr.index][G:], f.data[cr.index][G:])
    # Reads assemble the slot part and the home part.
    assert torch.equal(f.read(cr), f.data[cr.index])
    # The slot waits for the copy's event: a granule admitted into it waits.
    assert got.events == [event]
    assert f.cache.admit([f.region([G])]).events == [event]


def test_admission_into_a_slot_still_under_a_copy_lists_its_event():
    f = Fixture("lru", slots=1)
    c = f.cache
    old = f.region([G])
    f.offer(old)
    c.advance_tick()
    new = f.region([G])
    adm = f.cache.admit([new])
    assert adm.admitted == [1] and adm.victims == (0, 0, 1)
    (_, _, _, event), = f.writer.calls
    assert adm.events == [event]


# -- admission -----------------------------------------------------------

def test_sort_shaped_admission_takes_a_prefix_and_keeps_ties():
    # A partition sort's ticks: one region each, B buckets of one granule,
    # every granule tying on the consumer (the join core).
    B = 8
    f = Fixture("dataflow", slots=B)
    c = f.cache
    first = f.region([G] * B, nu=5)
    assert f.offer(first).admitted == [B]
    c.advance_tick()
    second = f.region([G] * B, nu=5)
    adm = f.offer(second)
    # Bucket k displaces the resident bucket B-1-k while strictly sooner.
    assert adm.admitted == [4] and adm.victims == (0, 0, 4)
    assert c.state(_gids(second)) == ["dirty"] * 4 + ["evicted"] * 4
    assert c.state(_gids(first)) == ["dirty"] * 4 + ["evicted"] * 4
    evicted = f.writer.calls[0][1] - f.homes[first.index].data_ptr()
    assert sorted(evicted.tolist()) == [4 * G, 5 * G, 6 * G, 7 * G]
    c.advance_tick()
    # The residents are buckets 0-3 of two ticks: the third tick's buckets
    # 0 and 1 beat the two bucket 3s, its bucket 2 ties with a bucket 2.
    third = f.region([G] * B, nu=5)
    assert f.offer(third).admitted == [2]
    for cr in (first, second, third):
        assert torch.equal(f.read(cr), f.data[cr.index])


def test_ties_keep_the_incumbent():
    # One stream per region: every granule ties on consumer and hint.
    f = Fixture("dataflow", slots=2)
    c = f.cache
    f.offer(f.region([2 * G], nu=5))
    c.advance_tick()
    refused = f.region([2 * G], nu=5)
    adm = f.offer(refused)
    assert adm.admitted == [0] and adm.victims == (0, 0, 0)
    assert not f.writer.calls
    assert c.state(_gids(refused)) == ["evicted", "evicted"]
    assert torch.equal(f.read(refused), f.data[refused.index])


def test_dataflow_admission_rules():
    f = Fixture("dataflow", slots=1)
    c = f.cache
    far = f.region([G], nu=9)
    f.offer(far)
    c.advance_tick()
    # A later consumer loses to the resident one.
    assert f.offer(f.region([G], nu=12)).admitted == [0]
    # A sooner consumer displaces it.
    sooner = f.region([G], nu=4)
    assert f.offer(sooner).admitted == [1]
    c.advance_tick()
    # A dead victim always loses.
    c.note_read(sooner, 0, G)
    assert f.offer(f.region([G], nu=20)).admitted == [1]


def test_dataflow_protects_the_running_consumers_granules():
    f = Fixture("dataflow", slots=1)
    c = f.cache
    cur_input = f.region([G], nu=2)
    f.offer(cur_input)
    c.advance_tick()
    for _ in range(3):
        c.advance()                              # position 2
    assert f.offer(f.region([G], nu=3)).admitted == [0]


@pytest.mark.parametrize("policy", ["lru", "mru"])
def test_ordered_policies_admit_while_reclaim_can_free(policy):
    f = Fixture(policy, slots=2)
    c = f.cache
    a = f.region([G, G])
    f.offer(a)
    c.advance_tick()
    c.replace_consume_pins(a, 0, G)
    # One granule is evictable: a prefix of one.
    assert f.offer(f.region([G, G, G])).admitted == [1]
    c.advance_tick()


def test_free_slots_go_to_the_most_valuable_region_first():
    f = Fixture("dataflow", slots=1)
    far, near = f.region([G], nu=9), f.region([G], nu=3)
    adm = f.offer(far, near)
    assert [r.index for r in adm.regions] == [near.index, far.index]
    assert adm.admitted == [1, 0]


# -- victim order against a reference order --------------------------------

def _old_entry(pid, nu, hint, tick):
    return SimpleNamespace(page_id=pid, next_use_op=nu, use_order_hint=hint,
                           tick_admitted=tick)


def _dataflow_key(e, cur):
    """The reference Dataflow victim key of one single-granule region
    (smaller first): dead, then future ones furthest consumer first, then
    the one being consumed; on equal consumers the higher hint, then the
    newer admission first."""
    nu, h = e.next_use_op, e.use_order_hint
    if nu < cur:
        return (0, 0, 0, -e.tick_admitted)
    if nu == cur:
        return (2, 0, -h, -e.tick_admitted)
    return (1, -nu, -h, -e.tick_admitted)


def _reference_order(policy, entries, cur):
    """Region ids in reference victim order: LRU oldest admission first,
    MRU newest first, Dataflow by ``_dataflow_key`` (ties: older first)."""
    if policy == "lru":
        return list(entries)
    if policy == "mru":
        return list(reversed(entries))
    return [e.page_id for e in sorted(entries.values(),
                                      key=lambda e: _dataflow_key(e, cur))]


@pytest.mark.parametrize("policy", ["lru", "mru", "dataflow"])
def test_victim_order_matches_the_reference_order(policy):
    rng = random.Random(7)
    n = 60
    f = Fixture(policy, slots=n)
    c = f.cache
    old_entries = OrderedDict()
    regions = []
    tick = 0
    for i in range(n):
        if rng.random() < 0.3:
            c.advance_tick()
            tick += 1
        nu = rng.randrange(3, 9)
        hint = rng.randrange(0, 4)
        cr = c.register(G, 0, [0], [G], [0], next_use_op=nu,
                        expected_reads=1, hints=[hint])
        assert c.admit([cr]).admitted == [1]
        regions.append(cr)
        old_entries[i] = _old_entry(i, nu, hint, tick)
    c.advance_tick()
    cur = 5
    if policy == "dataflow":
        _move_to(c, regions, cur)
        # Some entries had their last read: dead on both sides.
        for i in rng.sample(range(n), 10):
            c.note_read(regions[i], 0, G)
            old_entries[i].next_use_op = -1
    expected = _reference_order(policy, old_entries, cur)
    got = _victim_regions(c, regions)
    assert got == expected
    # The first k by partition match the full order.
    for k in (1, 7, 31):
        part = _victim_regions(c, regions, k)
        assert part == expected[:k]


# -- cleaning --------------------------------------------------------------

def test_cleaning_takes_the_front_quarter_and_counts_clean_ones():
    f = Fixture("lru", slots=8)
    c = f.cache
    regions = [f.region([G]) for _ in range(8)]
    for cr in regions:
        f.offer(cr)
        c.advance_tick()
    targets = c.cleaning_targets()
    assert sorted(targets.gids.tolist()) == [regions[0].first, regions[1].first]
    for s, d, n in zip(targets.src.tolist(), targets.dst.tolist(),
                       targets.size.tolist()):
        ctypes.memmove(d, s, n)
    event = Event()
    c.commit_cleaning(targets, event)
    # In flight: not clean yet, and not offered again.
    assert c.state(targets.gids) == ["dirty", "dirty"]
    assert len(c.cleaning_targets()) == 0
    event.fired = True
    c.poll()
    assert c.state(targets.gids) == ["clean", "clean"]
    # The two clean ones still cover the front quarter.
    assert len(c.cleaning_targets()) == 0
    for cr in regions[:2]:
        assert torch.equal(f.homes[cr.index], f.data[cr.index])


def test_cleaning_skips_pinned_and_dead_granules():
    f = Fixture("lru", slots=4, percent=100)
    c = f.cache
    a, b, d = f.region([G]), f.region([G]), f.region([G])
    f.offer(a, b, d)
    c.note_read(d, 0, G)
    c.replace_consume_pins(a, 0, G)
    assert c.cleaning_targets().gids.tolist() == [b.first]


def test_reclaiming_a_granule_under_cleaning_needs_no_second_copy():
    f = Fixture("lru", slots=2, percent=100)
    c = f.cache
    cr = f.region([G, G])
    f.offer(cr)
    c.advance_tick()
    targets = c.cleaning_targets()
    for s, d, n in zip(targets.src.tolist(), targets.dst.tolist(),
                       targets.size.tolist()):
        ctypes.memmove(d, s, n)
    event = Event()
    c.commit_cleaning(targets, event)
    got = c.reclaim(2 * G)
    assert (got.clean, got.dirty) == (2, 0) and not f.writer.calls
    # Both slots wait for the cleaning copy.
    assert got.events == [event]
    assert _free(c) == 2
    assert f.cache.admit([f.region([G, G])]).events == [event]
    event.fired = True
    assert f.cache.admit([f.region([G])]).events == []


# -- the dead rule, pieces, drop -------------------------------------------

def test_a_granule_is_dead_once_every_stream_in_it_is_read():
    f = Fixture("lru", slots=4)
    c = f.cache
    # Granule 0 holds stream 0 and the head of stream 1; granule 1 the rest
    # of stream 1 and stream 2.
    cr = f.region([G // 2, G, G // 2], reads=2)
    f.offer(cr)
    c.note_read(cr, 0, G // 2)
    c.note_read(cr, 0, G // 2)
    assert c.state(_gids(cr)) == ["dirty", "dirty"]
    # A read that stops inside stream 1 does not finish it.
    c.note_read(cr, G // 2, G)
    assert c._core.seg_reads().tolist() == [0, 2, 2]
    c.note_read(cr, G // 2, 2 * G)
    c.note_read(cr, G // 2, 3 * G // 2)
    assert c.state(_gids(cr)) == ["dead", "dirty"]
    c.note_read(cr, 3 * G // 2, 2 * G)
    assert c.state(_gids(cr)) == ["dead", "dead"]


def test_slots_fill_from_the_region_top():
    f = Fixture("lru", slots=8, below=3)  # the region is slots 3..10
    c = f.cache
    pad = f.region([3 * G])
    f.offer(pad)
    assert [_slot(c, g) for g in _gids(pad)] == [8, 9, 10]
    mid = f.region([G])
    f.offer(mid)
    assert _slot(c, mid.first) == 7
    assert c.slot_addrs(_gids(pad)).tolist() == [
        f.region_.base + s * G for s in (8, 9, 10)]
    assert _free(c) == 4                  # slots 3..6


def test_pieces_follow_slot_runs_and_the_home():
    f = Fixture("lru", slots=8)
    c = f.cache
    pad = f.region([3 * G])
    f.offer(pad)                          # slots 5 6 7
    mid = f.region([G])
    f.offer(mid)                          # slot 4
    c.advance_tick()
    c.drop(pad)                           # free: 0..3 and 5..7
    cr = f.region([G] * 5)
    adm = f.offer(cr)                     # no run holds 5: the top run, then
    assert adm.admitted == [5]            # the next one down
    assert [_slot(c, g) for g in _gids(cr)] == [5, 6, 7, 2, 3]
    c.advance_tick()
    c.drop(mid)
    p = c.pieces(cr, 100, 5 * G - 100)
    assert p.offset.tolist() == [100, 3 * G]
    assert p.on_gpu.tolist() == [True, True]
    # Evict the tail and read across slots and home.
    c.reclaim(2 * G)
    p = c.pieces(cr, 0, 5 * G)
    assert p.on_gpu.tolist() == [True, False]
    assert p.offset.tolist() == [0, 3 * G]
    assert torch.equal(f.read(cr, 10, 5 * G - 10),
                       f.data[cr.index][10:5 * G - 10])


def test_clear_takes_the_regions_new_blocks_and_old_regions_stay_harmless():
    f = Fixture("lru", slots=4, below=4)  # region 4..7
    c = f.cache
    old = f.region([2 * G])
    f.offer(old)
    assert [_slot(c, g) for g in _gids(old)] == [6, 7]
    c.clear()
    f.region_.boundary = 2                # the region grew to 2..7
    c.clear()
    assert _free(c) == 6
    new = f.region([G] * 6)
    f.offer(new)
    assert [_slot(c, g) for g in _gids(new)] == [2, 3, 4, 5, 6, 7]
    # The region registered before the clear: freeing it touches nothing.
    assert c.home_events(old) == []
    c.drop(old)
    assert c.state(_gids(new)) == ["dirty"] * 6 and _free(c) == 0


def test_drop_releases_every_slot():
    f = Fixture("dataflow", slots=4)
    c = f.cache
    cr = f.region([G] * 3)
    f.offer(cr)
    assert _free(c) == 1
    c.drop(cr)
    assert _free(c) == 4 and cr.freed
    assert c.state(_gids(cr)) == ["dead"] * 3
    c.clear()
    assert not c.regions and c._core.granule_count() == 0


def test_policy_names():
    region = TensorRegion(2, G)
    for name, want in (("LRU", "lru"), ("mru", "mru"),
                       ("dataflow", "dataflow")):
        assert NativeGranuleCache(region, name).policy == want
    with pytest.raises(ValueError):
        NativeGranuleCache(region, "fifo")


@pytest.mark.parametrize("policy", ["lru", "mru", "dataflow"])
def test_victim_order_with_clean_granules_matches_the_reference_order(
        policy):
    # Clean granules keep their place: the old order, with a live clean
    # entry moved ahead of a dirty one only where their keys are equal.
    rng = random.Random(11)
    n = 60
    f = Fixture(policy, slots=n)
    c = f.cache
    old_entries = OrderedDict()
    regions = []
    tick = 0
    for i in range(n):
        if rng.random() < 0.3:
            c.advance_tick()
            tick += 1
        nu, hint = rng.randrange(3, 9), rng.randrange(0, 3)
        cr = c.register(G, 0, [0], [G], [0], next_use_op=nu,
                        expected_reads=1, hints=[hint])
        assert c.admit([cr]).admitted == [1]
        regions.append(cr)
        old_entries[i] = _old_entry(i, nu, hint, tick)
    c.advance_tick()
    clean = set(rng.sample(range(n), 20))
    c.commit_cleaning(SimpleNamespace(
        gids=np.array([regions[i].first for i in sorted(clean)])),
        Event(fired=True))
    c.poll()
    cur = 5
    if policy == "dataflow":
        _move_to(c, regions, cur)
    old_order = _reference_order(policy, old_entries, cur)
    if policy == "dataflow":
        key = {e.page_id: _dataflow_key(e, cur)
               for e in old_entries.values()}
    else:
        key = {pid: pos for pos, pid in enumerate(old_order)}
    pos = {pid: p for p, pid in enumerate(old_order)}
    dirty_live = {i for i in old_order if i not in clean
                  and not (policy == "dataflow" and key[i][0] == 0)}
    expected = sorted(old_order,
                      key=lambda i: (key[i], i in dirty_live, pos[i]))
    got = _victim_regions(c, regions)
    assert got == expected
    for k in (1, 9, 33):
        part = _victim_regions(c, regions, k)
        assert part == expected[:k]


def test_dataflow_admission_passes_a_clean_running_input():
    # The running operator's clean input stays; the dirty granule with the
    # furthest consumer is the victim, as the reference order has it.
    f = Fixture("dataflow", slots=2)
    c = f.cache
    running, future = f.region([G], nu=2), f.region([G], nu=8)
    f.offer(running, future)
    c.advance_tick()
    c.commit_cleaning(SimpleNamespace(gids=np.array([running.first])),
                      Event(fired=True))
    c.poll()
    for _ in range(3):
        c.advance()                              # position 2
    adm = f.offer(f.region([G], nu=5))
    assert adm.admitted == [1] and adm.victims == (0, 0, 1)
    assert c.state(_gids(running)) == ["clean"]
    assert c.state(_gids(future)) == ["evicted"]


def test_dead_granules_release_their_slots_when_reads_settle():
    f = Fixture("dataflow", slots=4)
    c = f.cache
    a, b, d = f.region([G], nu=3), f.region([G], nu=3), f.region([G], nu=6)
    f.offer(a, b, d)
    c.advance_tick()
    c.replace_consume_pins(b, 0, G)
    c.defer_read(a, 0, G)
    c.defer_read(b, 0, G)
    assert _free(c) == 1
    c.advance_tick()                             # reads settle
    # a is dead and released; b is dead but pinned, so it keeps its slot.
    assert _slot(c, a.first) == -1 and _slot(c, b.first) >= 0
    assert _free(c) == 2
    assert c.state(_gids(a) + _gids(b)) == ["dead", "dead"]
    c.replace_consume_pins()
    # d's consumer runs and ends: past it, d is dead and released.
    for _ in range(8):
        c.advance()
    assert _slot(c, d.first) == -1 and _slot(c, b.first) == -1
    assert _free(c) == 4
    assert not f.writer.calls


@pytest.mark.parametrize("policy", ["lru", "mru", "dataflow"])
def test_pipelined_reads_release_each_granule_once_its_pins_go(policy):
    """The event-ordered tick loop's pattern: each tick settles the reads
    the tick before bound, then pins the granules of the chunk it issues and
    of the one it computes.  A granule dead while pinned keeps its slot and
    is released at the first settle after its last pin goes (the core's
    candidates, no scan of the region), and a granule holding two streams
    lives until both are read."""
    f = Fixture(policy, slots=8)
    c = f.cache
    chunks = [f.region([G]) for _ in range(5)]
    shared = f.region([G // 2, G // 2])
    f.offer(*chunks, shared)
    prev = np.empty(0, dtype=np.int64)
    for i, cr in enumerate(chunks):
        c.advance_tick()
        # Settled so far: chunks 0..i-1.  Chunks i-2 and i-1 are dead but
        # still pinned (the previous tick pinned them); older ones are gone.
        for j, other in enumerate(chunks):
            held = _slot(c, other.first) >= 0
            assert held == (j >= i - 2), (i, j)
        gids = np.array([cr.first], dtype=np.int64)
        c.replace_consume_pins(gids=np.concatenate([prev, gids]))
        prev = gids
        c.defer_read(cr, 0, G)
        if i == 0:
            c.defer_read(shared, 0, G // 2)
        if i == 2:
            c.defer_read(shared, G // 2, G)
        # The shared granule's second stream is read at tick 2 and settled
        # at the top of tick 3, which releases it.
        assert (_slot(c, shared.first) >= 0) == (i <= 2), i
    c.advance_tick()
    # The pipe's end: the pins go; the last two chunks are released by the
    # next release (a settle with reads, or an explicit one).
    c.replace_consume_pins()
    assert c.release_dead() == 2
    assert all(_slot(c, cr.first) == -1 for cr in chunks + [shared])
    assert _free(c) == 8
    assert not f.writer.calls


# -- the movement ledger ---------------------------------------------------

def _compressed_region(f):
    """A region of 2.5 granules holding two compressed records: bytes
    [0, 1.5G) decode to 6G, then 0.25G of padding, then [1.75G, 2.5G)
    decode to 1.5G.  Registered with the records' decoded sizes, as the
    producer does under experiment telemetry."""
    nbytes = 5 * G // 2
    data = torch.arange(nbytes, dtype=torch.int64).to(torch.uint8)
    home = torch.zeros(nbytes, dtype=torch.uint8)
    cr = f.cache.register(nbytes, home.data_ptr(), [0, 7 * G // 4],
                          [3 * G // 2, 3 * G // 4], [0, 1], next_use_op=10,
                          expected_reads=1, raw=[6 * G, 3 * G // 2])
    f.data[cr.index], f.homes[cr.index] = data, home
    return cr


def test_granules_count_their_records_decoded_bytes(telemetry):
    """A granule's decoded bytes are its records', spread over their stored
    bytes. Cleaning takes one granule; a reclaim of the region then releases
    it clean (``avoided_clean``) and copies the two dirty ones home
    (``spill_evict``): the region's decoded bytes and its bytes, between
    them."""
    f = Fixture("lru", slots=3)
    cr = _compressed_region(f)
    # Granule 0: 2/3 of record 0; granule 1: the rest of it, padding, and
    # 1/3 of record 1; granule 2 (half a granule): the rest of record 1.
    assert cr.decoded.tolist() == [4 * G, 2 * G + G // 2, G]
    assert f.cache.ledger_bytes(np.array(_gids(cr))) == (
        6 * G + 3 * G // 2, 5 * G // 2)
    assert f.cache.ledger_bytes(np.array([cr.first + 2])) == (G, G // 2)
    # Registered without them (a RAW region outside telemetry), a granule
    # counts its own bytes both ways.
    plain = f.region([G, G // 2])
    assert plain.decoded is None
    assert f.cache.ledger_bytes(np.array(_gids(plain))) == (
        3 * G // 2, 3 * G // 2)
    # Both regions' granules in one call, interleaved: each counted as its
    # own region counts it.
    mixed = np.array([plain.first + 1, cr.first + 2, plain.first, cr.first])
    assert f.cache.ledger_bytes(mixed) == (
        G // 2 + G + G + 4 * G, G // 2 + G // 2 + G + G)
    telemetry(experiment=True)
    movement.reset()
    f = Fixture("lru", slots=3)
    c = f.cache
    cr = _compressed_region(f)
    f.offer(cr)
    c.advance_tick()
    targets = c.cleaning_targets()
    assert len(targets) == 1
    for s, d, n in zip(targets.src.tolist(), targets.dst.tolist(),
                       targets.size.tolist()):
        ctypes.memmove(d, s, n)
    c.commit_cleaning(targets, Event(fired=True))
    got = c.reclaim(3 * G)
    assert (got.clean, got.dirty) == (1, 2)
    kinds = movement.report()["kinds"]
    clean, evict = kinds["avoided_clean"], kinds["spill_evict"]
    k = targets.gids[0] - cr.first
    assert clean == {"pages": 1, "logical_b": int(cr.decoded[k]),
                     "wire_b": int(targets.size[0])}
    assert evict["pages"] == 2
    assert clean["logical_b"] + evict["logical_b"] == 6 * G + 3 * G // 2
    assert clean["wire_b"] + evict["wire_b"] == 5 * G // 2
    assert torch.equal(f.read(cr), f.data[cr.index])
    movement.reset()
