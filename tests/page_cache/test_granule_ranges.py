"""The granule cache's range calls against their one-range, one-granule
forms: pieces and consume pins; the reader's cached spans; and the
cleaning gate.

CPU only, on the granule cache's test fixture (``test_granule_cache.py``).
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from IO.pages.region import CACHED
from IO.transfer.tick_layout import cached_read_ranges
from tests.page_cache.test_granule_cache import G, Fixture, _gids


def _region_bytes(rng, buckets, max_records=3):
    """Records of ``buckets`` streams in stream order: sizes per record and
    the stream of each (several records per stream)."""
    sizes, streams = [], []
    for b in range(buckets):
        for _ in range(rng.randint(1, max_records)):
            sizes.append(rng.randint(1, 2 * G) // 512 * 512 + 512)
            streams.append(b)
    return sizes, streams


def _register(f, rng, nu=10, reads=1, hints=None):
    sizes, streams = _region_bytes(rng, rng.randint(1, 6))
    offsets = np.r_[0, np.cumsum(sizes)[:-1]]
    nbytes = int(sum(sizes))
    home = np.zeros(nbytes, dtype=np.uint8)
    f.homes[len(f.cache.regions)] = home
    return f.cache.register(nbytes, home.ctypes.data, offsets, sizes, streams,
                            next_use_op=nu, expected_reads=reads, hints=hints)


def _pieces_one_by_one(c, cr, lo, hi):
    """The pieces of one range, granule by granule."""
    out = []
    if hi <= lo:
        return out
    for k in range(lo // G, (hi - 1) // G + 1):
        slot = int(c.slot_of([cr.first + k])[0])
        a, b = max(k * G, lo), min((k + 1) * G, hi)
        if out:
            p = out[-1]
            same_home = slot < 0 and p["slot"] < 0
            next_slot = (slot >= 0 and p["slot"] >= 0
                         and slot == p["last"] + 1)
            if same_home or next_slot:
                p["size"] += b - a
                p["last"] = slot
                continue
        src = (cr.home_ptr + a if slot < 0
               else int(c.slot_addrs([cr.first + k])[0]) + (a - k * G))
        out.append({"offset": a, "size": b - a, "src": src, "slot": slot,
                    "last": slot})
    return out


def _scatter(f, rng, regions):
    """Admit the regions over a small pool, evicting as it goes."""
    for cr in regions:
        f.cache.admit([cr])
        f.cache.advance_tick()
        if rng.random() < 0.5:
            f.cache.reclaim(rng.randint(1, 3) * G)


@pytest.mark.parametrize("seed", range(6))
def test_pieces_match_the_granule_by_granule_form(seed):
    rng = random.Random(seed)
    f = Fixture("lru", slots=12, below=5)
    c = f.cache
    regions = [_register(f, rng) for _ in range(8)]
    _scatter(f, rng, regions)
    ranges = []
    for _ in range(20):
        cr = rng.choice(regions)
        lo = rng.randrange(0, cr.nbytes)
        ranges.append((cr, lo, rng.randint(lo, cr.nbytes)))
    p = c.pieces(np.array([cr.index for cr, _, _ in ranges]),
                 np.array([lo for _, lo, _ in ranges]),
                 np.array([hi for _, _, hi in ranges]))
    for i, (cr, lo, hi) in enumerate(ranges):
        mine = p.run == i
        want = _pieces_one_by_one(c, cr, lo, hi)
        assert p.offset[mine].tolist() == [w["offset"] for w in want]
        assert p.size[mine].tolist() == [w["size"] for w in want]
        assert p.src[mine].tolist() == [w["src"] for w in want]
        assert p.on_gpu[mine].tolist() == [w["slot"] >= 0 for w in want]


def test_overlapping_ranges_pin_twice_and_unpin_together():
    f = Fixture("lru", slots=4)
    c = f.cache
    cr = f.region([G, G, G])
    f.offer(cr)
    c.advance_tick()
    c.replace_consume_pins(np.array([cr.index, cr.index]),
                           np.array([0, G]), np.array([2 * G, 3 * G]))
    assert c._core.pins_of(np.array(_gids(cr))).tolist() == [1, 2, 1]
    assert c.victim_order().tolist() == []
    c.replace_consume_pins()
    assert c._core.pins_of(np.array(_gids(cr))).tolist() == [0, 0, 0]
    assert len(c.victim_order()) == 3


# -- the slot pool and the reader -------------------------------------------

@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("settle", [False, True])
def test_cached_spans_match_record_by_record(shuffled, settle):
    rng = random.Random(7)
    cache = object()
    regions, region_of, offset, nbytes = [], [], [], []
    for k in range(6):
        cached = k % 3 != 1
        regions.append(SimpleNamespace(
            location=CACHED if cached else "host",
            release=(cache, SimpleNamespace(index=10 + k)) if cached
            else None))
        at = 0
        for _ in range(rng.randint(1, 4)):
            n = rng.randint(1, 9) * 512
            region_of.append(k)
            offset.append(at)
            nbytes.append(n)
            at += n
    table = SimpleNamespace(regions=regions, region=np.array(region_of),
                            offset=np.array(offset), nbytes=np.array(nbytes))
    recs = np.arange(2, len(region_of) - 1)
    if shuffled:
        recs = np.array(rng.sample(recs.tolist(), len(recs)))
    sel = SimpleNamespace(records=recs, tail_rows=3)
    col = SimpleNamespace(table_backed=True, record_table=lambda: table,
                          record_selection=lambda: sel)
    got = cached_read_ranges(col, settle=settle)
    assert cached_read_ranges(col, settle=settle) is got     # computed once
    want = {}
    for r in (recs[:-1] if settle else recs).tolist():
        k = region_of[r]
        if k % 3 == 1:
            continue
        lo, hi = want.get(10 + k, (1 << 62, 0))
        want[10 + k] = (min(lo, offset[r]), max(hi, offset[r] + nbytes[r]))
    assert got[0] is cache
    assert dict(zip(got[1].tolist(), zip(got[2].tolist(),
                                         got[3].tolist()))) == want


def test_cleaning_waits_for_the_gate_and_closes_when_pressure_ends():
    # Arm at two granules of live victims over a window of three admissions.
    f = Fixture("lru", slots=4, percent=100, gate=(2 * G, 3))
    c = f.cache
    first = f.region([G] * 4)
    f.offer(first)                               # free slots: no victims
    c.advance_tick()
    assert not c.cleaning_open() and len(c.cleaning_targets()) == 0
    f.offer(f.region([G]))                       # one live victim
    c.advance_tick()
    assert not c.cleaning_open()
    last = f.region([G])
    f.offer(last)                                # two over the window
    c.advance_tick()
    assert c.cleaning_open() and len(c.cleaning_targets()) > 0
    # Admissions with no live victim (the last region read, its slot
    # released dead): the window empties after three.
    for i in range(3):
        c.note_read(last, 0, G)
        assert c.release_dead() == 1
        c.advance_tick()
        last = f.region([G])
        f.offer(last)
        c.advance_tick()
        assert c.cleaning_open() == (i < 2)
    assert len(c.cleaning_targets()) == 0


def test_live_victims_of_a_reclaim_outside_admission_arm_the_gate():
    # A reclaim outside any admission that takes live granules counts
    # toward the gate as well.
    f = Fixture("lru", slots=4, percent=100, gate=(2 * G, 3))
    c = f.cache
    f.offer(f.region([G] * 4))
    c.advance_tick()
    assert not c.cleaning_open()
    got = c.reclaim(2 * G)
    assert got.slots == 2 and got.dead == 0
    assert c.cleaning_open()


def test_dead_victims_do_not_arm_the_gate():
    f = Fixture("lru", slots=2, percent=100, gate=(G, 4))
    c = f.cache
    a = f.region([G, G])
    f.offer(a)
    c.advance_tick()
    c.note_read(a, 0, 2 * G)                     # both granules dead
    c.replace_consume_pins(a, 0, 2 * G)          # kept in their slots
    c.replace_consume_pins()
    f.offer(f.region([G, G]))
    assert not c.cleaning_open()


def test_coalesce_merges_copies_that_continue_each_other():
    from IO.transfer.tick_layout import _coalesce
    src = np.array([100, 110, 300, 310, 500])
    dst = np.array([0, 10, 20, 40, 50])
    size = np.array([10, 10, 20, 10, 5])
    got = _coalesce(src, dst, size)
    assert [x.tolist() for x in got] == [[100, 300, 310, 500], [0, 20, 40, 50],
                                         [20, 20, 10, 5]]
