"""The granule cache (``IO/cache/native_cache.py`` over the C++ core)
against its reference model in Python (``reference_cache.py``): the same
random sequence of registrations, admissions, reads, settles, drops,
reclaims and cleaning on both, the reference over a CPU slot pool and the
native cache over a CPU region of as many slots (above slots outside it, as
the arena's cache region sits above the operator space), must make the same
decisions at every step: the same admissions, victims, cleaning picks,
pieces (their ranges, sizes, and which are in slots or in the home), granule
states and residency.  Slot ids are not compared: which slot a granule takes
is the cache's own business (the reference fills its pool from the bottom,
the native cache its region from the top), and neither is the order of the
cleaning picks and of the copies home, which follows the slots.

CPU only, with stand-in events the test fires.  Needs ``TQPlib.tqpcache``
(built by ``cmake --build build --target tqpcache``).
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from tests.page_cache.reference_slot_pool import SlotRunPool
from tests.page_cache.reference_cache import GranuleCache
from tests.page_cache.reference_policies import make_policy
from tests.page_cache.regions import TensorRegion

tqpcache = pytest.importorskip("TQPlib.tqpcache")
from IO.cache.native_cache import NativeGranuleCache  # noqa: E402

G = 1024


class Event:
    def __init__(self):
        self.fired = False

    def query(self):
        return self.fired

    def synchronize(self):
        self.fired = True


class Writer:
    """``copy_home`` that copies nothing and hands out events the test
    fires later, in order."""

    def __init__(self, created):
        self.created = created
        self.calls = []

    def __call__(self, src, dst, size):
        e = Event()
        self.created.append(e)
        self.calls.append((np.asarray(src).tolist(), np.asarray(dst).tolist(),
                           np.asarray(size).tolist()))
        return e


class Side:
    """One implementation under test."""

    def __init__(self, native: bool, policy: str, slots: int, percent: int,
                 gate):
        self.created = []           # every event, in creation order
        self.writer = Writer(self.created)
        if native:
            self.region = TensorRegion(slots + 5, G, boundary=5)
            self.cache = NativeGranuleCache(
                self.region, policy, copy_home=self.writer,
                clean_percent=percent, clean_gate=gate)
        else:
            self.pool = SlotRunPool(slots * G, G, device="cpu")
            self.cache = GranuleCache(
                self.pool, make_policy(policy), copy_home=self.writer,
                clean_percent=percent, clean_gate=gate)
        self.regions = []
        self.home = 1 << 40

    def event(self):
        e = Event()
        self.created.append(e)
        return e

    def fire_all(self):
        for e in self.created:
            e.fired = True

    def fire_oldest(self, k):
        """Fire the k oldest unfired events in creation order: the writer
        stream completes the batches in order."""
        for e in [e for e in self.created if not e.fired][:k]:
            e.fired = True


def _records(rng, nbytes):
    """Records of a few streams covering ``nbytes``: offsets, sizes,
    streams (stream runs in byte order)."""
    sizes, streams = [], []
    left = nbytes
    s = 0
    while left > 0:
        n = min(left, rng.randint(1, 3) * 512)
        sizes.append(n)
        streams.append(s)
        left -= n
        if rng.random() < 0.4:
            s += 1
    offsets = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int64)
    return offsets, np.asarray(sizes, dtype=np.int64), np.asarray(streams, dtype=np.int64)


def _pieces(p):
    """Pieces as ``(run, offset, size, home address or None)``, contiguous
    pieces of a run from the same side merged: which slots a granule sits
    in, and so which slot pieces continue each other, is not compared."""
    out = []
    for run, off, size, src, gpu in sorted(zip(
            p.run.tolist(), p.offset.tolist(), p.size.tolist(),
            p.src.tolist(), p.on_gpu.tolist())):
        home = None if gpu else int(src)
        if out:
            r0, o0, n0, h0 = out[-1]
            if (r0 == run and o0 + n0 == off and (h0 is None) == (home is None)
                    and (home is None or h0 + n0 == home)):
                out[-1] = (r0, o0, n0 + size, h0)
                continue
        out.append((run, off, size, home))
    return out


def _snapshot(side):
    c = side.cache
    out = {"resident": c.resident_bytes()}
    for cr in side.regions:
        if cr.freed:
            continue
        p = c.pieces(cr, 0, cr.nbytes)
        out[("pieces", cr.index)] = _pieces(p)
        gids = list(range(cr.first, cr.first + cr.n))
        out[("state", cr.index)] = c.state(gids)
        out[("admitted", cr.index)] = cr.admitted
    out["victims"] = c.victim_order().tolist()
    return out


def _run(policy, seed, native, slots=12, percent=25, gate=None, steps=60):
    rng = random.Random(seed)
    side = Side(native, policy, slots, percent, gate)
    c = side.cache
    c.advance()
    nu = 1
    log = []
    for step in range(steps):
        op = rng.random()
        live = [cr for cr in side.regions if not cr.freed]
        if op < 0.35 or not live:
            # register one to three regions and offer them together
            crs = []
            for _ in range(rng.randint(1, 3)):
                nbytes = rng.randint(1, 4 * G)
                off, sz, st = _records(rng, nbytes)
                cr = c.register(nbytes, side.home, off, sz, st,
                                next_use_op=nu + rng.randint(0, 3),
                                expected_reads=rng.randint(1, 2))
                side.home += nbytes + 4096
                side.regions.append(cr)
                crs.append(cr)
            adm = c.admit(crs)
            log.append(("admit", [x.index for x in adm.regions], adm.admitted,
                        adm.victims, bool(adm.events)))
            refused = [cr for cr in crs if cr.admitted < cr.n]
            if refused:
                c.note_home_copy(refused, side.event())
            cl = c.cleaning_targets()
            log.append(("clean", sorted(cl.gids.tolist())))
            if len(cl):
                c.commit_cleaning(cl, side.event())
        elif op < 0.6:
            # read ranges of a few regions, deferred, then settle at a tick
            picks = rng.sample(live, min(len(live), rng.randint(1, 3)))
            ri = np.array([cr.index for cr in picks], dtype=np.int64)
            lo = np.zeros(len(picks), dtype=np.int64)
            hi = np.array([cr.nbytes for cr in picks], dtype=np.int64)
            gids = c.granules(ri, lo, hi)
            c.replace_consume_pins(gids=gids)
            ps = c.pieces(ri, lo, hi)
            log.append(("pieces", _pieces(ps), bool(ps.events)))
            c.defer_read(ri, lo, hi)
            c.advance_tick()
            c.replace_consume_pins()
        elif op < 0.7:
            cr = rng.choice(live)
            c.drop(cr)
            log.append(("drop", cr.index, bool(c.home_events(cr))))
        elif op < 0.8:
            got = c.reclaim(rng.randint(1, 3) * G)
            log.append(("reclaim", got.slots, got.dead, got.clean, got.dirty))
        elif op < 0.9:
            c.advance()
            nu += 1
            log.append(("advance", nu))
        else:
            # events fire: either all, or the oldest few
            if rng.random() < 0.5:
                side.fire_all()
            else:
                side.fire_oldest(rng.randint(0, 3))
            c.poll()
            log.append(("fire",))
        log.append(_snapshot(side))
    side.fire_all()
    c.clear()
    free = (side.cache._core.free_count() if native
            else side.pool.free_count())
    log.append(("cleared", c.resident_bytes(), free))
    return log, side


@pytest.mark.parametrize("policy", ["lru", "mru", "dataflow"])
@pytest.mark.parametrize("seed", range(6))
def test_native_core_matches_the_reference(policy, seed):
    ref_log, ref_side = _run(policy, seed, native=False)
    nat_log, nat_side = _run(policy, seed, native=True)
    assert len(ref_log) == len(nat_log)
    for i, (a, b) in enumerate(zip(ref_log, nat_log)):
        assert a == b, f"step {i} ({policy}, seed {seed}):\n reference {a}\n native {b}"
    # The copies home: the same bytes into the same homes (from whichever
    # slots, in whichever order within a batch).
    def norm(side):
        return [sorted(zip(dst, size)) for _, dst, size in side.writer.calls]
    assert norm(ref_side) == norm(nat_side)


@pytest.mark.parametrize("policy", ["lru", "dataflow"])
def test_native_core_with_cleaning_gate(policy):
    gate = (2 * G, 4)
    ref_log, _ = _run(policy, 11, native=False, percent=50, gate=gate, steps=80)
    nat_log, _ = _run(policy, 11, native=True, percent=50, gate=gate, steps=80)
    assert ref_log == nat_log
