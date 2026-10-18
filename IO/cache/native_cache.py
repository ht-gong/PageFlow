"""The granule cache: which bytes of cached regions sit in GPU slots, its
bookkeeping in a C++ core (``TQPlib.tqpcache.GranuleCore``,
``IO/cache/granule_core.cc``).  ``IO/cache/README.md`` describes its
semantics; ``tests/page_cache/reference_cache.py`` is the same cache in
Python, the reference model the core is tested against.

What stays here is everything CUDA: the events of the copies into homes,
which the core knows as *batches* (integers, created in order), the waits a
reader or a slot fill must make, and the home writer's ``copy_home``.  The
batches' events are recorded on the home writer's one stream, in batch
order, so they fire in order: the newest fired batch marks every earlier one
fired (``mark_fired_upto``), and waiting on the newest live batch of a set
waits on the whole set.  The core returns that one batch (``wait_batch``),
and a caller gets at most one event to wait on.

The slots are the blocks of the arena's cache region
(``IO/cache/cache_region.py``): slot ``i`` is arena block ``i``, at the
arena's base plus ``i`` blocks, and the core hands out the region's highest
free slots first (the arena's operator allocations come from the bottom).
Between queries the region moves while the cache is empty: ``clear`` takes
the region's current blocks as the free slots.

Within a query the arena lends the region's lowest blocks to an allocation
that misses (``attach``; ``memory/README.md``, "Lending"): from C++, with
no Python and no GIL, the core evicts its surplus in victim order (a dirty
granule copied home first) and moves the granules left in those blocks to
free slots higher up, on the arena's lend stream (the home writer's); a
pinned or same-tick granule is never evicted or moved, so a lend that
would need one gives nothing.  The lent blocks come back at the next
pipeline start (``advance``), and only there: an admission short of slots
evicts as it would with nothing lent.  What a lend could give now is the
arena stats' ``lendable_b`` (the core's ``lendable_slots``).  The arena
holds the core's address while attached: ``detach``, and the finalizer
``attach`` registers, take it back before the core can go.

Nothing in Python holds a slot address across an arena allocation unless
its granules are pinned or were admitted this tick: a tick's reads resolve
their pieces under the tick's consume pins, a read outside the ticks pins
its granules for its bind and copies (``tick_layout.pin_cached_reads``),
the producer writes only this tick's slots, and cleaning and the copies
home issue their addresses before anything allocates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import weakref

import numpy as np

import TQPlib.tqpcache as tqpcache
from IO.telemetry import gates, movement

_I64 = np.int64
_EMPTY = np.zeros(0, dtype=_I64)
_STATE = ("dirty", "clean", "evicted", "dead")
_POLICIES = ("lru", "mru", "dataflow")

# The cleaning share of the live residents the engine builds the cache with.
CLEAN_PERCENT = 25
# The cleaning gate (the mirrors' eviction gate): cleaning starts once the
# live victims of the last CLEAN_GATE_WINDOW admissions reach
# CLEAN_GATE_ARM_B bytes, and stops once they are 0.  Set at SF100 (22
# queries, 1 MiB slots) so that live write-backs stay under 1% of the
# evicted live bytes under Dataflow and LRU (0.3% and 0.2%) with cleaning
# traffic below the page cache's mirrors (6.2 and 7.0 GiB against 10.7 and
# 24.3); the mirrors' 64 MiB over 32 left 3.9% under Dataflow.
CLEAN_GATE_ARM_B = 4 << 20
CLEAN_GATE_WINDOW = 64


@dataclass
class Admission:
    """What ``admit`` did: per region (in offer order) the granules
    admitted, a prefix; the slot writes must wait on ``events`` (copies
    still on reused slots); ``victims`` counts the granules reclaimed
    (dead, clean, dirty)."""
    regions: list
    admitted: List[int]
    events: list
    victims: tuple


@dataclass
class Reclaimed:
    """What ``reclaim`` freed: slots, bytes, the count per class, and the
    events the freed slots wait on before reuse."""
    slots: int
    nbytes: int
    dead: int
    clean: int
    dirty: int
    events: list


@dataclass
class Pieces:
    """Byte ranges of cached regions as pieces: the range each piece is
    from (``run``, its index in the request), region offsets, sizes, source
    addresses (a slot or the home) and whether each is in a slot; and the
    events of copies into the homes the home pieces read, not yet seen to
    end (a reader waits on them)."""
    run: np.ndarray
    offset: np.ndarray
    size: np.ndarray
    src: np.ndarray
    on_gpu: np.ndarray
    events: list

    def __len__(self) -> int:
        return len(self.offset)


@dataclass
class Cleaning:
    """Dirty granules to copy home: ids, slot and home addresses, sizes."""
    gids: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    size: np.ndarray

    def __len__(self) -> int:
        return len(self.gids)


def _granule_decoded(offsets, sizes, raw, nbytes: int, G: int) -> np.ndarray:
    """The decoded bytes of each granule of a region whose records
    (``offsets``, ``sizes``: stored bytes) decode to ``raw`` bytes each: a
    record's decoded bytes spread evenly over its stored bytes, the padding
    between records none.  They sum to ``raw``'s sum."""
    off = np.asarray(offsets, dtype=_I64)
    order = np.argsort(off, kind="stable")
    off = off[order]
    size = np.asarray(sizes, dtype=_I64)[order]
    raw = np.asarray(raw, dtype=_I64)[order]
    # The decoded bytes before byte x of the region: linear inside a
    # record, flat between records.
    xp = np.column_stack((off, off + size)).ravel()
    end = np.cumsum(raw)
    fp = np.column_stack((end - raw, end)).ravel()
    edges = np.minimum(np.arange(-(-nbytes // G) + 1, dtype=_I64) * G, nbytes)
    return np.diff(np.floor(np.interp(edges, xp, fp)).astype(_I64))


class NativeRegion:
    """A region registered with the cache: the fields its callers read;
    ``decoded``, each granule's decoded bytes when ``register`` was given
    the records' decoded sizes, else None."""

    __slots__ = ("index", "nbytes", "first", "n", "home_ptr", "next_use_op",
                 "seq", "min_hint", "admitted", "freed", "epoch", "decoded")

    def __repr__(self) -> str:
        return (f"NativeRegion({self.index}, {self.nbytes} B, granules "
                f"{self.first}..{self.first + self.n - 1}, admitted "
                f"{self.admitted}{', freed' if self.freed else ''})")


def _detach(core, ctx: int) -> None:
    """Detach ``core`` (at ``ctx``) from the arena, if it is the cache
    attached, and take the arena's functions from it.  ``attach``'s
    finalizer: it holds ``core``, so the core outlives every arena
    reference to it."""
    try:
        import TQPlib.tqpmemory as tqpmemory
        tqpmemory.clear_lend_hook(ctx)
    except Exception:
        pass
    core.set_arena(0, 0)


def _span_args(regions, lo, hi):
    if (type(regions) is np.ndarray and type(lo) is np.ndarray
            and type(hi) is np.ndarray):
        return regions, lo, hi
    if isinstance(regions, NativeRegion):
        regions = regions.index
    ri = np.atleast_1d(np.asarray(regions, dtype=_I64))
    lo = np.broadcast_to(np.asarray(lo, dtype=_I64), ri.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=_I64), ri.shape)
    return ri, lo, hi


class NativeGranuleCache:
    """The granule cache over the C++ core (module docstring), with the
    eviction policy ``policy`` names: ``lru``, ``mru`` or ``dataflow``.

    ``region`` gives the slots' geometry: ``base`` (slot 0's address),
    ``block`` (the slot size), ``num_blocks`` (the slot ids, the arena's
    blocks) and ``boundary`` (the region's lowest block); the engine passes
    a ``CacheRegion``."""

    def __init__(self, region, policy: str = "lru", copy_home=None,
                 clean_percent: int = CLEAN_PERCENT,
                 clean_gate: Optional[tuple] = (CLEAN_GATE_ARM_B,
                                                CLEAN_GATE_WINDOW)):
        policy = policy.lower()
        if policy not in _POLICIES:
            raise ValueError(f"granule cache policy {policy!r}: expected one "
                             f"of {', '.join(_POLICIES)}")
        self.policy = policy
        self.region = region
        self.G = int(region.block)
        self.copy_home = copy_home
        self.clean_percent = int(clean_percent)
        self.clean_gate = clean_gate
        arm, window = clean_gate if clean_gate is not None else (0, 1)
        self._core = tqpcache.GranuleCore(
            int(region.num_blocks), self.G, self.policy, self.clean_percent,
            int(arm), int(window))
        self._core.set_base(int(region.base))
        self._core.set_region(int(region.boundary))
        # batch id -> its event (``GranuleCore`` numbers batches from 0).
        self._events: list = []
        self.regions: List[NativeRegion] = []
        # Regions registered before the last ``clear`` belong to an earlier
        # epoch: freeing one does nothing.
        self._epoch = 0
        # Lending the region to the arena (``attach``): the finalizer that
        # detaches the core, while attached.
        self._detacher = None

    # -- batches ---------------------------------------------------------------

    def _new_batch(self, b: int, event) -> None:
        assert b == len(self._events), (
            f"native cache: batch {b} out of order ({len(self._events)} known)")
        self._events.append(event)
        # Its copies are issued: a lend may count them ended once it has
        # synchronised the device.
        self._core.note_issued(b)

    def _sync(self) -> None:
        """Tell the core which batches have fired: the newest first (it
        covers every earlier one), else the oldest live ones in order."""
        core = self._core
        newest = core.newest_batch()
        up = core.fired_upto()
        if newest <= up:
            return
        if self._events[newest].query():
            core.mark_fired_upto(newest)
            return
        for b in range(up + 1, newest):
            if self._events[b].query():
                core.mark_fired(b)
            else:
                break

    def _wait(self, b: int) -> list:
        """The events to wait on for batch ``b`` and every earlier one:
        none when they fired, else the newest."""
        if b < 0 or b <= self._core.fired_upto():
            return []
        e = self._events[b]
        if e.query():
            self._core.mark_fired_upto(b)
            return []
        return [e]

    def _victims(self, v) -> tuple:
        """Act on a reclaim's victims (``GranuleCore``'s tuple): copy the
        dirty ones home, count the ledger.  Returns ``(slots, dead, clean,
        dirty, events)``."""
        (slots, dead, clean, dirty, src, dst, size, dirty_batch, wait,
         dead_b, clean_b, dead_lb, clean_lb, dirty_lb) = v
        if dirty_batch >= 0:
            assert self.copy_home is not None, (
                "granule cache: a dirty granule must be copied home, but no "
                "home writer is set")
            if gates.EXPERIMENT:
                movement.record("spill_evict", dirty, dirty_lb,
                                int(size.sum()))
            self._new_batch(dirty_batch, self.copy_home(src, dst, size))
        if gates.EXPERIMENT:
            if dead:
                movement.record("avoided_dead", dead, dead_lb, dead_b)
            if clean:
                movement.record("avoided_clean", clean, clean_lb, clean_b)
        return slots, dead, clean, dirty, self._wait(wait)

    def _released(self, r) -> None:
        n, lb, wb = r
        if n and gates.EXPERIMENT:
            movement.record("avoided_dead", int(n), int(lb), int(wb))

    # -- regions ---------------------------------------------------------------

    def register(self, nbytes: int, home_ptr: int, offsets, sizes, streams,
                 next_use_op: int, expected_reads: int,
                 hints=None, raw=None) -> NativeRegion:
        """``raw``, the records' decoded sizes, for the movement ledger
        (``ledger_bytes``; the producer passes them under experiment
        telemetry only)."""
        decoded = None if raw is None else _granule_decoded(
            offsets, sizes, raw, int(nbytes), self.G)
        index, first, n, seg0, nseg, min_hint, seq = self._core.register(
            int(nbytes), int(home_ptr), np.asarray(offsets, dtype=_I64),
            np.asarray(sizes, dtype=_I64),
            np.atleast_1d(np.asarray(streams, dtype=_I64)),
            None if hints is None else np.asarray(hints, dtype=_I64),
            int(next_use_op), int(expected_reads), decoded)
        cr = NativeRegion()
        cr.decoded = decoded
        cr.index, cr.first, cr.n, cr.nbytes = index, first, n, int(nbytes)
        cr.home_ptr, cr.next_use_op = int(home_ptr), int(next_use_op)
        cr.seq, cr.min_hint, cr.admitted, cr.freed = seq, min_hint, 0, False
        cr.epoch = self._epoch
        assert index == len(self.regions)
        self.regions.append(cr)
        return cr

    def drop(self, cr: NativeRegion) -> None:
        if cr.freed or cr.epoch != self._epoch:
            return
        self._core.drop(cr.index)
        cr.freed = True

    def note_home_copy(self, crs: Sequence[NativeRegion], event) -> None:
        b = self._core.note_home_copy(
            np.fromiter((cr.index for cr in crs), dtype=_I64, count=len(crs)))
        if b >= 0:
            self._new_batch(b, event)

    def home_events(self, cr: NativeRegion) -> list:
        if cr.epoch != self._epoch:
            return []
        return self._wait(self._core.home_wait_batch(cr.index))

    # -- positions and ticks -------------------------------------------------

    def advance(self, node=None) -> None:
        """A pipeline starts: its position (``node``'s, under Dataflow)
        and lent blocks no longer in use back into the region."""
        pos = (node.position
               if node is not None and self.policy == "dataflow" else -1)
        self._released(self._core.advance(int(pos)))

    def reset_position(self) -> None:
        self._core.reset_position()

    def advance_tick(self) -> None:
        self._sync()
        self._released(self._core.advance_tick())

    @property
    def tick(self) -> int:
        return self._core.tick()

    # -- reads ---------------------------------------------------------------

    def slot_addrs(self, gids) -> np.ndarray:
        return self._core.slot_addrs(np.asarray(gids, dtype=_I64))

    def granules(self, regions, lo, hi) -> np.ndarray:
        return self._core.granules(*_span_args(regions, lo, hi))

    def pieces(self, regions, lo, hi) -> Pieces:
        run, off, size, src, on_gpu, wait = self._core.pieces(
            *_span_args(regions, lo, hi))
        return Pieces(run, off, size, src, on_gpu, self._wait(wait))

    def note_read(self, regions, lo, hi) -> None:
        self._core.note_read(*_span_args(regions, lo, hi))

    def defer_read(self, regions, lo, hi) -> None:
        ri, lo, hi = _span_args(regions, lo, hi)
        if len(ri):
            self._core.defer_read(ri, lo, hi)

    def settle_reads(self) -> None:
        self._released(self._core.settle_reads())

    def release_dead(self) -> int:
        r = self._core.release_dead()
        self._released(r)
        return int(r[0])

    def pin(self, gids) -> None:
        self._core.pin(np.asarray(gids, dtype=_I64))

    def unpin(self, gids) -> None:
        self._core.unpin(np.asarray(gids, dtype=_I64))

    def replace_consume_pins(self, regions=(), lo=(), hi=(),
                             gids=None) -> np.ndarray:
        if gids is None:
            gids = self.granules(regions, lo, hi)
        return self._core.replace_consume_pins(np.asarray(gids, dtype=_I64))

    # -- victims -------------------------------------------------------------

    def victim_order(self, k: Optional[int] = None) -> np.ndarray:
        self._sync()
        return self._core.victim_order(-1 if k is None else int(k))

    def reclaim(self, nbytes: int) -> Reclaimed:
        self._sync()
        slots, dead, clean, dirty, events = self._victims(
            self._core.reclaim(int(nbytes)))
        return Reclaimed(slots=slots, nbytes=slots * self.G, dead=dead,
                         clean=clean, dirty=dirty, events=events)

    def evictable_count(self) -> int:
        self._sync()
        return self._core.evictable_count()

    # -- admission ---------------------------------------------------------

    def admit(self, regions: Sequence[NativeRegion]) -> Admission:
        self._sync()
        order, counts, v = self._core.admit(
            np.fromiter((cr.index for cr in regions), dtype=_I64,
                        count=len(regions)))
        offered = [self.regions[i] for i in order.tolist()]
        admitted = counts.tolist()
        for cr, a in zip(offered, admitted):
            cr.admitted = a
        slots, dead, clean, dirty, events = self._victims(v)
        return Admission(regions=offered, admitted=admitted, events=events,
                         victims=(dead, clean, dirty))

    # -- cleaning ----------------------------------------------------------

    def cleaning_open(self) -> bool:
        return self._core.cleaning_open()

    def cleaning_targets(self) -> Cleaning:
        if self._core.cleaning_open():
            self._sync()
        return Cleaning(*self._core.cleaning_targets())

    def commit_cleaning(self, cleaning: Cleaning, event) -> None:
        if not len(cleaning.gids):
            return
        b = self._core.commit_cleaning(np.asarray(cleaning.gids, dtype=_I64))
        if b >= 0:
            self._new_batch(b, event)

    def poll(self) -> None:
        self._sync()
        self._core.poll()

    # -- lending the region to the arena ---------------------------------------

    def attach(self) -> None:
        """Lend the region to the arena from now on: register the core's
        lend hooks (``tqpcache.lend_hooks``) with the arena, over the home
        writer's stream, and give the core the arena's ``lend_api``.  An
        allocation that misses then borrows the region's lowest blocks
        (module docstring).  The region must be the arena's (a
        ``CacheRegion``) and the core's region its current one (``clear``
        after a move).  One cache is attached at a time; ``detach`` before
        the region moves.  The arena holds the core's address: a finalizer
        detaches it should this cache go while attached, and keeps the
        core alive until then."""
        import TQPlib.tqpmemory as tqpmemory
        from IO.codec.home_writer import writer_stream
        assert int(self._core.region_lo()) == int(self.region.boundary), (
            "native cache: attached over a region it does not hold "
            "(clear it after the boundary moved)")
        self.detach()
        self._core.set_arena(*tqpmemory.lend_api())
        lend, lendable = tqpcache.lend_hooks()
        ctx = self._core.ctx_ptr()
        tqpmemory.set_lend_hook(lend, lendable, ctx,
                                writer_stream().cuda_stream)
        self._detacher = weakref.finalize(self, _detach, self._core, ctx)

    @property
    def attached(self) -> bool:
        """Whether this cache is the one the arena lends from now (the
        arena's lend ctx is its core)."""
        import TQPlib.tqpmemory as tqpmemory
        return int(tqpmemory.lend_ctx()) == int(self._core.ctx_ptr())

    def detach(self) -> None:
        """No more lending from this cache (``attach``); a no-op when it is
        not attached."""
        if self._detacher is not None:
            self._detacher()
            self._detacher = None

    def lend_stats(self, reset: bool = False) -> tuple:
        """The core's lend counters since the last reset (``GranuleCore.
        lend_stats``): (lends asked, refused, blocks given, blocks taken
        back, granules relocated, copies issued, then per class dead / clean
        / dirty the granules evicted, their ledger bytes, their bytes)."""
        return self._core.lend_stats(reset)

    def take_lend_stats(self) -> dict:
        """The lends since the last call (both counters reset): the arena's
        (``tqpmemory.lend_stats``: misses, those lent for, short, failed,
        blocks lent and returned, blocks lent now, ns) and the core's
        (evictions per class, relocations, copies).  Under experiment
        telemetry the evictions go to the movement ledger as a reclaim's do:
        ``spill_evict`` for the dirty ones (copied home), ``avoided_clean``
        and ``avoided_dead``."""
        import TQPlib.tqpmemory as tqpmemory
        out = dict(tqpmemory.lend_stats(True))
        (lends, refused, blocks, returned, relocated, copies, evicted,
         logical, wire) = self._core.lend_stats(True)
        out.update(core_lends=lends, core_refused=refused,
                   relocated=relocated, copies=copies,
                   evicted_dead=evicted[0], evicted_clean=evicted[1],
                   evicted_dirty=evicted[2], dirty_b=wire[2])
        if gates.EXPERIMENT:
            for kind, c in (("avoided_dead", 0), ("avoided_clean", 1),
                            ("spill_evict", 2)):
                if evicted[c]:
                    movement.record(kind, evicted[c], logical[c], wire[c])
        return out

    # -- teardown ------------------------------------------------------------

    def clear(self) -> None:
        """Drop every region and granule, once the copies home are done;
        the free slots are then the region's blocks as they are now (the
        boundary may have moved since the last ``clear``).  A region
        registered before stays harmless: freeing it does nothing."""
        core = self._core
        live = core.live_batches().tolist()
        for b in live:
            self._events[b].synchronize()
        if live:
            core.mark_fired_upto(live[-1])
        core.clear()
        core.set_region(int(self.region.boundary))
        self._events = []
        for cr in self.regions:
            cr.freed = True
        self.regions = []
        self._epoch += 1

    # -- state, for tests and telemetry ----------------------------------

    def state(self, gids) -> List[str]:
        codes = self._core.state(np.asarray(gids, dtype=_I64))
        return [_STATE[c] for c in codes.tolist()]

    def resident_bytes(self) -> int:
        return self._core.resident_bytes()

    def ledger_bytes(self, gids) -> Tuple[int, int]:
        """What the movement ledger counts for granules ``gids``:
        ``logical_b``, their decoded bytes (``NativeRegion.decoded``; a
        granule of a region registered without them counts its own bytes),
        and ``wire_b``, their bytes."""
        lb, wb = self._core.ledger_bytes(np.asarray(gids, dtype=_I64))
        return int(lb), int(wb)

    def reset_peak(self) -> None:
        """Rebaseline the high-water mark of slots in use to the slots in
        use now."""
        self._core.reset_peak()

    def used_bytes(self) -> int:
        """Bytes of the slots in use (a slot waiting on a copy is not)."""
        return self._core.used_slots() * self.G

    def peak_used_bytes(self) -> int:
        """Bytes of the most slots in use at once since ``reset_peak``."""
        return self._core.peak_used_slots() * self.G

    def slot_of(self, gids) -> np.ndarray:
        return self._core.slot_of(np.asarray(gids, dtype=_I64))
