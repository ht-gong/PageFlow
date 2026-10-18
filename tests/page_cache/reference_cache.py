"""The granule cache in Python, the reference model of the engine's cache
(``IO/cache/native_cache.py`` over the C++ core ``IO/cache/granule_core.cc``):
``test_native_core.py`` runs both on the same random sequences and compares
their decisions.  It is not an engine path.

A cached region is a region (``IO/pages/region.py``) whose bytes have a
reserved pinned home and may also sit, in part, in GPU slots.  The cache
cuts it into granules, bytes ``[k*G, (k+1)*G)`` with ``G`` the slot size of
its pool (``reference_slot_pool.py``'s ``SlotRunPool``), and binds each
granule to at most one slot.  The granule is the unit of residency,
eviction and reclaim, and nothing else: reads are cut by records, and a
record can be partly in slots and partly in its home.

Every granule has one row in numpy arrays (``Granules``): ``slot`` (or -1),
``clean`` (the home holds its bytes), ``live`` (some stream in it still has
a read to come), ``pins`` (consume pins), and the keys the policies read
(``tests/page_cache/reference_policies.py``).  Its states:

  dirty    in a slot; the home is reserved but not written
  clean    in a slot and in the home
  evicted  in the home only (a granule admission refused starts here)
  dead     no read will come: its slot is released once its reads have
           settled (``release_dead``), or by the next reclaim if it was
           pinned or admitted this tick

Transitions: dirty -> clean (cleaning), clean -> evicted (release), dirty ->
evicted (copy home, then release after the copy's event), any -> dead.  No
granule goes from the home back into a slot: the cache admits only what a
producer offers.  ``drop`` frees a region's granules when its column is
freed.

**Admission.**  A producer registers each region it creates (``register``:
its size, its home and its records' layout by stream), then offers the
tick's regions together (``admit``).  Each region's granules are offered in
byte order and the cache admits a prefix, stopping at the first granule the
policy refuses: a free slot always admits; otherwise the granule is compared
with the victim reclaim would free for it (``GranulePolicy.admits``).  The
refused rest stays evicted, one byte range for the producer's dense buffer
and home copy.

**Reclaim.**  ``reclaim(nbytes)`` frees dead granules first, then live ones
in policy order, a clean one before a dirty one only on an equal key (it
needs no copy).  A dirty granule is copied home
through ``copy_home`` (the home writer) and its slot is released behind the
copy's event.  Pinned granules and granules admitted in the current tick
are never chosen.  For the slot governor, ``block_order`` ranks the pool's
blocks by the victim order of the granules they hold and ``reclaim_blocks``
frees every granule in the blocks it chose, so it can lend them to the
arena.

**Copies into homes.**  Every copy into a home is a batch of granules with
one event (``Granules.batch``): the producer's copy of a region's refused
part (``note_home_copy``), cleaning's and reclaim's.  A read of a home piece
waits on its granules' batches; a slot released from a granule still being
copied home goes back to the pool behind its batch, and a grant that takes
such a slot lists the events for the writer to wait on.

**Cleaning.**  After a tick's admissions, ``cleaning_targets`` names the dirty
granules among the front ``clean_percent`` of live, unpinned, resident
granules in policy order (clean ones there count as covered), and only
while the cleaning gate is open: the live victims (granules reclaimed with
a read to come; a reclaim outside admission counts with the latest one) of
the last ``CLEAN_GATE_WINDOW`` admissions reached
``CLEAN_GATE_ARM_B`` bytes, and have not dropped to 0 since; the caller
copies them home behind the event of the tick's slot fills and passes the
copy's event to ``commit_cleaning``.  They count as clean once it fires.

**Reads.**  The reader pins the granules the next tick reads
(``replace_consume_pins``), gets each byte range as pieces in slots or in the
home (``pieces``, with the events of the copies still writing those home
bytes), and reports each finished read (``note_read``): a stream whose last
record in the region has been read as often as expected is dead in that
region, and a granule is dead once every stream in it is.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from tests.page_cache.reference_policies import TICK_LIMIT, GranuleLRU, GranulePolicy
from IO.telemetry import gates, movement

_I64 = np.int64
_CLASS_SHIFT = 60
# The clean-before-dirty tie-break sits above every granule's ``tb``.
_TB_SHIFT = 50
# A segment's ``key``: its region index above this bit, its end below.
_KEY_SHIFT = 40
_NO_HINT = np.iinfo(np.int64).max
_ZERO = np.zeros(1, dtype=np.int64)
DEAD, CLEAN, DIRTY = 0, 1, 2
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
# A read of at most this many granules is cut into pieces granule by
# granule (``GranuleCache._pieces_small``), a larger one by array
# operations.
_SMALL_READ = 16
CLEAN_GATE_WINDOW = 64


class _Columns:
    """Growable numpy columns, one row per item; rows are handed out in
    order and never reused until ``reset``."""

    _FIELDS: tuple = ()

    def __init__(self, capacity: int = 1024):
        self.n = 0
        self._alloc(capacity)

    def _alloc(self, capacity: int) -> None:
        for name, dtype, fill in self._FIELDS:
            arr = np.full(capacity, fill, dtype=dtype)
            old = getattr(self, name, None)
            if old is not None:
                arr[:self.n] = old[:self.n]
            setattr(self, name, arr)
        self.capacity = capacity

    def grow(self, count: int) -> int:
        """Rows for ``count`` new items; returns the first."""
        first = self.n
        if self.n + count > self.capacity:
            self._alloc(max(2 * self.capacity, self.n + count))
        self.n += count
        return first

    def reset(self) -> None:
        self.n = 0
        for name, dtype, fill in self._FIELDS:
            getattr(self, name)[:] = fill


class Granules(_Columns):
    """Per-granule arrays, indexed by granule id; ids are handed out in
    region order."""

    _FIELDS = (("slot", _I64, -1), ("clean", bool, False),
               ("live", bool, False), ("pins", np.int32, 0),
               ("nu", _I64, 0), ("hint", _I64, 0), ("tick", _I64, 0),
               ("seq", _I64, 0), ("tb", _I64, 0), ("region", _I64, -1),
               ("cleaning", bool, False), ("batch", _I64, -1))


class _Regions(_Columns):
    """Per-region arrays, indexed by region index: its first granule, size,
    home, first segment, and whether its column was freed."""

    _FIELDS = (("first", _I64, 0), ("nbytes", _I64, 0), ("home", _I64, 0),
               ("seg0", _I64, 0), ("freed", bool, False))


class _Segments(_Columns):
    """Per-segment arrays: a segment is the byte range of one run of a
    stream's records in a region, ``[lo, hi)``, with its first and last
    granule ids (``glo``, ``ghi``), its stream's hint and the reads still
    expected.  Segments are stored in region order and, inside a region,
    in byte order, so ``glo``, ``ghi`` and ``key`` (the region index above
    ``_KEY_SHIFT``, ``hi`` below it) never decrease: every lookup is a
    ``searchsorted``."""

    _FIELDS = (("region", _I64, -1), ("stream", _I64, 0), ("lo", _I64, 0),
               ("hi", _I64, 0), ("key", _I64, 0), ("hint", _I64, 0),
               ("reads", _I64, 0), ("glo", _I64, 0), ("ghi", _I64, 0))


def _expand(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``starts[i] .. starts[i] + counts[i] - 1`` for every ``i``, back to
    back."""
    counts = np.asarray(counts, dtype=_I64)
    total = int(counts.sum())
    if not total:
        return np.zeros(0, dtype=_I64)
    shift = np.asarray(starts, dtype=_I64) - (np.cumsum(counts) - counts)
    return np.repeat(shift, counts) + np.arange(total, dtype=_I64)


def _starts(brk: np.ndarray) -> np.ndarray:
    """The first index of each group, given ``brk[i]``: a new group starts
    at ``i + 1``."""
    return np.concatenate((_ZERO, np.flatnonzero(brk) + 1))


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


def _span_args(regions, lo, hi):
    """``regions`` (a ``CachedRegion``, or region indices) and the byte
    bounds as equal-length int64 arrays."""
    if (type(regions) is np.ndarray and type(lo) is np.ndarray
            and type(hi) is np.ndarray):
        return regions, lo, hi
    if isinstance(regions, CachedRegion):
        regions = regions.index
    ri = np.atleast_1d(np.asarray(regions, dtype=_I64))
    lo = np.broadcast_to(np.asarray(lo, dtype=_I64), ri.shape)
    hi = np.broadcast_to(np.asarray(hi, dtype=_I64), ri.shape)
    return ri, lo, hi


class CachedRegion:
    """A region registered with the cache: its granules ``first .. first +
    n - 1``, its home, and its streams as segments (byte ranges, one per run
    of a stream's records, in byte order; the cache's segment arrays
    ``seg0 .. seg0 + nseg - 1``, read here as views); ``decoded``, each
    granule's decoded bytes when ``register`` was given them, else None."""

    __slots__ = ("index", "nbytes", "first", "n", "home_ptr",
                 "next_use_op", "seq", "seg0", "nseg", "min_hint", "_segs",
                 "admitted", "freed", "decoded")

    def _seg(self, name: str) -> np.ndarray:
        return getattr(self._segs, name)[self.seg0:self.seg0 + self.nseg]

    seg_stream = property(lambda self: self._seg("stream"))
    seg_lo = property(lambda self: self._seg("lo"))
    seg_hi = property(lambda self: self._seg("hi"))
    seg_hint = property(lambda self: self._seg("hint"))
    seg_reads = property(lambda self: self._seg("reads"))
    seg_g_lo = property(lambda self: self._seg("glo") - self.first)
    seg_g_hi = property(lambda self: self._seg("ghi") - self.first)

    def __repr__(self) -> str:
        return (f"CachedRegion({self.index}, {self.nbytes} B, granules "
                f"{self.first}..{self.first + self.n - 1}, admitted "
                f"{self.admitted}{', freed' if self.freed else ''})")


@dataclass
class Admission:
    """What ``admit`` did: per region (in offer order) the granules
    admitted, a prefix; the slot writes must wait on ``events`` (copies
    still on reused slots); ``victims`` counts the granules reclaimed
    (dead, clean, dirty)."""
    regions: List[CachedRegion]
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


class GranuleCache:
    """The cache over one ``SlotRunPool`` (module docstring).

    ``copy_home(src_ptrs, dst_ptrs, sizes)`` copies dirty granules from
    their slots into their homes and returns an event recorded after the
    copies (the home writer, behind the slots' fills).  ``clean_percent`` is
    the cleaning share of the live residents, in [0, 100].
    """

    def __init__(self, pool, policy: Optional[GranulePolicy] = None,
                 copy_home: Optional[Callable] = None,
                 clean_percent: int = 25,
                 clean_gate: Optional[tuple] = (CLEAN_GATE_ARM_B,
                                                CLEAN_GATE_WINDOW)):
        assert 0 <= clean_percent <= 100, (
            f"clean_percent must be in [0, 100], got {clean_percent}")
        assert clean_gate is None or (clean_gate[0] > 0 and clean_gate[1] > 0)
        # The cleaning gate, ``(arm bytes, window)``, or None: always open.
        self.clean_gate = clean_gate
        self._gate_history: deque = deque()
        self._gate_bytes = 0
        self._gate_armed = False
        self.pool = pool
        self.G = pool.slot_size
        self.policy = policy or GranuleLRU()
        self.copy_home = copy_home
        self.clean_percent = int(clean_percent)
        self.g = Granules()
        self.r = _Regions(256)
        self.s = _Segments()
        # The granule in each slot, or -1.
        self.owner = np.full(pool.num_slots, -1, dtype=_I64)
        self.regions: List[CachedRegion] = []
        self.tick = 0
        self._seq = 0
        # Copies of granules into their homes (the producer's refused part,
        # cleaning, reclaim), by batch (``Granules.batch``): each one's event
        # and whether it was seen to end; and the cleaning batches not yet
        # polled, as (granule ids, batch).
        self._batch_events: list = []
        self._batch_fired: list = []
        self._cleaning: list = []
        self._consume = np.zeros(0, dtype=_I64)
        # Reads issued and not yet counted (``defer_read``): arrays of
        # region indices, starts and ends.
        self._reads: list = []
        # Whether a resident granule may be dead and releasable: set when
        # reads end a stream, the position moves, pins drop or a dead
        # granule is admitted; ``release_dead`` clears it.
        self._maybe_dead = False

    # -- regions -----------------------------------------------------------

    def register(self, nbytes: int, home_ptr: int, offsets, sizes, streams,
                 next_use_op: int, expected_reads: int,
                 hints=None, raw=None) -> CachedRegion:
        """A new cached region of ``nbytes`` with its home at ``home_ptr``:
        its records' ``offsets``, ``sizes`` and ``streams``; its consumer's
        position and expected reads; ``hints`` per stream (default: the
        stream index, a partition sort's bucket); ``raw``, the records'
        decoded sizes, for the movement ledger (``ledger_bytes``; the
        producer passes them under experiment telemetry only).  Every
        granule starts evicted, until ``admit``."""
        nbytes = int(nbytes)
        assert 0 < nbytes < (1 << _KEY_SHIFT)
        # Regions hold few records and granules: plain Python over lists
        # costs less here than numpy calls on tiny arrays.
        off = np.asarray(offsets, dtype=_I64).tolist()
        size = np.asarray(sizes, dtype=_I64).tolist()
        st = np.asarray(streams, dtype=_I64)
        st = [int(st)] * len(off) if st.ndim == 0 else st.tolist()
        assert off and len(size) == len(off) == len(st)
        recs = list(zip(off, size, st))
        if any(a[0] > b[0] for a, b in zip(recs, recs[1:])):
            recs.sort(key=lambda r: r[0])
        # Segments: runs of one stream's records, in byte order.
        seg_lo, seg_hi, seg_st = [], [], []
        for o, n, t in recs:
            assert o >= 0 and o + n <= nbytes, "records overrun the region"
            if seg_st and t == seg_st[-1]:
                seg_hi[-1] = max(seg_hi[-1], o + n)
            else:
                assert not seg_hi or o >= seg_hi[-1], (
                    "a region's stream runs overlap")
                seg_lo.append(o)
                seg_hi.append(o + n)
                seg_st.append(t)
        if hints is None:
            seg_hint = seg_st
        else:
            hints = np.asarray(hints, dtype=_I64).tolist()
            seg_hint = [hints[t] for t in seg_st]
        cr = CachedRegion()
        cr.index = len(self.regions)
        assert cr.index < (1 << (63 - _KEY_SHIFT))
        cr.nbytes = nbytes
        G = self.G
        cr.n = n = -(-nbytes // G)
        cr.first = first = self.g.grow(n)
        cr.home_ptr = int(home_ptr)
        cr.next_use_op = int(next_use_op)
        cr.seq = self._seq
        self._seq += 1
        cr.min_hint = min(seg_hint)
        self.policy.check_region(cr.next_use_op, seg_hint, cr.seq)
        segs = self.s
        m = len(seg_lo)
        cr.seg0 = segs.grow(m)
        cr.nseg = m
        cr._segs = segs
        sl = slice(cr.seg0, cr.seg0 + m)
        seg_glo = [first + x // G for x in seg_lo]
        seg_ghi = [first + (x - 1) // G for x in seg_hi]
        base = cr.index << _KEY_SHIFT
        segs.region[sl] = cr.index
        segs.stream[sl] = seg_st
        segs.lo[sl] = seg_lo
        segs.hi[sl] = seg_hi
        segs.key[sl] = [base + x for x in seg_hi]
        segs.hint[sl] = seg_hint
        segs.reads[sl] = int(expected_reads)
        segs.glo[sl] = seg_glo
        segs.ghi[sl] = seg_ghi
        cr.admitted = 0
        cr.freed = False
        cr.decoded = None if raw is None else _granule_decoded(
            offsets, sizes, raw, nbytes, G)
        self.regions.append(cr)
        ri = self.r.grow(1)
        r = self.r
        r.first[ri] = first
        r.nbytes[ri] = nbytes
        r.home[ri] = cr.home_ptr
        r.seg0[ri] = cr.seg0
        ids = slice(first, first + n)
        g = self.g
        g.region[ids] = cr.index
        g.nu[ids] = cr.next_use_op
        g.seq[ids] = cr.seq
        # Older region first, later granule first inside one.
        g.tb[ids] = np.arange(first + n - 1, first - 1, -1, dtype=_I64)
        # Every segment expects its reads: a granule is live where one
        # covers it, keyed by the lowest hint among them (``_refresh``).
        if int(expected_reads) > 0:
            low = [_NO_HINT] * n
            for h, a, b in zip(seg_hint, seg_glo, seg_ghi):
                for k in range(a - first, b - first + 1):
                    if h < low[k]:
                        low[k] = h
            g.live[ids] = [x != _NO_HINT for x in low]
            g.hint[ids] = [0 if x == _NO_HINT else x for x in low]
        else:
            g.live[ids] = False
            g.hint[ids] = 0
        return cr

    def _refresh(self, gids: np.ndarray) -> None:
        """Recompute ``live`` and ``hint`` of granules ``gids`` (sorted)
        from the segments over them: live while one of them expects a read,
        keyed by the lowest hint among those."""
        segs = self.s
        m = segs.n
        a = np.searchsorted(segs.ghi[:m], gids, side="left")
        cnt = np.maximum(
            np.searchsorted(segs.glo[:m], gids, side="right") - a, 0)
        live = np.zeros(len(gids), dtype=bool)
        hint = np.zeros(len(gids), dtype=_I64)
        has = cnt > 0
        if has.any():
            c = cnt[has]
            idx = _expand(a[has], c)
            at = np.cumsum(c) - c
            reading = segs.reads[idx] > 0
            any_live = np.logical_or.reduceat(reading, at)
            low = np.minimum.reduceat(
                np.where(reading, segs.hint[idx], _NO_HINT), at)
            live[has] = any_live
            hint[has] = np.where(any_live, low, 0)
        self.g.live[gids] = live
        self.g.hint[gids] = hint

    def drop(self, cr: CachedRegion) -> None:
        """The region's column is freed: release its slots; its granules
        are dead."""
        if cr.freed:
            return
        gids = np.arange(cr.first, cr.first + cr.n, dtype=_I64)
        resident = gids[self.g.slot[gids] >= 0]
        if len(resident):
            assert not (self.g.pins[resident] > 0).any(), (
                f"region {cr.index} freed with pinned granules")
            self._release(resident)
        self.g.live[gids] = False
        self.s.reads[cr.seg0:cr.seg0 + cr.nseg] = 0
        cr.freed = True
        self.r.freed[cr.index] = True

    def note_home_copy(self, crs: Sequence[CachedRegion], event) -> None:
        """The producer copied the refused part of each region in ``crs``
        (its granules from ``admitted`` on) into its home; ``event``
        follows the copies: one batch, which reads of those home bytes
        wait on."""
        gids = [np.arange(cr.first + cr.admitted, cr.first + cr.n, dtype=_I64)
                for cr in crs]
        if gids:
            self._new_batch(np.concatenate(gids), event)

    def home_events(self, cr: CachedRegion) -> list:
        """Events of the copies into the region's home not yet seen to end
        (the producer's, cleaning's and reclaim's)."""
        batch = self.g.batch[cr.first:cr.first + cr.n]
        return self._batch_live(np.unique(batch[batch >= 0]).tolist())

    # -- positions and ticks -------------------------------------------------

    def advance(self, node=None) -> None:
        """A pipeline starts: count the reads issued before it
        (``settle_reads``) and move the policy's position.  Under Dataflow
        the position must be the node's graph position."""
        self.settle_reads()
        self.policy.advance()
        if node is not None and self.policy.uses_position:
            assert node.position == self.policy.cur, (
                f"GranuleCache.advance: {node.name!r} has graph position "
                f"{node.position} but the counter is at {self.policy.cur}")
        if self.policy.uses_position:
            # Granules whose consumer has run are dead now.
            self._maybe_dead = True
            self.release_dead()

    def reset_position(self) -> None:
        self.policy.reset()

    def advance_tick(self) -> None:
        """The tick boundary: this tick's admissions become evictable, and
        the reads issued before it are counted (``settle_reads``)."""
        self.settle_reads()
        self.poll()
        self.tick += 1
        assert self.tick < TICK_LIMIT, "granule tick out of the key's range"

    # -- reads -------------------------------------------------------------

    def slot_addrs(self, gids) -> np.ndarray:
        """Device addresses of the slots granules ``gids`` sit in (every one
        must be resident)."""
        slots = self.g.slot[np.asarray(gids, dtype=_I64)]
        assert (slots >= 0).all(), "a granule packed into no slot"
        return self.pool.slot_ptrs(slots)

    def granules(self, regions, lo, hi) -> np.ndarray:
        """Ids of the granules over bytes ``[lo, hi)`` of each region
        (``regions``: a ``CachedRegion`` or region indices), range by range;
        an empty range has none."""
        ri, lo, hi = _span_args(regions, lo, hi)
        first = self.r.first[ri]
        g0 = first + lo // self.G
        cnt = np.where(hi > lo, first + (hi - 1) // self.G + 1 - g0, 0)
        return _expand(g0, cnt)

    def pieces(self, regions, lo, hi) -> Pieces:
        """Bytes ``[lo, hi)`` of each region (``regions``: a
        ``CachedRegion`` or region indices) as pieces: consecutive granules
        in consecutive slots of one block, or in the home; ``run`` names the
        range a piece belongs to."""
        ri, lo, hi = _span_args(regions, lo, hi)
        r = self.r
        assert not r.freed[ri].any(), "a cached region read after its free"
        assert ((0 <= lo) & (lo <= hi) & (hi <= r.nbytes[ri])).all()
        G = self.G
        first = r.first[ri]
        g0 = first + lo // G
        cnt = np.where(hi > lo, first + (hi - 1) // G + 1 - g0, 0)
        if int(cnt.sum()) <= _SMALL_READ:
            return self._pieces_small(ri, lo, hi)
        return self._pieces_vec(ri, lo, hi, first, g0, cnt)

    def _pieces_vec(self, ri, lo, hi, first, g0, cnt) -> Pieces:
        """``pieces`` by array operations over the granules."""
        r = self.r
        G = self.G
        gids = _expand(g0, cnt)
        if not len(gids):
            z = np.zeros(0, dtype=_I64)
            return Pieces(z, z, z, z, np.zeros(0, dtype=bool), [])
        slots = self.g.slot[gids]
        res = slots >= 0
        bs = self.pool.block_slots
        # A piece ends where the range, the residency or the slot run ends.
        nxt = slots[:-1] + 1
        brk = (res[1:] != res[:-1]) | (res[1:] & (
            (slots[1:] != nxt) | (nxt % bs == 0)))
        if len(ri) > 1:
            run = np.repeat(np.arange(len(ri), dtype=_I64), cnt)
            brk |= run[1:] != run[:-1]
        else:
            run = np.zeros(len(gids), dtype=_I64)
        starts = _starts(brk)
        last = np.concatenate((starts[1:], [len(gids)])) - 1
        prun = run[starts]
        pfirst = first[prun]
        g_lo = (gids[starts] - pfirst) * G
        off = np.maximum(g_lo, lo[prun])
        size = np.minimum((gids[last] - pfirst + 1) * G, hi[prun]) - off
        on_gpu = res[starts]
        if on_gpu.all():
            src = self.pool.slot_ptrs(slots[starts]) + (off - g_lo)
            return Pieces(prun, off, size, src, on_gpu, [])
        src = r.home[ri[prun]] + off
        if on_gpu.any():
            src[on_gpu] = (self.pool.slot_ptrs(slots[starts[on_gpu]])
                           + (off[on_gpu] - g_lo[on_gpu]))
        batch = self.g.batch[gids[~res]]
        batch = batch[batch >= 0]
        events = (self._batch_live(np.unique(batch).tolist())
                  if len(batch) else [])
        return Pieces(prun, off, size, src, on_gpu, events)

    def _pieces_small(self, ri, lo, hi) -> Pieces:
        """``pieces`` granule by granule, for a read of few granules (array
        operations cost more than a loop there): the same pieces, sources
        and events."""
        r = self.r
        G = self.G
        bs = self.pool.block_slots
        slot_of = self.g.slot
        batch_of = self.g.batch
        prun, off, size, slot0, g_lo, last, region = [], [], [], [], [], [], []
        batches = set()
        for i, (k, a, b) in enumerate(zip(ri.tolist(), lo.tolist(),
                                          hi.tolist())):
            if b <= a:
                continue
            first = int(r.first[k])
            new_run = True
            for gid in range(first + a // G, first + (b - 1) // G + 1):
                s = int(slot_of[gid])
                ga = (gid - first) * G
                x0, x1 = max(ga, a), min(ga + G, b)
                if s < 0:
                    bt = int(batch_of[gid])
                    if bt >= 0:
                        batches.add(bt)
                if not new_run:
                    p = last[-1]
                    if (s < 0 and p < 0) or (
                            s >= 0 and p >= 0 and s == p + 1 and s % bs):
                        size[-1] += x1 - x0
                        last[-1] = s
                        continue
                new_run = False
                prun.append(i)
                off.append(x0)
                size.append(x1 - x0)
                slot0.append(s)
                g_lo.append(ga)
                last.append(s)
                region.append(k)
        if not prun:
            z = np.zeros(0, dtype=_I64)
            return Pieces(z, z, z, z, np.zeros(0, dtype=bool), [])
        slot0 = np.array(slot0, dtype=_I64)
        off = np.array(off, dtype=_I64)
        on_gpu = slot0 >= 0
        region = np.array(region, dtype=_I64)
        src = r.home[region] + off
        if on_gpu.any():
            src[on_gpu] = (self.pool.slot_ptrs(slot0[on_gpu])
                           + (off[on_gpu] - np.array(g_lo, dtype=_I64)[on_gpu]))
        events = self._batch_live(sorted(batches)) if batches else []
        return Pieces(np.array(prun, dtype=_I64), off,
                      np.array(size, dtype=_I64), src, on_gpu, events)

    def note_read(self, regions, lo, hi) -> None:
        """A consumer read bytes ``[lo, hi)`` of each region (``regions``: a
        ``CachedRegion`` or region indices): every stream whose last record
        in a region ends inside its range has had one more read."""
        ri, lo, hi = _span_args(regions, lo, hi)
        segs = self.s
        key = segs.key[:segs.n]
        base = ri << _KEY_SHIFT
        a = np.searchsorted(key, base + lo, side="right")
        cnt = np.maximum(np.searchsorted(key, base + hi, side="right") - a, 0)
        idx = _expand(a, cnt)
        if not len(idx):
            return
        # A segment in several ranges loses one read per range, down to 0.
        reads = segs.reads
        before = reads[idx]
        np.subtract.at(reads, idx, 1)
        after = np.maximum(reads[idx], 0)
        reads[idx] = after
        ended = idx[(before > 0) & (after == 0)]
        if len(ended):
            self._maybe_dead = True
            self._refresh(np.unique(_expand(
                segs.glo[ended], segs.ghi[ended] - segs.glo[ended] + 1)))

    def defer_read(self, regions, lo, hi) -> None:
        """Reads of bytes ``[lo, hi)`` of each region have been issued:
        ``note_read`` them at the next ``settle_reads``, once the copies
        have ended.  The reader issues its copies under the tick barrier,
        and the cache settles at the next tick boundary or pipeline start,
        behind it."""
        ri, lo, hi = _span_args(regions, lo, hi)
        if len(ri):
            self._reads.append((ri.copy(), lo.copy(), hi.copy()))

    def settle_reads(self) -> None:
        """``note_read`` every deferred read (their copies have ended)."""
        if not self._reads:
            return
        reads, self._reads = self._reads, []
        ri, lo, hi = (np.concatenate(x) for x in zip(*reads))
        # A freed region's reads are moot (its segments expect none).
        self.note_read(ri, lo, hi)
        self.release_dead()

    def release_dead(self) -> int:
        """Release the slots of resident granules that turned dead: no
        pins, not admitted this tick; each slot behind its granule's
        events.  The movement ledger counts them as ``avoided_dead``
        (``ledger_bytes``).  Returns the slots released."""
        if not self._maybe_dead:
            return 0
        g = self.g
        gids = self.owner[self.owner >= 0]
        gids = gids[self.policy.dead(g, gids)]
        free = (g.pins[gids] == 0) & (g.tick[gids] < self.tick)
        # Dead granules kept for their pins or their tick are retried.
        self._maybe_dead = not free.all()
        gids = gids[free]
        if not len(gids):
            return 0
        self.poll()
        if gates.EXPERIMENT:
            movement.record("avoided_dead", len(gids),
                            *self.ledger_bytes(gids))
        self._release(gids)
        return len(gids)

    def pin(self, gids) -> None:
        gids = np.asarray(gids, dtype=_I64)
        np.add.at(self.g.pins, gids, 1)

    def unpin(self, gids) -> None:
        """Drop one pin on each granule.  Nothing waits on the reads made
        under it: no admission or release can take the slot before the
        tick barrier, which ends those reads."""
        gids = np.asarray(gids, dtype=_I64)
        assert (self.g.pins[gids] > 0).all(), "unpin of an unpinned granule"
        np.subtract.at(self.g.pins, gids, 1)
        # A dead granule this unpins can be released now.
        self._maybe_dead = True

    def replace_consume_pins(self, regions=(), lo=(), hi=(),
                             gids=None) -> np.ndarray:
        """Pin the resident granules over bytes ``[lo, hi)`` of each region
        (``regions``: a ``CachedRegion`` or region indices; none by
        default), or among ``gids`` (``granules``' result) when given, for
        the next tick and drop the previous set's pins.  Returns the
        granules pinned, once per range they are in."""
        if gids is None:
            gids = self.granules(regions, lo, hi)
        if not len(gids) and not len(self._consume):
            return gids
        gids = gids[self.g.slot[gids] >= 0]
        # A granule in two ranges is pinned twice and unpinned twice.
        self.pin(gids)
        if len(self._consume):
            self.unpin(self._consume)
        self._consume = gids
        return gids

    # -- victims -------------------------------------------------------------

    def _classes(self, gids: np.ndarray):
        dead = self.policy.dead(self.g, gids)
        cls = np.where(dead, DEAD,
                       np.where(self.g.clean[gids] | self.g.cleaning[gids],
                                CLEAN, DIRTY)).astype(_I64)
        return dead, cls

    def _order(self, gids: np.ndarray, k: int):
        """The first ``k`` of ``gids`` in reclaim order, and their classes.
        Dead granules come first; the rest follow the policy key.  On an
        equal key a clean granule goes before a dirty one, then the older
        region and, inside one, the later granule."""
        if not len(gids) or k <= 0:
            return np.zeros(0, dtype=_I64), np.zeros(0, dtype=_I64)
        dead, cls = self._classes(gids)
        key = ((~dead).astype(_I64) << _CLASS_SHIFT) | self.policy.keys(
            self.g, gids, dead)
        tb = ((cls == DIRTY).astype(_I64) << _TB_SHIFT) | self.g.tb[gids]
        if k < len(gids):
            t = np.partition(key, k - 1)[k - 1]
            below = np.flatnonzero(key < t)
            tie = np.flatnonzero(key == t)
            tie = tie[np.argsort(tb[tie], kind="stable")[:k - len(below)]]
            sel = np.concatenate([below, tie])
        else:
            sel = np.arange(len(gids))
        sel = sel[np.lexsort((tb[sel], key[sel]))]
        return gids[sel], cls[sel]

    def _evictable(self) -> np.ndarray:
        gids = self.owner[self.owner >= 0]
        g = self.g
        return gids[(g.pins[gids] == 0) & (g.tick[gids] < self.tick)]

    def victim_order(self, k: Optional[int] = None) -> np.ndarray:
        """The granules reclaim would free first, in order (all evictable
        ones without ``k``)."""
        self.poll()
        gids = self._evictable()
        return self._order(gids, len(gids) if k is None else k)[0]

    def _release(self, gids: np.ndarray) -> list:
        """Unbind ``gids`` from their slots and return the slots, each
        behind its granule's copy home if still in flight.  Returns the
        events the slots wait on."""
        g = self.g
        slots = g.slot[gids]
        waits = []
        plain = np.ones(len(gids), dtype=bool)
        batch = g.batch[gids]
        for b in np.unique(batch[batch >= 0]).tolist():
            events = self._batch_live([b])
            if events:
                sel = batch == b
                self.pool.release_slots(slots[sel], events)
                waits += events
                plain[sel] = False
        if plain.any():
            self.pool.release_slots(slots[plain])
        self.owner[slots] = -1
        g.slot[gids] = -1
        g.cleaning[gids] = False
        return waits

    def _new_batch(self, gids: np.ndarray, event) -> None:
        """``event`` follows copies of ``gids`` into their homes: reads of
        those homes wait on it, and the granules' slots are reused only
        after it."""
        self.g.batch[gids] = len(self._batch_events)
        self._batch_events.append(event)
        self._batch_fired.append(False)

    def _batch_live(self, batches) -> list:
        """The events of copies into homes (by batch) not yet seen to
        end."""
        out = []
        for b in batches:
            if self._batch_fired[b]:
                continue
            e = self._batch_events[b]
            if e.query():
                self._batch_fired[b] = True
            else:
                out.append(e)
        return out

    def _reclaim_victims(self, victims: np.ndarray, cls: np.ndarray):
        """Free ``victims``; dirty ones are copied home first.  Returns the
        events their slots wait on.  The movement ledger counts the dirty
        ones' copies (``spill_evict``) and the releases with no copy
        (``avoided_dead``, ``avoided_clean``; ``ledger_bytes``)."""
        if not len(victims):
            return []
        if gates.EXPERIMENT:
            for kind, c in (("avoided_dead", DEAD), ("avoided_clean", CLEAN)):
                sel = victims[cls == c]
                if len(sel):
                    movement.record(kind, len(sel), *self.ledger_bytes(sel))
        events = []
        dirty = victims[cls == DIRTY]
        if len(dirty):
            assert self.copy_home is not None, (
                "granule cache: a dirty granule must be copied home, but no "
                "home writer is set")
            src, dst, size = self._home_copies(dirty)
            if gates.EXPERIMENT:
                movement.record("spill_evict", len(dirty),
                                *self.ledger_bytes(dirty))
            event = self.copy_home(src, dst, size)
            self._new_batch(dirty, event)
            events += self._release(dirty)
        rest = victims[cls != DIRTY]
        if len(rest):
            events += [e for e in self._release(rest)
                       if not any(e is x for x in events)]
        return events

    def _home_copies(self, gids: np.ndarray):
        """Per granule: its slot's address, its place in the home, and its
        size."""
        g = self.g
        ri = g.region[gids]
        off = (gids - self.r.first[ri]) * self.G
        size = np.minimum(self.G, self.r.nbytes[ri] - off)
        return self.pool.slot_ptrs(g.slot[gids]), self.r.home[ri] + off, size

    def reclaim(self, nbytes: int) -> Reclaimed:
        """Free about ``nbytes`` of slots in the victim order (``_order``:
        dead granules first, then live ones by the policy key, a clean one
        before a dirty one on an equal key); dirty victims are copied home
        first.  Pinned granules and this tick's admissions are skipped.
        Frees fewer when not enough can be freed."""
        self.poll()
        k = -(-int(nbytes) // self.G)
        victims, cls = self._order(self._evictable(), k)
        events = self._reclaim_victims(victims, cls)
        counts = [int((cls == c).sum()) for c in (DEAD, CLEAN, DIRTY)]
        self._observe_reclaim((counts[1] + counts[2]) * self.G)
        return Reclaimed(slots=len(victims), nbytes=len(victims) * self.G,
                         dead=counts[0], clean=counts[1], dirty=counts[2],
                         events=events)

    # -- blocks, for the slot governor --------------------------------------

    def evictable_count(self) -> int:
        """Resident granules reclaim may take now (unpinned, not admitted
        this tick)."""
        self.poll()
        return len(self._evictable())

    def block_order(self, exclude=()) -> List[tuple]:
        """The pool's present blocks the governor may have emptied, least
        valuable first, as ``(block, resident granules)``.  A block holding
        a granule reclaim may not take (pinned, or admitted this tick) is
        left out.  A block's value is the place in the victim order of the
        most valuable granule it holds; on a tie, fewer resident granules
        first, then the lower index."""
        self.poll()
        pool = self.pool
        bs = pool.block_slots
        nb = pool.num_blocks
        ev = self._evictable()
        order = self._order(ev, len(ev))[0]
        rank = np.full(pool.num_slots, -1, dtype=_I64)
        rank[self.g.slot[order]] = np.arange(len(order), dtype=_I64)
        used = np.flatnonzero(self.owner >= 0)
        blk = used // bs
        resident = np.bincount(blk, minlength=nb)
        value = np.full(nb, -1, dtype=_I64)
        np.maximum.at(value, blk, rank[used])
        blocked = np.zeros(nb, dtype=bool)
        blocked[blk[rank[used] < 0]] = True
        skip = set(int(b) for b in exclude)
        cands = [b for b in range(nb)
                 if pool.block_is_present(b) and not blocked[b]
                 and b not in skip]
        cands.sort(key=lambda b: (int(value[b]), int(resident[b]), b))
        return [(b, int(resident[b])) for b in cands]

    def reclaim_blocks(self, blocks) -> Reclaimed:
        """Free every granule in the slots of ``blocks`` (as ``reclaim``
        does: dead and clean ones released, dirty ones copied home first,
        each slot behind its events), so the governor can lend the blocks
        once ``events`` have fired.  No granule in them may be pinned or
        admitted this tick (``block_order`` leaves such blocks out)."""
        self.poll()
        pool = self.pool
        bs = pool.block_slots
        slots = np.concatenate(
            [np.arange(b * bs, b * bs + pool._slots_in_block(b), dtype=_I64)
             for b in blocks]) if len(blocks) else np.zeros(0, dtype=_I64)
        gids = self.owner[slots]
        gids = gids[gids >= 0]
        g = self.g
        assert not ((g.pins[gids] > 0) | (g.tick[gids] >= self.tick)).any(), (
            f"reclaim_blocks: blocks {list(blocks)} hold pinned or "
            "same-tick granules")
        _, cls = self._classes(gids)
        events = self._reclaim_victims(gids, cls)
        counts = [int((cls == c).sum()) for c in (DEAD, CLEAN, DIRTY)]
        self._observe_reclaim((counts[1] + counts[2]) * self.G)
        return Reclaimed(slots=len(gids), nbytes=len(gids) * self.G,
                         dead=counts[0], clean=counts[1], dirty=counts[2],
                         events=events)

    # -- admission ---------------------------------------------------------

    def admit(self, regions: Sequence[CachedRegion]) -> Admission:
        """Offer ``regions`` (registered, not yet admitted): admit a prefix
        of each one's granules (module docstring)."""
        self.poll()
        offered = self.policy.offer_order(self.g, regions)
        free = self.pool.free_count()
        total = sum(cr.n for cr in offered)
        if total > free:
            victims, vcls = self._order(self._evictable(), total - free)
        else:
            victims = vcls = np.zeros(0, dtype=_I64)
        vi = 0
        counts = []
        for cr in offered:
            assert not cr.freed and cr.admitted == 0, (
                f"region {cr.index} offered twice or after its free")
            assert not self.policy.uses_position or (
                cr.next_use_op > self.policy.cur), (
                f"granule cache: region {cr.index} offered with its consumer "
                f"(position {cr.next_use_op}) already run")
            f = min(free, cr.n)
            free -= f
            a = f
            m = min(cr.n - f, len(victims) - vi)
            if m > 0:
                cands = np.arange(cr.first + f, cr.first + f + m, dtype=_I64)
                v = victims[vi:vi + m]
                ok = self.policy.admits(self.g, cands, v,
                                        vcls[vi:vi + m] == DEAD)
                j = m if ok.all() else int(np.argmin(ok))
                vi += j
                a += j
            counts.append(a)
        taken = victims[:vi]
        taken_cls = vcls[:vi]
        if vi:
            self._reclaim_victims(taken, taken_cls)
        if total:
            self._observe_pressure(int((taken_cls != DEAD).sum()) * self.G)
        events = []
        for cr, a in zip(offered, counts):
            if not a:
                continue
            grant = self.pool.acquire_runs(a)
            assert grant is not None, "granule cache: admission ran out of slots"
            for e in grant.events:
                if not any(e is x for x in events):
                    events.append(e)
            slots = grant.slots()
            gids = slice(cr.first, cr.first + a)
            self.g.slot[gids] = slots
            self.owner[slots] = np.arange(cr.first, cr.first + a, dtype=_I64)
            self.g.clean[gids] = False
            self.g.tick[gids] = self.tick
            if not self.g.live[gids].all():
                self._maybe_dead = True
            cr.admitted = a
        return Admission(
            regions=list(offered), admitted=counts, events=events,
            victims=tuple(int((taken_cls == c).sum())
                          for c in (DEAD, CLEAN, DIRTY)) if vi else (0, 0, 0))

    # -- cleaning ----------------------------------------------------------

    def _observe_pressure(self, live_b: int) -> None:
        """One admission's live victims (granules reclaimed with a read
        still to come, clean or dirty), in granule bytes, into the cleaning
        gate's window: it arms once the window's sum reaches its arm bytes
        and disarms once the sum is 0."""
        if self.clean_gate is None:
            return
        arm_b, window = self.clean_gate
        if len(self._gate_history) >= window:
            self._gate_bytes -= self._gate_history.popleft()
        self._gate_history.append(live_b)
        self._gate_bytes += live_b
        if not self._gate_armed and self._gate_bytes >= arm_b:
            self._gate_armed = True
        elif self._gate_armed and self._gate_bytes == 0:
            self._gate_armed = False

    def _observe_reclaim(self, live_b: int) -> None:
        """A reclaim outside admission (``reclaim``, ``reclaim_blocks``):
        its live victims count toward the cleaning gate with the latest
        admission's, so lending slots to the arena arms the gate as
        admission pressure does."""
        if self.clean_gate is None or not live_b:
            return
        if not self._gate_history:
            self._gate_history.append(0)
        self._gate_history[-1] += live_b
        self._gate_bytes += live_b
        if not self._gate_armed and self._gate_bytes >= self.clean_gate[0]:
            self._gate_armed = True

    def cleaning_open(self) -> bool:
        """Whether the cleaning gate lets cleaning run (always without a
        gate)."""
        return self.clean_gate is None or self._gate_armed

    def cleaning_targets(self) -> Cleaning:
        """The dirty granules among the front ``clean_percent`` of the live,
        unpinned resident granules in victim order (this tick's admissions
        included; clean ones in the front count as covered); none while the
        cleaning gate is closed (``cleaning_open``)."""
        if not self.cleaning_open():
            z = np.zeros(0, dtype=_I64)
            return Cleaning(z, z, z, z)
        self.poll()
        g = self.g
        gids = self.owner[self.owner >= 0]
        gids = gids[(g.pins[gids] == 0) & ~self.policy.dead(g, gids)]
        n = ceil(len(gids) * self.clean_percent / 100)
        want = ~(g.clean[gids] | g.cleaning[gids])
        if not n:
            want[:] = False
        elif n < len(gids) and want.any():
            # The front: the n smallest keys, a tie going to the smaller
            # ``tb`` (``_order`` with every granule live, clean or not).
            key = self.policy.keys(g, gids, np.zeros(len(gids), dtype=bool))
            t = np.partition(key, n - 1)[n - 1]
            front = key < t
            tie = np.flatnonzero(key == t)
            tie = tie[np.argsort(g.tb[gids[tie]], kind="stable")[
                :n - int(front.sum())]]
            front[tie] = True
            want &= front
        pick = gids[want]
        if not len(pick):
            z = np.zeros(0, dtype=_I64)
            return Cleaning(z, z, z, z)
        src, dst, size = self._home_copies(pick)
        return Cleaning(pick, src, dst, size)

    def commit_cleaning(self, cleaning: Cleaning, event) -> None:
        """The caller issued ``cleaning``'s copies; ``event`` follows them.
        The granules are clean once it fires; their slots are reused only
        after it."""
        gids = cleaning.gids
        if not len(gids):
            return
        assert (self.g.slot[gids] >= 0).all(), "cleaning a granule not in a slot"
        self.g.cleaning[gids] = True
        self._new_batch(gids, event)
        self._cleaning.append((gids, len(self._batch_events) - 1))

    def poll(self) -> None:
        """Mark the granules of every finished cleaning copy clean."""
        if not self._cleaning:
            return
        keep = []
        for gids, b in self._cleaning:
            if not self._batch_live([b]):
                self.g.clean[gids] = True
                self.g.cleaning[gids] = False
            else:
                keep.append((gids, b))
        self._cleaning = keep

    # -- teardown ------------------------------------------------------------

    def clear(self) -> None:
        """Drop every region (between queries), once the copies into their
        homes have ended (the homes go back to the pinned pool next).  Slots
        still under a copy stay pending in the pool."""
        self._reads = []
        for event in self._batch_live(range(len(self._batch_events))):
            event.synchronize()
        self.poll()
        if len(self._consume):
            self.g.pins[self._consume] = 0
            self._consume = np.zeros(0, dtype=_I64)
        for cr in self.regions:
            if not cr.freed:
                gids = np.arange(cr.first, cr.first + cr.n, dtype=_I64)
                self.g.pins[gids] = 0
                self.drop(cr)
        self.regions = []
        self.g.reset()
        self.r.reset()
        self.s.reset()
        self.owner[:] = -1
        self._cleaning = []
        self._batch_events = []
        self._batch_fired = []
        self._maybe_dead = False
        self._gate_history.clear()
        self._gate_bytes = 0
        self._gate_armed = False
        self.tick = 0
        self._seq = 0

    # -- state, for tests and telemetry ----------------------------------

    def state(self, gids) -> List[str]:
        """Each granule's state: ``dirty``, ``clean``, ``evicted`` or
        ``dead`` (a dead granule still holding a slot is ``dead``)."""
        g = self.g
        gids = np.asarray(gids, dtype=_I64)
        dead = self.policy.dead(g, gids)
        out = []
        for gid, d in zip(gids.tolist(), dead.tolist()):
            if d:
                out.append("dead")
            elif g.slot[gid] < 0:
                out.append("evicted")
            elif g.clean[gid]:
                out.append("clean")
            else:
                out.append("dirty")
        return out

    def resident_bytes(self) -> int:
        return int((self.owner >= 0).sum()) * self.G

    def ledger_bytes(self, gids: np.ndarray) -> Tuple[int, int]:
        """Granules ``gids`` as the movement ledger counts them:
        ``logical_b``, their decoded bytes (``CachedRegion.decoded``; a
        region registered without them counts its bytes), and ``wire_b``,
        their bytes (a region's last granule is shorter)."""
        g = self.g
        ri = g.region[gids]
        k = gids - self.r.first[ri]
        size = np.minimum(self.G, self.r.nbytes[ri] - k * self.G)
        decoded = size.copy()
        # Each region's granules as one slice of a sort by region, so a
        # reclaim over many regions stays O(n log n).
        order = np.argsort(ri, kind="stable")
        regions, starts = np.unique(ri[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        for r, a, b in zip(regions.tolist(), starts.tolist(), ends.tolist()):
            dec = self.regions[r].decoded
            if dec is not None:
                at = order[a:b]
                decoded[at] = dec[k[at]]
        return int(decoded.sum()), int(size.sum())
