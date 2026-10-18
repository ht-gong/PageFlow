"""Spillable GPU regions: the free-list baseline (``--buffer_strategy freelist``).

In the free-list baseline a pipe's outputs stay on the GPU.  Each region is
one arena allocation (whole arena blocks, first fit, like every operator
temporary: the two share the arena's one free list), with a pinned home of
its size reserved beside it and not written (``Region.home``).  The region
producer registers each one once its writes are queued (``register``).

When an allocation misses after the arena has folded in every pending free,
the arena synchronises the device and the allocation stalls in a loop: it
calls ``evict_one`` (installed as its miss hook by ``install``), which
evicts the least recently used of the registered regions no read holds
(``pin``) — registration and each read make a region the most recent; the
regions made recent by one read are ordered arbitrarily among themselves —
copying it home on one stream, waiting for the copy, and making it a host
region whose GPU tensor is dropped; the arena folds the free in (it
coalesces with its free neighbours) and tries the allocation again.  The
memory is freed as the search goes: the loop stops at the first fit, or
when no unpinned region is left, and an allocation that still misses then
raises the arena's OOM, which fails the query.  An evicted region stays on
the host: its readers copy it H2D from then on.  An eviction frees the
region's blocks only when nothing else holds its GPU tensor (the producer
copies an output that is a view into a larger allocation, but an operator
may still hold an exact one); the arena then just goes on to the next
region, and ``arena_stats`` (which counts every unpinned region free) is
optimistic.

A region's records are offsets into it, so a move leaves its record table
valid.  What caches a region's address keys it on ``epoch()``, which each
eviction bumps (``tick_layout._region_bases``,
``RecordTable.uva_segments``).  A read that holds an address across an
allocation pins the regions it reads until its copies or kernel are queued:
a tick's bind until its launch (``tick_layout.bind_tick_sources``,
``BoundTickPlan.unpin``) and a gather's source for the gather
(``rearrange_pipeline``).

Single-threaded like the engine: the hook runs on the thread that
allocates, inside whatever torch call allocated.

``TQP_FREELIST_SQUEEZE_GB`` (a correctness stress, never a timing setting)
holds that many GiB of the arena through each query's timed run
(``begin_query`` to ``release_all``), so the query runs in a smaller arena
while its scan, before the timer, has the whole one.  The arena peaks the
``[freelist]`` line reports include the squeeze.

The reserve (``--freelist_reserve_gb``, ``install``'s ``reserve_b``) keeps
the arena's top blocks free through each query's timed run, for the
allocations eviction cannot serve: one that finds every registered region
pinned, or the free space split by pinned regions and live temporaries.
``begin_query`` moves the arena's region boundary (``memory/``,
``region_move``; the free list has no cache region, so the boundary is the
reserve's) down over them, and no allocation lands above it.  An
allocation that still misses once ``evict_one`` has no unpinned region left
draws on the reserve (``_draw``): the boundary moves up by what the request
lacks beyond the free run just below it (all of the reserve when that is
not enough), those blocks coalesce with that run, and the arena tries
again.  While the reserve is short it refills (``_refill``) from the free
run just below the boundary at each ``register`` and ``unpin``, so it is
whole again once the blocks a draw gave out are freed or evicted.
``release_all`` gives it back at the query's end, so the next scan has the
whole arena.  The arena's stats (``arena_stats``, the pass-2 admission's)
describe the operator space below the boundary: the reserve is never
counted free.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Iterable, List

import torch

from IO.pages.region import DEVICE, Region

# region_id -> Region, every spillable region not yet evicted or freed,
# least recently used first (``register`` and ``pin`` move one to the end).
_regions: "OrderedDict[int, Region]" = OrderedDict()
_epoch = 0
_stream = None
_installed = False
# Since the last ``take_stats``: misses that evicted something, the
# regions and bytes moved (stored, and logical: what their records decode
# to), host seconds in evict_one, the most bytes
# registered at once, and the most bytes a miss found pinned (left where
# they were); the reserve's draws, the bytes they gave out and took back,
# and the least it held.
_stats = dict(evictions=0, regions=0, bytes=0, logical_bytes=0, seconds=0.0,
              peak_live_b=0,
              peak_live_arena_b=0, pinned_at_miss_b=0, reserve_draws=0,
              reserve_drawn_b=0, reserve_refilled_b=0, reserve_min_b=0)
_live_b = 0
# region_id -> logical bytes, for the registered compressed regions (a RAW
# region's are its stored bytes).
_logical: dict = {}
# The registered regions' arena footprint: each rounded up to whole arena
# blocks (``install``'s ``block_b``), what an eviction gives back.
_live_arena_b = 0
_block_b = 1
_SQUEEZE_B = int(float(os.environ.get("TQP_FREELIST_SQUEEZE_GB") or 0)
                 * 1024 ** 3)
_squeeze = None
# The reserve (module docstring): the blocks it should hold (0: none), the
# blocks it holds (the arena's region; nothing else moves the boundary under
# the free list), and whether a query's timed run is holding it.
_reserve_blocks = 0
_held = 0
_reserve_on = False


def armed() -> bool:
    """True once ``install`` has made ``evict_one`` the arena's miss hook."""
    return _installed


def install(block_b: int, reserve_b: int = 0) -> None:
    """Make ``evict_one`` the arena's miss hook (``TQPlib.tqpmemory``);
    ``block_b`` is the arena's block size, ``reserve_b`` the reserve's
    size (whole blocks, rounded down; 0 for none)."""
    global _installed, _stream, _block_b, _reserve_blocks
    import TQPlib.tqpmemory as cmp
    _block_b = int(block_b)
    _reserve_blocks = int(reserve_b) // _block_b
    if _stream is None:
        _stream = torch.cuda.Stream()
    cmp.set_miss_hook(evict_one)
    _installed = True


def uninstall() -> None:
    """Remove the miss hook, give the reserve back and forget every region
    (``reset``)."""
    global _installed, _reserve_blocks
    import TQPlib.tqpmemory as cmp
    cmp.set_miss_hook(None)
    _installed = False
    _give_back()
    _reserve_blocks = 0
    reset()


def reserve_b() -> int:
    """Bytes the reserve holds now."""
    return _held * _block_b


def _refill() -> int:
    """Take back what the reserve lacks, as far as the free run just below
    the boundary goes; returns the blocks taken."""
    global _held
    import TQPlib.tqpmemory as cmp
    short = _reserve_blocks - _held
    if short <= 0:
        return 0
    k = min(short, int(cmp.get_arena_stats(True)["top_free_run_b"]) // _block_b)
    if k <= 0 or not cmp.region_move(-k):
        return 0
    _held += k
    _stats["reserve_refilled_b"] += k * _block_b
    return k


def _draw(need_bytes: int) -> int:
    """Give an allocation of ``need_bytes`` that eviction cannot serve the
    reserve's blocks: what it lacks beyond the free run just below the
    boundary, or all of them when they are not enough.  Returns the bytes
    given, 0 when the reserve is empty."""
    global _held
    import TQPlib.tqpmemory as cmp
    if not _held:
        return 0
    need = -(-int(need_bytes) // _block_b)
    top = int(cmp.get_arena_stats(True)["top_free_run_b"]) // _block_b
    k = need - top
    if k <= 0 or k > _held:
        k = _held
    cmp.region_move(k)
    _held -= k
    _stats["reserve_draws"] += 1
    _stats["reserve_drawn_b"] += k * _block_b
    _stats["reserve_min_b"] = min(_stats["reserve_min_b"], _held * _block_b)
    return k * _block_b


def _give_back() -> None:
    """The reserve's blocks back to the operator space (none hold data)."""
    global _held, _reserve_on
    import TQPlib.tqpmemory as cmp
    if _held:
        cmp.region_move(_held)
        _held = 0
    _reserve_on = False


def reset() -> int:
    """Forget every registered region (between queries: their tables are
    freed or abandoned); returns how many were still registered."""
    global _live_b, _live_arena_b
    left = len(_regions)
    for r in _regions.values():
        r.pins = 0
    _regions.clear()
    _logical.clear()
    _live_b = _live_arena_b = 0
    return left


def begin_query() -> None:
    """Where a query's timer starts: take the reserve (as much of it as the
    free run at the arena's top holds; the rest refills later), zero the
    counters, and under ``TQP_FREELIST_SQUEEZE_GB`` take that much of the
    arena until ``release_all``."""
    global _squeeze, _reserve_on
    if _reserve_blocks:
        # The scan's frees complete, so its blocks are free.
        torch.cuda.synchronize()
        _refill()
        _reserve_on = True
    take_stats()
    if _SQUEEZE_B and _squeeze is None:
        _squeeze = torch.empty(_SQUEEZE_B, dtype=torch.uint8, device="cuda")


def release_all() -> int:
    """Free the GPU bytes of every region still registered (the tables a
    query left behind, between queries) and forget them, the squeeze, and
    give the reserve back; returns how many regions there were."""
    global _squeeze
    _squeeze = None
    left = list(_regions.values())
    for r in left:
        r.free(None, None)
    reset()
    _give_back()
    return len(left)


def epoch() -> int:
    """Bumped by each eviction that moved a region: an address cached at
    another epoch may be stale."""
    return _epoch


def _arena_b(region: Region) -> int:
    return -(-max(region.nbytes, 1) // _block_b) * _block_b


def register(region: Region, logical_b: int = None) -> None:
    """A spillable region (``Region.home`` set) whose writes are queued;
    ``logical_b``, what a compressed region's records decode to (None: its
    stored bytes)."""
    global _live_b, _live_arena_b
    assert region.spillable and region.location == DEVICE, (
        f"{region} is not a spillable GPU region")
    _regions[region.region_id] = region
    if logical_b is not None:
        _logical[region.region_id] = int(logical_b)
    _live_b += region.nbytes
    _live_arena_b += _arena_b(region)
    if _live_b > _stats["peak_live_b"]:
        _stats["peak_live_b"] = _live_b
    if _live_arena_b > _stats["peak_live_arena_b"]:
        _stats["peak_live_arena_b"] = _live_arena_b
    if _reserve_on and _held < _reserve_blocks:
        _refill()


def unregister(region: Region) -> None:
    """A spillable region freed before any eviction moved it."""
    global _live_b, _live_arena_b
    _logical.pop(region.region_id, None)
    if _regions.pop(region.region_id, None) is not None:
        _live_b -= region.nbytes
        _live_arena_b -= _arena_b(region)


def live_bytes() -> int:
    """Bytes of the registered (GPU-resident, spillable) regions."""
    return _live_b


def pin(regions: Iterable[Region]) -> List[Region]:
    """Hold the spillable ones of ``regions`` on the GPU until ``unpin``, and
    make them the most recently used (a pin is a read); returns them (what
    to hand ``unpin``)."""
    held = [r for r in regions if r.home is not None]
    for r in held:
        r.pins += 1
        if r.region_id in _regions:
            _regions.move_to_end(r.region_id)
    return held


def unpin(held: Iterable[Region]) -> None:
    for r in held:
        if r.pins > 0:
            r.pins -= 1
    if _reserve_on and _held < _reserve_blocks:
        _refill()


@contextmanager
def pinned(regions: Iterable[Region]):
    held = pin(regions)
    try:
        yield held
    finally:
        unpin(held)


def evict_one(need_bytes: int = 0, first: bool = True) -> int:
    """The arena's miss hook, with the device synchronised and an
    allocation of ``need_bytes`` stalled (module docstring): copy the least
    recently used unpinned region home, wait, and make it a host region;
    returns its bytes.  With no unpinned region left, the allocation draws
    on the reserve (``_draw``); 0 when that is empty too.  The arena tries
    the allocation after each call and calls again until it fits; ``first``
    marks a miss's first call."""
    global _epoch, _live_b, _live_arena_b
    if first:
        pinned_b = sum(r.nbytes for r in _regions.values() if r.pins)
        _stats["pinned_at_miss_b"] = max(_stats["pinned_at_miss_b"],
                                         pinned_b)
    victim = next((r for r in _regions.values() if r.pins == 0), None)
    if victim is None:
        return _draw(need_bytes)
    t0 = time.perf_counter()
    with torch.cuda.stream(_stream):
        victim.spill_to_home()
    _stream.synchronize()
    del _regions[victim.region_id]
    _live_b -= victim.nbytes
    _live_arena_b -= _arena_b(victim)
    victim.settle_on_host()
    _epoch += 1
    if first:
        _stats["evictions"] += 1
    _stats["regions"] += 1
    _stats["bytes"] += victim.nbytes
    _stats["logical_bytes"] += _logical.pop(victim.region_id, victim.nbytes)
    _stats["seconds"] += time.perf_counter() - t0
    return victim.nbytes


def spillable_ptrs() -> List[int]:
    """The GPU addresses of the regions an eviction would move now."""
    return [r.data_ptr() for r in _regions.values() if r.pins == 0]


def arena_stats(drain_completed_first: bool = True) -> dict:
    """``TQPlib.tqpmemory.get_arena_stats`` with the spillable regions
    counted free (``free_b``, ``largest_free_run_b`` as an eviction of them
    would leave the arena), for the decisions that admit an allocation by
    the arena's free space: the arena evicts them for it on a miss.  Adds
    ``spillable_b``."""
    import TQPlib.tqpmemory as cmp
    stats = cmp.get_arena_stats(drain_completed_first)
    ptrs = spillable_ptrs()
    stats["spillable_b"] = sum(r.nbytes for r in _regions.values()
                               if r.pins == 0)
    if ptrs:
        stats.update(cmp.stats_if_freed(ptrs))
    return stats


def take_stats() -> dict:
    """The eviction counters since the last call, with the arena's miss
    counters (``miss_*``) and the reserve's size (``reserve_b``), the
    counters reset."""
    import TQPlib.tqpmemory as cmp
    out = dict(_stats)
    for k, v in cmp.miss_stats(True).items():
        out[f"miss_{k}"] = v
    out["reserve_b"] = _reserve_blocks * _block_b
    _stats.update(evictions=0, regions=0, bytes=0, logical_bytes=0,
                  seconds=0.0,
                  peak_live_b=_live_b, peak_live_arena_b=_live_arena_b,
                  pinned_at_miss_b=0, reserve_draws=0, reserve_drawn_b=0,
                  reserve_refilled_b=0, reserve_min_b=_held * _block_b)
    return out
