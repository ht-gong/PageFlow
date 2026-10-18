"""The residency report: the most a query's timed run held in each memory
tier, for the perf report's ``residency`` section (``main.py``, under
``TQP_EXPERIMENT_TELEMETRY``).

``begin`` runs where the query's timer starts, right after the arena's
``reset_peak``: it rebaselines the pinned pool's and the granule cache's
peaks and notes what each tier holds then (the scan's pinned bytes, say).
``report`` reads the tiers once the query's operators have run, before the
pools are released (or where a failed query stops).  A tier's peak covers
the output step too: the timer leaves the main plan's out, the peaks do not.

============== ==============================================================
section        what
============== ==============================================================
``gpu_arena``  the allocator's operator space, the arena below the cache
               region (``memory/cuda_memory_allocator.cc``): ``capacity_b``,
               ``at_start_b`` (allocated when the timer starts) and
               ``peak_b`` (``get_peak_bytes``), in whole arena blocks.  cuDF
               allocates from it too (``utility/rmm_arena.py``); the cache
               region is never an allocation.
``host_pool``  the pinned host pool (``IO/pages/pinned_mem.py``):
               ``capacity_b``, ``block_b``, and per tag (``intermediate``:
               the chunks the pipes' regions are carved from, the unwritten
               rest of each open chunk included; ``scan``: the scanned tables)
               ``at_start_b`` and ``peak_b``; ``peak_total_b`` over all tags.
               Whole blocks.
``slot_pool``  the granule cache's slots, the arena's cache region
               (``IO/cache/native_cache.py``), None without a cache:
               ``capacity_b`` and ``slab_b``, the region's bytes when the
               timer starts (the slot governor may lend blocks of it later,
               each back by its pipe's end); ``slot_b``, a slot (an arena
               block); ``at_start_used_b`` and ``peak_used_b``, the slots
               in use then and at most since.
============== ==============================================================

``host_pool`` is None when the run has no pinned pool (``main.py --uvm``).
``timed`` is False when the query failed before its timer started (then
``at_start_b`` is None and the peaks are the pools' since their last reset).
The module imports nothing from ``IO/``: the pools and the cache are passed
in.
"""
from __future__ import annotations

# What the tiers held when the timer started (``begin``); None before.
_start = None


def reset() -> None:
    """Forget the last query's start (top of each query)."""
    global _start
    _start = None


def begin(host_pool, cache) -> None:
    """The timer starts: rebaseline the pinned pool's and the granule
    cache's peaks (the caller has reset the arena's) and note what each tier
    holds.  ``cache``: the query's ``NativeGranuleCache``, None without
    one."""
    global _start
    import TQPlib.tqpmemory as cmp
    if host_pool is not None:
        host_pool.reset_peaks()
    if cache is not None:
        cache.reset_peak()
    _start = {
        "arena_b": cmp.get_arena_stats(False)["allocated_b"],
        "host_b": (None if host_pool is None
                   else host_pool.usage()["in_use_b"]),
        "slab_b": None if cache is None else cache.region.nbytes,
        "slots_b": None if cache is None else cache.used_bytes(),
    }


def report(host_pool, cache) -> dict:
    """The perf report's ``residency`` section (module docstring)."""
    import TQPlib.tqpmemory as cmp
    start = _start or {}
    out = {
        "timed": _start is not None,
        "gpu_arena": {
            "capacity_b": cmp.get_arena_stats(False)["capacity_b"],
            "at_start_b": start.get("arena_b"),
            "peak_b": cmp.get_peak_bytes(),
        },
        "host_pool": None,
        "slot_pool": None,
    }
    if host_pool is not None:
        host = host_pool.usage()
        out["host_pool"] = {
            "capacity_b": host["capacity_b"],
            "block_b": host["block_b"],
            "at_start_b": start.get("host_b"),
            "peak_b": host["peak_b"],
            "peak_total_b": host["peak_total_b"],
        }
    if cache is not None:
        out["slot_pool"] = {
            "capacity_b": start.get("slab_b"),
            "slab_b": start.get("slab_b"),
            "slot_b": cache.G,
            "at_start_used_b": start.get("slots_b"),
            "peak_used_b": cache.peak_used_bytes(),
        }
    return out
