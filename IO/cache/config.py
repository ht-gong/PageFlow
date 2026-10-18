"""The cache setting of the process: ``--buffer_strategy``, which ``main.py``
sets once (``set_global_buffer_strategy``) before any query runs.  ``cpu`` is
no GPU cache; ``lru``, ``mru`` and ``dataflow`` build the granule cache with
that policy.  ``freelist`` is the free-list baseline: no granule cache and no
cache region; intermediates stay on the GPU as arena regions, and an
allocation that misses stalls while the least recently used of them are
evicted to their homes until it fits (``IO/pages/device_regions.py``).

The four switches below are the cache-core work's
(``docs/cache-host-overhead-investigation.md``), each read at its use so
``experiments/self-verification/helpers/switch_ab.py`` can set it per query
occurrence; their defaults come from the environment, for ``tests/e2e``:

- ``CORE`` (``TQP_CACHE_CORE``, default ``native``): ``native``, the granule
  cache (``IO/cache/native_cache.py``); ``cpu``, this occurrence runs with no
  cache and no cache region, as a ``--buffer_strategy cpu`` run does: the
  boundary moves to the arena's top at its start, and the next occurrence
  with the cache moves it back.  ``main.py`` reads it at the top of each
  occurrence.
- ``FLOOR_GRANULES`` (``TQP_CACHE_FLOOR``, default 1): a cacheable output
  region of fewer bytes than this many granules is not offered to the cache
  and goes home as a plain region (``IO/codec/region_producer.py``); 0
  offers every region.
- ``PLAN_AT_COMPILE`` (``TQP_CACHE_PLAN``, on-word, default on): the reader
  derives each tick's cached spans, settle spans and granule ids when the
  tick's layout is compiled (``IO/transfer/tick_layout.py``), and binds the
  stages that read only cached regions together (one ``pieces`` call per
  tick), instead of per column per tick from the record selection.
- ``CLEANING`` (``TQP_CACHE_CLEAN``, on-word, default off): after a tick's
  admissions the producer copies the dirty granules at the front of the
  victim order home (``RegionProducer._clean``), so their eviction needs no
  copy; off, a dirty granule goes home only when it is evicted.

At SF100 (30 GiB arena, 512 MiB chunks, one 22-query pass) the floor took
0.44 s off with a 20 GB cache and 0.32 s with 2 GiB, and turning cleaning
off 0.25 s more at 2 GiB, where cleaning wrote 22 GB home to spare
14.5 GB of eviction copies (at 20 GB it never runs).  Cleaning was not
measured at larger scale factors.
"""
from __future__ import annotations

import os
from enum import Enum


class BufferStrategy(Enum):
    CPU = "cpu"
    LRU = "lru"
    MRU = "mru"
    DATAFLOW = "dataflow"
    FREELIST = "freelist"


_buffer_strategy = None


def set_global_buffer_strategy(strat: BufferStrategy) -> None:
    global _buffer_strategy
    _buffer_strategy = strat


def buffer_strategy():
    """The strategy ``set_global_buffer_strategy`` set (None before)."""
    return _buffer_strategy


CORE = os.environ.get("TQP_CACHE_CORE", "native")
# The policy of this occurrence's cache (``lru``, ``mru``, ``dataflow``), or
# None for ``--buffer_strategy``'s; ``main.py`` reads it with ``CORE``.
POLICY = os.environ.get("TQP_CACHE_POLICY") or None
FLOOR_GRANULES = int(os.environ.get("TQP_CACHE_FLOOR", "1"))
PLAN_AT_COMPILE = os.environ.get("TQP_CACHE_PLAN", "1").lower() in (
    "1", "true", "yes", "on")
CLEANING = os.environ.get("TQP_CACHE_CLEAN", "").lower() in (
    "1", "true", "yes", "on")


def freelist() -> bool:
    """The free-list baseline (``--buffer_strategy freelist``)."""
    return _buffer_strategy is BufferStrategy.FREELIST
