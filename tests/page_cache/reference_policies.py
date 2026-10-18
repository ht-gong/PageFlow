"""Eviction and admission policies of the reference granule cache
(``reference_cache.py``), as key functions; the C++ core
(``IO/cache/granule_core.cc``) implements the same keys and rules.

The reference cache keeps one row per granule
in numpy arrays (``Granules``).  A policy reads those arrays for a set of
granule ids and returns one int64 key per granule: smaller keys are evicted
first.  The cache puts dead granules above the key (they go first) and,
on an equal key, a clean granule before a dirty one; it orders with one
partition over the keys, never a Python object per granule.  The policies:

- **LRU**: the oldest admitted region first (FIFO; reads never promote).
- **MRU**: the newest admitted region first (LIFO).
- **Dataflow**: by graph distance, from the granule's region's consumer
  position ``nu`` and the current position ``cur``.  A granule is *dead*
  when every stream in it has had its last read or its consumer has run
  (``nu < cur``); among live ones, the furthest consumer goes first, then
  the latest stream (the highest ``use_order_hint``, a partition sort's
  bucket), then the newest admission; a granule whose consumer is running
  (``nu == cur``) goes after every future one.  A granule is keyed by its
  most valuable live stream: the lowest hint among them.

Ties on the key go to the older region first and, inside a region, to the
later granule first (``tb``), so evicted bytes stay a suffix of each region
and resident bytes a prefix.

Admission compares a granule with the victim reclaim would free for it.
LRU and MRU admit whenever there is one.  Dataflow's rules: a dead victim is always displaced; a live one only by a granule with
a sooner consumer, or on a tie by one with a strictly lower hint; otherwise
the incumbent stays (tie-keep: every granule of a partition sort's output
ties on ``nu``, and displacing on ties churns).
"""
from __future__ import annotations

import numpy as np

_I64 = np.int64

# Key layout (int64, smaller = evicted first).  The cache adds the class
# above bit 60, so a policy key stays below 2**60.
NU_BITS = 16
HINT_BITS = 16
TICK_BITS = 27
SEQ_LIMIT = 1 << 60
TICK_LIMIT = 1 << TICK_BITS
_NU_MAX = (1 << NU_BITS) - 1
_HINT_MAX = (1 << HINT_BITS) - 1
_TICK_MAX = (1 << TICK_BITS) - 1


class GranulePolicy:
    """LRU's and MRU's shared behaviour; subclasses set ``newest_first``."""

    name = "base"
    newest_first = False
    uses_position = False

    def __init__(self):
        self.cur = -1

    def advance(self) -> None:
        """A pipeline starts (the Dataflow position moves)."""
        self.cur += 1

    def reset(self) -> None:
        """A new plan starts."""
        self.cur = -1

    def dead(self, g, gids: np.ndarray) -> np.ndarray:
        """Granules no read will come for."""
        return ~g.live[gids]

    def keys(self, g, gids: np.ndarray, dead: np.ndarray) -> np.ndarray:
        seq = g.seq[gids]
        if self.newest_first:
            return (SEQ_LIMIT - 1) - seq
        return seq.copy()

    def admits(self, g, cands: np.ndarray, victims: np.ndarray,
               victim_dead: np.ndarray) -> np.ndarray:
        """Per pair: does candidate ``cands[i]`` displace ``victims[i]``?"""
        return np.ones(len(cands), dtype=bool)

    def check_region(self, next_use_op: int, hints, seq: int) -> None:
        """A new region's key fields fit their bits (the keys do not check
        them again)."""
        assert 0 <= seq < SEQ_LIMIT, f"region order {seq} out of the key's range"

    def offer_order(self, g, regions):
        """The order regions offered together are admitted in."""
        return list(regions)


class GranuleLRU(GranulePolicy):
    name = "lru"


class GranuleMRU(GranulePolicy):
    name = "mru"
    newest_first = True


class GranuleDataflow(GranulePolicy):
    name = "dataflow"
    uses_position = True

    def dead(self, g, gids: np.ndarray) -> np.ndarray:
        return ~g.live[gids] | (g.nu[gids] < self.cur)

    def keys(self, g, gids: np.ndarray, dead: np.ndarray) -> np.ndarray:
        # The fields' ranges are checked where they are set
        # (``check_region``; the cache bounds the tick).
        nu = g.nu[gids]
        hint = g.hint[gids]
        newest = _TICK_MAX - g.tick[gids]
        consumed_now = (nu == self.cur).astype(_I64)
        live_key = ((consumed_now << (NU_BITS + HINT_BITS + TICK_BITS))
                    | ((_NU_MAX - nu) << (HINT_BITS + TICK_BITS))
                    | ((_HINT_MAX - hint) << TICK_BITS)
                    | newest)
        return np.where(dead, newest, live_key)

    def check_region(self, next_use_op, hints, seq):
        super().check_region(next_use_op, hints, seq)
        assert 0 <= next_use_op <= _NU_MAX, (
            f"consumer position {next_use_op} out of the key's range")
        assert 0 <= min(hints) and max(hints) <= _HINT_MAX, (
            "use_order_hint out of the key's range")

    def admits(self, g, cands, victims, victim_dead):
        c_nu, v_nu = g.nu[cands], g.nu[victims]
        assert (c_nu > self.cur).all(), (
            f"granule cache: a granule offered with its consumer already run "
            f"(cur={self.cur})")
        sooner = c_nu < v_nu
        tie_wins = (c_nu == v_nu) & (g.hint[cands] < g.hint[victims])
        return victim_dead | sooner | tie_wins

    def offer_order(self, g, regions):
        # Most valuable first: nearest consumer, then the lowest hint.
        return sorted(regions, key=lambda r: (r.next_use_op, r.min_hint))


def make_policy(name: str) -> GranulePolicy:
    """The policy behind a ``--buffer_strategy`` name."""
    return {"lru": GranuleLRU, "mru": GranuleMRU,
            "dataflow": GranuleDataflow}[name.lower()]()
