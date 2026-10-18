"""Positional RAW-vs-candidate decisions over one pipe's decision menu.

The *decision* half of the compression plane: which subset of a pipe's
searchable columns to actually compress, given that the menu already carries
each column's single candidate.  This module owns no bytes and launches no
kernels; it sees no tags, plans, or column names — everything it consumes
arrives on the published :class:`~adaptive.decision_menu.DecisionMenu`, and
everything it emits is a positional pick vector returned through the
per-tick observer (the only actuation channel).

Decision universe: exactly the **two-entry input ranges** — the pipeline
publishes singletons for untyped/RAW-policy columns and for
operator-output-fed inputs (single materialized rep), so an illegal choice is
inexpressible here by construction (likewise, a within-range pick can never
invent a codec).

Benefit ranking: decision columns are ordered by the transfer bytes
compression removes, ``benefit = raw_bytes - stored_bytes[candidate]``
(≡ ``bytes·(1 - 1/ratio)`` — actual per-rep byte counts carried on the menu,
superseding the profiled ratio table).  Benefit-ordering keeps each leg
monotone in *k*, which is what makes ``f(k)`` unimodal.

CPU-only, torch-free — unit-testable without CUDA.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from adaptive.decision_menu import DecisionMenu, RAW


class CompressionController:
    """Per-pipe subset decisions over the published menu.

    Pure and stateless beyond the menu reference: every product is derived
    from the menu's ranges and byte counts.
    """

    def __init__(self, menu: DecisionMenu):
        self._menu = menu
        self._inputs = menu.inputs

    # -- decision universe -------------------------------------------------

    def decision_indices(self) -> List[int]:
        """Positions of the searchable input columns — exactly the two-entry
        ranges; ascending, deterministic."""
        return [i for i, c in enumerate(self._inputs) if c.is_decision]

    def candidate(self, idx: int) -> Optional[str]:
        """Input column ``idx``'s single non-RAW entry, or ``None``."""
        col = self._inputs[idx]
        return col.candidate if col.is_decision else None

    # -- benefit ranking ---------------------------------------------------

    def benefit(self, idx: int) -> float:
        """Transfer bytes compression removes from column ``idx``'s leg:
        ``raw_bytes - stored_bytes[candidate]``.  A candidate with no stored
        byte count (or one that doesn't shrink) ranks zero."""
        col = self._inputs[idx]
        cand = self.candidate(idx)
        if cand is None:
            return 0.0
        stored = col.stored_bytes.get(cand)
        if stored is None:
            return 0.0
        return float(max(0, col.raw_bytes - stored))

    def ranked_indices(self) -> List[int]:
        """Decision columns, highest benefit first; deterministic stable
        tiebreak on the position itself."""
        return sorted(self.decision_indices(),
                      key=lambda i: (-self.benefit(i), i))

    def decision_count(self) -> int:
        """N — the search dimension (|decision columns|)."""
        return len(self.decision_indices())

    # -- pick vectors (the observer return value) --------------------------

    def pick_vector(self, k: int) -> Tuple[str, ...]:
        """The full-length input-pick vector for subset size ``k``: the
        top-``k`` benefit-ranked decision columns pick their candidate, the
        remaining decision columns pick RAW, and every singleton column picks
        its only entry.  Every element is within its column's range by
        construction."""
        top = set(self.ranked_indices()[:k])
        out: List[str] = []
        for i, col in enumerate(self._inputs):
            if col.is_decision:
                out.append(col.candidate if i in top else RAW)
            else:
                out.append(col.choices[0])
        return tuple(out)

    # -- the write leg (read-and-write) ------------------------------------

    def write_decision_indices(self) -> List[int]:
        """Positions of the searchable OUTPUT columns — the two-entry
        output ranges.  Slot-group non-base slots publish locked singletons,
        so this is one position per logical output column."""
        return [i for i, c in enumerate(self._menu.outputs) if c.is_decision]

    def write_decision_count(self) -> int:
        return len(self.write_decision_indices())

    def write_reads(self) -> List[int]:
        """Each output slot's consumer reads (the menu's ``reads``),
        positional: the write leg's readback-price input."""
        return [c.reads for c in self._menu.outputs]

    def ranked_write_indices(self, weights=None) -> List[int]:
        """Write decision columns, highest benefit first.  ``weights`` is
        the in-run measured per-slot byte view (the menu carries no output
        bytes at construction); positional, indexed by output slot.
        ``None`` or missing entries rank zero (position tiebreak)."""
        def w(i):
            if weights is None or i >= len(weights):
                return 0.0
            return float(weights[i] or 0)
        return sorted(self.write_decision_indices(), key=lambda i: (-w(i), i))

    def write_pick_vector(self, k: int,
                          ranked: Optional[List[int]] = None
                          ) -> Tuple[str, ...]:
        """The full-length output-pick vector for write subset size ``k``:
        the top-``k`` of ``ranked`` (a frozen :meth:`ranked_write_indices`
        order — the search MUST rank once and hold it, or ``f(k)`` loses
        unimodality) pick their candidate, the remaining write decisions
        pick RAW, singletons pick their only entry."""
        if ranked is None:
            ranked = self.ranked_write_indices()
        top = set(ranked[:k])
        out: List[str] = []
        for i, col in enumerate(self._menu.outputs):
            if col.is_decision:
                out.append(col.candidate if i in top else RAW)
            else:
                out.append(col.choices[0])
        return tuple(out)
