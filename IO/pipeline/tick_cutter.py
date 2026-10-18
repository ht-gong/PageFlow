"""The tick cutter: how a pipe's input rows split into ticks.

Every record holds a power-of-two number of rows, cut from the start of its
row space (a scan's row 0) or of its producer slice (one tick's bucket)
(``IO/codec/region_producer.py``), so the record boundaries of the columns a
pipe reads nest, and the boundaries every column shares are known from the
columns' record tables.  The cutter packs whole *quanta* into each tick, up
to the chunk budget:

- **Points.** The candidate tick ends are the record boundaries every
  representation of every input column shares (``common_bounds``).  Between
  two of them lies a *piece*: in a row space cut from row 0, one record of
  the input with the most rows per record.
- **Quanta.** A pipe's quantum is the largest ``record_rows`` among the
  widths it reads and the masks it writes (1 byte per row), so the masks it
  writes, cut from each tick start, keep their table's boundaries.  The
  points kept are those a whole number of quanta past the last *origin*: row
  0, or the end of a piece shorter than a full record (a producer slice's
  last record), where the next slice's records start again
  (``quantum_points``).
- **Packing.** Each tick takes the most quanta whose raw bytes fit the
  budget, so the budget rounds down to whole quanta; the join core's bucket
  packing is the same greedy rule (``pack``).  Every tick then reads whole
  records of every column: its first and last rows are record boundaries,
  so the reader selects whole records and trims nothing
  (``variable.TableVariable.whole_records``).
- **Fallback.** A pipe whose quantum (its largest piece between two kept
  points) costs more raw bytes than its budget, or whose inputs share no
  record boundaries (one is not a record table of regions), is cut by
  rows: ``budget // row bytes`` rows per tick, the reader trimming at most
  one record per column at each end.  A pipe smaller than one quantum is
  one piece.  Those ticks are *fallback* ticks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from IO.codec import region_producer

_I64 = np.int64


@dataclass
class Ticks:
    """A pipe's ticks over rows ``[start, n)``: each tick's end row
    (``ends``, ascending, the last ``n``), whether it is a whole number of
    quanta (``whole``; False: a fallback tick, cut by rows), and whether
    the budget closed it (``full``: the next quantum, or row, would not have
    fit)."""

    ends: np.ndarray
    whole: np.ndarray
    full: np.ndarray

    def __len__(self) -> int:
        return len(self.ends)


def pack(points: np.ndarray, limit: int) -> np.ndarray:
    """Greedy packing of the pieces between consecutive ``points``
    (ascending, the first the start): each group takes the most pieces
    whose span is at most ``limit``, and at least one.  Returns the index in
    ``points`` of each group's end.  Empty pieces join the group before
    them."""
    ends = []
    i, last = 0, len(points) - 1
    while i < last:
        j = int(np.searchsorted(points, points[i] + limit, side="right")) - 1
        if j <= i:
            j = i + 1
        ends.append(j)
        i = j
    return np.asarray(ends, dtype=_I64)


def cut_rows(start: int, n: int, row_bytes: int, budget_b: int) -> Ticks:
    """Ticks of ``budget_b // row_bytes`` rows (at least one) over
    ``[start, n)``, the last shorter: the cut by rows."""
    per = max(1, budget_b // row_bytes) if row_bytes > 0 else max(1, n - start)
    ends = np.arange(start + per, n, per, dtype=_I64)
    ends = np.append(ends, _I64(n)) if n > start else ends
    full = np.diff(np.r_[start, ends]) == per
    return Ticks(ends, np.zeros(len(ends), dtype=bool), full)


def rep_bounds(rep) -> Optional[np.ndarray]:
    """The record boundaries of one representation in its own rows, with 0
    and its row count; None when it is not a record table of regions."""
    fn = getattr(rep, "record_bounds", None)
    return fn() if fn is not None else None


def common_bounds(columns: Sequence) -> Optional[np.ndarray]:
    """The record boundaries every representation of every column
    (``CombinedColumnRep``) shares, with 0 and the row count; None when one
    is not a record table of regions."""
    out = None
    for ccr in columns:
        for rep in ccr.reps:
            b = rep_bounds(rep)
            if b is None:
                return None
            out = b if out is None else np.intersect1d(out, b,
                                                       assume_unique=True)
    return out


def input_record_rows(columns: Sequence) -> int:
    """The most rows per record among the columns' representations."""
    rows = 1
    for ccr in columns:
        for rep in ccr.reps:
            t = rep.record_table() if getattr(rep, "table_backed", False) \
                else None
            if t is not None:
                rows = max(rows, region_producer.record_rows(
                    int(t.row_bytes)))
    return rows


def quantum_points(bounds: np.ndarray, piece_rows: int,
                   quantum: int) -> np.ndarray:
    """The points of ``bounds`` a whole number of ``quantum`` rows past the
    last origin (the first point, or the end of a piece shorter than
    ``piece_rows``), each such end, and the last point."""
    if quantum <= piece_rows:
        return bounds
    lengths = np.diff(bounds)
    origin_at = np.ones(len(bounds), dtype=bool)
    origin_at[1:] = lengths < piece_rows
    origin = np.maximum.accumulate(np.where(origin_at, bounds, 0))
    keep = origin_at | ((bounds - origin) % quantum == 0)
    keep[-1] = True
    return bounds[keep]


def cut(points: Optional[np.ndarray], start: int, n: int, row_bytes: int,
        budget_b: int, floor_b: int = 0) -> Ticks:
    """The pipe's ticks over rows ``[start, n)`` (module docstring): whole
    pieces between ``points`` (the quantum points), each tick the most that
    fit ``budget_b`` raw bytes; by rows when ``points`` is None or one piece
    costs more than the budget.  Under a sizing search's class floor
    ``floor_b``, a budget whose whole quanta would fall under the floor is
    raised to the next whole quantum
    (``adaptive.sizing_controller.whole_quanta``, the quantum being the
    largest piece)."""
    if n <= start:
        return Ticks(np.zeros(0, dtype=_I64), np.zeros(0, dtype=bool),
                     np.zeros(0, dtype=bool))
    if points is None or row_bytes <= 0:
        return cut_rows(start, n, row_bytes, budget_b)
    pts = points[(points > start) & (points < n)]
    pts = np.concatenate(([start], pts, [n])).astype(_I64)
    if floor_b:
        from adaptive.sizing_controller import whole_quanta
        budget_b = max(budget_b, whole_quanta(
            budget_b, max_piece(points) * row_bytes, floor_b))
    max_rows = budget_b // row_bytes
    if int(np.diff(pts).max()) > max_rows:
        return cut_rows(start, n, row_bytes, budget_b)
    ends = pack(pts, max_rows)
    whole = np.ones(len(ends), dtype=bool)
    # A start off the points (after ticks cut by rows) begins inside a
    # piece: that tick does not start on a record boundary.
    whole[0] = bool(np.isin(start, points))
    full = ends < len(pts) - 1
    return Ticks(pts[ends], whole, full)


def max_piece(points: Optional[np.ndarray]) -> int:
    """The most rows between two consecutive points (0 without points)."""
    if points is None or len(points) < 2:
        return 0
    return int(np.diff(points).max())


def slice_column(ccr, starts: np.ndarray, ends: np.ndarray):
    """``ccr``'s slices ``[starts[t], ends[t])``, one per tick: a tick whose
    ends are record boundaries of every representation reads whole records
    (``whole_records``), the others ``get_sliced_rep``.  Returns the slices and, per tick, whether every
    representation reads whole records."""
    from IO.pages.combined_column_rep import CombinedColumnRep
    k = len(starts)
    per_rep = [rep.whole_records(starts, ends)
               if getattr(rep, "table_backed", False) else None
               for rep in ccr.reps]
    exact = np.full(k, bool(per_rep), dtype=bool)
    for sl in per_rep:
        exact &= (np.array([s is not None for s in sl], dtype=bool)
                  if sl is not None else False)
    out = []
    for t in range(k):
        s, e = int(starts[t]), int(ends[t])
        reps = []
        for rep, sl in zip(ccr.reps, per_rep):
            if exact[t] and s < e:
                reps.append(sl[t])
            else:
                reps.append(rep.get_sliced_rep(slice(s, e)))
        out.append(CombinedColumnRep(reps, tag=ccr.tag))
    return out, exact
