"""The sweep's cost function, shared by ``profiling/sweep.py`` and
``experiments/paper_scripts/figures/make_regret_fig.py``.

One measured round trip of a column through the engine's compress path has
four legs, each timed with CUDA events:

  compress_s    GPU compress into pages
  d2h_s         the compressed page bytes, device -> pinned host
  h2d_s         the same bytes, pinned host -> device
  decompress_s  GPU decompress back to the column

**Cost = the sum of the four.**  RAW's cost is the measured D2H + H2D of the
raw bytes.  Every leg is kept in ``sweep_per_column.csv`` so the formula can
change here without re-running the sweep.
"""
from __future__ import annotations

from typing import Mapping

LEGS = ("compress_s", "d2h_s", "h2d_s", "decompress_s")
RAW_LEGS = ("raw_d2h_s", "raw_h2d_s")


def cost_s(row: Mapping[str, object]) -> float:
    """Round-trip seconds of one (column, run, codec, chunk) measurement."""
    return sum(float(row[k]) for k in LEGS)


def raw_cost_s(row: Mapping[str, object]) -> float:
    """Seconds to move the column's raw bytes D2H and back H2D."""
    return sum(float(row[k]) for k in RAW_LEGS)
