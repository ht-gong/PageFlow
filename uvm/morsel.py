"""Morsels: the rows one kernel call of the UVM executor takes.

A morsel is the chunk budget (``--chunk_size_mb``) over the bytes per row
of the operator's input, at least one row and at most ``MORSEL_ROWS_MAX``,
below every INT_MAX element limit of the kernels the executor calls
(``torch.sort``, ``unique``, ``nonzero``, CUB) with a factor of two to
spare.  Elementwise operators take morsels too, so their temporaries stay
bounded as ticks bound them in the streaming engine.
"""
from __future__ import annotations

import constants

MORSEL_ROWS_MAX = 1 << 30


def morsel_rows(row_bytes: int, chunk_b: int = None) -> int:
    """Rows per morsel of an input of ``row_bytes`` bytes a row."""
    if chunk_b is None:
        chunk_b = constants.get_chunk_size_b()
    return max(1, min(MORSEL_ROWS_MAX, chunk_b // max(1, row_bytes)))


def morsels(rows: int, row_bytes: int, chunk_b: int = None):
    """``(start, stop)`` of each morsel of ``rows`` rows, in order; none
    for no rows."""
    step = morsel_rows(row_bytes, chunk_b)
    for start in range(0, rows, step):
        yield start, min(rows, start + step)
