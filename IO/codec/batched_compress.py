"""Layout constants of the region producer's batched compress.

``_MAX_OUT_BUF_BYTES`` caps the worst-case output of one batched nvCOMP
manager call (``IO.codec.region_producer``); :func:`aligned_partition_layout`
is the padded bucket layout a partition-sort output is written in, each
bucket starting on a whole ``lcm(row bytes, 512)`` unit, so every bucket's
records start 512-aligned.
"""
from __future__ import annotations

from typing import List, Sequence

# Cap peak worst-case compressed output per batched manager compress call.
# Above this, the producer splits a tick's records into pieces, one call and
# one slab each.  This bounds BOTH the staging slab and the
# manager's retained scratch (nvCOMP scratch scales with the batch size —
# an uncapped ~4 GB sort tick demanded a single 5.5 GB scratch block at
# SF100 Q7).  256 MiB picked to keep the slab small enough to fit in any
# contiguous hole of the arena under typical SF100-SF300 workloads.  HARD memory rule — do not raise casually.
_MAX_OUT_BUF_BYTES = 256 * 1024 * 1024  # 256 MiB


# ---------------------------------------------------------------------------
# Layout helpers (CPU)
# ---------------------------------------------------------------------------

def aligned_partition_layout(partition_row_counts: Sequence[int],
                             row_bytes: int,
                             align_unit_b: int):
    """Compute a padded partition layout for the partitioned-compress path.

    Each partition starts at a byte offset that is a multiple of
    ``align_unit_b`` (which itself is a multiple of nvCOMP's 512-byte
    alignment). Padding rows fill the gap between a partition's last
    actual row and the next partition's aligned start; nvCOMP never
    reads them because records are built over each partition's actual
    bytes only.

    Returns:
        padded_row_counts: per-partition row totals after rounding up.
        partition_byte_starts: byte offset of each partition in the buffer.
        total_padded_rows: total rows in the padded buffer.
    """
    assert align_unit_b % row_bytes == 0, (
        f"align_unit_b={align_unit_b} must be a multiple of row_bytes={row_bytes}"
    )
    rows_per_align = align_unit_b // row_bytes
    padded_counts: List[int] = []
    starts: List[int] = []
    cum_rows = 0
    for c in partition_row_counts:
        starts.append(cum_rows * row_bytes)
        if c == 0:
            padded_counts.append(0)
            continue
        padded = ((c + rows_per_align - 1) // rows_per_align) * rows_per_align
        padded_counts.append(padded)
        cum_rows += padded
    return padded_counts, starts, cum_rows
