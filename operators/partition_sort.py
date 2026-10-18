"""The join sort pipe's hash partition on the fused kernel
(``operators/kernels/partition_sort.cu``, the ``TQPlib.tqpkernels`` module).

One call per sort-pipe tick puts the tick's transfer set into the padded
per-bucket layout the region producer reads
(``IO/codec/batched_compress.aligned_partition_layout``: each non-empty bucket
starts on a whole ``lcm(row bytes, 512)``-byte unit): a row's bucket is the top
log2(B) bits of mix64(key) (bucket 0 for every row when B is 1), rows keep
their source order inside a bucket, and the global index (``goff`` + the row's
pre-mask position) is computed in the kernel.  Padding rows are left
unwritten.

The kernel runs three launches (K1 per-tile histogram, K2 bucket scan, K3
staged scatter; five or six with a mask prologue) and one host wait: K2 writes
the bucket counts into a pinned buffer through its UVA alias, and the call
waits for them while K3 runs.

Under a mask a column arrives raw, since the sort pipe folds the mask and
hands over the mask's kept rows as ``n_valid``, or compacted by a zero-copy
read.  Below a selectivity
threshold the kernel first collects the kept rows' pre-mask ids
(select-first): ``SELECT_FIRST_RAW_BELOW`` when a column is raw (its rows are
then gathered), ``SELECT_FIRST_BELOW`` when every column is compacted.  Above
it, the kernel walks the pre-mask rows: raw columns take the mask as a
predicate (the fold), compacted ones are read at each tile's offset
(compacted-input).
"""
from __future__ import annotations

from typing import List, Optional

import torch

import TQPlib.tqpkernels as tqpkernels
from IO.pipeline.common import PartitionedTensor
from operators.partitions import JOIN_B_MAX as _B_MAX
from utility.tensor_utils import row_bytes

# Kept rows per mask row below which a masked tick with compacted columns
# collects its kept rows' ids first (select-first) rather than walking the
# pre-mask rows (compacted-input).  The two cross at 0.50-0.55 on H100 for
# 16.8M and 33.5M pre-mask rows and one or two 8-B columns.
SELECT_FIRST_BELOW = 0.5
# The same for a masked tick with a raw column: select-first against the
# mask as a predicate (the fold), which cross at 0.27-0.31 there.
SELECT_FIRST_RAW_BELOW = 0.3

# The kernel's limits (partition_sort.cu); its BMAX, _B_MAX, is the join's B
# cap (imported above from operators/partitions.py).
_MAX_COLS = 6
_MAX_WIDTHS = 4          # row widths, the gidx's 8 B included
_MAX_ROWS = (1 << 31) - 1

# Where the kernel writes a call's counts, padded totals and, after them, the
# kept rows a mask prologue found: one buffer for the process, since a call
# has read it back before it returns.
_host: Optional[torch.Tensor] = None


def _host_buffer() -> torch.Tensor:
    global _host
    if _host is None:
        _host = torch.empty(_B_MAX + _MAX_WIDTHS + 1, dtype=torch.int64,
                            pin_memory=True)
    return _host


def partition(cols: List[torch.Tensor], key_pos: int,
              mask: Optional[torch.Tensor], goff: int, B: int,
              n_valid: Optional[int] = None) -> List[PartitionedTensor]:
    """``cols``: the tick's transfer set in output order, the int64 join key
    at ``key_pos``; with ``mask`` (the pre-mask mask), each column either
    compacted or raw (the mask's rows), and ``n_valid`` the mask's kept rows
    (needed when a column is raw); ``goff`` is the tick's first global row.
    Returns one ``PartitionedTensor`` per column, then the gidx's, each in
    the padded layout (its ``bucket_starts()`` give each bucket's first
    row).  Raises ValueError for a tick outside the kernel's limits."""
    key = cols[key_pos]
    n_pre = mask.shape[0] if mask is not None else key.shape[0]
    # The kernel's row-bytes classes, in the order it numbers them: the
    # columns' widths, then the gidx's 8 B.
    widths = []
    for t in cols:
        rb = row_bytes(t)
        if rb not in widths:
            widths.append(rb)
    if 8 not in widths:
        widths.append(8)
    if (not 1 <= B <= _B_MAX or B & (B - 1) or len(cols) > _MAX_COLS
            or len(widths) > _MAX_WIDTHS or key.dtype != torch.int64
            or key.dim() != 1 or n_pre > _MAX_ROWS
            or any(t.dim() not in (1, 2) for t in cols)
            or (mask is not None and (mask.dim() != 1
                                      or mask.element_size() != 1))):
        raise ValueError(
            f"partition_sort: the tick is outside the kernel's limits (B={B}, "
            f"{len(cols)} columns of widths {widths}, a {key.dtype} key, "
            f"{n_pre} rows; at most B={_B_MAX}, {_MAX_COLS} columns, "
            f"{_MAX_WIDTHS} widths with the gidx's 8 B, a 1-D int64 key, "
            f"{_MAX_ROWS} rows, 1-D or 2-D columns, a 1-D 1-byte mask)")
    select_first = False
    if mask is None:
        comp = [0] * len(cols)
        n_valid = n_pre
    else:
        comp = [int(t.shape[0] != n_pre) for t in cols]
        lengths = {t.shape[0] for t, c in zip(cols, comp) if c}
        if len(lengths) > 1:
            raise ValueError(
                f"partition_sort: compacted columns of {sorted(lengths)} rows")
        if lengths:
            rows = lengths.pop()
            if n_valid is not None and n_valid != rows:
                raise RuntimeError(
                    f"partition_sort: compacted columns hold {rows} rows, "
                    f"the mask keeps {n_valid}")
            n_valid = rows
        elif n_valid is None:
            raise ValueError("partition_sort: raw columns under a mask need "
                             "the mask's kept rows")
        mask = mask.contiguous()
        below = SELECT_FIRST_BELOW if all(comp) else SELECT_FIRST_RAW_BELOW
        select_first = n_valid < below * n_pre
    host = _host_buffer()
    outs = tqpkernels.partition_sort([t.contiguous() for t in cols], comp,
                                     key_pos, mask, n_valid, goff, B,
                                     select_first, host)
    counts = host[:B].tolist()
    totals = host[B:B + len(widths)].tolist()
    kept = sum(counts)
    # The kept rows the kernel found: a mask prologue's count, else K1's.  A
    # caller's count that disagrees fails here; the kernel neither read nor
    # wrote past what that count sized.
    found = int(host[B + _MAX_WIDTHS])
    if found < 0:
        found = kept
    if found != n_valid or kept != n_valid:
        raise RuntimeError(
            f"partition_sort: the mask keeps {found} rows and {kept} were "
            f"partitioned; the caller gave {n_valid}")
    # Each output, the gidx's last, narrowed to its class's padded total.
    return [PartitionedTensor(t[:totals[widths.index(row_bytes(t))]], counts)
            for t in outs]
