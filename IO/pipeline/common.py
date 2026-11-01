from __future__ import annotations
import os
from math import gcd
import torch
import constants
from IO.codec.batched_compress import aligned_partition_layout
from utility.tensor_utils import row_bytes
import IO.transfer.data_exchange as data_exchange
from enum import Enum
from IO.transfer.in_stage import DirectIn
from adaptive.instage_optimization import (
    get_fixed_compression, find_rep_by_algorithm,
)
from IO.pages.combined_column_rep import CombinedColumnRep
import TQPlib.tqpmemory as cmp


def _is_raw_algo(algo) -> bool:
    return not algo or algo == "RAW"


class PartitionedTensor:
    """Operator-output marker: one fat tensor holding the B buckets of one
    output instance, ``partition_row_counts[b]`` rows each, in bucket
    order, in the padded layout of
    ``batched_compress.aligned_partition_layout``.
    ``VortexPipeline._execute_post_comp`` hands it to the region producer as
    one instance of B streams (hashjoin's partitioned ``sort_kernel``, and
    a partitioned aggregation's pass 1, ``aggregate._partition_outputs``)."""
    __slots__ = ("tensor", "partition_row_counts")

    def __init__(self, tensor, partition_row_counts):
        self.tensor = tensor
        self.partition_row_counts = list(partition_row_counts)

    def bucket_starts(self):
        """Each bucket's first row in ``tensor`` (an empty bucket's is where
        the next one starts): the padded layout's, each non-empty bucket a
        whole number of ``lcm(row bytes, COMPRESSION_ALIGNMENT)``-byte
        units."""
        row_b = row_bytes(self.tensor)
        align = constants.COMPRESSION_ALIGNMENT
        _, starts, _ = aligned_partition_layout(
            self.partition_row_counts, row_b, row_b * align // gcd(row_b, align))
        return [start // row_b for start in starts]


class PipeStep(str, Enum):
    PLAN = "metadata calc"
    APPLY_OP = "apply op"
    SYNC = "transfer sync"

    def __str__(self):
        return self.value


def col_to_gpu(ccr: CombinedColumnRep) -> torch.Tensor:
    rep = find_rep_by_algorithm(ccr, get_fixed_compression())
    in_stage = DirectIn(rep)
    # One-stage utility path: compile explicitly before entering exchange so
    # its geometry does not fall back onto the launch-critical path.
    DirectIn.install_compiled_tick_layout([in_stage])
    ex = data_exchange.exchange()
    # Its cached granules stay where the bind finds them through the
    # launch's allocations (a lend moves only unpinned ones).
    from IO.transfer.tick_layout import pin_cached_reads
    unpin = pin_cached_reads(rep)
    try:
        ex.launch([in_stage])
    finally:
        unpin()
    ex.sync()
    in_stage.decompress()
    torch.cuda.synchronize()
    return in_stage.gpu_col.data()


def decode_windows_off() -> bool:
    """``TQP_NO_DECODE_WINDOWS=1``: the gather's index is decoded whole
    (``col_to_gpu``), never window by window (``col_to_gpu_windowed``).
    The sort's payload windows (``operators.sort``) are not affected."""
    return (os.environ.get("TQP_NO_DECODE_WINDOWS") or "") not in ("", "0")


def col_to_gpu_windowed(ccr: CombinedColumnRep) -> torch.Tensor:
    """The whole column on the GPU, as ``col_to_gpu`` gives it, but a
    compressed column larger than the chunk budget is read and decoded a
    window of whole records at a time (``get_sliced_rep``, then
    ``col_to_gpu``), each window copied into its rows of the one output
    buffer and freed before the next.  ``col_to_gpu`` holds the whole
    compressed column (its agg_buf) beside the whole decoded one (its
    shared output), both contiguous; this holds the decoded column and one
    window.  A RAW column, or one within the budget, is ``col_to_gpu``'s
    (a RAW read is a single buffer anyway)."""
    from adaptive.sizing_controller import window_rows
    from IO.codec.region_producer import record_rows
    rep = find_rep_by_algorithm(ccr, get_fixed_compression())
    n = ccr.col_length()
    row_b = max(1, rep.col_width() * rep.orig_elem_size())
    chunk_b = constants.get_chunk_size_b()
    if (decode_windows_off() or not rep.is_compressed or n * row_b <= chunk_b
            or not getattr(rep, "table_backed", False)):
        return col_to_gpu(ccr)
    w = window_rows(chunk_b, row_b, record_rows(row_b))
    # The output first, before any window's buffers: the one large
    # contiguous allocation takes (or lends next to) the run at the cache
    # region's boundary while nothing of this read sits there.  Its shape
    # is col_to_gpu's: rows, or rows x width.
    table = rep.record_table()
    width = int(table.width)
    out = torch.empty((n,) if width == 1 else (n, width),
                      dtype=table.stored_dtype, device="cuda")
    for lo in range(0, n, w):
        part = col_to_gpu(ccr.get_sliced_rep(slice(lo, min(lo + w, n))))
        out[lo:lo + part.shape[0]].copy_(
            part.reshape((-1,) + tuple(out.shape[1:])))
        del part
    return out


class _WindowPeakTracker:
    """Composes per-iteration arena-peak brackets with the pipe-level
    ``peak_gpu_bytes`` record.

    The allocator exposes ONE windowed high-water counter
    (``cmp.reset_window_peak``/``get_window_peak_bytes``), and the rearrange
    pipe brackets it whole (reset at pipe entry, read at pipe exit).  The
    gather-window sizing search additionally needs per-gather-iteration
    transients (``iter_peak − live_at_entry``), so while a sizing session is
    live the counter is rebaselined per iteration and a Python-side running
    max preserves the whole-pipe high-water mark for the final perf record.
    With no session the tracker is never advanced and ``final_peak()``
    reads the untouched pipe-level bracket — bit-identical to the
    pre-sizing behavior.
    """

    def __init__(self):
        self._running = 0

    def iter_begin(self) -> int:
        """Fold the window so far into the running max, rebaseline, and
        return the live occupancy at the new baseline (``live_at_entry``)."""
        self._running = max(self._running, cmp.get_window_peak_bytes())
        cmp.reset_window_peak()
        return cmp.get_window_peak_bytes()

    def iter_end(self) -> int:
        """The iteration's arena peak; folded into the running max."""
        p = cmp.get_window_peak_bytes()
        self._running = max(self._running, p)
        return p

    def final_peak(self) -> int:
        return max(self._running, cmp.get_window_peak_bytes())
