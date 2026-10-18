from __future__ import annotations
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
