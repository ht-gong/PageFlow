"""Variable-block pinned-host allocator.

``MemoryPool`` is the variable-block first-fit allocator backing
:class:`PinnedMemory`; the host budget (about 0.69 GB per scale factor,
``main._pinned_capacity_gb``) and the heterogeneous sizes of scan batches
and region chunks make uniform slotting impractical here.

The backing host memory is split into one or more *segments*, each its own
pinned, device-mapped region whose pages all sit on the GPU's NUMA node
(``TQPlib.tqpmemory.pinned_host_alloc_tensor``: an ``mmap`` bound to the
node with ``mbind``, pinned with ``cudaHostRegister``).  With
``TQP_NUMA_BIND`` off (``utility.numa.NUMA_BIND``) the mapping is not bound,
and its pages sit on the node of the thread that pins them.  One mapped region
for the whole pool has failed above ~500–590 GiB in a live CUDA context,
so segments stay well below that.

The first-fit free list spans all segments via a single global block
index, but allocations never straddle a segment boundary: the free list
is seeded with one run per segment and coalescing never merges across a
boundary.  So every allocation lives within a single mapped region and
``cudaHostGetDevicePointer`` (``zero_copy_reader.device_alias``, per region
or page the zero-copy readers read) keeps returning a valid device alias.

Under ``TQP_EXPERIMENT_TELEMETRY`` the pool counts the blocks each *tag*
holds (an allocation's tag, ``"intermediate"`` unless its caller names one:
the scan's region producer names ``"scan"``) and their high-water mark, per
tag and in total, for the perf report's ``residency`` section
(``IO/telemetry/residency.py``): ``usage``, ``reset_peaks``.  ``free_all``
keeps the peaks.
"""
import torch
import constants
import TQPlib.tqpmemory as cmp
from IO.telemetry import gates
from utility.tensor_utils import dtype_size
from utility.logger import message_logger
from typing_extensions import List

# Per-segment size for the pinned host pool.  Must comfortably exceed the
# largest single ``get_storage`` request (a 256 MiB scan batch, see
# IO/data_scanner.py, or a region producer's RAW tick output, which grows
# with the chunk budget) and stay well under the ~500–590 GiB
# monolithic-mapped ceiling.  Pools smaller than this are a single segment.
PINNED_SEGMENT_GB = 128


class PinnedPoolExhausted(MemoryError):
    """No free run of the pinned host pool fits a request."""


class MemoryPool:
    COALESCE_THRESHOLD = 5

    def __init__(self, capacity_gb: int, block_size_mb: int, seg_size_gb=None):
        self.pool_size_bytes = int(capacity_gb * 1024**3)
        self.block_size_bytes = int(block_size_mb * 1024**2)

        if self.pool_size_bytes < self.block_size_bytes:
            raise ValueError("Pool size must be larger than the block size.")

        self._num_blocks = self.pool_size_bytes // self.block_size_bytes

        # ── Segment layout ───────────────────────────────────────────────
        # Blocks are partitioned into fixed-size segments of ``_seg_blocks``
        # blocks each (the last segment may be shorter).  ``seg_size_gb=None``
        # means a single segment spanning the whole pool.
        if seg_size_gb is None:
            self._seg_blocks = self._num_blocks
        else:
            seg_bytes = int(seg_size_gb * 1024**3)
            self._seg_blocks = max(seg_bytes // self.block_size_bytes, 1)
        self._num_segments = max(
            (self._num_blocks + self._seg_blocks - 1) // self._seg_blocks, 1)

        self._free = self._seed_free()
        self._allocs = {}
        self._free_times = 0
        # Residency by tag, kept under TQP_EXPERIMENT_TELEMETRY only: each
        # allocation's tag, and the blocks each tag holds and has held at
        # most since ``reset_peaks``, also over all tags.
        self._tags = {}
        self._in_use = {}
        self._peak = {}
        self._in_use_total = 0
        self._peak_total = 0
        # Cached size (in blocks) of the largest contiguous free run.
        # ``None`` = stale, recompute on next ``free_space`` call.
        # Mutated in lock-step with ``_free`` to avoid the O(F) scan.
        self._largest_free_blocks: int | None = max(
            (length for _, length in self._free), default=0)

    def _seed_free(self):
        """One free run per segment so a run never spans a boundary."""
        free = []
        for i in range(self._num_segments):
            start = i * self._seg_blocks
            length = min(self._seg_blocks, self._num_blocks - start)
            if length > 0:
                free.append((start, length))
        return free

    def _storage_slice(self, start_block, actual_size_bytes):
        """Host tensor slice for a block run.  Overridden by
        :class:`PinnedMemory` to index the owning segment's tensor."""
        raise NotImplementedError

    def get_storage(self, actual_size_bytes, tag="intermediate"):
        """First-fit allocation of ``actual_size_bytes``: ``(ptr, storage)``.
        ``tag`` names what it holds, for the residency counts.  Raises
        :class:`PinnedPoolExhausted` when no free run fits."""
        got = self.try_get_storage(actual_size_bytes, tag)
        if got is None:
            gib = 1024**3
            raise PinnedPoolExhausted(
                f"pinned host pool exhausted: no free run fits a "
                f"{actual_size_bytes / gib:.3f} GiB request (pool "
                f"{self.pool_size_bytes / gib:.0f} GiB, largest free run "
                f"{self.free_space() / gib:.3f} GiB); raise --pinned_pool_gb")
        return got

    def try_get_storage(self, actual_size_bytes, tag="intermediate"):
        """As :meth:`get_storage`, but None when no free run fits, for
        allocations a caller can do without (the region producer's
        look-ahead chunk).

        Runs freed since the last :meth:`coalesce` may still sit unmerged
        beside their neighbours, so a miss merges them and looks once more
        before it reports no fit.  With none pending the list is already
        merged and a miss is final."""
        alloc_blocks = max((actual_size_bytes + self.block_size_bytes - 1) // self.block_size_bytes, 1)
        got = self._take_first_fit(alloc_blocks, actual_size_bytes)
        if got is None and self._free_times:
            self.coalesce()
            got = self._take_first_fit(alloc_blocks, actual_size_bytes)
        if got is not None and gates.EXPERIMENT:
            self._tags[got[0]] = tag
            self._count(tag, alloc_blocks)
        return got

    # -- residency (TQP_EXPERIMENT_TELEMETRY) ------------------------------

    def _count(self, tag, blocks):
        """``blocks`` more (or, negative, fewer) held by ``tag``; raises the
        peaks."""
        n = self._in_use.get(tag, 0) + blocks
        self._in_use[tag] = n
        if n > self._peak.get(tag, 0):
            self._peak[tag] = n
        self._in_use_total += blocks
        if self._in_use_total > self._peak_total:
            self._peak_total = self._in_use_total

    def reset_peaks(self):
        """Rebaseline every peak to the blocks held now (a query's timer
        starts)."""
        self._peak = dict(self._in_use)
        self._peak_total = self._in_use_total

    def usage(self) -> dict:
        """The residency counts in bytes: ``in_use_b`` and ``peak_b`` per
        tag, ``in_use_total_b`` and ``peak_total_b``, with the pool's
        ``capacity_b`` and ``block_b``.  Counted only under
        TQP_EXPERIMENT_TELEMETRY."""
        b = self.block_size_bytes
        return {"capacity_b": self._num_blocks * b, "block_b": b,
                "in_use_b": {t: n * b for t, n in self._in_use.items()},
                "peak_b": {t: n * b for t, n in self._peak.items()},
                "in_use_total_b": self._in_use_total * b,
                "peak_total_b": self._peak_total * b}

    def _take_first_fit(self, alloc_blocks, actual_size_bytes):
        """Allocate ``alloc_blocks`` from the first free run that holds them:
        ``(ptr, storage)``, or None when no run in the list is long enough."""
        for i, (start, length) in enumerate(self._free):
            if length >= alloc_blocks:
                ptr = start
                if length == alloc_blocks:
                    self._free.pop(i)
                else:
                    self._free[i] = (start + alloc_blocks, length - alloc_blocks)
                self._allocs[ptr] = alloc_blocks
                # Invalidate cache only if we just shrunk what was the
                # largest run; otherwise the prior max is still correct.
                if (self._largest_free_blocks is not None
                        and length == self._largest_free_blocks):
                    self._largest_free_blocks = None
                # Each free run lies wholly within one segment and
                # ``alloc_blocks <= length``, so this slice never straddles a
                # segment boundary.
                return ptr, self._storage_slice(start, actual_size_bytes)
        return None

    def free_all(self):
        self._free = self._seed_free()
        self._free_times = 0
        self._allocs.clear()
        if gates.EXPERIMENT:
            self._tags.clear()
            self._in_use = {t: 0 for t in self._in_use}
            self._in_use_total = 0
        self._largest_free_blocks = max(
            (length for _, length in self._free), default=0)

    def free(self, ptr: int):
        length = self._allocs.pop(ptr, None)
        if length is None:
            raise ValueError(f"{ptr} not found in pool")
        if gates.EXPERIMENT:
            tag = self._tags.pop(ptr, None)
            if tag is not None:
                self._count(tag, -length)

        self._free.append((ptr, length))
        # Optimistic update — coalesce may grow the largest run further but
        # never shrink it below ``length``.  If the cache was already
        # invalidated, leave it ``None`` and let ``free_space`` recompute.
        if self._largest_free_blocks is not None:
            if length > self._largest_free_blocks:
                self._largest_free_blocks = length

        # Merge once the frees since the last merge exceed the threshold;
        # ``coalesce`` restarts the count.
        self._free_times += 1
        if self._free_times > self.COALESCE_THRESHOLD:
            self.coalesce()

    def shrink(self, ptr: int, actual_size_bytes: int) -> None:
        """Keep the first ``actual_size_bytes`` of allocation ``ptr`` (at
        least one block) and return the blocks after them to the pool, as a
        free does.  The bytes kept and their tensor views stay valid."""
        length = self._allocs.get(ptr)
        if length is None:
            raise ValueError(f"{ptr} not found in pool")
        keep = max((actual_size_bytes + self.block_size_bytes - 1)
                   // self.block_size_bytes, 1)
        if keep >= length:
            return
        self._allocs[ptr] = keep
        tail = length - keep
        if gates.EXPERIMENT and ptr in self._tags:
            self._count(self._tags[ptr], -tail)
        self._free.append((ptr + keep, tail))
        if (self._largest_free_blocks is not None
                and tail > self._largest_free_blocks):
            self._largest_free_blocks = tail
        self._free_times += 1
        if self._free_times > self.COALESCE_THRESHOLD:
            self.coalesce()

    def coalesce(self):
        """Sort the free list by start block and merge runs that touch within
        one segment.  Restarts the count of frees since the last merge."""
        self._free_times = 0
        if not self._free:
            return
        self._free.sort(key=lambda x: x[0])
        coalesced = []
        cur_start, cur_len = self._free[0]
        for st, ln in self._free[1:]:
            # Merge only physically-adjacent runs that live in the SAME
            # segment — never across a boundary, or an allocation could span
            # two mapped regions and break cudaHostGetDevicePointer on it.
            same_seg = (cur_start // self._seg_blocks) == (st // self._seg_blocks)
            if cur_start + cur_len == st and same_seg:
                cur_len += ln
            else:
                coalesced.append((cur_start, cur_len))
                cur_start, cur_len = st, ln
        coalesced.append((cur_start, cur_len))
        self._free = coalesced
        # Coalescing reshapes the run distribution; recompute on demand.
        self._largest_free_blocks = None

    def free_space(self) -> int:
        """Return the largest contiguous free region in bytes.  Merges the
        runs freed since the last :meth:`coalesce` first, so that two
        touching runs count as one region."""
        if self._free_times:
            self.coalesce()
        if self._largest_free_blocks is None:
            self._largest_free_blocks = max(
                (length for _, length in self._free), default=0)
        return self._largest_free_blocks * self.block_size_bytes


class PinnedMemory(MemoryPool):
    def __init__(self, capacity_gb: int, block_size_mb: int,
                 seg_size_gb=PINNED_SEGMENT_GB, numa_node=None):
        super().__init__(capacity_gb, block_size_mb, seg_size_gb=seg_size_gb)
        # Every page of the pool sits on ``numa_node`` (default: GPU 0's
        # node), bound there unless TQP_NUMA_BIND is off.  A bound pool
        # cannot spill to another node, so the node must be able to hold it.
        from utility import numa
        self.numa_node = numa.gpu_numa_node(0) if numa_node is None else numa_node
        self.numa_bound = numa.NUMA_BIND
        if self.numa_bound:
            numa.check_node_capacity(self.numa_node,
                                     self._num_blocks * self.block_size_bytes)
        # One mapped region per segment; each returned tensor owns its memory
        # and frees it when all references (including page views/slices) are
        # gone.  Node -1 pins a region unbound.
        self._segments = []
        for i in range(self._num_segments):
            start = i * self._seg_blocks
            length = min(self._seg_blocks, self._num_blocks - start)
            self._segments.append(cmp.pinned_host_alloc_tensor(
                length * self.block_size_bytes,
                self.numa_node if self.numa_bound else -1))

    def _storage_slice(self, start_block, actual_size_bytes):
        seg = start_block // self._seg_blocks
        local_byte = (start_block % self._seg_blocks) * self.block_size_bytes
        return self._segments[seg][local_byte: local_byte + actual_size_bytes]
