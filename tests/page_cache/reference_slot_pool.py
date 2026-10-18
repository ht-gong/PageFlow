"""The reference model's slot pool: the granule cache's GPU slots as they
were before the cache moved into the arena's region.

The reference cache (``reference_cache.py``) allocates its slots here, and
the slot governor's lease-return tests drive its blocks by hand.  One slab,
partitioned at init into ``num_slots`` equal slots of ``slot_size`` bytes,
handed out in runs of consecutive slots (``SlotRunPool``).

Block-backed elasticity.  The slab is stored as ``num_blocks`` block
tensors of ``block_slots`` slots each; slot indices are stable across the
pool's lifetime via ``slot_idx -> (block, offset)``.  A block holding no
live slot can be *released* — its tensor returns to the shared arena,
immediately allocatable by compute — and later *restored*.

With ``block_slots=None`` (the default) the pool is a single block.

Under ``TQP_EXPERIMENT_TELEMETRY`` the pool keeps the high-water mark of
its slots in use since ``reset_peak`` (``peak_used_bytes``), for the perf
report's ``residency`` section (``IO/telemetry/residency.py``).
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np
import torch

from IO.telemetry import gates


def _fired(events) -> bool:
    """True once every event in ``events`` has completed."""
    return all(e.query() for e in events)


class SlotGrant:
    """Slots handed out by :meth:`SlotRunPool.acquire_runs`: ``runs`` as
    ``(start, count)`` pairs in the order they were taken, and ``events``,
    the copies still reading or writing any of them.  A caller that writes
    into the slots orders its writes after every event."""

    __slots__ = ("runs", "events")

    def __init__(self, runs, events):
        self.runs = runs
        self.events = events

    def slots(self):
        """Every slot granted, run by run."""
        if len(self.runs) == 1:
            s, n = self.runs[0]
            return np.arange(s, s + n, dtype=np.int64)
        if not self.runs:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([np.arange(s, s + n, dtype=np.int64)
                               for s, n in self.runs])


class SlotRunPool:
    """Slots of their own size, handed out in runs of consecutive slots.

    The reference cache's pool: one slot holds
    one granule, a ``slot_size_b`` piece of a cached region, and a region's
    granules take consecutive slots where a free run allows.  The free list
    is a list of ``(start, count)`` runs, allocated first fit and merged
    after every ``COALESCE_THRESHOLD`` frees, as the pinned pool's
    (``IO/pages/pinned_mem.py``).  A run never spans two blocks, because each
    block is its own tensor.

    A slot is reused only after every copy reading or writing it has
    completed.  A release names the events of the copies still in flight on
    the slots (``events``); such slots wait in a pending list and return to
    the free list once every event has fired.  :meth:`acquire_runs` hands
    out free slots first, then pending ones, whose events the grant lists
    for the writer to order its writes after.

    The block surface (``empty_block_indices``, ``release_block``,
    ``restore_block``) is the slot governor's; a block with pending slots is
    not empty.

    Parameters
    ----------
    capacity_b : int
        Total bytes to reserve on the GPU.  Trimmed to a multiple of
        ``slot_size_b`` (any slack at the tail is unused).
    slot_size_b : int
        Bytes per slot.
    device : str
        CUDA device for the slab.
    block_slots : int | None
        Slots per backing block.  ``None`` (default) = one block backing
        the whole pool.  The last block may hold fewer slots when
        ``num_slots`` is not a multiple.
    """

    DEVICE = 'cuda:0'
    COALESCE_THRESHOLD = 5

    def __init__(self, capacity_b: int, slot_size_b: int,
                 device: str | None = None,
                 block_slots: Optional[int] = None):
        if slot_size_b <= 0:
            raise ValueError(f"slot_size_b must be positive, got {slot_size_b}")
        if capacity_b < slot_size_b:
            raise ValueError(
                f"capacity_b ({capacity_b}) must be >= slot_size_b ({slot_size_b})"
            )
        self.slot_size = int(slot_size_b)
        self.num_slots = int(capacity_b) // self.slot_size
        if block_slots is None:
            block_slots = self.num_slots
        if block_slots <= 0:
            raise ValueError(f"block_slots must be positive, got {block_slots}")
        self.block_slots = int(block_slots)
        self.num_blocks = -(-self.num_slots // self.block_slots)  # ceil div
        self._device = device or self.DEVICE
        # Exactly num_slots * slot_size across the blocks, each its own
        # arena tensor (torch.empty: readers see only bytes written into
        # their slots).
        self._blocks: List[Optional[torch.Tensor]] = [
            torch.empty(
                self._slots_in_block(b) * self.slot_size,
                dtype=torch.uint8,
                device=self._device,
            )
            for b in range(self.num_blocks)
        ]
        # Per-block live-slot counts: O(1) block-emptiness checks for the
        # governor (only a block with no live slot is borrowable).
        self._block_live: List[int] = [0] * self.num_blocks
        self._used = np.zeros(self.num_slots, dtype=bool)
        self._runs: List[tuple] = self._seed_runs()
        self._free_n = self.num_slots
        self._frees = 0
        # Released runs whose copies are still in flight:
        # (start, count, events), oldest first.
        self._pending: deque = deque()
        self._pending_n = 0
        self._block_pending: List[int] = [0] * self.num_blocks
        self._bases = None
        self._bases_of = None
        # Slots in use (the sum of ``_block_live``), and at most since
        # ``reset_peak`` (experiment telemetry).
        self._live_n = 0
        self._peak_used = 0

    # -- geometry --------------------------------------------------------

    def _slots_in_block(self, block_idx: int) -> int:
        if block_idx < self.num_blocks - 1:
            return self.block_slots
        return self.num_slots - self.block_slots * (self.num_blocks - 1)

    def _block_of(self, slot_idx: int) -> tuple[int, int]:
        return divmod(slot_idx, self.block_slots)

    def block_nbytes(self, block_idx: int) -> int:
        """Bytes backing ``block_idx`` (last block may be short)."""
        return self._slots_in_block(block_idx) * self.slot_size

    def _seed_runs(self) -> List[tuple]:
        return [(b * self.block_slots, self._slots_in_block(b))
                for b in range(self.num_blocks)
                if self._blocks[b] is not None]

    def slot_ptrs(self, slots):
        """Device addresses of ``slots`` (an int array), as an int64 array."""
        slots = np.asarray(slots, dtype=np.int64)
        bases = self._block_bases()
        block = slots // self.block_slots
        assert (bases[block] != 0).all(), (
            "slot_ptrs: a slot's backing block is released")
        return bases[block] + (slots % self.block_slots) * self.slot_size

    def _block_bases(self) -> np.ndarray:
        """Each block's device address (0: released), rebuilt when a block
        is released or restored."""
        ptrs = [0 if b is None else b.data_ptr() for b in self._blocks]
        if self._bases is None or self._bases_of != ptrs:
            self._bases = np.array(ptrs, dtype=np.int64)
            self._bases_of = ptrs
        return self._bases

    # -- queries ---------------------------------------------------------

    @property
    def pool_size_bytes(self) -> int:
        """Total bytes managed by this pool at configured (fully-restored)
        size."""
        return self.num_slots * self.slot_size

    def block_is_present(self, block_idx: int) -> bool:
        return self._blocks[block_idx] is not None

    def block_live_count(self, block_idx: int) -> int:
        return self._block_live[block_idx]

    def released_block_indices(self) -> List[int]:
        return [b for b in range(self.num_blocks) if self._blocks[b] is None]

    def borrowed_bytes(self) -> int:
        """Bytes currently released back to the arena (exact accounting)."""
        return sum(self.block_nbytes(b) for b in range(self.num_blocks)
                   if self._blocks[b] is None)

    def free_count(self) -> int:
        """Slots not in use: free ones plus pending ones (released, with a
        copy still in flight on them)."""
        return self._free_n + self._pending_n

    def empty_block_indices(self) -> List[int]:
        """Present blocks with no slot in use and no copy in flight."""
        self.reap()
        return [b for b in range(self.num_blocks)
                if self._blocks[b] is not None and self._block_live[b] == 0
                and self._block_pending[b] == 0]

    # -- the free list ---------------------------------------------------

    def coalesce(self) -> None:
        """Sort the free runs and merge those that touch within one block."""
        self._frees = 0
        if len(self._runs) < 2:
            return
        runs = np.array(self._runs, dtype=np.int64)
        runs = runs[np.argsort(runs[:, 0], kind="stable")]
        start, count = runs[:, 0], runs[:, 1]
        bs = self.block_slots
        join = ((start[:-1] + count[:-1] == start[1:])
                & (start[:-1] // bs == start[1:] // bs))
        first = np.concatenate(([0], np.flatnonzero(~join) + 1))
        self._runs = list(zip(start[first].tolist(),
                              np.add.reduceat(count, first).tolist()))

    def _add_free(self, start: int, count: int) -> None:
        self._runs.append((start, count))
        self._free_n += count
        self._frees += 1
        if self._frees > self.COALESCE_THRESHOLD:
            self.coalesce()

    def reap(self) -> int:
        """Move every pending run whose events have all fired to the free
        list.  Returns the slots freed."""
        if not self._pending_n:
            return 0
        freed = 0
        keep = deque()
        for start, count, events in self._pending:
            if _fired(events):
                self._block_pending[start // self.block_slots] -= count
                self._add_free(start, count)
                freed += count
            else:
                keep.append((start, count, events))
        self._pending = keep
        self._pending_n -= freed
        return freed

    def wait_pending(self) -> None:
        """Wait on the host for every pending run's copies, and free them."""
        for _, _, events in self._pending:
            for e in events:
                e.synchronize()
        self.reap()

    def _take_from_runs(self, count: int) -> List[tuple]:
        """Take ``count`` free slots: one run that holds them if there is
        one (first fit), else runs in free-list order.  The caller checked
        that ``count`` slots are free."""
        got = self._first_fit(count)
        if got is None and self._frees:
            self.coalesce()
            got = self._first_fit(count)
        if got is not None:
            return [got]
        taken = []
        left = count
        while left:
            s, n = self._runs[0]
            k = min(n, left)
            if k == n:
                self._runs.pop(0)
            else:
                self._runs[0] = (s + k, n - k)
            taken.append((s, k))
            left -= k
        self._free_n -= count
        return taken

    def _first_fit(self, count: int):
        for i, (s, n) in enumerate(self._runs):
            if n >= count:
                if n == count:
                    self._runs.pop(i)
                else:
                    self._runs[i] = (s + count, n - count)
                self._free_n -= count
                return (s, count)
        return None

    def _mark_used(self, runs) -> None:
        for s, n in runs:
            assert not self._used[s:s + n].any(), (
                f"slots {s}..{s + n - 1} handed out twice")
            self._used[s:s + n] = True
            self._block_live[s // self.block_slots] += n
            self._live_n += n
        if gates.EXPERIMENT:
            self._peak_used = max(self._peak_used, self._live_n)

    def reset_peak(self) -> None:
        """Rebaseline the high-water mark to the slots in use now."""
        self._peak_used = self._live_n

    def peak_used_bytes(self) -> int:
        """Bytes of the most slots in use at once since ``reset_peak``
        (counted under TQP_EXPERIMENT_TELEMETRY only)."""
        return self._peak_used * self.slot_size

    # -- allocation ------------------------------------------------------

    def acquire_runs(self, count: int) -> Optional[SlotGrant]:
        """``count`` slots, in as few runs as the free list allows (one run
        when one fits), then pending slots, oldest release first, whose
        events the grant lists.  ``None`` (and nothing taken) when not
        enough slots are available."""
        count = int(count)
        if count <= 0:
            return SlotGrant([], [])
        self.reap()
        if self._free_n + self._pending_n < count:
            return None
        from_free = min(count, self._free_n)
        runs = self._take_from_runs(from_free) if from_free else []
        events = []
        left = count - from_free
        while left:
            start, n, evs = self._pending.popleft()
            k = min(n, left)
            if k < n:
                self._pending.appendleft((start + k, n - k, evs))
            self._block_pending[start // self.block_slots] -= k
            self._pending_n -= k
            runs.append((start, k))
            for e in evs:
                if not any(e is x for x in events):
                    events.append(e)
            left -= k
        self._mark_used(runs)
        return SlotGrant(runs, events)

    def release_slots(self, slots, events=()) -> None:
        """Return ``slots`` (any order), grouped into runs of consecutive
        slots within a block, all behind the same ``events``."""
        slots = np.sort(np.asarray(slots, dtype=np.int64))
        if not len(slots):
            return
        if not self._used[slots].all() or (np.diff(slots) == 0).any():
            raise ValueError(
                f"release_slots({slots.tolist()}): a slot is not allocated "
                "(double free or never acquired)")
        bs = self.block_slots
        brk = (np.diff(slots) != 1) | ((slots[1:] % bs) == 0)
        starts = np.concatenate(([0], np.flatnonzero(brk) + 1))
        counts = np.diff(np.concatenate((starts, [len(slots)])))
        self._used[slots] = False
        per_block = np.bincount(slots // bs, minlength=self.num_blocks)
        touched = np.flatnonzero(per_block).tolist()
        for blk in touched:
            self._block_live[blk] -= int(per_block[blk])
        self._live_n -= len(slots)
        runs = list(zip(slots[starts].tolist(), counts.tolist()))
        events = [e for e in events if not e.query()]
        if events:
            for blk in touched:
                self._block_pending[blk] += int(per_block[blk])
            self._pending.extend((a, n, events) for a, n in runs)
            self._pending_n += len(slots)
        else:
            self._runs.extend(runs)
            self._free_n += len(slots)
            self._frees += len(runs)
            if self._frees > self.COALESCE_THRESHOLD:
                self.coalesce()

    def free_all(self) -> None:
        """Release every slot, once the copies on pending slots are done."""
        self.wait_pending()
        self._used[:] = False
        self._runs = self._seed_runs()
        self._free_n = sum(n for _, n in self._runs)
        self._frees = 0
        self._block_live = [0] * self.num_blocks
        self._live_n = 0
        self._block_pending = [0] * self.num_blocks

    # -- elasticity (governor-only surface) ------------------------------

    def release_block(self, block_idx: int) -> int:
        """Free ``block_idx``'s backing tensor to the shared arena and pull
        its slots out of circulation.  Returns the bytes released.

        Preconditions (the governor's to establish): the block holds no
        live or pending slot, and no in-flight kernel or copy can still
        read a view into it.
        """
        if not (0 <= block_idx < self.num_blocks):
            raise ValueError(f"release_block({block_idx}): no such block")
        if self._blocks[block_idx] is None:
            raise ValueError(f"release_block({block_idx}): already released")
        self.reap()
        if self._block_live[block_idx] or self._block_pending[block_idx]:
            raise ValueError(
                f"release_block({block_idx}): {self._block_live[block_idx]} "
                f"live and {self._block_pending[block_idx]} pending slots — "
                "only empty blocks are borrowable")
        if self._frees:
            self.coalesce()
        lo = block_idx * self.block_slots
        hi = lo + self._slots_in_block(block_idx)
        kept = [(s, n) for s, n in self._runs if not (lo <= s < hi)]
        self._free_n -= sum(n for s, n in self._runs if lo <= s < hi)
        self._runs = kept
        nbytes = self.block_nbytes(block_idx)
        self._blocks[block_idx] = None
        return nbytes

    def restore_block(self, block_idx: int) -> int:
        """Re-allocate ``block_idx``'s backing tensor and return its slots
        to circulation.  Returns the bytes re-acquired.  Raises whatever
        the allocator raises when the arena cannot host the block (the
        governor tolerates this as deferred re-grow).
        """
        if not (0 <= block_idx < self.num_blocks):
            raise ValueError(f"restore_block({block_idx}): no such block")
        if self._blocks[block_idx] is not None:
            raise ValueError(f"restore_block({block_idx}): not released")
        nbytes = self.block_nbytes(block_idx)
        self._blocks[block_idx] = torch.empty(
            nbytes, dtype=torch.uint8, device=self._device)
        self._add_free(block_idx * self.block_slots,
                       self._slots_in_block(block_idx))
        return nbytes
