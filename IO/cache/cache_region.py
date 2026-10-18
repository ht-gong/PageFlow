"""The arena's cache region: the GPU memory of the granule cache.

The region is the arena's top blocks, behind a boundary
(``memory/cuda_memory_allocator.cc``, ``region_move``).
No operator allocation lands in it, and the arena's stats and window peaks
describe the operator space below the boundary.  One granule takes one arena
block: slot ``i`` of the cache is arena block ``i``, at ``base + i * block``,
so moving the boundary never renumbers a slot.

Operator allocations come from the arena's bottom (first fit) and the
cache's granules from the region's top, so the blocks next to the boundary
are the last either side takes.

``CacheRegion`` is the engine's handle on it: the geometry the granule
cache's core needs (``base``, ``block``, ``num_blocks``, ``boundary``),
``set_size`` to move the boundary between queries, and ``view``, one
non-owning tensor over the region's bytes.  ``main.py`` builds one per
process.  Within a query only the arena moves the boundary: an allocation
that misses borrows the region's lowest blocks through the attached cache
and the cache takes them back (``NativeGranuleCache.attach``), so
``boundary`` and ``nbytes`` are read live from the arena, never kept.
"""
from __future__ import annotations

import torch

import TQPlib.tqpmemory as tqpmemory


class _DeviceBytes:
    """A device byte range in the CUDA array interface."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,), "typestr": "|u1", "data": (ptr, False),
            "version": 3, "strides": None, "stream": None}


class CacheRegion:
    """The arena's cache region (module docstring).  Over the arena
    ``TQPlib.tqpmemory`` has initialised."""

    def __init__(self):
        self.base = int(tqpmemory.arena_base())
        self.block = int(tqpmemory.block_size())
        self.num_blocks = int(tqpmemory.region_boundary()
                              + tqpmemory.region_blocks())

    @property
    def boundary(self) -> int:
        """The region's lowest block (``num_blocks`` when there is none)."""
        return int(tqpmemory.region_boundary())

    @property
    def nbytes(self) -> int:
        return int(tqpmemory.region_blocks()) * self.block

    def set_size(self, nbytes: int) -> bool:
        """Move the boundary so the region holds ``nbytes // block`` blocks;
        True when it moved.  A shrink always succeeds; a grow, from no
        region too, takes the blocks just below the boundary, all or none,
        and raises ``RuntimeError`` when the arena refuses them (an operator
        allocation sits there).  Only with the cache empty and detached
        (between queries).  It ends any lending even when the size does not
        change (no block counts as lent after it): a region lent whole by
        the query's end is empty already, and its lent blocks are the
        operator space's from then on."""
        want = int(nbytes) // self.block
        if want > self.num_blocks:
            raise ValueError(
                f"cache region of {want} blocks: the arena has "
                f"{self.num_blocks}")
        have = int(tqpmemory.region_blocks())
        if want == have:
            tqpmemory.region_move(0)          # only resets the lent count
            return False
        if want < have:
            tqpmemory.region_move(have - want)
        elif tqpmemory.region_move(-(want - have)) == 0:
            raise RuntimeError(
                f"the arena refused to grow the cache region from {have} to "
                f"{want} blocks: blocks below the boundary are in use")
        return True

    def view(self) -> torch.Tensor:
        """The region's bytes as one uint8 tensor on the GPU, zero-copy:
        slot ``i``'s bytes are ``view()[(i - boundary) * block:][:block]``.
        Valid until the boundary moves, which any arena allocation can do
        while a cache is attached: take it, use it and drop it with no
        allocation in between."""
        lo = self.boundary
        nbytes = (self.num_blocks - lo) * self.block
        if not nbytes:
            return torch.empty(0, dtype=torch.uint8, device="cuda")
        return torch.as_tensor(
            _DeviceBytes(self.base + lo * self.block, nbytes), device="cuda")
