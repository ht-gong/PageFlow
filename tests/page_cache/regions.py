"""A cache region over a tensor of its own, for the granule cache's tests.

It has the geometry ``NativeGranuleCache`` reads off the engine's
``IO.cache.cache_region.CacheRegion``: ``num_blocks`` slots of ``block``
bytes from ``base``, the region being ``[boundary, num_blocks)``."""
from __future__ import annotations

import torch


class TensorRegion:
    def __init__(self, num_blocks: int, block: int, boundary: int = 0,
                 device: str = "cpu"):
        self.tensor = torch.zeros(num_blocks * block, dtype=torch.uint8,
                                  device=device)
        self.base = self.tensor.data_ptr()
        self.block = int(block)
        self.num_blocks = int(num_blocks)
        self.boundary = int(boundary)

    @property
    def nbytes(self) -> int:
        return (self.num_blocks - self.boundary) * self.block

    def shrink(self, k: int) -> None:
        self.boundary += int(k)

    def grow(self, k: int) -> int:
        k = int(k)
        if k > self.boundary:
            return 0
        self.boundary -= k
        return k
