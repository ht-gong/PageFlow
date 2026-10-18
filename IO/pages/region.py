"""Regions: byte ranges of records, the unit of allocation and lifetime.

A region holds records back to back. It is written once, by one producer
step, and never changed afterwards. Which records it holds, and where, is
not stored here: a :class:`IO.pages.record_table.RecordTable` row names its
region by index and gives each record's offset into it.

A region outside the granule cache is one uint8 tensor, in pinned host
memory or on the GPU. ``release`` says how that memory goes back:

- an ``int`` with ``location == "host"``: a pinned-pool block index, returned
  with ``cpu_pool.free``;
- ``None``: an arena tensor (or a view of memory someone else owns), freed
  when its last reference dies;
- a ``(cache, cached region)`` pair with ``location == "cached"``: a region
  the granule cache (``IO/cache/native_cache.py``) holds, its storage the
  pinned home (carved by the producer, returned with the table's chunks).
  Its free drops it from the cache, which releases its slots, and waits for
  the copies still writing its home.

A *spillable* region (the free-list baseline, ``IO/pages/device_regions.py``)
is a GPU region with a ``home``: pinned host bytes of its size, reserved but
not written.  It is the one kind of region that moves: an eviction copies
it into its home and makes it a host region (``spill_to_home``, then
``settle_on_host``).  Its records are offsets into the region, so the move
leaves its record table valid; what caches a region's address keys it on
``device_regions.epoch()``.

The record table that indexes the region frees it (``RecordTable.free``).

``uva_ptr`` is a host region's device alias, for the zero-copy readers
(``IO/zero_copy_reader.cu``).
"""
from __future__ import annotations

from typing import Optional

import torch

HOST = "host"
DEVICE = "device"
CACHED = "cached"
_LOCATIONS = (HOST, DEVICE, CACHED)

# ---------------------------------------------------------------------------
# Global region-ID counter
# ---------------------------------------------------------------------------
_next_region_id: int = 0


def _assign_region_id() -> int:
    """Return a fresh, process-unique region ID."""
    global _next_region_id
    rid = _next_region_id
    _next_region_id += 1
    return rid


class Region:
    """One region: ``nbytes`` of records in one allocation.

    Attributes:
        region_id  process-unique id, for telemetry and as an owner index.
        nbytes     size of the region in bytes.
        location   ``host``, ``device`` or ``cached``.
        storage    the region's bytes as a 1-D uint8 tensor (``None`` once
                   freed).
        release    how the memory goes back (module docstring).
        freed      True once the memory has been returned (``free``).
    """

    __slots__ = ("region_id", "nbytes", "location", "storage", "release",
                 "freed", "_uva_ptr", "home", "pins")

    def __init__(self, nbytes: int, location: str,
                 storage: Optional[torch.Tensor] = None,
                 release=None, home: Optional[torch.Tensor] = None):
        assert location in _LOCATIONS, f"unknown region location {location!r}"
        assert nbytes >= 0, f"negative region size {nbytes}"
        if storage is not None:
            assert storage.dtype == torch.uint8 and storage.dim() == 1, (
                "region storage must be a 1-D uint8 tensor")
            assert storage.numel() == nbytes, (
                f"region storage holds {storage.numel()} B, expected {nbytes}")
        self.region_id = _assign_region_id()
        self.nbytes = int(nbytes)
        self.release = release
        self.freed = False
        self.storage = storage
        self.location = location
        self._uva_ptr = None
        # A spillable GPU region's reserved host bytes (module docstring),
        # else None; reads that hold its address pin it (``pins`` > 0) so
        # an eviction leaves it where it is.
        if home is not None:
            assert location == DEVICE, "only a GPU region has a home"
            assert home.dtype == torch.uint8 and home.dim() == 1 \
                and home.numel() == nbytes and not home.is_cuda, (
                    "a home is nbytes of host memory as 1-D uint8")
        self.home = home
        self.pins = 0

    def data_ptr(self) -> int:
        """Address of the region's first byte (host or device)."""
        storage = self.storage
        assert storage is not None, f"region {self.region_id} has no storage"
        return storage.data_ptr()

    @property
    def uva_ptr(self) -> int:
        """The device alias of a host region's first byte
        (``cudaHostGetDevicePointer``), for zero-copy reads: computed once
        per region."""
        assert self.location == HOST, (
            f"region {self.region_id} is not in host memory")
        if self._uva_ptr is not None:
            return self._uva_ptr
        import TQPlib.zero_copy_reader as zero_copy_reader
        storage = self.storage
        assert storage is not None, f"region {self.region_id} has no storage"
        self._uva_ptr = zero_copy_reader.device_alias(storage)
        return self._uva_ptr

    @property
    def spillable(self) -> bool:
        """A GPU region an eviction may move to its home."""
        return self.home is not None

    @property
    def device_ptr(self) -> int:
        """The address a kernel reads the region at: its own on the GPU, the
        device alias (``uva_ptr``) in host memory."""
        if self.location == DEVICE:
            return self.data_ptr()
        return self.uva_ptr

    def spill_to_home(self) -> None:
        """Queue the copy of a spillable region into its home on the current
        stream; the caller waits for it, then calls ``settle_on_host``."""
        assert self.home is not None and self.storage is not None, (
            f"region {self.region_id} is not spillable")
        self.home.copy_(self.storage, non_blocking=True)

    def settle_on_host(self) -> None:
        """After ``spill_to_home``'s copy has ended: the home is the region's
        storage and its GPU tensor is dropped (freed once nothing else holds
        it)."""
        self.storage = self.home
        self.home = None
        self.location = HOST
        self._uva_ptr = None

    def free(self, cpu_pool, gpu_pool) -> None:
        """Return the memory. A second call does nothing."""
        if self.freed:
            return
        if self.home is not None:
            from IO.pages import device_regions
            device_regions.unregister(self)
            self.home = None
        if self.release is not None:
            if self.location == HOST:
                cpu_pool.free(self.release)
            else:
                cache, cr = self.release
                cache.drop(cr)
                for event in cache.home_events(cr):
                    event.synchronize()
        self.release = None
        self.storage = None
        self._uva_ptr = None
        self.freed = True

    def __repr__(self) -> str:
        return (f"Region(id={self.region_id}, {self.nbytes} B, "
                f"{self.location}"
                f"{', spillable' if self.home is not None else ''}"
                f"{', freed' if self.freed else ''})")
