"""RMM, and so cuDF, allocates from the engine's GPU arena.

cuDF (``pylibcudf``) takes every device buffer from RMM's current device
resource, by default ``cudaMalloc``: outside the arena, in the about 2 GB the
arena leaves free on the card.  ``install`` replaces it, on device 0, with a
callback resource over the arena's stream-ordered pair (``pa_cuda_malloc`` /
``pa_cuda_free``, the functions torch's pluggable allocator calls, bound as
``TQPlib.tqpmemory.stream_malloc`` / ``stream_free``):

- each allocation and free is ordered on the stream RMM names, as torch's
  are, so a freed block is reused only after the work queued on that stream
  before the free;
- every arena allocation starts on a block boundary, so RMM's 256-byte
  alignment holds;
- the arena's out-of-memory ``RuntimeError`` is re-raised as
  ``MemoryError``, which RMM turns into ``rmm::out_of_memory`` for cuDF;
- every allocation is rounded up to whole arena blocks (1 MiB under
  ``main.py`` and ``tests/conftest.py``), so each of cuDF's small buffers
  costs a whole block.

Call it after ``cmp.init``, with the arena's block size, and before any cuDF
call (``main.py``, ``tests/conftest.py``).  A Python callback runs per
allocation and per free.

**Small buffers.**  Inside ``small_buffers()`` (the scan,
``IO/data_scanner.scan_to_ram``) RMM's resource is a binning resource over
the callback resource: a buffer of at most ``SMALL_MAX_B`` comes from a bin
of its power-of-two size class (from 256 B), which carves its buffers from
slabs of one arena block each, so small buffers share blocks; a larger one
goes to the arena as above.  Every bin's slab holds at least one buffer
when ``SMALL_MAX_B`` is at most the block (at 1 MiB blocks the 1 MiB bin's
slab holds exactly one); with smaller blocks (``TQP_ARENA_BLOCK_B``) a small
buffer wastes little on rounding and the scope bins nothing.  The chunked parquet reader holds one small
buffer per part file for the length of a pass, and a pass can take every
file of a small table (16,000 at SF1000): one block each would fill the
arena.  A bin gives no slab back while it lives.  When the scope ends, the
callback resource is RMM's again; the bins live on while a buffer they
served does (each buffer keeps its resource), and give every slab back to
the arena when the last one is freed (a bin's destructor waits for the
streams its buffers were freed on).
"""
from __future__ import annotations

import contextlib

SMALL_MIN_B = 256              # the smallest bin, RMM's alignment
SMALL_MAX_B = 1024 ** 2        # the largest bin; a larger buffer is the arena's

_resource = None
block_bytes = None             # the arena's block size, given to ``install``


def install(block: int) -> None:
    """Make the arena, of ``block``-byte blocks, RMM's device resource on
    device 0.  Idempotent: a later call changes nothing."""
    global _resource, block_bytes
    if _resource is not None:
        return
    import rmm.mr

    import TQPlib.tqpmemory as cmp

    def allocate(size: int, stream) -> int:
        try:
            return cmp.stream_malloc(size, stream.__cuda_stream__()[1])
        except RuntimeError as err:
            raise MemoryError(f"arena: {err} ({size} B)") from err

    def deallocate(ptr: int, size: int, stream) -> None:
        cmp.stream_free(ptr, stream.__cuda_stream__()[1])

    _resource = rmm.mr.CallbackMemoryResource(allocate, deallocate)
    block_bytes = block
    rmm.mr.set_per_device_resource(0, _resource)


@contextlib.contextmanager
def small_buffers():
    """RMM's buffers of at most ``SMALL_MAX_B`` share arena blocks while the
    scope lasts (module docstring).  Each bin takes one slab when the scope
    starts.  Arena blocks smaller than ``SMALL_MAX_B`` (``TQP_ARENA_BLOCK_B``)
    waste little on a small buffer and would give a bin no slab: the scope
    then changes nothing."""
    if block_bytes < SMALL_MAX_B:
        yield
        return
    import rmm.mr

    binning = rmm.mr.BinningMemoryResource(_resource)
    size = SMALL_MIN_B
    while size <= SMALL_MAX_B:
        binning.add_bin(size, rmm.mr.FixedSizeMemoryResource(
            _resource, block_size=size,
            blocks_to_preallocate=block_bytes // size))
        size *= 2
    rmm.mr.set_per_device_resource(0, binning)
    try:
        yield
    finally:
        rmm.mr.set_per_device_resource(0, _resource)
