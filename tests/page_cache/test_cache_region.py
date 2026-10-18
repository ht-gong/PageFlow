"""The arena's cache region as the engine holds it
(``IO/cache/cache_region.py``) and the granule cache over it: ``set_size``
moves the boundary, the cache's slots are the region's arena blocks, filled
from the top, and its tensor view covers them.

On the shared test arena; each test removes its region at the end, even on
failure."""
from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")

MIB = 1024 ** 2


@pytest.fixture
def region(cuda_allocator):
    """A ``CacheRegion`` over the test arena, with no region before the test
    and none after it."""
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    pytest.importorskip("TQPlib.tqpcache")
    import TQPlib.tqpmemory as cmp
    from IO.cache.cache_region import CacheRegion
    assert cmp.region_blocks() == 0
    torch.cuda.synchronize()
    yield CacheRegion()
    torch.cuda.synchronize()
    cmp.region_move(cmp.region_blocks())


def test_set_size_moves_the_boundary_and_the_operator_space_follows(region):
    import TQPlib.tqpmemory as cmp
    assert region.block == MIB
    before = cmp.get_arena_stats()
    region.set_size(64 * MIB)
    assert region.nbytes == 64 * MIB
    assert region.boundary == region.num_blocks - 64
    st = cmp.get_arena_stats()
    assert st["region_b"] == 64 * MIB
    assert st["capacity_b"] == before["capacity_b"] - 64 * MIB
    region.set_size(16 * MIB + 5)            # whole blocks
    assert region.nbytes == 16 * MIB
    region.set_size(80 * MIB)
    assert region.nbytes == 80 * MIB
    region.set_size(0)                       # no region
    assert region.nbytes == 0 and region.boundary == region.num_blocks
    assert cmp.get_arena_stats()["capacity_b"] == before["capacity_b"]


def _fill(piece_b):
    """Allocate ``piece_b`` tensors until the operator space has no run of
    that size left."""
    pieces = []
    while True:
        try:
            pieces.append(torch.empty(piece_b, dtype=torch.uint8,
                                      device="cuda:0"))
        except (RuntimeError, torch.OutOfMemoryError):
            return pieces


def test_a_refused_grow_raises(region):
    region.set_size(4 * MIB)
    pieces = _fill(64 * MIB) + _fill(MIB)    # the operator space is full
    try:
        assert max(p.data_ptr() + p.numel() for p in pieces) == (
            region.base + region.boundary * MIB)
        with pytest.raises(RuntimeError, match="refused to grow"):
            region.set_size(8 * MIB)
        assert region.nbytes == 4 * MIB
    finally:
        del pieces


def test_a_refused_grow_from_no_region_raises_the_same(region):
    pieces = _fill(64 * MIB) + _fill(MIB)    # the operator space is full
    try:
        with pytest.raises(RuntimeError, match="refused to grow"):
            region.set_size(4 * MIB)
        assert region.nbytes == 0
    finally:
        del pieces

def test_the_cache_fills_the_region_from_the_top(region):
    from IO.cache.native_cache import NativeGranuleCache
    region.set_size(8 * MIB)
    lo, top = region.boundary, region.num_blocks
    cache = NativeGranuleCache(region, "lru")
    home = torch.zeros(3 * MIB, dtype=torch.uint8)
    cr = cache.register(3 * MIB, home.data_ptr(), [0], [3 * MIB], [0],
                        next_use_op=1, expected_reads=1)
    assert cache.admit([cr]).admitted == [3]
    gids = np.arange(cr.first, cr.first + 3)
    slots = cache.slot_of(gids).tolist()
    assert slots == [top - 3, top - 2, top - 1]
    assert cache.slot_addrs(gids).tolist() == [
        region.base + s * MIB for s in slots]
    # The view covers the region: write the granules through it and read
    # them back at the addresses the pieces give.
    view = region.view()
    assert view.data_ptr() == region.base + lo * MIB
    assert view.numel() == 8 * MIB
    data = torch.randint(0, 256, (3 * MIB,), dtype=torch.uint8,
                         device="cuda:0")
    for k, s in enumerate(slots):
        view[(s - lo) * MIB:(s - lo + 1) * MIB].copy_(
            data[k * MIB:(k + 1) * MIB])
    p = cache.pieces(cr, 0, 3 * MIB)
    assert p.on_gpu.tolist() == [True]
    start = int(p.src[0]) - view.data_ptr()
    assert torch.equal(view[start:start + 3 * MIB], data)
    # Emptied, the cache follows the boundary.
    cache.clear()
    region.set_size(2 * MIB)
    cache.clear()
    assert cache._core.free_count() == 2
