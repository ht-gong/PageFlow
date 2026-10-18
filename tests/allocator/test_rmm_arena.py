"""RMM's device resource is the arena (utility/rmm_arena.py), so cuDF's
buffers come out of the arena's blocks, freed back to it, and an allocation
the arena cannot hold raises MemoryError.  Inside ``small_buffers()`` small
buffers share arena blocks."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")

MIB = 1024 ** 2


def test_rmm_and_cudf_buffers_are_arena_allocations(cuda_allocator):
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    import pylibcudf as plc
    import rmm
    import TQPlib.tqpmemory as cmp
    torch.cuda.synchronize()
    before = cmp.get_arena_stats()["allocated_b"]
    buf = rmm.DeviceBuffer(size=3 * MIB + 1)
    assert buf.ptr % 256 == 0
    col = plc.column_factories.make_numeric_column(
        plc.DataType(plc.TypeId.INT64), 1 << 20, plc.MaskState.UNALLOCATED)
    assert cmp.get_arena_stats()["allocated_b"] - before >= 11 * MIB + 1
    del buf, col
    torch.cuda.synchronize()
    assert cmp.get_arena_stats()["allocated_b"] == before


def test_rmm_beyond_the_arena_raises_memory_error(cuda_allocator):
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    import rmm
    import TQPlib.tqpmemory as cmp
    cap = cmp.get_arena_stats()["capacity_b"]
    with pytest.raises(MemoryError):
        rmm.DeviceBuffer(size=cap + MIB)


def test_small_buffers_share_arena_blocks(cuda_allocator):
    """Inside ``small_buffers()`` a small buffer comes from a bin's slab of
    one arena block, so 1,000 buffers of 4 KiB take a few blocks, not
    1,000; a buffer over ``SMALL_MAX_B`` is still the arena's own.  Outside
    the scope a small buffer costs a whole block again, the bins keep their
    slabs while a buffer they served lives, and once the buffers are freed
    the arena is as it was."""
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    import rmm
    import TQPlib.tqpmemory as cmp
    from utility import rmm_arena
    block = rmm_arena.block_bytes
    bins = 13                                   # 256 B .. 1 MiB

    def allocated():
        torch.cuda.synchronize()
        return cmp.get_arena_stats()["allocated_b"]

    before = allocated()
    with rmm_arena.small_buffers():
        assert allocated() - before == bins * block
        small = [rmm.DeviceBuffer(size=4096) for _ in range(1000)]
        per_slab = block // 4096
        slabs = -(-1000 // per_slab)            # the bin's first slab included
        assert allocated() - before == (bins - 1 + slabs) * block
        large = rmm.DeviceBuffer(size=rmm_arena.SMALL_MAX_B + 1)
        large_blocks = -(-(rmm_arena.SMALL_MAX_B + 1) // block)
        assert allocated() - before == (bins - 1 + slabs + large_blocks) * block
    outside = rmm.DeviceBuffer(size=4096)
    assert allocated() - before == (bins - 1 + slabs + large_blocks + 1) * block
    del small, large, outside
    assert allocated() == before
