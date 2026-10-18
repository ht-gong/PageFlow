"""The arena allocator's force-drain (memory/cuda_memory_allocator.cc).

A free is ordered on the stream that freed the block.  When the clean list
has no run large enough, the allocator force-drains every pending free into
it; a block whose free has not yet run on its stream keeps a guard, so any
stream that later takes it waits for that free, not only the stream that
force-drained.  Without the guard, a block a stream still uses (work queued
before its free) could be written by another stream."""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")

MIB = 1024 ** 2
PIECE = 64 * MIB


def _fill(stream):
    """Allocate PIECE-sized tensors on ``stream`` until the arena has no
    PIECE run left; returns them in allocation (address) order."""
    pieces = []
    with torch.cuda.stream(stream):
        while True:
            try:
                pieces.append(torch.empty(PIECE, dtype=torch.uint8,
                                          device="cuda:0"))
            except (RuntimeError, torch.OutOfMemoryError):
                break
    return pieces


def test_a_force_drained_block_waits_for_its_free(cuda_allocator):
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    import TQPlib.tqpmemory as cmp
    torch.cuda.synchronize()
    b = torch.cuda.Stream()
    c = torch.cuda.Stream()
    pieces = _fill(b)
    try:
        assert len(pieces) > 30
        assert cmp.get_arena_stats()["largest_free_run_b"] < PIECE
        ptrs = [p.data_ptr() for p in pieces]
        assert ptrs[21] == ptrs[20] + PIECE        # adjacent: one 2-piece run
        # Stream b is busy; then it frees one piece (f1) and a 2-piece run
        # (f2): all three frees wait behind the sleep.
        with torch.cuda.stream(b):
            for _ in range(2):
                torch.cuda._sleep(1_000_000_000)    # about a second
        f1 = ptrs[10]
        pieces[10] = pieces[20] = pieces[21] = None
        assert cmp.get_arena_stats(False)["pending_b"] >= 3 * PIECE
        # b asks for two pieces: no clean run fits, so it force-drains.
        # It takes f2 (first fit; f1 is one piece) with no wait, being the
        # freeing stream; f1 becomes clean but keeps its guard.
        with torch.cuda.stream(b):
            taken_b = torch.empty(2 * PIECE, dtype=torch.uint8,
                                  device="cuda:0")
        assert taken_b.data_ptr() == ptrs[20]
        # c takes f1 and must wait for b's free of it.
        with torch.cuda.stream(c):
            taken_c = torch.empty(PIECE, dtype=torch.uint8, device="cuda:0")
            after = torch.cuda.Event()
            after.record(c)
        assert taken_c.data_ptr() == f1
        assert not after.query(), "c took a block b's queued work still owns"
        assert not b.query()
        after.synchronize()
    finally:
        del pieces
        torch.cuda.synchronize()
