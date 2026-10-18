"""The arena's cache region (memory/cuda_memory_allocator.cc).

The region is the arena's top blocks, behind a boundary.  Its blocks are
never in the free list, so no allocation lands in them, and the arena stats,
the window peak and the OOM-diagnostic peak describe the operator space
below the boundary; ``top_free_run_b`` is its free run that ends at the
boundary.  ``region_move(delta)`` moves the boundary: a positive
delta moves it up (the region gives blocks to the operator space), a
negative one down (the region takes the blocks just below the boundary, all
or none, only when every one is free).

The tests on the shared test arena remove their region at the end, even on
failure.  ``free_all`` and the OOM dump run in a subprocess on an arena of
its own, since they would wipe the shared arena's live allocations."""
import os
import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")

MIB = 1024 ** 2
PIECE = 64 * MIB
REGION_B = 256 * MIB
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture
def cmp(cuda_allocator):
    """The arena module, with no region before the test and none after it."""
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    import TQPlib.tqpmemory as cmp
    assert cmp.region_blocks() == 0
    torch.cuda.synchronize()
    yield cmp
    torch.cuda.synchronize()
    cmp.region_move(cmp.region_blocks())


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


def test_region_init_takes_the_top_blocks_out_of_the_operator_space(cmp):
    block = cmp.block_size()
    assert block == MIB
    before = cmp.get_arena_stats()
    assert before["region_b"] == 0
    n_blocks = cmp.region_boundary()
    assert n_blocks * block == before["capacity_b"]

    assert cmp.region_init(REGION_B + block - 1) == REGION_B // block
    after = cmp.get_arena_stats()
    assert after["region_b"] == REGION_B
    assert after["capacity_b"] == before["capacity_b"] - REGION_B
    assert after["free_b"] == before["free_b"] - REGION_B
    assert after["allocated_b"] == before["allocated_b"]
    assert cmp.region_blocks() == REGION_B // block
    assert cmp.region_boundary() == n_blocks - REGION_B // block
    assert cmp.arena_base() > 0
    with pytest.raises(RuntimeError, match="already has a cache region"):
        cmp.region_init(REGION_B)


def test_moving_the_boundary_up_gives_the_blocks_back_coalesced(cmp):
    before = cmp.get_arena_stats()
    cmp.region_init(REGION_B)
    assert cmp.region_move(64) == 64
    mid = cmp.get_arena_stats()
    assert mid["region_b"] == REGION_B - 64 * MIB
    assert mid["capacity_b"] == before["capacity_b"] - REGION_B + 64 * MIB
    assert cmp.region_move(cmp.region_blocks()) == REGION_B // MIB - 64
    after = cmp.get_arena_stats()
    # One run again: the stats are those of the arena with no region.
    for key in ("capacity_b", "free_b", "largest_free_run_b", "region_b"):
        assert after[key] == before[key], key
    with pytest.raises(RuntimeError, match="the region holds 0"):
        cmp.region_move(1)


def test_no_allocation_lands_in_the_region(cmp):
    cmp.region_init(REGION_B)
    lo = cmp.arena_base() + cmp.region_boundary() * cmp.block_size()
    pieces = _fill(PIECE) + _fill(MIB)
    try:
        assert len(pieces) > 30
        assert cmp.get_arena_stats()["region_b"] == REGION_B
        assert max(p.data_ptr() + p.numel() for p in pieces) <= lo
        # The operator space is full up to the boundary.
        assert max(p.data_ptr() + p.numel() for p in pieces) == lo
        assert cmp.region_move(-1) == 0
    finally:
        del pieces


def test_moving_down_takes_only_free_blocks_all_or_none(cmp):
    cmp.region_init(REGION_B)
    boundary = cmp.region_boundary()
    pieces = _fill(PIECE) + _fill(MIB)
    try:
        top = max(range(len(pieces)), key=lambda i: pieces[i].data_ptr())
        n = pieces[top].numel() // MIB
        # A block whose free has not yet run on its stream is pending,
        # not free: the boundary does not move.
        s = torch.cuda.current_stream()
        torch.cuda._sleep(1_000_000_000)        # about a second
        pieces[top] = None
        assert cmp.region_move(-n) == 0
        s.synchronize()
        # Completed, it is drained and taken, but all or none: one block
        # more is allocated.
        assert cmp.region_move(-(n + 1)) == 0
        assert cmp.region_boundary() == boundary
        assert cmp.region_move(-n) == n
        assert cmp.region_boundary() == boundary - n
        assert cmp.get_arena_stats()["region_b"] == REGION_B + n * MIB
    finally:
        del pieces
    torch.cuda.synchronize()
    assert cmp.region_move(-8) == 8
    with pytest.raises(RuntimeError, match="the operator space holds"):
        cmp.region_move(-(cmp.region_boundary() + 1))
    # The most negative count is refused by the range check, not negated.
    with pytest.raises(RuntimeError, match="down by 9223372036854775808"):
        cmp.region_move(-(1 << 63))


def test_a_guarded_block_is_not_free(cmp):
    """A block force-drained while its free is still queued keeps a guard:
    it is in the clean list, but the region does not take it until the
    free has run."""
    cmp.region_init(REGION_B)
    b = torch.cuda.Stream()
    pieces = _fill(PIECE) + _fill(MIB)
    try:
        top = max(range(len(pieces)), key=lambda i: pieces[i].data_ptr())
        big = min((i for i, p in enumerate(pieces) if p.numel() == PIECE),
                  key=lambda i: pieces[i].data_ptr())
        assert big != top
        # Both frees queue behind a sleep on the stream that allocated them.
        s = torch.cuda.current_stream()
        torch.cuda._sleep(1_000_000_000)        # about a second
        pieces[top] = pieces[big] = None
        # On another stream no clean run fits: a force-drain makes both
        # clean with guards, and first fit takes the lowest 64 MiB run.
        with torch.cuda.stream(b):
            taken = torch.empty(PIECE, dtype=torch.uint8, device="cuda:0")
        assert cmp.get_arena_stats(False)["pending_b"] == 0
        assert cmp.region_move(-1) == 0
        s.synchronize()
        assert cmp.region_move(-1) == 1
        del taken
    finally:
        del pieces


def test_the_top_free_run_ends_at_the_boundary(cmp):
    """``top_free_run_b`` is the clean free run that ends at the region's
    boundary, which the region's lowest blocks join when it moves up: 0
    when an allocation, or a block whose free has not run, sits just below
    the boundary."""
    cmp.region_init(REGION_B)
    st = cmp.get_arena_stats()
    top = st["top_free_run_b"]
    assert 0 < top <= st["largest_free_run_b"]
    assert cmp.region_move(64) == 64
    assert cmp.get_arena_stats()["top_free_run_b"] == top + 64 * MIB
    assert cmp.region_move(-64) == 64
    assert cmp.get_arena_stats()["top_free_run_b"] == top
    pieces = _fill(PIECE) + _fill(MIB)
    try:
        assert cmp.get_arena_stats()["top_free_run_b"] == 0
        hi = max(range(len(pieces)), key=lambda i: pieces[i].data_ptr())
        n = pieces[hi].numel()
        s = torch.cuda.current_stream()
        torch.cuda._sleep(1_000_000_000)        # about a second
        pieces[hi] = None
        assert cmp.get_arena_stats(False)["top_free_run_b"] == 0
        s.synchronize()
        assert cmp.get_arena_stats()["top_free_run_b"] == n
        # The region's lowest blocks extend it.
        assert cmp.region_move(16) == 16
        assert cmp.get_arena_stats()["top_free_run_b"] == n + 16 * MIB
    finally:
        del pieces


def test_peaks_and_stats_count_operator_allocations_only(cmp):
    cmp.reset_window_peak()
    base = cmp.get_window_peak_bytes()
    assert base == cmp.get_arena_stats()["allocated_b"]
    cmp.region_init(REGION_B)
    assert cmp.get_window_peak_bytes() == base
    assert cmp.get_arena_stats()["allocated_b"] == base
    t = torch.empty(8 * MIB, dtype=torch.uint8, device="cuda:0")
    assert cmp.get_window_peak_bytes() == base + 8 * MIB
    del t
    cmp.region_move(-16)
    cmp.region_move(16)
    assert cmp.get_window_peak_bytes() == base + 8 * MIB


_OWN_ARENA = textwrap.dedent("""
    import torch
    import TQPlib.tqpmemory as cmp
    MIB = 1 << 20
    cmp.init(1024 * MIB, MIB)
    assert cmp.region_init(256 * MIB) == 256
    a = cmp.malloc(100 * MIB)
    assert cmp.get_arena_stats()["allocated_b"] == 100 * MIB
    cmp.free_all()
    s = cmp.get_arena_stats()
    assert s["region_b"] == 256 * MIB, s
    assert s["capacity_b"] == s["free_b"] == s["largest_free_run_b"] == 768 * MIB, s
    assert s["allocated_b"] == 0, s
    try:
        cmp.malloc(800 * MIB)
    except RuntimeError as e:
        assert "Pool out of memory" in str(e)
    else:
        raise AssertionError("an allocation reached into the region")
    cmp.region_move(256)
    assert cmp.malloc(800 * MIB) != 0
    print("ok")
""")


def test_free_all_leaves_the_region_and_the_oom_dump_names_it(cuda_allocator):
    if not cuda_allocator:
        pytest.skip("needs the arena allocator")
    env = dict(os.environ, PYTHONPATH=_REPO_ROOT)
    r = subprocess.run([sys.executable, "-c", _OWN_ARENA], cwd=_REPO_ROOT,
                       env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
    assert "Cache region: 256 MB (256 blocks" in r.stderr
