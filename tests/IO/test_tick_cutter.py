"""The tick cutter (``IO/pipeline/tick_cutter.py``) and whole-record reads.

A pipe's ticks are whole quanta of the record boundaries every input column
shares, packed up to the chunk budget (which rounds down to whole quanta),
and a tick's columns select their records whole: no row search at read
time, no trim after decoding, no row offset in a zero-copy read.  A pipe
whose quantum costs more than its budget is cut by rows and trims (the
fallback).  The join core packs whole buckets
by the same greedy rule.

The CPU tests check the cut over synthetic row spaces; the GPU tests read
whole-record slices of a scanned table; the SF1 test runs the 22 queries at
the default budget and checks every pipe's ticks and every tick read.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch

import constants
from IO.codec import region_producer
from IO.pipeline import tick_cutter
from adaptive.sizing_controller import whole_quanta, window_rows

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")

_REPO = pathlib.Path(__file__).resolve().parents[2]
MB = 1 << 20


@pytest.fixture(autouse=True)
def _default_record_cap(monkeypatch):
    monkeypatch.setattr(region_producer, "RECORD_MB", None)


def _scan_bounds(n, rows_per_record):
    """A scan's record boundaries: every ``rows_per_record`` from row 0."""
    return np.r_[np.arange(0, n, rows_per_record), n].astype(np.int64)


def _slice_bounds(slices, rows_per_record):
    """A producer table's: each slice cut from its own first row."""
    b = [0]
    for rows in slices:
        s = b[-1]
        b.extend(range(s + rows_per_record, s + rows, rows_per_record))
        b.append(s + rows)
    return np.asarray(b, dtype=np.int64)


def _check(ticks, start, points, row_b, budget_b):
    """Every tick ends on a point, fits the budget, and is maximal: the
    next piece would not have fit."""
    ends = ticks.ends
    starts = np.r_[start, ends[:-1]]
    assert np.isin(ends, points).all()
    assert ((ends - starts) * row_b <= budget_b).all()
    pts = points[points > start]
    for s, e, full in zip(starts.tolist(), ends.tolist(), ticks.full):
        nxt = pts[pts > e]
        if len(nxt):
            assert (int(nxt[0]) - s) * row_b > budget_b
            assert full
    assert ticks.whole.all()


# ── CPU: the cut ──────────────────────────────────────────────────────────


def test_a_scan_row_space_is_cut_in_whole_quanta():
    """Columns of 8 bytes and a 1-byte mask (56 bytes a row): the
    quantum is the mask's record, 2^20 rows, and a 256 MiB budget holds
    four of them (224 MiB)."""
    n = 6_001_215
    q = region_producer.record_rows(1)
    assert q == 1 << 20 and region_producer.record_rows(8) == 1 << 19
    bounds = np.intersect1d(_scan_bounds(n, 1 << 19), _scan_bounds(n, q))
    points = tick_cutter.quantum_points(bounds, q, q)
    ticks = tick_cutter.cut(points, 0, n, 56, 256 * MB)
    _check(ticks, 0, points, 56, 256 * MB)
    assert ticks.ends.tolist() == [4 * q, n]
    assert 4 * q * 56 == 224 * MB


def test_the_masks_a_pipe_writes_set_its_quantum():
    """8-byte inputs with 2^17-row records (a smaller cap's); a pipe that
    writes a mask keeps its ticks on 2^20 rows, so the mask's records (cut
    from each tick start) are its table's."""
    n = 5_000_000
    bounds = _scan_bounds(n, 1 << 17)
    reads_only = tick_cutter.quantum_points(bounds, 1 << 17, 1 << 17)
    assert np.array_equal(reads_only, bounds)
    writes_mask = tick_cutter.quantum_points(bounds, 1 << 17, 1 << 20)
    assert writes_mask.tolist() == list(range(0, n, 1 << 20)) + [n]
    ticks = tick_cutter.cut(writes_mask, 0, n, 17, 100 * MB)
    _check(ticks, 0, writes_mask, 17, 100 * MB)
    mask_records = np.unique(np.concatenate(
        [np.arange(s, e, 1 << 20) for s, e in
         zip(np.r_[0, ticks.ends[:-1]], ticks.ends)] + [[n]]))
    assert np.array_equal(mask_records, writes_mask)


def test_producer_slices_restart_the_quanta():
    """A producer table's records start again at each slice: the kept
    points are whole quanta from each slice start, and each slice end."""
    slices = [3_000_123, 2_500_000, 700_001, 1 << 20]
    bounds = _slice_bounds(slices, 1 << 17)
    points = tick_cutter.quantum_points(bounds, 1 << 17, 1 << 20)
    starts = np.r_[0, np.cumsum(slices)]
    want = sorted({int(s) + k for s, e in zip(starts[:-1], starts[1:])
                   for k in range(0, int(e - s), 1 << 20)} | {int(starts[-1])}
                  | set(starts.tolist()))
    assert points.tolist() == want
    ticks = tick_cutter.cut(points, 0, int(bounds[-1]), 16, 32 * MB)
    _check(ticks, 0, points, 16, 32 * MB)


def test_a_quantum_over_the_budget_is_cut_by_rows():
    n = 3_000_000
    points = _scan_bounds(n, 1 << 20)
    ticks = tick_cutter.cut(points, 0, n, 16, 4 * MB)
    rows = tick_cutter.cut_rows(0, n, 16, 4 * MB)
    assert np.array_equal(ticks.ends, rows.ends)
    assert not ticks.whole.any()
    assert ticks.ends[0] == 4 * MB // 16
    # Without record tables (points None): by rows too.
    assert np.array_equal(tick_cutter.cut(None, 0, n, 16, 64 * MB).ends,
                          tick_cutter.cut_rows(0, n, 16, 64 * MB).ends)


def test_a_table_smaller_than_a_quantum_is_one_tick():
    n = 200_000
    points = tick_cutter.quantum_points(np.array([0, n]), 1 << 20, 1 << 20)
    ticks = tick_cutter.cut(points, 0, n, 300, 256 * MB)
    assert ticks.ends.tolist() == [n] and ticks.whole.all()


def test_a_cut_from_inside_a_piece_starts_with_a_partial_tick():
    """A resection after ticks cut by rows starts off the points."""
    n = 4 << 20
    points = _scan_bounds(n, 1 << 20)
    ticks = tick_cutter.cut(points, 123, n, 8, 16 * MB)
    assert ticks.ends.tolist() == [2 << 20, 4 << 20]
    assert ticks.whole.tolist() == [False, True]


def test_pack_is_the_join_cores_bucket_packing():
    """Whole buckets up to the budget, at least one per tick, empty
    buckets with the tick before them: the loop the join core ran."""
    rng = np.random.default_rng(7)
    for _ in range(200):
        sizes = rng.integers(0, 50, size=int(rng.integers(1, 40)))
        sizes[rng.random(len(sizes)) < 0.2] = 0
        budget = int(rng.integers(1, 120))
        ref, cur, cur_b = [], [], 0
        for b, pb in enumerate(sizes.tolist()):
            if cur and cur_b + pb > budget:
                ref.append(cur)
                cur, cur_b = [], 0
            cur.append(b)
            cur_b += pb
        ref.append(cur)
        ends = tick_cutter.pack(np.r_[0, np.cumsum(sizes)], budget).tolist()
        got = [list(range(a, e)) for a, e in zip([0] + ends[:-1], ends)]
        assert got == ref


def test_gather_windows_are_whole_quanta():
    q = region_producer.record_rows(8)
    assert window_rows(256 * MB, 8 + 24, q) == (256 * MB // 32) // q * q
    # Smaller than one quantum: by rows.
    assert window_rows(2 * MB, 32, q) == 2 * MB // 32
    assert window_rows(256 * MB, 32) == 256 * MB // 32


def test_under_a_sizing_floor_a_budget_rounds_up_a_quantum():
    """A sizing search never realizes a tick or window under its class
    floor: when whole quanta round under it, the next whole quantum."""
    q = 48 * MB
    assert whole_quanta(80 * MB, q) == 48 * MB             # no floor: down
    assert whole_quanta(80 * MB, q, 64 * MB) == 96 * MB    # under it: up
    assert whole_quanta(100 * MB, q, 64 * MB) == 96 * MB   # down is enough
    assert whole_quanta(80 * MB, q, 64 * MB, 90 * MB) == 48 * MB  # cap
    assert whole_quanta(30 * MB, q, 64 * MB) == 30 * MB    # under a quantum
    # The same rounding in a gather window ...
    rows = 1 << 17
    assert window_rows(80 * MB, 384, rows, 64 * MB) * 384 == 96 * MB
    assert window_rows(80 * MB, 384, rows) * 384 == 48 * MB
    # ... and in the cutter (the quantum is the largest piece).
    n = 10 * rows
    points = _scan_bounds(n, rows)
    assert tick_cutter.cut(points, 0, n, 384, 80 * MB).ends[0] == rows
    assert tick_cutter.cut(points, 0, n, 384, 80 * MB,
                           64 * MB).ends[0] == 2 * rows


# ── GPU: whole-record reads of a scanned table ─────────────────────────────


@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


@pytest.fixture
def small_records(monkeypatch):
    """64 KiB records: an 8-byte column holds 8192 rows a record."""
    from IO import data_scanner
    from IO.transfer.data_exchange import set_exchange_to_naive
    from utility.logger import set_message_logger, set_perf_logger
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()
    monkeypatch.setattr(data_scanner, "COMPRESSION_BATCH_SIZE_B", 1 << 20)
    monkeypatch.setattr(region_producer, "RECORD_MB", 1 / 16)


@requires_cuda
@pytest.mark.parametrize("codec", ["RAW", "BITCOMP"])
def test_whole_record_slices_read_without_trim(cuda_allocator, pool,
                                                small_records, codec):
    from tests.scan_table import scan_table
    from IO.transfer.in_stage import DirectIn
    from IO.transfer.tick_layout import _plan_stage_read
    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, 5000, (300_000, 1), dtype=constants.int_dtype,
                      generator=g)
    var = scan_table(x, "int", pool, codec)
    bounds = var.record_bounds()
    assert bounds[1] == 8192 and bounds[-1] == 300_000
    starts = np.array([0, 3 * 8192, 7 * 8192, 290_000])
    ends = np.array([3 * 8192, 30 * 8192, 290_000, 300_000])
    slices = var.whole_records(starts, ends)
    # 290_000 is not a record boundary.
    assert [s is None for s in slices] == [False, False, True, True]
    for s, e, sl in zip(starts[:2], ends[:2], slices[:2]):
        sel = sl.record_selection()
        assert sel.head_rows == 0 and sel.tail_rows == 0
        assert sel.rows == sel.decoded_rows == e - s
        read = _plan_stage_read(DirectIn(sl))
        assert read.row_range is None
        got = sl.col_to_gpu().cpu().reshape(-1)
        assert torch.equal(got, x[s:e, 0])
    if codec == "RAW":
        from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
        from variable import Variable
        sl = slices[1]
        src = sl.uva_source()
        assert src.row_start == 0 and src.row_stop == ends[1] - starts[1]
        mask = torch.arange(300_000) % 5 == 1
        lo, hi = int(starts[1]), int(ends[1])
        stage = ZeroCopyMaskIn(sl, Variable(tensor_type="bool"), 1)
        stage.mask_gpu = mask[lo:hi].to("cuda:0")
        stage.mask_count = int(stage.mask_gpu.sum())
        stage.execute_zero_copy()
        assert torch.equal(stage.gpu_col.data().cpu(),
                           x[lo:hi][mask[lo:hi]].squeeze(1))
    var.free_underlying_mem(pool, None)


# ── SF1: every pipe's ticks and every tick read ────────────────────────────


@pytest.mark.parametrize("chunk_mb", [256, 64, 16])
def test_sf1_ticks_read_whole_records(sf1_parquet, monkeypatch, capsys,
                                      chunk_mb):
    """The 22 SF1 queries at the default 256 MiB budget and at smaller ones
    (more ticks per pipe, and pipes whose quantum does not fit): every tick
    of a pipe cut in quanta reads whole records of every column (no trim;
    the records cover exactly its rows), fits the budget, and only a pipe
    whose quantum costs more than its budget is cut by rows; the join
    core's bucket ticks read whole records.  Results match the reference
    (at 10 significant figures at the default budget, 6 at the others: a
    sum's order follows the ticks)."""
    import main
    from IO.cache.config import set_global_buffer_strategy, BufferStrategy
    from IO.pipeline.vortex_pipeline import VortexPipeline
    from IO.transfer import tick_layout
    from IO.transfer.data_exchange import set_exchange_to_naive
    from adaptive.instage_optimization import set_adaptive_compression
    from tests.sigfig_compare import (
        SIGFIGS_GEOMETRY, SIGFIGS_THRESHOLD, compare_results)
    from utility.logger import set_message_logger
    set_message_logger(None, False)
    set_global_buffer_strategy(BufferStrategy.CPU)
    set_exchange_to_naive()

    cuts = []
    real_cut = tick_cutter.cut

    def cut(points, start, n, row_bytes, budget_b, floor_b=0):
        ticks = real_cut(points, start, n, row_bytes, budget_b, floor_b)
        cuts.append((points, start, n, row_bytes, budget_b, ticks))
        return ticks
    monkeypatch.setattr(tick_cutter, "cut", cut)

    pipes = []
    outside = []                 # reads outside a pipe's build
    current = [outside]          # where reads go now
    real_init = VortexPipeline.__init__

    def init(self, *args, **kwargs):
        n_cuts = len(cuts)
        mine, prev = [], current[0]
        current[0] = mine        # the build compiles every tick's reads
        try:
            real_init(self, *args, **kwargs)
        finally:
            current[0] = prev
        pipes.append((self.name, list(self.tick_whole[:self.stages - 2]),
                      cuts[n_cuts] if len(cuts) > n_cuts else None, mine))
    monkeypatch.setattr(VortexPipeline, "__init__", init)

    real_plan = tick_layout._plan_stage_read

    def plan(stage):
        sel = stage.src_col.record_selection()
        # A tick's slice of whole records, or a whole column (a stage the
        # build makes over a whole column, e.g. to probe a zero-copy read).
        whole = (getattr(stage.src_col, "_records", None) is not None
                 or sel.rows == sel.table.num_rows)
        current[0].append((whole, sel.head_rows, sel.tail_rows, sel.rows,
                           sel.decoded_rows))
        return real_plan(stage)
    monkeypatch.setattr(tick_layout, "_plan_stage_read", plan)

    set_adaptive_compression(True)
    default_chunk = constants.get_chunk_size_b()
    constants.set_chunk_size_b(chunk_mb * MB)
    try:
        main.exec(1, list(range(1, 23)), None, None, True, None)
    finally:
        constants.set_chunk_size_b(default_chunk)
        set_adaptive_compression(False)

    def whole_reads(rs, name):
        for whole_rec, head, tail, rows, decoded in rs:
            assert head == 0 and tail == 0 and rows == decoded, name

    budget = chunk_mb * MB
    by_rows = rows_ticks = whole_ticks = n_reads = trimmed = 0
    for name, whole, got, rs in pipes:
        n_reads += len(rs)
        if got is None:                       # the join core: buckets
            assert all(whole), name
            whole_reads(rs, name)
            whole_ticks += len(whole)
            continue
        points, start, n, row_b, budget_b, ticks = got
        assert budget_b <= budget, name
        if not ticks.whole.all():             # cut by rows
            assert not ticks.whole.any(), name
            assert points is None or (
                tick_cutter.max_piece(points) * row_b > budget_b), name
            by_rows += 1
            rows_ticks += len(ticks)
            trimmed += sum(1 for r in rs if r[1] or r[2])
            continue
        assert whole == [True] * len(ticks), name
        assert all(r[0] for r in rs), name    # whole-record slices
        whole_reads(rs, name)
        _check(ticks, start, points, row_b, budget_b)
        whole_ticks += len(ticks)
    assert whole_ticks > 0 and n_reads > 0
    whole_reads(outside, "outside a pipe")    # whole columns
    with capsys.disabled():
        print(f"\n[ticks] SF1 at {chunk_mb} MiB: {len(pipes)} pipes, {whole_ticks} whole-"
              f"quanta ticks; {by_rows} pipes ({rows_ticks} ticks) cut by "
              f"rows, {trimmed} of their reads trimmed; {n_reads} reads")

    res = (_REPO / "results.out").read_text(encoding="utf-8")
    ref = (_REPO / "tests" / "verification" / "SF1_reference.out").read_text(
        encoding="utf-8")
    compare_results(res, ref,
                    SIGFIGS_THRESHOLD if chunk_mb == 256 else SIGFIGS_GEOMETRY)
