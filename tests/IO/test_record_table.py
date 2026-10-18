"""Record tables and regions (``IO/pages/record_table.py``, ``region.py``).

The tests are CPU-only: synthetic tables from the builder, checked against
brute-force row arithmetic. They cover the builder and freeze, row
selections, runs and freeing.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import constants
from IO.pages.record_table import RecordTable, RecordTableBuilder
from IO.pages.region import DEVICE, HOST, Region
from tests.record_tables import region_records, stream_row_offsets

ALIGN = constants.COMPRESSION_ALIGNMENT
ANS = (1, 8, 131072)


class FakePool:
    """Records what is returned to it."""

    def __init__(self):
        self.freed = []

    def free(self, ptr):
        self.freed.append(("pinned", ptr))

    def release(self, slot):
        self.freed.append(("slot", slot))


def _region(nbytes, location=HOST, release=None):
    return Region(nbytes, location,
                  storage=torch.zeros(nbytes, dtype=torch.uint8),
                  release=release)


def _aligned(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _bucket_major_table(ticks, num_streams, rng, codec_key=ANS, release=None):
    """One region per tick, records of every stream back to back in stream
    order (1-3 records per stream, 512-aligned), int64 rows."""
    b = RecordTableBuilder(torch.int64, 1, "int", num_streams=num_streams)
    expect = {s: [] for s in range(num_streams)}   # (region, rows) per record
    for t in range(ticks):
        offs, sizes, raws, streams = [], [], [], []
        cur = 0
        for s in range(num_streams):
            for _ in range(int(rng.integers(1, 4))):
                rows = int(rng.integers(1, 50))
                size = int(rng.integers(1, 3 * ALIGN))
                offs.append(cur)
                sizes.append(size)
                raws.append(rows * 8)
                streams.append(s)
                expect[s].append((t, rows))
                cur += _aligned(size)
        ri = b.add_region(_region(cur, release=release(t) if release else None))
        assert ri == t
        b.append(ri, offs, sizes, raws, stream=streams, codec_key=codec_key,
                 handles=np.arange(1, len(offs) + 1) + 1000 * t)
    return b.freeze(), expect


# ---------------------------------------------------------------------------
# Builder and freeze
# ---------------------------------------------------------------------------

def test_freeze_orders_by_stream_and_numbers_rows():
    rng = np.random.default_rng(1)
    table, expect = _bucket_major_table(4, 3, rng)
    assert table.num_streams == 3
    for s in range(3):
        lo, hi = table.stream_ptr[s], table.stream_ptr[s + 1]
        got = list(zip(table.region[lo:hi].tolist(), table.rows[lo:hi].tolist()))
        assert got == expect[s]                    # tick order kept per stream
        rows = table.rows[lo:hi]
        assert table.row_start[lo:hi].tolist() == (
            np.cumsum(rows) - rows).tolist()
        assert table.stream_rows()[s] == rows.sum()
    assert table.num_rows == table.rows.sum()
    assert (table.global_row_start() == np.cumsum(table.rows) - table.rows).all()
    assert (table.record_stream() == np.repeat(np.arange(3),
                                               np.diff(table.stream_ptr))).all()


def test_builder_rejects_bad_records():
    b = RecordTableBuilder(torch.int32, 2)         # 8-byte rows
    ri = b.add_region(_region(4096))
    with pytest.raises(AssertionError):
        b.append(ri, 0, 100, 12)                   # not whole rows
    with pytest.raises(AssertionError):
        b.append(ri, 4000, 200, 16)                # overruns the region
    with pytest.raises(AssertionError):
        b.append(ri, 8, 100, 16, codec_key=ANS)    # compressed, unaligned
    b.append(ri, 0, 100, 16, codec_key=ANS)
    with pytest.raises(AssertionError):
        b.append(ri, 512, 100, 16, codec_key=(1, 4, 65536))   # other key
    b.freeze()
    with pytest.raises(AssertionError):
        b.append(ri, 1024, 100, 16)


def test_overlapping_records_rejected_at_freeze():
    b = RecordTableBuilder(torch.int64, 1)
    ri = b.add_region(_region(2048))
    b.append(ri, [0, 512], [600, 100], [8, 8], codec_key=ANS)
    with pytest.raises(AssertionError):
        b.freeze()


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def _record_rows(table):
    start = table.global_row_start()
    return [range(int(a), int(a + n)) for a, n in zip(start, table.rows)]


def _check_row_selection(table, start, stop):
    sel = table.select_rows(start, stop)
    spans = _record_rows(table)
    want = [r for r in range(table.num_records)
            if spans[r].start < stop and spans[r].stop > start
            and start < stop]
    assert sel.records.tolist() == want
    assert sel.rows == stop - start
    if want:
        assert table.global_row_start()[want[0]] + sel.head_rows == start
        assert sel.head_rows == start - spans[want[0]].start
        assert sel.tail_rows == spans[want[-1]].stop - stop
        decoded = [row for r in want for row in spans[r]]
        kept = decoded[sel.head_rows:sel.head_rows + sel.rows]
        assert kept == list(range(start, stop))
    else:
        assert sel.num_records == 0 and sel.head_rows == 0


def test_select_rows_matches_brute_force():
    rng = np.random.default_rng(2)
    table, _ = _bucket_major_table(3, 2, rng)
    n = table.num_rows
    for start in range(n + 1):
        for stop in range(start, min(n, start + 90) + 1):
            _check_row_selection(table, start, stop)
    with pytest.raises(AssertionError):
        table.select_rows(0, n + 1)


def test_selection_bytes():
    rng = np.random.default_rng(4)
    table, _ = _bucket_major_table(2, 2, rng)
    sel = table.select_rows(3, table.num_rows - 2)
    assert sel.stored_bytes() == table.nbytes[sel.records].sum()
    assert sel.decoded_rows * 8 == table.raw_bytes[sel.records].sum()


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------

def _check_runs(table, records, runs, align):
    """Each run is one region's byte range holding exactly the given
    records, laid out align-aligned; each record lands inside its run."""
    for k in range(runs.num_runs):
        assert runs.dst_offset[k] % align == runs.src_offset[k] % align
        mine = records[runs.record_run == k]
        assert (table.region[mine] == runs.region[k]).all()
        lo = table.offset[mine].min()
        hi = (table.offset[mine] + table.nbytes[mine]).max()
        assert (lo, hi - lo) == (runs.src_offset[k], runs.nbytes[k])
        # no record of the region outside the selection lies inside the run
        others = region_records(table, int(runs.region[k]))
        inside = others[(table.offset[others] >= lo) & (table.offset[others] < hi)]
        assert sorted(inside.tolist()) == sorted(mine.tolist())
    assert (runs.record_dst == runs.dst_offset[runs.record_run]
            + table.offset[records] - runs.src_offset[runs.record_run]).all()
    ends = runs.dst_offset + runs.nbytes
    assert (ends[:-1] <= runs.dst_offset[1:]).all()
    assert runs.total_bytes >= (ends[-1] if runs.num_runs else 0)


def test_runs_one_per_region_for_rows_and_adjacent_streams():
    rng = np.random.default_rng(5)
    table, _ = _bucket_major_table(4, 5, rng)
    offs = stream_row_offsets(table)
    for start, stop in ((offs[1], offs[4]), (offs[0], offs[5]),
                        (offs[2] + 1, offs[3] - 1)):
        sel = table.select_rows(int(start), int(stop))
        runs = table.runs(sel.records)
        regions = np.unique(table.region[sel.records])
        assert runs.region.tolist() == regions.tolist()
        _check_runs(table, sel.records, runs, ALIGN)


def test_runs_split_at_gaps_and_pack_dense():
    rng = np.random.default_rng(6)
    table, _ = _bucket_major_table(2, 3, rng)
    region0 = region_records(table, 0)
    assert len(region0) >= 3
    picked = np.sort(region0[[0, 2]])              # a gap at region0[1]
    recs = np.sort(np.r_[picked, region_records(table, 1)])
    runs = table.runs(recs)
    assert runs.num_runs == 3
    _check_runs(table, recs, runs, ALIGN)
    dense = table.runs(recs, align=1)
    assert dense.dst_offset.tolist() == [0] + np.cumsum(dense.nbytes)[:-1].tolist()
    assert dense.total_bytes == dense.nbytes.sum()
    empty = table.runs(np.zeros(0, dtype=np.int64))
    assert empty.num_runs == 0 and empty.total_bytes == 0


def test_runs_keep_source_alignment():
    """A run that starts at an unaligned RAW record keeps the compressed
    record after it 512-aligned in the staging buffer."""
    b = RecordTableBuilder(torch.int64, 1)
    first = b.add_region(_region(ALIGN))
    mixed = b.add_region(_region(2 * ALIGN))
    b.append(first, 0, 24, 24)                     # RAW, 3 rows
    b.append(mixed, 112, 400, 400)                 # RAW at 112, ends at 512
    b.append(mixed, ALIGN, 100, 80, codec_key=ANS)
    table = b.freeze()
    recs = np.arange(table.num_records)
    runs = table.runs(recs)
    assert runs.num_runs == 2
    _check_runs(table, recs, runs, ALIGN)
    assert runs.record_dst.tolist() == [0, ALIGN + 112, 2 * ALIGN]
    assert runs.total_bytes == 3 * ALIGN


# ---------------------------------------------------------------------------
# Free
# ---------------------------------------------------------------------------

def test_free_returns_every_region_once():
    rng = np.random.default_rng(7)
    table, _ = _bucket_major_table(3, 4, rng, release=lambda t: 100 + t)
    cpu, gpu = FakePool(), FakePool()
    table.free(cpu, gpu)
    assert cpu.freed == [("pinned", 100), ("pinned", 101), ("pinned", 102)]
    assert all(r.freed and r.storage is None for r in table.regions)
    table.free(cpu, gpu)                           # a second free: nothing
    assert len(cpu.freed) == 3


def test_free_releases_each_kind_once():
    b = RecordTableBuilder(torch.int64, 1, num_streams=2)
    host = b.add_region(_region(ALIGN, HOST, release=5))
    b.add_region(_region(ALIGN, DEVICE))
    b.add_region(_region(ALIGN, HOST, release=6))   # holds no record
    b.append(host, 0, 64, 64, stream=0)
    b.append(1, 0, 64, 64, stream=1)
    table = b.freeze()
    table.host_blocks = [11]
    cpu, gpu = FakePool(), FakePool()
    table.free(cpu, gpu)
    assert cpu.freed == [("pinned", 5), ("pinned", 6), ("pinned", 11)]
    assert gpu.freed == [] and all(r.freed for r in table.regions)
    table.free(cpu, gpu)
    assert len(cpu.freed) == 3
