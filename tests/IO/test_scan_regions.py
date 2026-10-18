"""The scan's record tables and the zero-copy readers' segments.

The scan (``IO/data_scanner.py``) builds each representation of a column as
a record table of regions, one region per scan batch, written by a region
producer, one tick per batch, its regions carved from the producer's pinned
chunks (``RecordTable.host_blocks``); records are a power-of-two number of
rows within the record cap, cut from row 0, and batches whole row quanta, so
every boundary of a narrower column is one of each wider column of its
table.  A tick slice of such a column holds no reference, so freeing it
frees nothing; the column's own Variable frees every batch.  A column built
from a tensor (``tests/scan_table.py``) is cut as the scan cuts the
parquet's.  On SF1 a table's columns are scanned together, as a query scans
them, against the reader's decode (``IO/parquet_reader.py``), also with
batches that span the reader's chunks and part files.

The zero-copy readers (``IO/zero_copy_reader.cu``) read segments: device
aliases of host regions, with explicit row starts, and the rows a read
covers.  A gather's indices count from the source's first row, also when
that row is not the first of a region.  The movement
ledger files a scanned table's reads as scans, a producer's as restores.
"""
from __future__ import annotations

import json
import pathlib
import pickle
from collections import defaultdict

import numpy as np
import pytest
import torch

import constants
from IO import data_scanner
from IO.pages.combined_column_rep import CombinedColumnRep
from IO.pages.record_table import RecordTableBuilder
from IO.pages.region import HOST, Region
from IO.telemetry import movement
from tests.scan_table import scan_table
from utility.logger import set_message_logger, set_perf_logger
from variable import TableVariable, Variable
from tests.record_tables import region_records, stream_row_offsets

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _host_table(base):
    """A one-record RAW table over a host region (no GPU needed)."""
    builder = RecordTableBuilder(torch.int64, 1, "int")
    k = builder.add_region(Region(64, HOST, storage=torch.zeros(
        64, dtype=torch.uint8)))
    builder.append(k, 0, 64, 64)
    table = builder.freeze()
    table.base = base
    return table


def test_the_ledger_files_a_scanned_table_as_scans():
    scanned = TableVariable(_host_table(True), "int")
    produced = TableVariable(_host_table(False), "int")
    assert movement.uva_kind(scanned) == "scan_uva"
    assert movement.uva_kind(produced) == "restore_uva"
    assert movement.uva_kind(scanned.get_sliced_rep(slice(2, 6))) \
        == "scan_uva"
    assert movement.table_read_kind(scanned.record_table()) == "scan_h2d"
    assert movement.table_read_kind(produced.record_table()) \
        == "restore_h2d"


# ── GPU ────────────────────────────────────────────────────────────────────

@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


@pytest.fixture
def engine(monkeypatch):
    """Small batches and records, so a column spans several of each: a
    64 KiB record cap, which cuts an 8-byte column into 8192 rows a record
    and 131072 a batch."""
    from IO.transfer.data_exchange import set_exchange_to_naive
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()
    monkeypatch.setattr(data_scanner, "COMPRESSION_BATCH_SIZE_B", 1 << 20)
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1 / 16)


def _count_scan_allocs(pool, monkeypatch):
    """Count the pool's allocations from now on (a scan makes only its
    producers' chunks)."""
    calls = [0]
    real_try = pool.try_get_storage

    def try_get_storage(nbytes, *tag):       # get_storage's too
        got = real_try(nbytes, *tag)
        calls[0] += got is not None
        return got
    monkeypatch.setattr(pool, "try_get_storage", try_get_storage)
    return calls


def _allocs_of(table):
    """The pool allocations a scan table holds: its producer's chunks."""
    return len(table.host_blocks)


def _column(rows, width=1, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 5000, (rows, width), dtype=constants.int_dtype,
                         generator=g)


def _read(var):
    """The rows of a Variable (pages or a table), decoded, on the host, as
    (rows, width)."""
    from IO.pipeline.common import col_to_gpu
    if var.table_backed:
        out = var.col_to_gpu().cpu()
    else:
        out = col_to_gpu(CombinedColumnRep([var])).cpu()
    return out.reshape(var.col_length(), -1)


@requires_cuda
@pytest.mark.parametrize("codec", ["RAW", "BITCOMP", "ANS"])
def test_a_scan_is_one_region_per_batch_in_producer_chunks(
        cuda_allocator, pool, engine, monkeypatch, codec):
    """One region per batch, carved from the producer's chunks, which the
    table holds and frees; 64 KiB records (8192 rows) within batches."""
    x = _column(400_000)
    allocs = _count_scan_allocs(pool, monkeypatch)
    var = scan_table(x, "int", pool, codec)
    table = var.record_table()
    batch_rows = (1 << 20) // 8
    batches = -(-400_000 // batch_rows)
    assert len(table.regions) == batches
    assert allocs[0] == len(table.host_blocks) >= 1
    assert all(r.release is None and r.location == HOST
               for r in table.regions)
    want = []
    for lo in range(0, 400_000, batch_rows):
        n = min(batch_rows, 400_000 - lo)
        want += [min(8192, n - s) for s in range(0, n, 8192)]
    assert table.rows.tolist() == want
    assert table.base
    if codec == "RAW":
        assert not table.is_compressed and var.is_host_raw()
    else:
        assert table.is_compressed
    assert torch.equal(_read(var).squeeze(1), x.squeeze(1))
    tick = var.get_sliced_rep(slice(130_000, 270_001))
    assert torch.equal(_read(tick).squeeze(1), x[130_000:270_001, 0])


@requires_cuda
@pytest.mark.parametrize("codec", ["RAW", "BITCOMP"])
def test_freeing_a_tick_slice_of_a_scan_representation_frees_nothing_else(
        cuda_allocator, pool, engine, monkeypatch, codec):
    x = _column(400_000)
    var = scan_table(x, "int", pool, codec)
    held = dict(pool._allocs)
    first = var.get_sliced_rep(slice(0, 100_000))
    first.free_underlying_mem(pool, None)
    assert pool._allocs == held
    rest = var.get_sliced_rep(slice(100_000, 400_000))
    assert torch.equal(_read(rest).squeeze(1), x[100_000:, 0])
    var.free_underlying_mem(pool, None)
    assert not pool._allocs
    assert all(r.freed for r in var.record_table().regions)


def _gather(src, idx):
    import TQPlib.zero_copy_reader as zero_copy_reader
    u = src.uva_source()
    return zero_copy_reader.zero_copy_rearrange(
        u.seg_ptrs, u.seg_rows, idx.to("cuda:0"), u.row_start, u.row_stop,
        u.dtype, u.width).cpu()


@requires_cuda
def test_zero_copy_rearrange_honours_row_start(cuda_allocator, pool, engine):
    """Indices count from the source's first row, for a slice that starts
    inside a record of a region; a null index stays null."""
    x = _column(300_000, width=2)
    table = scan_table(x, "int", pool, "RAW")
    g = torch.Generator().manual_seed(5)
    lo, hi = 70_001, 230_007
    idx = torch.randint(0, hi - lo, (50_000,), generator=g)
    idx[::97] = constants.null
    want = x[lo:hi][idx.clamp(min=0)]
    want[idx == constants.null] = constants.null
    src = table.get_sliced_rep(slice(lo, hi))
    assert src.uva_source().row_start > 0
    assert torch.equal(_gather(src, idx), want)


@requires_cuda
def test_zero_copy_mask_reads_a_slice_of_a_scan_table(cuda_allocator, pool,
                                                      engine):
    from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
    x = _column(300_000)
    var = scan_table(x, "int", pool, "RAW")
    lo, hi = 123_457, 291_011
    mask = torch.arange(300_000) % 7 == 3
    stage = ZeroCopyMaskIn(var.get_sliced_rep(slice(lo, hi)),
                           Variable(tensor_type="bool"), 1)
    stage.mask_gpu = mask[lo:hi].to("cuda:0")
    stage.mask_count = int(stage.mask_gpu.sum())
    stage.execute_zero_copy()
    assert torch.equal(stage.gpu_col.data().cpu(), x[lo:hi][mask[lo:hi]]
                       .squeeze(1))


# ── SF1: every scanned column ────────────────────────────────────────────


@pytest.fixture(scope="module")
def schema():
    return json.loads((_REPO / "queries" / "schema.json").read_text())


def _scanned_columns():
    """``{TABLE: [COLUMN, ...]}`` of every column the 22 queries scan."""
    cols = defaultdict(set)
    for qid in range(1, 23):
        with open(_REPO / "queries" / "pickles" / f"planobj-q{qid}.pkl",
                  "rb") as f:
            for plan in pickle.load(f):
                for st in plan.disk_read_cols:
                    cols[st.table.upper()].add(st.name.upper())
    return {t: sorted(c) for t, c in sorted(cols.items())}


def _decoded(root, schema, table, columns):
    """``{COLUMN: tensor}``: ``columns`` of the SF1 ``table`` read whole
    through the scan's reader, cast and 2-D as the scan takes them, on the
    host."""
    parts = defaultdict(list)
    for _, chunk in data_scanner.parquet_table(root, 1, table).batches(
            columns):
        for name, tensor in chunk.items():
            parts[name].append(tensor)
    out = {}
    for name, p in parts.items():
        tensor_type = data_scanner.TYPE_MAPPING[
            schema[table][name].split(',')[0]]
        out[name] = data_scanner.ensure_2d(
            data_scanner.cast(torch.cat(p), tensor_type)).cpu()
    return out


def _sf1_scan(root, schema, table, columns, pool):
    """``columns`` of the SF1 ``table`` in one scan, as a query scans them,
    under the adaptive policy: ``{COLUMN: [CombinedColumnRep, ...]}``."""
    from compression.format_policy import CompressionPolicy
    from IO.transfer.data_exchange import set_exchange_to_naive
    from parsing import ScannedTuple
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()
    to_scan = [ScannedTuple(c, i, table) for i, c in enumerate(columns)]
    return data_scanner.scan_to_ram(
        {"tables path": root}, schema, 1, pool, to_scan,
        policy=CompressionPolicy.adaptive())[table]


def _free(scan, pool):
    for ccrs in scan.values():
        for ccr in ccrs:
            ccr.free_underlying_mem(pool, None)
    assert not pool._allocs


def test_sf1_scan_tables_hold_every_scanned_column(sf1_parquet, schema,
                                                    monkeypatch):
    """Every SF1 column the queries scan, each table's columns in one scan
    under the adaptive policy: each representation's table, one region per
    batch, reads back the column's rows as the reader decodes them, and
    plans the same reads region by region as record by record; its records
    are cut on rows, the same in both representations, and nest across the
    columns of a table."""
    from IO.pages.pinned_mem import PinnedMemory
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", None)
    pool = PinnedMemory(capacity_gb=4, block_size_mb=0.5)
    allocs = _count_scan_allocs(pool, monkeypatch)
    reps = 0
    cuts = defaultdict(dict)       # table -> column -> (row bytes, bounds)
    for table_name, columns in _scanned_columns().items():
        want = _decoded(sf1_parquet, schema, table_name, columns)
        before = allocs[0]
        scan = _sf1_scan(sf1_parquet, schema, table_name, columns, pool)
        assert allocs[0] - before == sum(
            _allocs_of(tv.record_table()) for ccrs in scan.values()
            for ccr in ccrs for tv in ccr.reps)
        for col in columns:
            (tables,) = scan[col]
            for tv in tables.reps:
                got = tv.record_table()
                assert got.base
                assert torch.equal(_read(tv), want[col]), col
                _check_selections(tv, _slices(np.random.default_rng(1),
                                              tv.col_length(), 1_000_003))
                rows, row_b = tv.col_length(), int(got.row_bytes)
                per_batch = data_scanner._batch_rows(row_b)
                assert len(got.regions) == -(-rows // per_batch), col
                bounds = _check_row_records(got, row_b, per_batch)
                if col not in cuts[table_name]:
                    cuts[table_name][col] = (row_b, bounds)
                # Both representations cut the same records.
                assert np.array_equal(cuts[table_name][col][1], bounds), col
                reps += 1
        _free(scan, pool)
    assert reps == 106
    # Record boundaries nest across the columns of a table: each narrower
    # column's are among each wider one's, and a 1-byte mask's (one
    # quantum) among every column's.
    from IO.codec.region_producer import row_quantum
    for table_name, by_col in cuts.items():
        q = row_quantum()
        for row_b, bounds in by_col.values():
            rows = int(bounds[-1])
            mask = np.r_[np.arange(0, rows, q), rows]
            assert np.isin(mask, bounds).all(), table_name
            for other_b, other in by_col.values():
                if other_b >= row_b:
                    assert np.isin(bounds, other).all(), table_name


@pytest.mark.parametrize("table", ["LINEITEM", "ORDERS", "NATION"])
def test_sf1_batches_span_reader_chunks(sf1_parquet, schema, monkeypatch,
                                        table):
    """Batches of one 2^20-row quantum over reader chunks of 8 MiB (passes
    of 32 MiB): batches start inside chunks and span chunk and part-file
    boundaries, and still hold the column's rows, both representations
    cutting the same records.  A column scanned twice gets two
    representations."""
    from IO import parquet_reader
    from IO.pages.pinned_mem import PinnedMemory
    columns = _scanned_columns()[table]
    want = _decoded(sf1_parquet, schema, table, columns)
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", None)
    monkeypatch.setattr(data_scanner, "COMPRESSION_BATCH_SIZE_B", 1)
    firsts = []
    read = parquet_reader.ParquetTable.batches

    def batches(self, names):
        for first, chunk in read(self, names, chunk_read_limit=8 << 20,
                                 pass_read_limit=32 << 20):
            firsts.append(first)
            yield first, chunk
    monkeypatch.setattr(parquet_reader.ParquetTable, "batches", batches)
    pool = PinnedMemory(capacity_gb=2, block_size_mb=0.5)
    scan = _sf1_scan(sf1_parquet, schema, table, columns + columns[:1], pool)
    once, twice = scan[columns[0]]
    assert once is not twice
    for col in columns:
        for ccr in scan[col]:
            bounds = []
            for tv in ccr.reps:
                got = tv.record_table()
                assert torch.equal(_read(tv), want[col]), col
                assert len(got.regions) == -(-got.num_rows // (1 << 20)), col
                bounds.append(_check_row_records(got, int(got.row_bytes),
                                                 1 << 20))
            assert all(np.array_equal(b, bounds[0]) for b in bounds), col
    if table == "LINEITEM":
        # Several batches, and chunk starts that are no batch start.
        assert len(once.reps[0].record_table().regions) > 4
        assert len(firsts) > 8 and any(f % (1 << 20) for f in firsts)
    _free(scan, pool)


def _check_row_records(table, row_b, per_batch):
    """The records of one scanned representation: every record a
    power of two rows, the largest within 4 MiB and 2^20 rows (512-byte
    multiples), full
    but for the column's last, each starting at a multiple of its rows from
    row 0; every batch (region) a whole number of 2^20 row quanta, so it
    starts a record.  Returns the record boundaries in rows."""
    from IO.codec.region_producer import record_rows, row_quantum
    rr = record_rows(row_b)
    assert rr & (rr - 1) == 0 and rr * row_b % 512 == 0
    assert rr * row_b <= 4 << 20 and (rr == 1 << 20
                                      or 2 * rr * row_b > 4 << 20)
    assert row_quantum() == 1 << 20 and (per_batch % (1 << 20) == 0)
    starts = table.global_row_start()
    rows = table.rows
    assert (rows[:-1] == rr).all() and 0 < rows[-1] <= rr
    assert (starts % rr == 0).all()
    first = np.array([region_records(table, k)[0]
                      for k in range(len(table.regions))])
    assert (starts[first] % per_batch == 0).all()
    return np.r_[starts, table.num_rows]


def test_a_scan_leaves_the_arena_as_it_found_it(sf1_parquet, schema):
    """While a scan's representations are alive, as when the query that
    follows starts, the GPU arena holds what it held before the scan: the
    same allocated and free bytes and the same largest free run (cuDF gives
    back its buffers and the scan releases its producers' nvCOMP scratch)."""
    import gc
    import TQPlib.tqpmemory as cmp
    from IO.pages.pinned_mem import PinnedMemory
    from TQPlib.tqpcompress import mgr_release_scratch

    def arena():
        gc.collect()
        torch.cuda.synchronize()
        stats = cmp.get_arena_stats(True)
        return {k: stats[k] for k in
                ("allocated_b", "free_b", "largest_free_run_b")}

    # As a query's teardown in main.py leaves it: no manager holds scratch.
    mgr_release_scratch()
    pool = PinnedMemory(capacity_gb=4, block_size_mb=0.5)
    before = arena()
    for table in ("LINEITEM", "ORDERS", "PART"):
        scan = _sf1_scan(sf1_parquet, schema, table,
                         _scanned_columns()[table], pool)
        assert arena() == before, table
        _free(scan, pool)


@requires_cuda
def test_a_scan_of_many_small_files_shares_arena_blocks(tmp_path, schema,
                                                       cuda_allocator):
    """cuDF holds a small buffer per part file for a whole read pass, and one
    pass takes every file of a small table (16,000 of them at SF1000): the
    scan packs those buffers into shared arena blocks
    (``rmm_arena.small_buffers``), so a table of 2,000 one-row files peaks
    at a few blocks, not one a file."""
    import os
    import pylibcudf as plc
    from pylibcudf.gpumemoryview import gpumemoryview
    import TQPlib.tqpmemory as cmp
    from IO.pages.pinned_mem import PinnedMemory
    from IO.parquet_reader import table_dir
    from utility import rmm_arena

    files = 2000
    keys = torch.arange(files, dtype=torch.int64, device="cuda")
    tbl = plc.Table([plc.Column(plc.DataType(plc.TypeId.INT64), files,
                                gpumemoryview(keys), None, 0, 0, [])])
    meta = plc.io.types.TableInputMetadata(tbl)
    meta.column_metadata[0].set_name("s_suppkey")
    src = table_dir(str(tmp_path), 1, "supplier")
    os.makedirs(src)
    options = plc.io.parquet.ParquetWriterOptions.builder(
        plc.io.SinkInfo([os.path.join(src, f"part-{i:05d}.parquet")
                         for i in range(files)]), tbl).metadata(meta).build()
    options.set_partitions([plc.io.types.PartitionInfo(i, 1)
                            for i in range(files)])
    plc.io.parquet.write_parquet(options)

    pool = PinnedMemory(capacity_gb=1, block_size_mb=0.5)
    cmp.reset_window_peak()
    before = cmp.get_arena_stats()["allocated_b"]
    scan = _sf1_scan(str(tmp_path), schema, "SUPPLIER", ["S_SUPPKEY"], pool)
    peak = cmp.get_window_peak_bytes() - before
    assert peak < files // 20 * rmm_arena.block_bytes, peak
    _free(scan, pool)


# ── The tick reader: region by region over tables ────────────────────────

_READ_FIELDS = ("codec_key", "is_raw", "data_bytes", "total_rows",
                "row_range", "data_type", "data_width", "row_bytes",
                "raw_rows")
_READ_ARRAYS = ("comp_ids", "comp_data_off", "comp_out_off", "comp_handles",
                "comp_raw_bytes")


def assert_same_read(fast, slow):
    """Two plans of one table stage's read lay out, bind and count the
    same: every field, and the runs ``_bind_regions`` binds."""
    for name in _READ_FIELDS:
        assert getattr(fast, name) == getattr(slow, name), name
    for name in _READ_ARRAYS:
        assert np.array_equal(getattr(fast, name), getattr(slow, name)), name
    assert fast.runs == slow.runs == []
    assert len(fast.run_arr) == len(slow.run_arr) == 5
    for i, (a, b) in enumerate(zip(fast.run_arr, slow.run_arr)):
        assert np.array_equal(a, b), ("run_arr", i)


def _check_selections(var, starts_stops):
    """Plan each ``[start, stop)`` slice of ``var`` both ways."""
    from IO.transfer import tick_layout
    for start, stop in starts_stops:
        if start >= stop:
            continue
        sel = var.get_sliced_rep(slice(start, stop)).record_selection()
        got = tick_layout._plan_table_stage(sel)
        want = tick_layout._plan_table_read(sel)
        assert_same_read(got, want)


def _slices(rng, n, tick):
    out = [(0, n), (0, 1), (n - 1, n)]
    out += [(lo, min(n, lo + tick)) for lo in range(0, n, tick)]
    out += [tuple(sorted(int(v) for v in rng.integers(0, n + 1, 2)))
            for _ in range(40)]
    return out


@requires_cuda
@pytest.mark.parametrize("codec", ["RAW", "BITCOMP", "ANS"])
def test_the_region_planner_reads_a_scan_table_as_the_table_planner(
        cuda_allocator, pool, engine, codec):
    from IO.transfer import tick_layout
    x = _column(400_000, width=3)
    var = scan_table(x, "int", pool, codec)
    assert tick_layout._RegionIndex(var.record_table()).ok
    _check_selections(var, _slices(np.random.default_rng(7), 400_000,
                                   37_011))


@requires_cuda
@pytest.mark.parametrize("B", [1, 8])
@pytest.mark.parametrize("codec", ["RAW", "BITCOMP"])
def test_the_region_planner_reads_a_producer_table_as_the_table_planner(
        cuda_allocator, pool, B, codec):
    """A producer's table: one stream (page-shaped, region by region) or
    eight (bucket-major regions: the table planner); either way the same
    read."""
    from IO.codec.region_producer import RegionProducer
    from IO.transfer import tick_layout
    from IO.transfer.data_exchange import set_exchange_to_naive
    from IO.codec.batched_compress import aligned_partition_layout
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()
    rng = np.random.default_rng(3)
    producer = RegionProducer(pool, "t 0 sort pipeline")
    for _ in range(6):
        # One tick, bucket b's rows from an lcm(row bytes, 512)-aligned
        # start, as the sort pipe lays a partitioned output out.
        counts = [int(rng.integers(1, 9000)) for _ in range(B)]
        _, starts, total = aligned_partition_layout(counts, 8, 512)
        out = torch.full((total, 1), -1, dtype=constants.int_dtype,
                         device="cuda:0")
        for c, st in zip(counts, starts):
            out[st // 8:st // 8 + c] = torch.from_numpy(
                rng.integers(0, 50, (c, 1))).to("cuda:0")
        if producer.has_pending():
            producer.pack()
        producer.compress([(0, out, counts, codec)])
        torch.cuda.current_stream().synchronize()
    producer.drain()
    table = producer.tables()[0]
    var = TableVariable(table, "int")
    assert tick_layout._RegionIndex(table).ok is (B == 1)
    _check_selections(var, _slices(rng, table.num_rows, 5_003))
    streams = stream_row_offsets(table).tolist()
    _check_selections(var, list(zip(streams[:-1], streams[1:]))
                      + [(streams[1], streams[-2])])
