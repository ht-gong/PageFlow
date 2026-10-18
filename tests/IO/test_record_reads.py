"""The DirectIn reader over record tables: a tick copies and decodes only the
records its rows overlap.

CPU tests cover the compiled geometry over record tables of regions and
the header-parse fail-safe; the GPU tests read RAW, compressed and mixed
columns a region producer wrote, host and GPU resident, over random row
ranges, and the zero-copy mask stage's mask, against the source rows and
the movement ledger."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from IO.pages.record_table import RecordTableBuilder
from IO.pages.region import DEVICE, HOST, Region
from IO.telemetry import movement
from IO.transfer.in_stage import DirectIn
from IO.transfer.tick_layout import compile_tick_layout
from tests.record_tables import stream_row_offsets
from variable import TableVariable, Variable

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")


@pytest.fixture
def ledger(telemetry):
    telemetry(experiment=True)
    movement.reset()
    yield
    movement.reset()


@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


# ── CPU: compiled geometry ───────────────────────────────────────────────

def _host_raw_column(values: torch.Tensor, rows_per_region: int):
    """``values`` (2-D int64) as a RAW host column: one region and one
    record per ``rows_per_region`` rows."""
    builder = RecordTableBuilder(values.dtype, values.shape[1], "int")
    for lo in range(0, values.shape[0], rows_per_region):
        t = values[lo:lo + rows_per_region].clone()
        k = builder.add_region(Region(t.nbytes, HOST,
                                      t.reshape(-1).view(torch.uint8)))
        builder.append(k, 0, t.nbytes, t.nbytes)
    return TableVariable(builder.freeze(), "int")


def test_raw_rows_copied_by_rows():
    values = torch.arange(300, dtype=torch.int64).unsqueeze(1)
    col = _host_raw_column(values, 100)
    stage = DirectIn(col.get_sliced_rep(slice(150, 260)))
    layout = compile_tick_layout([stage])
    view = layout.per_stage_views[0]
    runs = layout.per_stage_bindings[0]
    assert runs.region.tolist() == [1, 2]
    # Rows 50..99 of region 1, rows 0..59 of region 2, back to back.
    assert list(zip(runs.src_offset.tolist(), runs.nbytes.tolist())) == [
        (400, 400), (0, 480)]
    assert view['total_rows'] == 110 and view['row_range'] is None
    assert layout.pack_args[1] == [0, 400]
    assert int(runs.logical_b.sum()) == view['total_rows'] * 8


def _compressed_regions(regions, handles=True):
    """A one-stream table of ``regions`` host regions, each four
    512-aligned compressed records of 16 int64 rows (100 stored bytes)."""
    builder = RecordTableBuilder(torch.int64, 1, "int")
    for _ in range(regions):
        k = builder.add_region(Region(
            3 * 512 + 100, HOST, torch.zeros(3 * 512 + 100,
                                             dtype=torch.uint8)))
        builder.append(k, np.arange(4, dtype=np.int64) * 512, 100, 128,
                       codec_key=(1, 8, 1024),
                       handles=np.arange(1, 5) if handles else None)
    return builder.freeze()


def test_compressed_runs_cover_only_overlapping_records():
    col = TableVariable(_compressed_regions(2), "int")
    # Rows 20..70: records 1-3 of region 0 and record 0 of region 1.
    stage = DirectIn(col.get_sliced_rep(slice(20, 70)))
    layout = compile_tick_layout([stage])
    runs = layout.per_stage_bindings[0]
    view = layout.per_stage_views[0]
    template = layout.codec_groups[0]['mgr_template']
    assert list(zip(runs.region.tolist(), runs.src_offset.tolist(),
                    runs.nbytes.tolist(), runs.logical_b.tolist())) == [
        (0, 512, 2 * 512 + 100, 3 * 128), (1, 0, 100, 128)]
    assert view['total_rows'] == 64
    assert view['row_range'] == slice(4, 54)
    # Region 0's run is 1124 B, padded to 1536.
    assert template.compressed_relative.tolist() == [0, 512, 1024, 1536]
    assert template.output_relative.tolist() == [
        i * 128 for i in range(len(template.output_relative))]
    # Every run starts 512-aligned in agg_buf.
    assert all(d % 512 == 0 for d in layout.pack_args[1])


# ── GPU: rows read back, by codec and residency ──────────────────────────

def _on_gpu(table):
    """``table`` with each region's bytes copied to a GPU region."""
    builder = RecordTableBuilder(table.stored_dtype, table.width,
                                 table.logical_type,
                                 num_streams=table.num_streams)
    for region in table.regions:
        builder.add_region(Region(region.nbytes, DEVICE,
                                  region.storage.to("cuda:0")))
    streams = table.record_stream()
    i = 0
    while i < table.num_records:
        j = i
        while j < table.num_records and table.codec[j] == table.codec[i]:
            j += 1
        cid = int(table.codec[i])
        builder.append(table.region[i:j], table.offset[i:j],
                       table.nbytes[i:j], table.raw_bytes[i:j],
                       stream=streams[i:j],
                       codec_key=table.codec_keys[cid] if cid else None,
                       handles=table.handle[i:j] if cid else None)
        i = j
    return builder.freeze()


def _produced(pool, values, codecs, tensor_type="int", device="host"):
    """``values`` (2-D, CPU) written by a region producer, one tick per
    entry of ``codecs`` (equal row splits): one region per tick, its
    records of at most ``TQP_MGR_RECORD_MB``."""
    from IO.codec.region_producer import RegionProducer
    producer = RegionProducer(pool, "test")
    for part, codec in zip(torch.tensor_split(values, len(codecs)), codecs):
        producer.compress([(0, part.contiguous().to("cuda:0"),
                            [part.shape[0]], codec)])
        producer.pack()
    producer.drain()
    table = producer.tables()[0]
    if device == "gpu":
        table = _on_gpu(table)
    return TableVariable(table, tensor_type)


def _read(stages):
    """One tick's read, as the exchange runs it, on one stream."""
    DirectIn.install_compiled_tick_layout(stages)
    agg_buf, plan = DirectIn.plan_input_data_movement(stages)
    if plan is not None:
        DirectIn.launch_h2d(agg_buf, plan)
        DirectIn.launch_d2d(agg_buf, plan)
    for s in stages:
        s.phase_finalize(None)
        s.decompress()
    torch.cuda.synchronize()
    return [s.gpu_col.data() for s in stages]


def _column(pool, kind, values, residency):
    """A region producer's column of eight ticks: all RAW, all one codec, or
    a write flip (BITCOMP ticks, then RAW ticks)."""
    codecs = (["BITCOMP"] * 4 + ["RAW"] * 4 if kind == "mixed"
              else [kind] * 8)
    return _produced(pool, values, codecs, device=residency)


@requires_cuda
@pytest.mark.parametrize("residency", ["host", "gpu"])
@pytest.mark.parametrize("kind", ["RAW", "BITCOMP", "ANS", "mixed"])
def test_rows_read_back_exactly(cuda_allocator, ledger, pool, kind,
                                residency):
    rng = np.random.default_rng(7)
    rows = 200_000
    values = torch.from_numpy(
        rng.integers(0, 1 << 20, rows).astype(np.int64)).unsqueeze(1)
    col = _column(pool, kind, values, residency)
    table = col.record_table()
    assert table.num_records > 3
    bounds = [0, rows] + table.global_row_start().tolist()
    cases = [(0, rows), (1, rows - 1)]
    cases += [(max(b - 3, 0), min(b + 5, rows)) for b in bounds]
    cases += [tuple(sorted(int(x) for x in rng.integers(0, rows + 1, 2)))
              for _ in range(12)]
    for start, stop in cases:
        if start == stop:
            continue
        movement.reset()
        (got,) = _read([DirectIn(col.get_sliced_rep(slice(start, stop)))])
        assert torch.equal(got.cpu(), values[start:stop].squeeze(1)), (
            f"{kind} {residency} rows [{start}, {stop})")
        reads = movement.report()["kinds"]
        read = {f: sum(reads[k][f] for k in movement.READ_KINDS)
                for f in ("logical_b", "wire_b")}
        sliced = col.get_sliced_rep(slice(start, stop))
        # Compressed records whole, RAW records only the rows used.
        sel = sliced.record_selection()
        t, recs = sel.table, sel.records
        g0 = t.global_row_start()[recs]
        first = int(g0[0]) + sel.head_rows
        kept = np.clip(np.minimum(g0 + t.rows[recs], first + sel.rows)
                       - np.maximum(g0, first), 0, None)
        raw = t.codec[recs] == 0
        assert read["logical_b"] == int(t.raw_bytes[recs][~raw].sum()
                                        + kept[raw].sum() * 8)
        assert read["wire_b"] <= sel.stored_bytes()


@requires_cuda
def test_one_tick_of_several_codecs_and_stages(cuda_allocator, pool):
    rng = np.random.default_rng(3)
    rows = 120_000
    cols, values = [], []
    for kind in ("RAW", "BITCOMP", "ANS", "BITCOMP", "mixed"):
        v = torch.from_numpy(
            rng.integers(0, 1 << 30, rows).astype(np.int64)).unsqueeze(1)
        values.append(v)
        cols.append(_column(pool, kind, v, "host"))
    start, stop = 31_337, 77_777
    got = _read([DirectIn(c.get_sliced_rep(slice(start, stop))) for c in cols])
    for g, v in zip(got, values):
        assert torch.equal(g.cpu(), v[start:stop].squeeze(1))


@requires_cuda
@pytest.mark.parametrize("compressed", [True, False])
def test_zero_copy_mask_reads_only_its_records(cuda_allocator, ledger, pool,
                                               monkeypatch, compressed):
    from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
    # 1 MiB records: four per tick's region.
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    rng = np.random.default_rng(5)
    rows = 8 << 20
    mask = torch.from_numpy(rng.random(rows) < 0.05).unsqueeze(1)
    col = _produced(pool, mask, ["BITCOMP" if compressed else "RAW"] * 2,
                    tensor_type="bool")
    start, stop = 1_234_567, 2_345_678
    stage = ZeroCopyMaskIn(None, col.get_sliced_rep(slice(start, stop)), 1)
    ZeroCopyMaskIn.launch_h2d(None, [stage])
    torch.cuda.synchronize()
    assert torch.equal(stage.mask_gpu.cpu(), mask[start:stop].squeeze(1))
    read = movement.report()["kinds"]["restore_h2d"]
    # Less than the regions the rows lie in.
    sel = col.get_sliced_rep(slice(start, stop)).record_selection()
    table = sel.table
    whole = int(table.raw_bytes[np.isin(
        table.region, np.unique(table.region[sel.records]))].sum())
    assert read["logical_b"] < whole
    if not compressed:
        assert read["logical_b"] == stop - start


# ── missing handles, empty reads ─────────────────────────────────────────

def _fake_ingest(monkeypatch):
    """Header parses replaced by fresh handle values; returns the record
    count of each parse."""
    import TQPlib.tqpcompress as tqpcompress
    from IO.transfer import tick_layout
    parsed = []
    issued = [100]

    def ingest(codec_idx, elem_size, chunk_b, ptrs):
        parsed.append(ptrs.numel())
        out = torch.arange(1, ptrs.numel() + 1, dtype=torch.int64) + issued[0]
        issued[0] += 100
        return out

    monkeypatch.setattr(tqpcompress, "mgr_ingest_configs", ingest,
                        raising=False)
    monkeypatch.setattr(tick_layout.tqpcompress, "mgr_decompress",
                        lambda *a: None)
    return parsed


def _launch(layout):
    from IO.transfer.tick_layout import _launch_mgr_group
    template = layout.codec_groups[0]['mgr_template']
    _launch_mgr_group({'agg_buf': torch.empty(4096, dtype=torch.uint8),
                       'mgr_template': template},
                      torch.empty(4096, dtype=torch.uint8))
    return template


def test_a_record_without_a_handle_is_parsed_at_its_launch(monkeypatch):
    col = TableVariable(_compressed_regions(1, handles=False), "int")
    parsed = _fake_ingest(monkeypatch)
    # Two ticks compiled up front, as a pipe does: both lack handles.
    t1 = compile_tick_layout([DirectIn(col.get_sliced_rep(slice(0, 40)))])
    t2 = compile_tick_layout([DirectIn(col.get_sliced_rep(slice(30, 64)))])
    templates = []
    for layout in (t1, t2):
        assert layout.codec_groups[0]['mgr_template'].launch_args is None
        templates.append(_launch(layout))
    # Each launch parses its own three records, for its template only: the
    # parsed configs are never shared, and the table does not keep them.
    assert parsed == [3, 3]
    assert set(templates[0].handles.tolist()).isdisjoint(
        templates[1].handles.tolist())
    assert col.record_table().handle.tolist() == [0, 0, 0, 0]


@requires_cuda
def test_an_empty_column_can_be_read_again(cuda_allocator):
    from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
    stage = DirectIn(Variable(tensor_type="bool"))
    DirectIn.install_compiled_tick_layout([stage])
    for _ in range(2):
        stage.gpu_col = None
        assert DirectIn.plan_input_data_movement([stage]) == (None, None)
        assert stage.gpu_col.data().numel() == 0
    mask = ZeroCopyMaskIn(None, Variable(tensor_type="bool"), 1)
    for _ in range(2):
        mask._read_mask()
        assert mask.mask_gpu.numel() == 0


# ── CPU: the table planner over a producer's bucket-major regions ────────

def _mixed_bucket_table(regions, buckets, rng, per=None):
    """Bucket-major regions as the region producer writes them: per region,
    every bucket's records, 8-row int64 records; RAW regions (a write flip)
    between compressed ones.  Row ``g`` of the row space holds ``g``: the
    stored bytes of each compressed record are random, and its decoded
    rows are returned per record id."""
    from IO.pages.record_table import RecordTableBuilder
    from IO.pages.region import HOST, Region
    builder = RecordTableBuilder(torch.int64, 1, "int", num_streams=buckets)
    plan = []
    for k in range(regions):
        n_per = per if per is not None else int(rng.integers(1, 3))
        streams = np.repeat(np.arange(buckets), n_per)
        n = len(streams)
        raw = k % 3 == 2                     # a RAW region, rows back to back
        size = n * (64 if raw else 512)
        i = builder.add_region(Region(size, HOST,
                                      torch.zeros(size, dtype=torch.uint8)))
        if raw:
            builder.append(i, np.arange(n) * 64, 64, 64, stream=streams)
        else:
            builder.append(i, np.arange(n) * 512, rng.integers(64, 400, n),
                           64, stream=streams, codec_key=(3, 8, 1024),
                           handles=np.arange(1, n + 1))
        plan.append(raw)
    table = builder.freeze()
    # Fill the regions: RAW records hold their rows, compressed records
    # random bytes; each record's rows are its global row ids.
    gstart = table.global_row_start()
    decoded = {}
    for r in range(table.num_records):
        rows = np.arange(gstart[r], gstart[r] + table.rows[r], dtype=np.int64)
        storage = table.regions[int(table.region[r])].storage
        off, nb = int(table.offset[r]), int(table.nbytes[r])
        if table.codec[r] == 0:
            storage[off:off + nb] = torch.from_numpy(rows.view(np.uint8))
        else:
            storage[off:off + nb] = torch.from_numpy(
                rng.integers(0, 256, nb).astype(np.uint8))
            decoded[r] = rows
    return table, decoded


def _simulate(read, table, decoded):
    """Copy a planned read's runs out of the regions and decode it as the
    reader would; returns the kept rows."""
    region, src, nbytes, dst, _ = read.run_arr
    buf = np.zeros(read.data_bytes, dtype=np.uint8)
    for k, s, n, d in zip(region.tolist(), src.tolist(), nbytes.tolist(),
                          dst.tolist()):
        buf[d:d + n] = table.regions[k].storage[s:s + n].numpy()
    out_b = read.total_rows * read.row_bytes
    if read.is_raw:
        assert read.data_bytes == out_b
        out = buf
    else:
        out = np.zeros(out_b, dtype=np.uint8)
        for rid, d, o in zip(read.comp_ids.tolist(),
                             read.comp_data_off.tolist(),
                             read.comp_out_off.tolist()):
            off, nb = int(table.offset[rid]), int(table.nbytes[rid])
            region_bytes = table.regions[int(table.region[rid])].storage
            assert np.array_equal(buf[d:d + nb],
                                  region_bytes[off:off + nb].numpy())
            raw = decoded[rid].view(np.uint8)
            out[o:o + len(raw)] = raw
        for d, o, n in read.raw_rows:
            out[o:o + n] = buf[d:d + n]
    rows = out.view(np.int64)
    return rows if read.row_range is None else rows[read.row_range]


def test_the_table_planner_reads_bucket_major_regions():
    from IO.transfer.tick_layout import _plan_table_read
    rng = np.random.default_rng(23)
    for _ in range(6):
        table, decoded = _mixed_bucket_table(int(rng.integers(2, 8)), 6, rng)
        n = table.num_rows
        cases = [(0, n)] + [tuple(sorted(int(x) for x in
                                         rng.integers(0, n + 1, 2)))
                            for _ in range(30)]
        offs = stream_row_offsets(table)
        cases += [(int(offs[a]), int(offs[b])) for a in range(6)
                  for b in range(a + 1, 7)]
        for start, stop in cases:
            if start == stop:
                continue
            read = _plan_table_read(table.select_rows(start, stop))
            assert np.array_equal(_simulate(read, table, decoded),
                                  np.arange(start, stop)), (start, stop)
            # A range of whole buckets copies one compressed run per
            # region it touches (a region here is all RAW or all
            # compressed).
            if start in offs and stop in offs and not read.is_raw:
                comp_regions = np.unique(table.region[read.comp_ids])
                assert np.isin(read.run_arr[0], comp_regions).sum() == \
                    len(comp_regions)


def test_the_table_stage_plans_page_shaped_regions_region_by_region():
    """One-stream tables (a filter's regions) are page-shaped: the table
    stage plans them region by region, to the same read as the table
    planner."""
    from IO.transfer.tick_layout import (
        _index_of, _plan_table_read, _plan_table_stage)
    rng = np.random.default_rng(29)
    for _ in range(6):
        table, decoded = _mixed_bucket_table(int(rng.integers(2, 8)), 1, rng,
                                             per=1)
        assert _index_of(table).ok
        n = table.num_rows
        for _ in range(30):
            start, stop = sorted(int(x) for x in rng.integers(0, n + 1, 2))
            if start == stop:
                continue
            sel = table.select_rows(start, stop)
            a, b = _plan_table_stage(sel), _plan_table_read(sel)
            assert all(np.array_equal(x, y)
                       for x, y in zip(a.run_arr, b.run_arr)), (start, stop)
            for f in ("codec_key", "is_raw", "data_bytes", "total_rows",
                      "row_range", "raw_rows"):
                assert getattr(a, f) == getattr(b, f), (f, start, stop)
            for f in ("comp_ids", "comp_data_off", "comp_out_off",
                      "comp_handles", "comp_raw_bytes"):
                assert np.array_equal(getattr(a, f), getattr(b, f)), f
            assert np.array_equal(_simulate(a, table, decoded),
                                  np.arange(start, stop))
