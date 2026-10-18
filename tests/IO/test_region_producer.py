"""The region producer and the home writer (IO/codec/region_producer.py,
IO/codec/home_writer.py).

A producer is fed partitioned ticks as a sort pipe feeds it (each bucket
padded to an aligned start) or plain ones as a filter's: compressed at tick
t, packed at t+1 after a barrier, drained at the end.  Each tick of each
column must become one region whose records sit at the cumulative aligned
sizes, bucket by bucket; the frozen table must read back every row in
partition-major order, RAW, compressed, mixed across ticks, and with a codec
group split into pieces.  The writer must hold every buffer it copies from
until its copy has ended, keep at most two shipments in flight and make the
producer wait on the oldest beyond that.  ``compute-sanitizer`` runs this
file for illegal accesses (it sees the arena as one allocation, so the data
checks are what catch a buffer reused too early)."""
from __future__ import annotations

import gc
import weakref
from math import gcd

import numpy as np
import pytest
import torch

import constants
from IO.codec import region_producer
from IO.codec.home_writer import HomeWriter
from IO.codec.region_producer import RegionProducer, record_cap_b
from tests.record_tables import region_records, stream_row_offsets

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("row_b,rows", [(1, 1 << 20), (4, 1 << 20),
                                         (8, 1 << 19), (25, 1 << 17),
                                         (16, 1 << 18), (199, 1 << 14)])
def test_row_cut_records_default_to_4_mib_and_2_20_rows(
        monkeypatch, row_b, rows):
    """By default a record is the largest power of two rows within 4 MiB
    raw and 2^20 rows: a mask's and a 4-byte column's hold 2^20 rows, wider
    columns fill 4 MiB."""
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", None)
    assert region_producer.RECORD_ROWS_MAX == 1 << 20
    assert region_producer.record_rows(row_b) == rows
    assert record_cap_b(row_b) == rows * row_b


def test_row_cut_records_nest_across_row_widths(monkeypatch):
    """Every width's rows per record is a power of two, the largest within
    the cap (or the fewest that keep 512-byte multiples), a 512-byte
    multiple, and divides the row quantum; up to rows of cap/512 bytes
    (2 KiB at 1 MiB; no column is wider), it divides every narrower
    width's too."""
    monkeypatch.setattr("IO.codec.region_producer.RECORD_ROWS_MAX", None)
    for mb in (None, 4, 1 / 16):
        monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", mb)
        cap = int((mb or 4) * (1 << 20))
        prev = None
        for row_b in range(1, 2 * cap // 512 + 1):
            rr = region_producer.record_rows(row_b)
            assert rr & (rr - 1) == 0 and rr * row_b % 512 == 0
            assert rr * row_b <= cap < 2 * rr * row_b or \
                (rr * row_b > cap and (rr // 2) * row_b % 512)
            if prev is not None and row_b <= cap // 512:
                assert prev % rr == 0
            assert region_producer.row_quantum(row_b) % rr == 0
            prev = rr


@pytest.mark.parametrize("mb,rows_max", [(4, 1 << 20), (4, 1 << 18),
                                         (1, 1 << 20), (1 / 16, 1 << 20)])
def test_row_cut_records_fit_both_the_byte_and_the_row_cap(
        monkeypatch, mb, rows_max):
    """Under a row cap, rows per record are the largest power of two within
    both caps (or the fewest that keep 512-byte multiples), so a mask's
    records stop at the row cap while wide columns fill the byte cap; the
    boundaries still nest and divide the row quantum."""
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", mb)
    monkeypatch.setattr("IO.codec.region_producer.RECORD_ROWS_MAX", rows_max)
    cap = int(mb * (1 << 20))
    prev = None
    for row_b in range(1, 2 * cap // 512 + 1):
        rr = region_producer.record_rows(row_b)
        fit = min(cap // row_b, rows_max)
        want = max(1 << (fit.bit_length() - 1), 512 // gcd(row_b, 512))
        assert rr == want and rr & (rr - 1) == 0 and rr * row_b % 512 == 0
        assert rr <= rows_max or rr == 512 // gcd(row_b, 512)
        if prev is not None and row_b <= cap // 512:
            assert prev % rr == 0
        assert region_producer.row_quantum(row_b) % rr == 0
        prev = rr
    assert region_producer.record_rows(1) == min(rows_max, cap)
    if mb == 4:
        # 8-byte rows fill the 4 MiB cap under a 2^20 row cap.
        assert region_producer.record_rows(8) == min(1 << 19, rows_max)


# ── GPU: ticks in, regions and tables out ────────────────────────────────

@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


def _padded(parts, dtype):
    """One tick of a partitioned column, laid out as the sort pipe lays it
    out: bucket b's rows from an lcm(row bytes, 512)-aligned start."""
    from IO.codec.batched_compress import aligned_partition_layout
    counts = [p.shape[0] for p in parts]
    rb = parts[0].shape[1] * parts[0].element_size()
    unit = rb * 512 // gcd(rb, 512)
    _, starts, total = aligned_partition_layout(counts, rb, unit)
    out = torch.full((total, parts[0].shape[1]), -1, dtype=dtype,
                     device="cuda:0")
    for p, s in zip(parts, starts):
        out[s // rb:s // rb + p.shape[0]] = p.to("cuda:0")
    return out, counts


def _ticks(rng, n_ticks, B, dtype, rows=(0, 4000), empty=0.15):
    """Per tick, per bucket, random rows of ``dtype`` (a bucket is empty
    with probability ``empty``)."""
    ticks = []
    for _ in range(n_ticks):
        parts = []
        for _ in range(B):
            n = int(rng.integers(*rows)) if rng.random() >= empty else 0
            vals = rng.integers(0, 50, n).astype(np.int64)
            parts.append(torch.from_numpy(vals).to(dtype).unsqueeze(1))
        ticks.append(parts)
    return ticks


def _run(producer, feeds):
    """Feed ``feeds`` (per tick, the producer's items; consumed) tick by
    tick: compress at t, the compute stream's barrier, pack at t+1, then the
    drain.  The tick's tensors are dropped right after their compress and
    their memory written over, as a pipe's next allocations would."""
    while feeds:
        if producer.has_pending():
            producer.pack()
        items = feeds.pop(0)
        producer.compress(items)
        del items
        scribble = torch.full((8 << 20,), 0x5A, dtype=torch.uint8,
                              device="cuda:0")
        del scribble
        torch.cuda.current_stream().synchronize()
    producer.drain()
    return producer.tables()


def _expected(ticks):
    """The column in partition-major order: bucket by bucket, each bucket's
    rows tick by tick."""
    B = len(ticks[0])
    return torch.cat([t[b] for b in range(B) for t in ticks]).squeeze(1)


def _read_all(table, tensor_type="int"):
    from variable import TableVariable
    return TableVariable(table, tensor_type).col_to_gpu().cpu()


@requires_cuda
@pytest.mark.parametrize("codec", ["BITCOMP", "ANS", "RAW"])
def test_each_tick_is_one_region_of_cumulative_records(cuda_allocator, pool,
                                                       codec):
    rng = np.random.default_rng(3)
    B, T = 6, 4
    cols = {0: _ticks(rng, T, B, torch.int64),
            B: _ticks(rng, T, B, torch.int64)}
    feeds = []
    for t in range(T):
        items = []
        for base, ticks in cols.items():
            tensor, counts = _padded(ticks[t], torch.int64)
            items.append((base, tensor, counts, codec))
        feeds.append(items)
    producer = RegionProducer(pool, "t 0 sort pipeline")
    tables = _run(producer, feeds)
    assert set(tables) == set(cols)
    align = constants.COMPRESSION_ALIGNMENT
    for base, ticks in cols.items():
        table = tables[base]
        assert table.num_streams == B
        assert len(table.regions) == T
        assert all(r.location == "host" for r in table.regions)
        for k in range(T):
            recs = region_records(table, k)
            streams = table.record_stream()[recs]
            # Bucket-major inside the region, each bucket's rows in order.
            assert (np.diff(streams) >= 0).all()
            if codec == "RAW":
                continue
            sizes = table.nbytes[recs]
            off = np.zeros(len(recs) + 1, dtype=np.int64)
            np.cumsum((sizes + align - 1) // align * align, out=off[1:])
            assert table.offset[recs].tolist() == off[:-1].tolist()
            assert table.regions[k].nbytes == off[-1]
        assert table.stream_rows().tolist() == [
            sum(t[b].shape[0] for t in ticks) for b in range(B)]
        assert torch.equal(_read_all(table), _expected(ticks))
        # A range of buckets, as a join-core tick reads it.
        from variable import TableVariable
        var = TableVariable(table, "int")
        rows = stream_row_offsets(table)
        part = var.get_sliced_rep(slice(int(rows[2]), int(rows[5])))
        got = part.col_to_gpu().cpu()
        assert torch.equal(got, torch.cat(
            [t[b] for b in range(2, 5) for t in ticks]).squeeze(1))
        var.free_underlying_mem(pool, None)
        assert all(r.freed for r in table.regions)


@requires_cuda
def test_a_write_flip_mixes_codecs_in_one_table(cuda_allocator, pool):
    rng = np.random.default_rng(5)
    codecs = ["RAW", "BITCOMP", "BITCOMP", "RAW", "BITCOMP"]
    ticks = _ticks(rng, len(codecs), 1, torch.int64, rows=(1000, 50_000),
                   empty=0)
    feeds = [[(0, t[0].to("cuda:0"), [t[0].shape[0]], c)]
             for t, c in zip(ticks, codecs)]
    tables = _run(RegionProducer(pool, "t filter pipe"), feeds)
    table = tables[0]
    assert sorted(set(table.codec.tolist())) == [0, 3]
    assert torch.equal(_read_all(table), _expected(ticks))


@requires_cuda
def test_a_codec_group_past_the_slab_limit_is_split(cuda_allocator, pool,
                                                    monkeypatch):
    from IO.codec import batched_compress
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 0)       # one-unit records
    monkeypatch.setattr(batched_compress, "_MAX_OUT_BUF_BYTES", 64 << 10)
    rng = np.random.default_rng(9)
    ticks = _ticks(rng, 3, 4, torch.int64, rows=(2000, 6000), empty=0)
    feeds = []
    for t in ticks:
        tensor, counts = _padded(t, torch.int64)
        feeds.append([(0, tensor, counts, "ANS")])
    producer = RegionProducer(pool, "t 0 sort pipeline")
    producer.compress(feeds.pop(0))
    # Several pieces, none waited for: their slabs stay until the pack.
    assert len(producer._tick.pieces) > 1
    assert all(p.slab is not None for p in producer._tick.pieces)
    torch.cuda.current_stream().synchronize()
    tables = _run(producer, feeds)
    table = tables[0]
    assert len(table.regions) == 3
    assert torch.equal(_read_all(table), _expected(ticks))


@requires_cuda
def test_a_big_tick_never_asks_for_more_than_one_piece(cuda_allocator, pool,
                                                       monkeypatch):
    """A tick of many pieces (two columns, one spanning several pieces)
    allocates no slab or dense buffer larger than one piece's worst case,
    and still packs each column into one region that reads back whole."""
    from IO.codec import batched_compress
    limit = 4 << 20
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    monkeypatch.setattr(batched_compress, "_MAX_OUT_BUF_BYTES", limit)
    sizes = []
    real_alloc = region_producer._alloc_slab

    def alloc(nbytes, device):
        sizes.append(nbytes)
        return real_alloc(nbytes, device)
    monkeypatch.setattr(region_producer, "_alloc_slab", alloc)
    rng = np.random.default_rng(41)
    ticks = [[_ticks(rng, 1, 1, torch.int64, rows=(1_500_000, 1_600_000),
                     empty=0)[0][0],
              _ticks(rng, 1, 1, torch.int64, rows=(200_000, 300_000),
                     empty=0)[0][0]] for _ in range(2)]
    feeds = [[(base, t[base].to("cuda:0"), [t[base].shape[0]], "BITCOMP")
              for base in (0, 1)] for t in ticks]
    producer = RegionProducer(pool, "t 0 project pipeline")
    tables = _run(producer, feeds)
    # 12+ MiB of records a tick: four pieces or more, each slab and dense
    # buffer within one piece (plus the dense buffer's offsets, 24 B per
    # record).
    assert len(sizes) >= 2 * 2 * 4
    assert max(sizes) <= limit + 24 * 8
    for base in (0, 1):
        table = tables[base]
        assert len(table.regions) == 2
        assert torch.equal(_read_all(table),
                           torch.cat([t[base] for t in ticks]).squeeze(1))


@requires_cuda
def test_a_failed_slab_leaves_no_ticket(cuda_allocator, pool, monkeypatch):
    """An allocation that fails while a tick's pieces are set up finishes
    every ticket begun, so the manager's registry holds none after it."""
    from IO.codec import batched_compress
    from TQPlib.tqpcompress import mgr_counters
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    monkeypatch.setattr(batched_compress, "_MAX_OUT_BUF_BYTES", 4 << 20)
    calls = []
    real_alloc = region_producer._alloc_slab

    def alloc(nbytes, device):
        calls.append(nbytes)
        if len(calls) == 3:
            raise torch.cuda.OutOfMemoryError("injected")
        return real_alloc(nbytes, device)
    monkeypatch.setattr(region_producer, "_alloc_slab", alloc)
    live0 = mgr_counters()["tickets_live"]
    x = torch.arange(2_000_000, dtype=torch.int64,
                     device="cuda:0").unsqueeze(1) % 977
    producer = RegionProducer(pool, "t 1 project pipeline")
    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected"):
        producer.compress([(0, x, [x.shape[0]], "BITCOMP")])
    assert len(calls) == 3
    assert not producer.has_pending()
    assert mgr_counters()["tickets_live"] == live0
    producer.abort()


# ── GPU: the home writer's frees and cap ─────────────────────────────────

def _blocked_event(ms=200):
    """An event that fires only after ``ms`` of GPU sleep on a side
    stream: copies waiting on it cannot start before then."""
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        torch.cuda._sleep(int(ms * 1.5e6))
        evt = torch.cuda.Event()
        evt.record()
    return evt


@requires_cuda
def test_a_source_is_held_until_its_copy_has_ended(cuda_allocator, pool):
    w = HomeWriter()
    src = torch.arange(1 << 20, dtype=torch.int64, device="cuda:0")
    expect = src.cpu()
    ref = weakref.ref(src)
    ptr, home = pool.get_storage(src.nbytes)
    s = w.ship([src.data_ptr()], [home.data_ptr()], [src.nbytes],
               keep=[src], after=_blocked_event(1000))
    del src
    w.reap()
    # The copy cannot have started: its source is still held.
    assert not s.end.query() and ref() is not None
    s.end.synchronize()
    w.reap()
    gc.collect()
    assert s.done and ref() is None
    assert torch.equal(home.view(torch.int64), expect)
    pool.free(ptr)


@requires_cuda
def test_a_fill_runs_on_the_writer_stream_before_the_copies(cuda_allocator,
                                                            pool):
    """``ship``'s fill (the producer's slot fills) runs with the writer's
    stream current, behind the shipment's event and before its copies; the
    buffers only it reads (``fill_keep``) are dropped once it has run, the
    copies' sources once they end, and the shipment's span, the pipe's d2h
    lane, covers both."""
    from IO.codec.home_writer import writer_stream
    w = HomeWriter()
    big = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda:0")
    slab = torch.empty(1 << 20, dtype=torch.uint8, device="cuda:0")
    big_ref, slab_ref = weakref.ref(big), weakref.ref(slab)
    ptr, home = pool.get_storage(big.nbytes)
    streams = []

    def fill():
        streams.append(torch.cuda.current_stream())
        torch.cuda._sleep(50_000_000)
    s = w.ship([big.data_ptr()], [home.data_ptr()], [big.nbytes],
               keep=[big], after=_blocked_event(10), fill=fill,
               fill_keep=[slab])
    del big, slab
    assert streams == [writer_stream()]
    assert torch.cuda.current_stream() != writer_stream()
    s.filled.synchronize()
    copying = not s.end.query()
    w.reap()
    gc.collect()
    assert slab_ref() is None
    if copying:
        # The 256 MiB copy home is still running: its source is held.
        assert big_ref() is not None
    s.end.synchronize()
    w.reap()
    gc.collect()
    assert big_ref() is None
    assert s.ms() > 10
    w.drain()
    pool.free(ptr)


@requires_cuda
def test_the_cap_holds_two_shipments_and_waits_on_the_oldest(cuda_allocator,
                                                             pool):
    w = HomeWriter(cap=2)
    homes = []
    ships = []
    for i in range(2):
        src = torch.full((1 << 18,), i, dtype=torch.int64, device="cuda:0")
        ptr, home = pool.get_storage(src.nbytes)
        homes.append(ptr)
        ships.append(w.ship([src.data_ptr()], [home.data_ptr()],
                            [src.nbytes], keep=[src],
                            after=_blocked_event(150)))
    assert w.in_flight() == 2
    assert w.wait_for_room()
    # It waited on the oldest, and only as long as it had to.
    assert ships[0].end.query() and w.in_flight() <= 1
    assert w.cap_waits == 1
    assert not w.wait_for_room()
    w.drain()
    for ptr in homes:
        pool.free(ptr)


class _HeldEnd:
    """A shipment's end event that reports its copies as running until
    someone waits on it: the copies stay in flight for as long as the test
    needs, whatever the GPU's timing (a sanitizer serializes kernels)."""

    def __init__(self, event):
        self.event = event
        self.waited = False

    def query(self):
        return self.waited and self.event.query()

    def synchronize(self):
        self.waited = True
        self.event.synchronize()

    def record(self, stream=None):
        self.event.record(stream)


@requires_cuda
def test_a_producer_waits_for_room_before_its_third_pack(cuda_allocator,
                                                         pool, monkeypatch):
    rng = np.random.default_rng(11)
    ticks = _ticks(rng, 4, 1, torch.int64, rows=(20_000, 40_000), empty=0)
    producer = RegionProducer(pool, "t filter pipe")
    held = []
    real_ship = producer.writer.ship

    def ship(*a, **kw):
        s = real_ship(*a, **kw)
        s.end = _HeldEnd(s.end)
        held.append(s.end)
        return s
    monkeypatch.setattr(producer.writer, "ship", ship)
    for t in ticks:
        if producer.has_pending():
            before = sum(e.waited for e in held)
            producer.pack()
            assert producer.writer.in_flight() <= 2
            if len(held) > 2:
                # The third pack found two shipments in flight and waited
                # on the oldest, and only on it.
                assert held[sum(e.waited for e in held) - 1] is \
                    held[before]
        producer.compress([(0, t[0].to("cuda:0"), [t[0].shape[0]],
                            "BITCOMP")])
        torch.cuda.current_stream().synchronize()
    producer.drain()
    # Four shipments (three packs in the loop, one in the drain): the third
    # and the fourth each waited on the oldest still in flight.
    assert len(held) == 4 and producer.writer.cap_waits == 2
    assert held[0].waited and held[1].waited
    assert torch.equal(_read_all(producer.tables()[0]), _expected(ticks))


@requires_cuda
def test_regions_freed_at_the_drain_take_no_late_copy(cuda_allocator, pool):
    """A table freed as soon as the pipe drains returns its homes; nothing
    still in flight writes into them once the pool hands them out again."""
    from variable import TableVariable
    rng = np.random.default_rng(13)
    ticks = _ticks(rng, 6, 4, torch.int64, rows=(20_000, 60_000), empty=0)
    feeds = []
    for t in ticks:
        tensor, counts = _padded(t, torch.int64)
        feeds.append([(0, tensor, counts, "BITCOMP")])
    producer = RegionProducer(pool, "t 0 sort pipeline")
    table = _run(producer, feeds)[0]
    homes = [(r.data_ptr(), r.nbytes) for r in table.regions]
    TableVariable(table, "int").free_underlying_mem(pool, None)
    assert all(r.freed for r in table.regions)
    # The pool hands the same bytes out again; fill them and wait for every
    # stream: no copy home may land after the drain.
    total = sum(n for _, n in homes)
    ptr, again = pool.get_storage(total)
    again.fill_(0xA5)
    torch.cuda.synchronize()
    assert int((again != 0xA5).sum()) == 0
    pool.free(ptr)


def _held_b(pool):
    """Bytes the pinned pool has handed out."""
    return sum(pool._allocs.values()) * pool.block_size_bytes


@requires_cuda
def test_homes_are_carved_from_chunks_trimmed_at_the_drain(cuda_allocator,
                                                           pool):
    """Small regions share pinned chunks sized for the column's expected
    output, so the pool's block rounding is paid per chunk, not per
    region; the drain gives every chunk's unused blocks back, and the
    table returns the chunks when it is freed."""
    from variable import TableVariable
    rng = np.random.default_rng(17)
    T = 12
    ticks = _ticks(rng, T, 1, torch.int64, rows=(20_000, 21_000), empty=0)
    feeds = [[(0, t[0].to("cuda:0"), [t[0].shape[0]], "BITCOMP")]
             for t in ticks]
    producer = RegionProducer(pool, "t filter pipe")
    held0 = _held_b(pool)
    peak = 0
    for i, items in enumerate(feeds):
        # As the pipeline counts it: the ticks whose output is still to be
        # packed, the one packed now included.
        producer.ticks_left = T - i + 1
        if producer.has_pending():
            producer.pack()
        producer.compress(items)
        torch.cuda.current_stream().synchronize()
        peak = max(peak, _held_b(pool) - held0)
    producer.drain()
    table = producer.tables()[0]
    block = pool.block_size_bytes
    stored = sum(r.nbytes for r in table.regions)
    held = _held_b(pool) - held0
    # A few chunks (the first region alone, then the rest, and one more
    # when compressed sizes run over the estimate), trimmed to whole blocks;
    # never more reserved than the column will write, give or take a region
    # or two.
    assert 2 <= len(table.host_blocks) <= 3
    assert held <= stored + 3 * block
    assert peak <= held + 2 * max(r.nbytes for r in table.regions) + block
    assert torch.equal(_read_all(table), _expected(ticks))
    TableVariable(table, "int").free_underlying_mem(pool, None)
    assert _held_b(pool) == held0 and not table.host_blocks


def _late_copies(producer, monkeypatch, ms=300):
    """Make every copy home wait ``ms`` of GPU time after its real event,
    so the copies are in flight on the GPU while the test goes on."""
    real_ship = producer.writer.ship

    def ship(*a, **kw):
        side = torch.cuda.Stream()
        side.wait_event(kw["after"])
        with torch.cuda.stream(side):
            torch.cuda._sleep(int(ms * 1.5e6))
            late = torch.cuda.Event()
            late.record()
        kw["after"] = late
        return real_ship(*a, **kw)
    monkeypatch.setattr(producer.writer, "ship", ship)


@requires_cuda
@pytest.mark.parametrize("codec", ["BITCOMP", "RAW"])
def test_buffers_stay_held_while_their_copies_wait(cuda_allocator, pool,
                                                   monkeypatch, codec):
    """The dense buffers (and RAW output tensors) copies still read are not
    handed out again: GPU memory allocated and written while every copy
    home waits leaves the homes' bytes intact."""
    rng = np.random.default_rng(19)
    ticks = _ticks(rng, 3, 4, torch.int64, rows=(30_000, 60_000), empty=0)
    producer = RegionProducer(pool, "t 0 sort pipeline")
    _late_copies(producer, monkeypatch)
    for t in ticks:
        if producer.has_pending():
            producer.pack()
        tensor, counts = _padded(t, torch.int64)
        producer.compress([(0, tensor, counts, codec)])
        del tensor
        for _ in range(4):
            scribble = torch.full((16 << 20,), 0x5A, dtype=torch.uint8,
                                  device="cuda:0")
            del scribble
        torch.cuda.current_stream().synchronize()
    producer.drain()
    assert torch.equal(_read_all(producer.tables()[0]), _expected(ticks))


@requires_cuda
def test_raw_outputs_are_held_to_the_cap_in_ticks(cuda_allocator, pool,
                                                  monkeypatch):
    """RAW outputs wait for room like packs do, and a tick's RAW and
    compressed shipments count as one tick of the two."""
    rng = np.random.default_rng(29)
    ticks = _ticks(rng, 5, 1, torch.int64, rows=(10_000, 20_000), empty=0)
    producer = RegionProducer(pool, "t filter pipe")
    held = []
    real_ship = producer.writer.ship

    def ship(*a, **kw):
        s = real_ship(*a, **kw)
        s.end = _HeldEnd(s.end)
        held.append(s)
        return s
    monkeypatch.setattr(producer.writer, "ship", ship)
    for t in ticks:
        if producer.has_pending():
            producer.pack()
        x = t[0].to("cuda:0")
        # Two columns: one RAW, one compressed, in the same tick.
        producer.compress([(0, x, [x.shape[0]], "RAW"),
                           (1, x.clone(), [x.shape[0]], "BITCOMP")])
        assert producer.writer.in_flight() <= 2
        torch.cuda.current_stream().synchronize()
    producer.drain()
    # Ten shipments over five ticks: a tick's RAW shipment and its packed
    # region share the tick's place under the cap.
    assert len(held) == 10
    assert producer.writer.cap_waits >= 2
    for base in (0, 1):
        assert torch.equal(_read_all(producer.tables()[base]),
                           _expected(ticks))


def _pinned_rep(raw_2d):
    """A scanned column: the scan's RAW record table, in a pinned pool of
    its own (the test watches ``pool``)."""
    from tests.scan_table import scan_table
    from IO.pages.combined_column_rep import CombinedColumnRep
    from IO.pages.pinned_mem import PinnedMemory
    own = PinnedMemory(capacity_gb=0.25, block_size_mb=0.5)
    return CombinedColumnRep([scan_table(
        raw_2d.contiguous(), "int", own, "RAW")])


@requires_cuda
def test_a_failed_pipe_waits_for_its_copies_home(cuda_allocator, pool,
                                                 monkeypatch):
    """A failure in the tick loop with copies home in flight waits for them
    before it leaves the pipe, and gives the homes back."""
    from IO.pipeline.vortex_pipeline import VortexPipeline
    from IO.transfer.data_exchange import set_exchange_to_naive
    from utility.logger import set_message_logger, set_perf_logger
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()
    rows = 1 << 17
    data = (torch.arange(rows, dtype=constants.int_dtype) % 7).unsqueeze(1)
    held0 = pool._allocs.copy()
    pipe = VortexPipeline(
        input_columns={0: _pinned_rep(data)},
        output_columns_type=[constants.int_dtype],
        operator=lambda d: [d[0]], cpu_mem_pool=pool,
        chunk_size=256 * 1024, name="t filter pipe",
        output_compression_algs=["BITCOMP"])
    producer = pipe._VortexPipeline__producer
    _late_copies(producer, monkeypatch, ms=500)
    shipped = []
    late_ship = producer.writer.ship

    def ship(*a, **kw):
        s = late_ship(*a, **kw)
        shipped.append((s, s.end))
        return s
    monkeypatch.setattr(producer.writer, "ship", ship)
    real_pack = producer.pack
    calls = []

    def pack():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected")
        return real_pack()
    monkeypatch.setattr(producer, "pack", pack)
    with pytest.raises(RuntimeError, match="injected"):
        pipe.do_exchange()
    # The first tick's copy was in flight when the second pack failed; it
    # has ended, nothing is held, and the homes are back in the pool.
    assert len(shipped) == 1
    s, end = shipped[0]
    assert s.done and end.query()
    assert producer.writer.in_flight() == 0
    assert pool._allocs == held0


@requires_cuda
def test_a_region_that_fits_is_carved_when_its_chunk_does_not():
    """At the edge of the pool a region whose own size fits gets an
    allocation of its own when the chunk sized for the column does not."""
    from IO.pages.pinned_mem import PinnedMemory
    from IO.codec.region_producer import _Column
    small = PinnedMemory(capacity_gb=1, block_size_mb=0.5)
    hog, _ = small.get_storage(small.pool_size_bytes - (2 << 20))
    col = _Column(0, 1)
    col.written, col.carved = 4 << 20, 1      # a 4 MiB mean region so far
    region, home = col.carve(1 << 20, small, ticks_left=100)
    assert region.nbytes == 1 << 20 and col.chunk.numel() == 1 << 20
    col.close(small)
    for ptr in col.home_ptrs:
        small.free(ptr)
    small.free(hog)


@requires_cuda
def test_without_ticks_left_regions_still_share_chunks(cuda_allocator, pool):
    """A caller that does not say how many ticks are left still gets
    chunks of several regions, not one allocation per region."""
    rng = np.random.default_rng(31)
    T = 10
    ticks = _ticks(rng, T, 1, torch.int64, rows=(20_000, 21_000), empty=0)
    feeds = [[(0, t[0].to("cuda:0"), [t[0].shape[0]], "BITCOMP")]
             for t in ticks]
    producer = RegionProducer(pool, "t 0 project")
    assert producer.ticks_left is None
    table = _run(producer, feeds)[0]
    assert len(table.regions) == T and len(table.host_blocks) <= 3
    assert torch.equal(_read_all(table), _expected(ticks))


# ── GPU: cached regions against plain ones ──────────────────────────────

def _random_feeds(seed):
    """Random ticks of a few columns, as a pipe feeds them: per column a
    dtype, a bucket count (one: a plain tensor; more: the sort pipe's
    padded layout) and a codec, RAW on some ticks."""
    rng = np.random.default_rng(seed)
    cols = []
    for c in range(int(rng.integers(1, 5))):
        dtype = [torch.int64, torch.int32][int(rng.integers(0, 2))]
        B = [1, 3, 8][int(rng.integers(0, 3))]
        codec = ["BITCOMP", "ANS", "CASCADED"][int(rng.integers(0, 3))]
        cols.append((c * 8, dtype, B, codec))
    feeds = []
    for _ in range(int(rng.integers(2, 6))):
        items = []
        for base, dtype, B, codec in cols:
            parts = _ticks(rng, 1, B, dtype, rows=(0, 30_000))[0]
            if B == 1:
                tensor, counts = parts[0].to("cuda:0"), [parts[0].shape[0]]
            else:
                tensor, counts = _padded(parts, dtype)
            items.append((base, tensor, counts,
                          "RAW" if rng.random() < 0.25 else codec))
        feeds.append(items)
    return feeds


class _CopyCodec:
    """A stand-in for the manager's compress calls whose output is a
    deterministic function of its input (nVCOMP's bytes, and some of its
    sizes, vary between two runs of the same input): a record "compresses"
    to its first ``min(size, size // 2 + size % 97)`` bytes."""

    def __init__(self, monkeypatch):
        import TQPlib.tqpcompress as native
        self.copy = native.batched_memcpy_abs
        self.tickets = {}
        for name in ("begin", "launch", "finish"):
            monkeypatch.setattr(native, f"mgr_compress_{name}",
                                getattr(self, name))

    def begin(self, codec, elem_size, chunk_b, sizes):
        tid = len(self.tickets) + 1
        self.tickets[tid] = sizes.numpy().copy()
        return tid, torch.from_numpy(self.tickets[tid] + 64)

    def launch(self, tid, in_ptrs, out_ptrs, comp_dev, comp_pin):
        sizes = self.tickets[tid]
        out = np.minimum(sizes, sizes // 2 + sizes % 97)
        n = len(sizes)
        arr = torch.from_numpy(np.concatenate(
            [in_ptrs.numpy(), out_ptrs.numpy(), out])).to("cuda:0")
        self.copy(arr[:n], arr[n:2 * n], arr[2 * n:])
        comp_dev[:n].copy_(arr[2 * n:])
        comp_pin[:n] = torch.from_numpy(out)
        return torch.arange(n, dtype=torch.int64)

    def finish(self, tid):
        return torch.zeros(len(self.tickets[tid]), dtype=torch.int64)


def _region_bytes(region):
    """A region's bytes as its readers see them: its home, or for a cached
    region its pieces in slots and at home."""
    if region.location != "cached":
        return region.storage.clone()
    from TQPlib.tqpcompress import batched_h2d_to_ptrs
    cache, cr = region.release
    out = torch.zeros(cr.nbytes, dtype=torch.uint8, pin_memory=True)
    p = cache.pieces(cr, 0, cr.nbytes)
    torch.cuda.synchronize()          # every copy into slots and homes
    batched_h2d_to_ptrs(p.src.tolist(),
                        (p.offset + out.data_ptr()).tolist(), p.size.tolist())
    torch.cuda.synchronize()
    return out


def _produce(pool, feeds, cache):
    producer = RegionProducer(pool, "t 0 sort pipeline", cache=cache)
    if cache is not None:
        producer.cached_outputs = {b: (5, 1) for b in range(0, 64, 8)}
    while feeds:
        if cache is not None:
            cache.advance_tick()
        if producer.has_pending():
            producer.pack()
        producer.compress(feeds.pop(0))
        torch.cuda.current_stream().synchronize()
    if cache is not None:
        cache.advance_tick()
    producer.drain()
    return {base: (t, [_region_bytes(r) for r in t.regions],
                   [r.nbytes for r in t.regions],
                   [r.release[1].admitted if r.location == "cached" else -1
                    for r in t.regions])
            for base, t in producer.tables().items()}


@requires_cuda
@pytest.mark.parametrize("codec", ["copy", "nvcomp"])
@pytest.mark.parametrize("seed", range(6))
def test_cached_regions_hold_the_records_plain_ones_do(
        cuda_allocator, pool, monkeypatch, seed, codec):
    """The same random ticks (records cut at the cap and codec groups split
    into pieces on some seeds) with and without the granule cache (prefixes
    in slots, the rest home; whole-admitted ticks filled by one kernel).
    With a deterministic stand-in compressor: the same record tables and
    every record's bytes the same, read from slots and homes.  With nvCOMP:
    the same rows decoded."""
    from IO.codec import batched_compress
    from tests.page_cache.test_granule_producer import _cache
    if seed % 2:
        monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 0)
    if seed % 3 == 0:
        monkeypatch.setattr(batched_compress, "_MAX_OUT_BUF_BYTES", 96 << 10)
    if codec == "copy":
        _CopyCodec(monkeypatch)
    got = {}
    for cached in (False, True):
        got[cached] = _produce(pool, _random_feeds(seed),
                               _cache(24) if cached else None)
    assert set(got[False]) == set(got[True])
    # Some prefix went into slots.
    assert any(a > 0 for v in got[True].values() for a in v[3])
    for base in got[False]:
        (t0, b0, n0, _), (t1, b1, n1, _) = got[False][base], got[True][base]
        if codec == "nvcomp":
            assert torch.equal(_read_all(t0), _read_all(t1))
            continue
        assert n0 == n1
        for name in ("region", "offset", "nbytes", "raw_bytes", "row_start",
                     "rows", "codec", "stream_ptr"):
            assert np.array_equal(getattr(t0, name), getattr(t1, name)), name
        # Every record's bytes (the alignment gaps are never written).
        for k, (x, y) in enumerate(zip(b0, b1)):
            for rec in region_records(t0, k).tolist():
                a, n = int(t0.offset[rec]), int(t0.nbytes[rec])
                assert torch.equal(x[a:a + n], y[a:a + n])


def test_a_single_buckets_rows_are_checked():
    """The one-bucket shortcut takes only a compact tensor; a padded one is
    read at its bucket start, and rows that match neither layout fail."""
    counts = np.array([5], dtype=np.int64)
    for rows in (5, 64):        # compact; padded to lcm(8, 512) / 8 rows
        col = region_producer._Column(0, 1)
        t = torch.zeros(rows, 1, dtype=torch.int64)
        col.take(t, 1)
        ptrs, sizes, buckets = RegionProducer._col_records(col, t, counts)
        assert ptrs.tolist() == [t.data_ptr()]
        assert sizes.tolist() == [40]
        assert buckets.tolist() == [0]
    t = torch.zeros(7, 1, dtype=torch.int64)
    with pytest.raises(AssertionError, match="match neither"):
        RegionProducer._col_records(col, t, counts)


def test_buckets_back_to_back_must_be_the_padded_layout():
    """Several buckets back to back are read only when that is the padded
    layout, every bucket but the last whole lcm(row bytes, 512) units: a
    bucket off its unit would start a compressed record off the alignment a
    codec reads it at.  One bucket, and the padded layout, read as ever."""
    def starts(rows, counts, width=1, dtype=torch.int64):
        return region_producer._bucket_starts(
            torch.zeros(rows, width, dtype=dtype),
            np.array(counts, dtype=np.int64)).tolist()
    # One bucket, back to back.
    assert starts(5, [5]) == [0]
    # Whole 64-row units of 8-byte rows but the last: back to back is
    # padded.
    assert starts(197, [64, 0, 128, 5]) == [0, 512, 512, 1536]
    # Buckets off their units, padded.
    assert starts(256, [5, 0, 70, 3]) == [0, 512, 512, 1536]
    # The same buckets back to back: refused.
    with pytest.raises(AssertionError, match="starts bucket 1 off its 64-row"):
        starts(78, [5, 0, 70, 3])
    # A 25-byte string's unit is 512 rows.
    assert starts(1030, [512, 512, 6], 25, torch.int8) == [
        0, 512 * 25, 1024 * 25]
    with pytest.raises(AssertionError, match="off its 512-row unit"):
        starts(1030, [500, 524, 6], 25, torch.int8)


@requires_cuda
def test_records_are_cut_from_each_slice_start(cuda_allocator, pool,
                                               monkeypatch):
    """Each (tick, bucket) slice of every output is cut into records from
    the slice's first row, full but for the slice's last, so the boundaries
    of a mask and of 4-, 8- and 25-byte columns of the same slices nest;
    RAW and compressed outputs alike."""
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1 / 64)
    rng = np.random.default_rng(17)
    B, T = 3, 3
    counts = [[int(rng.integers(0, 40_000)) if rng.random() > 0.1 else 0
               for _ in range(B)] for _ in range(T)]
    widths = {0: (torch.int8, 1), 1: (torch.int32, 1), 2: (torch.int64, 1),
              3: (torch.int8, 25)}
    data = {}
    for base, (dtype, w) in widths.items():
        data[base] = [[torch.from_numpy(rng.integers(
            0, 50, (n, w))).to(dtype) for n in tick] for tick in counts]
    for codec in ("RAW", "BITCOMP"):
        feeds = []
        for t in range(T):
            items = []
            for base, (dtype, _) in widths.items():
                tensor, cnt = _padded(data[base][t], dtype)
                items.append((base, tensor, cnt, codec))
            feeds.append(items)
        tables = _run(RegionProducer(pool, "t 0 sort pipeline"), feeds)
        bounds = {}
        for base, (dtype, w) in widths.items():
            table = tables[base]
            row_b = w * torch.tensor([], dtype=dtype).element_size()
            rr = region_producer.record_rows(row_b)
            got = []
            for b in range(B):
                recs = np.arange(table.stream_ptr[b], table.stream_ptr[b + 1])
                starts, rows = table.row_start[recs], table.rows[recs]
                lo, cuts = 0, []
                for t in range(T):
                    n = counts[t][b]
                    if not n:
                        continue
                    cuts.append(np.arange(lo, lo + n, rr))
                    lo += n
                want = np.concatenate(cuts) if cuts else np.zeros(0, int)
                assert starts.tolist() == want.tolist(), (codec, base, b)
                assert (np.r_[starts[1:], lo] - starts == rows).all()
                got.append(np.r_[starts, lo])
            bounds[base] = got
        assert torch.equal(_read_all(tables[2]), torch.cat(
            [data[2][t][b] for b in range(B) for t in range(T)]).squeeze(1))
        for b in range(B):
            for wide, narrow in ((3, 2), (2, 1), (1, 0)):
                assert np.isin(bounds[narrow][b], bounds[wide][b]).all()


@requires_cuda
@pytest.mark.parametrize("on_host", [False, True])
def test_abs_copy_reads_its_arrays_from_the_gpu_or_pinned_host(on_host):
    """The copy kernel over absolute addresses copies the same bytes whether
    its address arrays sit on the GPU or are read from pinned host memory."""
    import TQPlib.tqpcompress as native
    gen = torch.Generator().manual_seed(7)
    sizes = [1, 511, 4096, 70_000, 3]
    src = torch.randint(0, 255, (sum(sizes),), dtype=torch.uint8,
                        generator=gen).to("cuda:0")
    dst = torch.zeros(sum(sizes) + 64, dtype=torch.uint8, device="cuda:0")
    offs = np.cumsum([0] + sizes[:-1])
    # Write the chunks to the destination in reverse order, 13 bytes in.
    rev = np.cumsum([0] + sizes[::-1][:-1])[::-1]
    table = torch.tensor(np.concatenate([src.data_ptr() + offs,
                                         dst.data_ptr() + 13 + rev,
                                         sizes]), dtype=torch.int64)
    table = table.pin_memory() if on_host else table.to("cuda:0")
    n = len(sizes)
    native.batched_memcpy_abs(table[:n], table[n:2 * n], table[2 * n:])
    torch.cuda.synchronize()
    for o, r, s in zip(offs, rev, sizes):
        assert torch.equal(dst[13 + r:13 + r + s], src[o:o + s])


@requires_cuda
def test_abs_copy_launch_offsets_destinations_and_records_its_event(
        cuda_allocator, monkeypatch):
    """``abs_copy.launch``: the arrays staged and read over UVA, the
    destinations offset from a base, and the staging buffer's event recorded
    behind the kernel by the same native call."""
    from IO.pages import abs_copy
    gen = torch.Generator().manual_seed(8)
    sizes = np.array([1, 511, 4096, 70_000, 3], dtype=np.int64)
    src = torch.randint(0, 255, (int(sizes.sum()),), dtype=torch.uint8,
                        generator=gen).to("cuda:0")
    dst = torch.zeros(int(sizes.sum()) + 64, dtype=torch.uint8,
                      device="cuda:0")
    offs = np.r_[0, np.cumsum(sizes)[:-1]]
    rev = np.r_[0, np.cumsum(sizes[::-1])[:-1]][::-1]
    got = []
    orig = abs_copy.staging
    monkeypatch.setattr(abs_copy, "staging",
                        lambda n: (got.append(orig(n)), got[-1])[1])
    torch.cuda._sleep(200_000_000)
    abs_copy.launch(src.data_ptr() + offs, 13 + rev, sizes, dst.data_ptr())
    assert not got[0][2].query()
    torch.cuda.synchronize()
    assert got[0][2].query()
    for o, r, n in zip(offs, rev, sizes):
        assert torch.equal(dst[13 + r:13 + r + n], src[o:o + n])


@requires_cuda
def test_abs_copy_refuses_pageable_host_arrays():
    import TQPlib.tqpcompress as native
    table = torch.zeros(3, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="pinned host memory"):
        native.batched_memcpy_abs(table[:1], table[1:2], table[2:])

