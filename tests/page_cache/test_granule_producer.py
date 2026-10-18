"""The region producer with the granule cache (IO/codec/region_producer.py,
IO/cache/native_cache.py) and the reader that binds cached regions piece by
piece (IO/transfer/tick_layout.py).

A producer with a cache offers each tick's regions at pack time; the cache
admits a prefix of each region's granules, the pack writes that prefix into
slots and the rest through a dense buffer to the home.  Small slots make the
cut fall inside a record, so a record is read back from a slot (D2D) and
its home (H2D) at once; later ticks evict earlier ones (a dirty granule is
copied home first).  Without room for a dense buffer (lending included:
``tests/allocator/test_arena_lend.py`` has a lend inside the producer's
copy plan) the refused bytes are copied from the slab straight to the home,
with or without a cache.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from IO.codec import region_producer
from IO.codec.region_producer import RegionProducer
from tests.record_tables import region_records, stream_row_offsets

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")

G = 16 << 10


@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


def _cache(slots, policy="lru"):
    from IO.cache.native_cache import NativeGranuleCache
    from IO.codec.home_writer import copy_home
    from tests.page_cache.regions import TensorRegion
    return NativeGranuleCache(TensorRegion(slots, G, device="cuda:0"),
                              policy, copy_home=copy_home)


def _ticks(rng, n_ticks, B, rows):
    """Per tick, per bucket, ``rows`` random int64 rows of small values
    (they compress)."""
    return [[torch.from_numpy(rng.integers(0, 50, rows).astype(np.int64))
             .unsqueeze(1) for _ in range(B)] for _ in range(n_ticks)]


def _feed(ticks, codec):
    """The producer's items per tick: one column, its buckets laid out as
    the sort pipe lays them out, each from an lcm(row bytes, 512)-aligned
    start (``batched_compress.aligned_partition_layout``)."""
    from IO.codec.batched_compress import aligned_partition_layout
    feeds = []
    for t in ticks:
        counts = [p.shape[0] for p in t]
        _, starts, total = aligned_partition_layout(counts, 8, 512)
        out = torch.full((total, 1), -1, dtype=torch.int64, device="cuda:0")
        for p, s in zip(t, starts):
            out[s // 8:s // 8 + p.shape[0]] = p.to("cuda:0")
        feeds.append([(0, out, counts, codec)])
    return feeds


def _run(producer, feeds, cache):
    """Compress at t, the barrier, pack at t+1 (the cache's tick boundary
    first, as a pipe's), then the drain."""
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
    return producer.tables()[0]


def _expected(ticks):
    B = len(ticks[0])
    return torch.cat([t[b] for b in range(B) for t in ticks]).squeeze(1)


def _producer(pool, cache):
    producer = RegionProducer(pool, "t 0 sort pipeline", cache=cache)
    producer.cached_outputs = {0: (5, 1)}
    return producer


@requires_cuda
@pytest.mark.parametrize("codec", ["BITCOMP", "RAW"])
def test_a_record_split_between_slots_and_home_reads_back_exactly(
        cuda_allocator, pool, monkeypatch, codec):
    from variable import TableVariable
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    rng = np.random.default_rng(11)
    ticks = _ticks(rng, 3, 3, 150_000)
    cache = _cache(6)
    table = _run(_producer(pool, cache), _feed(ticks, codec), cache)
    assert all(r.location == "cached" for r in table.regions)
    crs = [r.release[1] for r in table.regions]
    # Each tick took the six slots, evicting the tick before.
    assert [cr.admitted for cr in crs] == [6, 6, 6]
    last = crs[-1]
    cut = last.admitted * G
    recs = region_records(table, len(crs) - 1)
    off, n = table.offset[recs], table.nbytes[recs]
    # The cut lies inside a record: its head in slots, its tail at home.
    assert ((off < cut) & (off + n > cut)).any()
    pieces = cache.pieces(last, 0, last.nbytes)
    assert pieces.on_gpu.any() and not pieces.on_gpu.all()
    # The earlier ticks are evicted: read from their homes.
    assert not cache.pieces(crs[0], 0, crs[0].nbytes).on_gpu.any()
    var = TableVariable(table, "int")
    assert torch.equal(var.col_to_gpu().cpu(), _expected(ticks))
    # A range of buckets, as a join-core tick reads it.
    rows = stream_row_offsets(table)
    part = var.get_sliced_rep(slice(int(rows[1]), int(rows[3])))
    assert torch.equal(part.col_to_gpu().cpu(), torch.cat(
        [t[b] for b in (1, 2) for t in ticks]).squeeze(1))
    var.free_underlying_mem(pool, None)
    assert all(cr.freed for cr in crs)
    assert cache._core.free_count() == cache._core.num_slots()


@requires_cuda
def test_an_uncached_output_is_a_host_region(cuda_allocator, pool):
    from variable import TableVariable
    rng = np.random.default_rng(12)
    ticks = _ticks(rng, 2, 2, 20_000)
    cache = _cache(64)
    producer = RegionProducer(pool, "t 0 sort pipeline", cache=cache)
    table = _run(producer, _feed(ticks, "BITCOMP"), cache)
    assert all(r.location == "host" for r in table.regions)
    assert cache.resident_bytes() == 0
    assert torch.equal(TableVariable(table, "int").col_to_gpu().cpu(),
                       _expected(ticks))


@requires_cuda
@pytest.mark.parametrize("cached", [False, True])
def test_a_dense_buffer_that_does_not_fit_is_copied_from_the_slab(
        cuda_allocator, pool, monkeypatch, cached):
    """The pack's dense buffer cannot be allocated: the refused records go
    from the slab straight to their homes (a cached region's slot prefix
    still by the copy kernel).  The cache is not asked to reclaim: slots it
    freed would stay in the slab."""
    from variable import TableVariable
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    rng = np.random.default_rng(13)
    ticks = _ticks(rng, 3, 2, 150_000)
    cache = _cache(4) if cached else None
    producer = _producer(pool, cache)
    real_alloc = region_producer._alloc_slab
    failed = []

    def alloc(nbytes, device):
        # The pack's allocations hold its dense buffers (and the copy
        # arrays beyond them); a small one is the kernel's arrays alone.
        if packing and nbytes > 4096:
            failed.append(nbytes)
            raise RuntimeError("Pool out of memory")
        return real_alloc(nbytes, device)
    monkeypatch.setattr(region_producer, "_alloc_slab", alloc)
    reclaims = []
    if cached:
        real_reclaim = cache.reclaim

        def reclaim(nbytes):
            reclaims.append(nbytes)
            return real_reclaim(nbytes)
        monkeypatch.setattr(cache, "reclaim", reclaim)
    real_pack = producer.pack
    packing = False

    def pack():
        nonlocal packing
        packing = True
        try:
            return real_pack()
        finally:
            packing = False
    monkeypatch.setattr(producer, "pack", pack)
    table = _run(producer, _feed(ticks, "BITCOMP"), cache)
    assert len(failed) >= len(ticks)       # every pack's
    assert not reclaims
    if cached:
        assert all(r.release[1].admitted for r in table.regions)
    assert torch.equal(TableVariable(table, "int").col_to_gpu().cpu(),
                       _expected(ticks))


@requires_cuda
def test_a_home_piece_read_waits_for_its_copy_home(cuda_allocator, pool,
                                                   monkeypatch):
    # The exchange binds on the current stream (the legacy default stream
    # here, as in a pipe) and copies on its own non-blocking H2D stream,
    # which a wait on the binding stream does not order: the H2D lane must
    # wait for the copies into the homes it reads (BoundTickPlan.home_waits).
    from IO.codec.home_writer import writer_stream
    from IO.transfer import data_exchange
    from IO.transfer.in_stage import DirectIn
    from IO.transfer.tick_layout import pin_cached_reads
    from variable import TableVariable
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    rng = np.random.default_rng(12)
    ticks = _ticks(rng, 3, 3, 150_000)
    cache = _cache(6)
    table = _run(_producer(pool, cache), _feed(ticks, "RAW"), cache)
    last = table.regions[-1].release[1]
    assert last.admitted == 6
    # The last tick's dirty granules are evicted behind a busy writer
    # stream: their copies home are still queued when the read binds.
    with torch.cuda.stream(writer_stream()):
        torch.cuda._sleep(500_000_000)
    assert cache.reclaim(6 * G).dirty == 6
    assert not cache._events[-1].query()
    assert not cache.pieces(last, 0, last.nbytes).on_gpu.any()
    var = TableVariable(table, "int")
    stage = DirectIn(var)
    DirectIn.install_compiled_tick_layout([stage])
    ex = data_exchange.exchange()
    unpin = pin_cached_reads(var)
    try:
        ex.launch([stage])
    finally:
        unpin()
    ex.sync()
    stage.decompress()
    torch.cuda.synchronize()
    assert torch.equal(stage.gpu_col.data().cpu(), _expected(ticks))
    var.free_underlying_mem(pool, None)


@requires_cuda
def test_merged_slot_pieces_copy_exactly(cuda_allocator):
    # Slot pieces that continue each other merge into copies of many MiB,
    # aligned or not; one launch copies them all exactly.
    from types import SimpleNamespace
    from IO.transfer.tick_layout import _coalesce, launch_slot_pieces
    MiB = 1 << 20
    src = torch.randint(0, 256, (64 * MiB,), dtype=torch.uint8,
                        device="cuda:0")
    agg = torch.zeros(48 * MiB, dtype=torch.uint8, device="cuda:0")
    # 24 one-MiB pieces back to back, 9 MiB + 3 B starting at an odd byte,
    # and a small piece.
    s0 = np.arange(24, dtype=np.int64) * MiB
    pieces = (np.r_[s0, 30 * MiB + 1, 50 * MiB],
              np.r_[s0, 25 * MiB + 1, 40 * MiB],
              np.r_[np.full(24, MiB), 9 * MiB + 3, 100].astype(np.int64))
    merged = _coalesce(pieces[0] + src.data_ptr(), *pieces[1:])
    assert len(merged[2]) == 3 and merged[2][0] == 24 * MiB
    launch_slot_pieces(agg, SimpleNamespace(slot_pieces=merged))
    want = torch.zeros_like(agg)
    for a, b, n in zip(*(x.tolist() for x in pieces)):
        want[b:b + n] = src[a:a + n]
    assert torch.equal(agg, want)


@requires_cuda
@pytest.mark.parametrize("codec", ["BITCOMP", "ANS"])
def test_whole_admitted_regions_are_filled_into_their_slots(
        cuda_allocator, pool, monkeypatch, codec):
    """A tick whose cached outputs are admitted whole is packed by one kernel
    into the slots (``_fill_slots``, every tick): every granule in a slot
    and nothing at home, and the rows read back exactly from the slots (a
    codec's bytes vary run to run, so the bytes are compared through the
    decode)."""
    from variable import TableVariable
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    rng = np.random.default_rng(5)
    ticks = _ticks(rng, 3, 3, 40_000)
    fills = []
    orig = RegionProducer._fill_slots
    monkeypatch.setattr(RegionProducer, "_fill_slots",
                        lambda self, *a: (fills.append(1), orig(self, *a)))
    cache = _cache(1024)
    table = _run(_producer(pool, cache), _feed(ticks, codec), cache)
    crs = [r.release[1] for r in table.regions]
    assert len(crs) == len(ticks)
    assert all(cr.admitted == cr.n for cr in crs)
    assert all(cache.pieces(cr, 0, cr.nbytes).on_gpu.all() for cr in crs)
    # No copy into a home: the cache made no copy batch.
    assert cache._core.newest_batch() < 0 and not cache._events
    var = TableVariable(table, "int")
    assert torch.equal(var.col_to_gpu().cpu(), _expected(ticks))
    var.free_underlying_mem(pool, None)
    assert len(fills) == len(ticks)


@requires_cuda
def test_a_tick_binds_its_fully_cached_stages_with_one_pieces_call(
        cuda_allocator, pool, monkeypatch):
    """Two stages over cached tables and one over a host table, in one tick:
    the cached stages are bound together (the layout's ``_TickCached``, one
    ``pieces`` call), the host stage on its own; every stage reads back
    exactly, and the reads settle at the next tick, so the cached granules
    die and their slots come back."""
    from IO.cache.native_cache import NativeGranuleCache
    from IO.transfer import data_exchange
    from IO.transfer.in_stage import DirectIn
    from IO.transfer.tick_layout import pin_cached_reads
    from variable import TableVariable
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    rng = np.random.default_rng(13)
    cache = _cache(1024)
    t1, t2, t3 = (_ticks(rng, 2, 2, 30_000) for _ in range(3))
    tab1 = _run(_producer(pool, cache), _feed(t1, "BITCOMP"), cache)
    tab2 = _run(_producer(pool, cache), _feed(t2, "RAW"), cache)
    tab3 = _run(_producer(pool, None), _feed(t3, "BITCOMP"), None)
    assert all(r.location == "cached" for r in tab1.regions + tab2.regions)
    assert not any(r.location == "cached" for r in tab3.regions)
    vars_ = [TableVariable(t, "int") for t in (tab1, tab2, tab3)]
    stages = [DirectIn(v) for v in vars_]
    layout = DirectIn.install_compiled_tick_layout(stages)
    assert layout.tick_cached is not None
    assert [b.in_tick for b in layout.per_stage_bindings] == [True, True,
                                                               False]
    calls = []
    orig = NativeGranuleCache.pieces
    monkeypatch.setattr(NativeGranuleCache, "pieces",
                        lambda self, *a: (calls.append(1), orig(self, *a))[1])
    ex = data_exchange.exchange()
    unpins = [pin_cached_reads(v) for v in vars_]
    try:
        ex.launch(stages)
    finally:
        for unpin in unpins:
            unpin()
    ex.sync()
    for stage in stages:
        stage.decompress()
    torch.cuda.synchronize()
    assert len(calls) == 1
    for stage, t in zip(stages, (t1, t2, t3)):
        assert torch.equal(stage.gpu_col.data().cpu(), _expected(t))
    cache.advance_tick()
    assert cache._core.free_count() == cache._core.num_slots()
    for v in vars_:
        v.free_underlying_mem(pool, None)


@requires_cuda
def test_a_slot_fill_waits_for_its_victims_on_the_writer_stream(
        cuda_allocator, pool, monkeypatch):
    """A pack whose admission evicts dirty granules fills their slots on the
    home writer's stream, behind the victims' copies home: the producer's
    stream does not wait on those copies (it ends while the writer stream is
    still busy), and every row reads back exactly."""
    from IO.codec.home_writer import writer_stream
    from variable import TableVariable
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    rng = np.random.default_rng(14)
    ticks = _ticks(rng, 2, 3, 150_000)
    feeds = _feed(ticks, "BITCOMP")
    cache = _cache(6)
    producer = _producer(pool, cache)
    for _ in range(2):
        cache.advance_tick()
        if producer.has_pending():
            producer.pack()
        producer.compress(feeds.pop(0))
        torch.cuda.current_stream().synchronize()
    torch.cuda.synchronize()
    # Tick 0's granules fill the six slots, dirty.  Tick 1's pack evicts
    # them: their copies home queue behind a busy writer stream.
    with torch.cuda.stream(writer_stream()):
        torch.cuda._sleep(500_000_000)
        busy = torch.cuda.Event()
        busy.record()
    cache.advance_tick()
    producer.pack()
    done = torch.cuda.Event()
    done.record()
    done.synchronize()
    assert not busy.query()
    cache.advance_tick()
    producer.drain()
    table = producer.tables()[0]
    crs = [r.release[1] for r in table.regions]
    assert [cr.admitted for cr in crs] == [6, 6]
    assert not cache.pieces(crs[0], 0, crs[0].nbytes).on_gpu.any()
    var = TableVariable(table, "int")
    assert torch.equal(var.col_to_gpu().cpu(), _expected(ticks))
    var.free_underlying_mem(pool, None)
