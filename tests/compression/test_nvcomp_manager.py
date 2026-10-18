"""nvCOMP 5.2 manager component tests (the engine compress path).

Validates the manager-format compress/decompress path end to end at the unit
level, through the engine's own compress (the region producer) and read (a
``DirectIn`` over the record table):

  t1  byte-exact round trips per codec x dtype x shape, incl. remainder
      records and empty partitions (never submitted)
  t2  incompressible input: S <= M honored, expansion placed correctly
  t3  record layout: 512-aligned in-bounds records, one stream per
      partition, each partition's rows in its own records
  t4  guard canaries: compression never writes outside its [out, out+M) span
  t5  DecompressionConfig portability across streams/managers; registry
      entries are process-lifetime (one per record, never erased)
  t6  enqueue paths hold no torch-level sync; zero header-parse syncs on the
      hot path (cached handles); the launch's actual sizes land in the pinned
      buffer (stored over UVA)
  t7  uva_store: a CUDA tensor's bytes land in a pinned tensor (a slice of
      one too) on the current stream; wrong arguments (sizes, devices,
      pinning, a dtype mismatch) are refused
  t8  a batch longer than one nvCOMP launch takes (65,535 buffers): one
      ticket and one decompress over 70,000 records round-trip byte-exact

Requires CUDA (shared custom-allocator arena from tests/conftest.py).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import constants
from tests.record_tables import region_records

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda:0"

# (name, codec id, chunk) — the ids mirror nvcomp_manager.cu's kCodec*
# constants and the chunks are format_policy._DEFAULT_SETTINGS, which the
# autouse fixture in tests/conftest.py pins for every test.
CODECS = [
    ("ANS", 1, 131072),
    ("CASCADED", 2, 32768),
    ("BITCOMP", 3, 131072),
    ("LZ4", 4, 65536),
    ("SNAPPY", 5, 65536),
    ("GDEFLATE", 6, 65536),
    ("DEFLATE", 7, 65536),
]


@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


def _mk(dtype, n, compressible=True):
    if compressible:
        base = torch.arange(n, device=DEVICE, dtype=torch.int64) % 977
    else:
        base = torch.randint(-2**62, 2**62, (n,), device=DEVICE,
                             dtype=torch.int64)
    return base.to(dtype).contiguous()


def _compress(pool, col, algo, counts=None):
    """``col`` (one tick, ``counts`` rows per partition) compressed by a
    region producer: the frozen record table of its regions."""
    from IO.codec.region_producer import RegionProducer
    producer = RegionProducer(pool, "test")
    if counts is None:
        counts = [col.shape[0]]
    producer.compress([(0, col, counts, algo)])
    producer.drain()
    return producer.tables()[0]


def _read(table):
    """The table's rows, read back through a ``DirectIn`` (the record
    reader: runs copied H2D, one batched decompress), flattened."""
    from variable import TableVariable
    got = TableVariable(table, "int").col_to_gpu()
    torch.cuda.synchronize()
    return got.reshape(-1)


# ---------------------------------------------------------------------------
# t1: round trips
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("algo,codec_idx,chunk_b", CODECS)
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int8])
def test_t1_roundtrip(cuda_allocator, pool, algo, codec_idx, chunk_b, dtype):
    if not cuda_allocator:
        pytest.skip("custom allocator unavailable")
    for n in (1, 64, 300_000):
        col = _mk(dtype, n)
        table = _compress(pool, col, algo)
        assert table.num_records > 0, f"{algo}/{dtype}/{n}: no records"
        assert (table.codec == codec_idx).all()
        assert (table.handle != 0).all()
        assert int(table.raw_bytes.sum()) == col.nbytes
        assert torch.equal(_read(table), col), f"{algo}/{dtype}/{n} mismatch"
        table.free(pool, None)

    # Partitioned (padded layout) with an empty partition (never submitted).
    from IO.codec.batched_compress import aligned_partition_layout
    from math import gcd
    elem = torch.empty(0, dtype=dtype).element_size()
    align_unit = elem * constants.COMPRESSION_ALIGNMENT // gcd(
        elem, constants.COMPRESSION_ALIGNMENT)
    counts = [5000, 0, 12345]
    _, starts, total_padded = aligned_partition_layout(counts, elem, align_unit)
    buf = torch.zeros(total_padded, dtype=dtype, device=DEVICE)
    truth = []
    for b, c in enumerate(counts):
        if c == 0:
            continue
        rows = _mk(dtype, c)
        buf[starts[b] // elem: starts[b] // elem + c] = rows
        truth.append(rows)
    table = _compress(pool, buf, algo, counts)
    assert table.stream_rows().tolist() == counts
    assert torch.equal(_read(table), torch.cat(truth))
    table.free(pool, None)


# ---------------------------------------------------------------------------
# t2: incompressible input
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("algo,codec_idx,chunk_b", CODECS)
def test_t2_incompressible(cuda_allocator, pool, algo, codec_idx, chunk_b):
    if not cuda_allocator:
        pytest.skip("custom allocator unavailable")
    from TQPlib.tqpcompress import mgr_max_compressed_size

    col = _mk(torch.int64, 400_000, compressible=False)
    table = _compress(pool, col, algo)
    for orig, s in zip(table.raw_bytes.tolist(), table.nbytes.tolist()):
        m = mgr_max_compressed_size(codec_idx, 8, chunk_b, orig)
        assert 0 < s <= m, f"S={s} outside (0, M={m}]"
    assert torch.equal(_read(table), col)
    table.free(pool, None)


# ---------------------------------------------------------------------------
# t3: record layout
# ---------------------------------------------------------------------------

@requires_cuda
def test_t3_record_layout(cuda_allocator, pool, monkeypatch):
    if not cuda_allocator:
        pytest.skip("custom allocator unavailable")
    # Small records -> several per partition.
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    counts = [300_000, 0, 200_000]
    from IO.codec.batched_compress import aligned_partition_layout
    _, starts, total_padded = aligned_partition_layout(counts, 8, 512)
    buf = torch.zeros(total_padded, dtype=torch.int64, device=DEVICE)
    truth = []
    for b, c in enumerate(counts):
        if c:
            rows = _mk(torch.int64, c, compressible=False)
            buf[starts[b] // 8: starts[b] // 8 + c] = rows
            truth.append(rows)
    table = _compress(pool, buf, "ANS", counts)
    assert table.num_streams == len(counts)
    assert table.stream_rows().tolist() == counts
    per_stream = np.diff(table.stream_ptr)
    assert per_stream.tolist()[1] == 0 and (per_stream[[0, 2]] >= 2).all()
    # Aligned, in-bounds, non-overlapping record extents in each region.
    assert (table.offset % constants.COMPRESSION_ALIGNMENT == 0).all()
    for k, region in enumerate(table.regions):
        recs = region_records(table, k)
        offs, ends = table.offset[recs], table.offset[recs] + table.nbytes[recs]
        assert int(ends.max()) <= region.nbytes
        order = np.argsort(offs)
        assert (offs[order][1:] >= ends[order][:-1]).all()
    assert torch.equal(_read(table), torch.cat(truth))
    table.free(pool, None)


# ---------------------------------------------------------------------------
# t4: guard canaries — compression never writes outside [out, out+M)
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("algo,codec_idx,chunk_b", CODECS)
def test_t4_canaries(cuda_allocator, algo, codec_idx, chunk_b):
    if not cuda_allocator:
        pytest.skip("custom allocator unavailable")
    from TQPlib.tqpcompress import (mgr_compress_begin, mgr_compress_launch,
                                    mgr_compress_finish)

    torch.manual_seed(4)
    recs = [torch.randint(-2**62, 2**62, (32_768,), dtype=torch.int64,
                          device=DEVICE).view(torch.uint8) for _ in range(4)]
    sizes = torch.tensor([r.nbytes for r in recs], dtype=torch.int64)
    tid, max_sizes = mgr_compress_begin(codec_idx, 8, chunk_b, sizes)

    guard = 1024
    max_np = max_sizes.numpy()
    spans = (max_np + 511) // 512 * 512 + guard
    slab = torch.full((int(spans.sum()),), 0xAB, dtype=torch.uint8,
                      device=DEVICE)
    offs = np.zeros(len(recs), dtype=np.int64)
    np.cumsum(spans[:-1], out=offs[1:])
    out_ptrs = torch.from_numpy(offs + slab.data_ptr())
    cs_dev = torch.empty(len(recs), dtype=torch.int64, device=DEVICE)
    cs_pin = torch.empty(len(recs), dtype=torch.int64, pin_memory=True)
    handles = mgr_compress_launch(
        tid, torch.tensor([r.data_ptr() for r in recs]), out_ptrs,
        cs_dev, cs_pin)
    torch.cuda.synchronize()
    statuses = mgr_compress_finish(tid)
    assert (statuses == 0).all()
    actual = cs_pin.numpy()
    assert (actual > 0).all() and (actual <= max_np).all()
    for i in range(len(recs)):
        tail = slab[int(offs[i] + max_np[i]): int(offs[i] + spans[i])]
        assert bool((tail == 0xAB).all()), (
            f"{algo}: record {i} wrote past its worst-case bound")


# ---------------------------------------------------------------------------
# t5: config portability + registry hygiene
# ---------------------------------------------------------------------------

@requires_cuda
def test_t5_config_portability(cuda_allocator):
    if not cuda_allocator:
        pytest.skip("custom allocator unavailable")
    import TQPlib.tqpcompress as tc

    base = int(dict(tc.mgr_counters())["live_configs"])
    data = torch.randint(0, 999, (1 << 19,), dtype=torch.int64,
                         device=DEVICE).view(torch.uint8)
    sA, sB = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(sA):
        tid, ms = tc.mgr_compress_begin(1, 1, 131072,
                                        torch.tensor([data.nbytes]))
        out = torch.empty(int(ms[0]), dtype=torch.uint8, device=DEVICE)
        cs_dev = torch.empty(1, dtype=torch.int64, device=DEVICE)
        cs_pin = torch.empty(1, dtype=torch.int64, pin_memory=True)
        handles = tc.mgr_compress_launch(
            tid, torch.tensor([data.data_ptr()]),
            torch.tensor([out.data_ptr()]), cs_dev, cs_pin)
        evt = torch.cuda.Event()
        evt.record(sA)
    sA.synchronize()
    assert (tc.mgr_compress_finish(tid) == 0).all()

    dst = torch.zeros_like(data)
    with torch.cuda.stream(sB):
        sB.wait_event(evt)
        # Decompress through a DIFFERENT same-spec manager (stream B) using
        # stream A's cached config.
        tc.mgr_decompress(1, 1, 131072, torch.tensor([out.data_ptr()]),
                          torch.tensor([dst.data_ptr()]), handles)
    sB.synchronize()
    assert torch.equal(dst, data)

    # Registry entries are process-lifetime: one per emitted record.
    assert int(dict(tc.mgr_counters())["live_configs"]) == base + 1


# ---------------------------------------------------------------------------
# t6: sync-freedom of the enqueue paths
# ---------------------------------------------------------------------------

@requires_cuda
def test_t6_sync_freedom(cuda_allocator, pool):
    if not cuda_allocator:
        pytest.skip("custom allocator unavailable")
    import TQPlib.tqpcompress as tc

    col = _mk(torch.int64, 200_000)
    table = _compress(pool, col, "ANS")
    assert len(table.regions) == 1
    # The region's bytes on the GPU, decoded from the table's cached
    # handles, as the reader's manager template does.
    stored = table.regions[0].storage.to(DEVICE)
    comp_ptrs = torch.from_numpy(table.offset + stored.data_ptr())
    output = torch.empty_like(col)
    out_offs = np.zeros(table.num_records, dtype=np.int64)
    np.cumsum(table.raw_bytes[:-1], out=out_offs[1:])
    out_ptrs = torch.from_numpy(out_offs + output.data_ptr())
    handles = torch.from_numpy(table.handle.copy())
    torch.cuda.synchronize()
    parses_before = int(dict(tc.mgr_counters())["header_parse_syncs"])

    # Hot decompress with cached handles: no torch-level sync, no header parse.
    codec_idx, elem_size, chunk_b = table.codec_keys[1]
    torch.cuda.set_sync_debug_mode("error")
    try:
        tc.mgr_decompress(codec_idx, elem_size, chunk_b, comp_ptrs, out_ptrs,
                          handles)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert torch.equal(output, col)
    assert int(dict(tc.mgr_counters())["header_parse_syncs"]) == parses_before
    table.free(pool, None)

    # Enqueue-only compress: begin/launch under sync-error mode.
    data = col.view(torch.uint8)
    torch.cuda.set_sync_debug_mode("error")
    try:
        tid, ms = tc.mgr_compress_begin(1, 8, 131072,
                                        torch.tensor([data.nbytes]))
        out = torch.empty(int(ms[0]), dtype=torch.uint8, device=DEVICE)
        cs_dev = torch.empty(1, dtype=torch.int64, device=DEVICE)
        cs_pin = torch.empty(1, dtype=torch.int64, pin_memory=True)
        handles = tc.mgr_compress_launch(
            tid, torch.tensor([data.data_ptr()]),
            torch.tensor([out.data_ptr()]), cs_dev, cs_pin)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert (tc.mgr_compress_finish(tid) == 0).all()
    assert 0 < int(cs_pin[0]) == int(cs_dev[0].item()) <= int(ms[0])


@requires_cuda
def test_t7_uva_store(cuda_allocator):
    import TQPlib.tqpcompress as tc
    src = torch.arange(37, dtype=torch.int64, device=DEVICE) * 3 + 1
    dst = torch.zeros(64, dtype=torch.int64, pin_memory=True)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        side.wait_stream(torch.cuda.current_stream())
        tc.uva_store(src, dst[5:42])
        done = torch.cuda.Event()
        done.record(side)
    done.synchronize()
    assert torch.equal(dst[5:42], src.cpu())
    assert not dst[:5].any() and not dst[42:].any()
    # A scalar (a mask count) into one slot.
    one = torch.tensor(12345, dtype=torch.int64, device=DEVICE)
    tc.uva_store(one.view(1), dst[0:1])
    torch.cuda.synchronize()
    assert int(dst[0]) == 12345
    with pytest.raises(RuntimeError):
        tc.uva_store(src, dst[:4])                     # too small
    with pytest.raises(RuntimeError):
        tc.uva_store(src, torch.zeros(64, dtype=torch.int64))  # not pinned
    with pytest.raises(RuntimeError):
        tc.uva_store(src.cpu(), dst)                    # not CUDA
    with pytest.raises(RuntimeError):
        tc.uva_store(src.int(), dst)                    # another dtype


# ---------------------------------------------------------------------------
# t8: batches longer than one nvCOMP launch
# ---------------------------------------------------------------------------

@requires_cuda
def test_t8_batch_longer_than_one_launch(cuda_allocator):
    """nvCOMP cannot launch a batched call over more than 65,535 buffers
    ("invalid configuration argument"); the manager surface slices it, so
    one ticket and one decompress over 70,000 records round-trip."""
    if not cuda_allocator:
        pytest.skip("custom allocator unavailable")
    import TQPlib.tqpcompress as tc

    n, rows = 70_000, 256                  # 2 KiB records, BITCOMP int64
    torch.manual_seed(8)
    data = torch.randint(0, 1 << 20, (n, rows), dtype=torch.int64,
                         device=DEVICE)
    rec_b = rows * 8
    tid, max_sizes = tc.mgr_compress_begin(
        3, 8, 131072, torch.full((n,), rec_b, dtype=torch.int64))
    spans = (max_sizes.numpy() + 511) // 512 * 512
    slab = torch.empty(int(spans.sum()), dtype=torch.uint8, device=DEVICE)
    offs = np.zeros(n, dtype=np.int64)
    np.cumsum(spans[:-1], out=offs[1:])
    comp = torch.from_numpy(offs + slab.data_ptr())
    rec = np.arange(n, dtype=np.int64) * rec_b
    handles = tc.mgr_compress_launch(
        tid, torch.from_numpy(rec + data.data_ptr()), comp,
        torch.empty(n, dtype=torch.int64, device=DEVICE),
        torch.empty(n, dtype=torch.int64, pin_memory=True))
    torch.cuda.synchronize()
    assert (tc.mgr_compress_finish(tid) == 0).all()
    out = torch.zeros_like(data)
    tc.mgr_decompress(3, 8, 131072, comp,
                      torch.from_numpy(rec + out.data_ptr()), handles)
    torch.cuda.synchronize()
    assert torch.equal(out, data)
