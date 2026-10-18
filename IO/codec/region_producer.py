"""The region producer: a pipe's outputs, one region per tick and column
instance.

Every pipe's outputs go through here: the sort and filter pipes', the
direct pipes' (join core, aggregation passes, project), and the gathers'
chunks (``IO/pipeline/rearrange_pipeline.py``); so do the scan's
representations (``IO/data_scanner.py``, one tick per batch) and the codec
sweep's cells (``profiling/sweep.py``).  With the granule cache
(``cache``) the producer offers its cacheable outputs' regions to the cache
(below).

One tick of one column instance becomes one *region*: its records back to
back, 512-aligned, the instance's buckets in bucket order inside it, each
bucket's records in row order.  The region's bytes live in the pinned host
pool (its *home*) and never change.  Homes are carved from per-column pinned
chunks of at most ``_CHUNK_MAX_B``, sized for what the column is expected to
write still and trimmed back when the column moves on to a new chunk or the
pipe ends (``_Column.carve``); a carved region has no allocation of its own,
and the table returns the chunks (``RecordTable.host_blocks``).  A pipe's
regions of one instance form one record table (``IO/pages/record_table.py``)
with one stream per bucket, frozen at the end of the pipe.

A tick, for a compressed instance:

1. **Tick t, the producer's stream** (``compress``; the pipe's compute
   stream).  Each bucket's rows are cut into records from the bucket's
   first row (``record_rows``: a power-of-two number of rows within
   ``TQP_MGR_RECORD_MB`` raw bytes, 4 MiB by default, and within
   ``RECORD_ROWS_MAX`` rows, 2^20), and the instances that share a codec key are compressed
   together, one batched manager call per piece of at most 256 MiB of
   worst-case output (``_MAX_OUT_BUF_BYTES``), each piece into a worst-case
   slab of its own (so no allocation is larger than one piece), with the
   actual sizes stored into pinned memory over UVA by one block on the
   stream (``tqpcompress.uva_store``): a copy would queue on the copy engine
   behind the copies home already issued, and the stream with it.  Nothing
   blocks, and every slab stays alive until the pack.
2. **Tick t+1, before the operator** (``pack``).  The sizes have landed (the
   pack waits for the compress's event).  Each instance's records
   get packed offsets from one cumulative sum and its home is carved; then,
   piece by piece, one ``batched_memcpy_abs`` compacts the piece's records
   from its slab into a dense GPU buffer of its own, and the slab goes.  The
   tick's rows are appended to each instance's table builder.
3. **Right behind the pack** (``home_writer.HomeWriter.ship``).  The home
   writer copies each instance's part of each dense buffer into its home
   (one copy per instance and piece) and drops the buffers once the copies
   have ended.

A RAW instance needs no deferral: its output tensor is the dense buffer
(each bucket's rows cut into records the same way, the padding between
buckets shipping with them), so ``compress`` ships it at once.  At the end of the pipe, ``drain``
packs the last tick's records and waits for every copy still in flight;
``tables`` then holds the frozen tables.

Under the free-list baseline (``on_gpu``: ``--buffer_strategy freelist``, a
pipe's producer, not the scan's) every region stays on the GPU in an arena
allocation of its own (so an eviction frees what the region holds), its
home carved as above but not written, and is registered as spillable
(``IO/pages/device_regions.py``).  A RAW output's region is its output
tensor (a copy of it when the tensor is a view into a larger allocation).
A compressed output's pack (``_keep_packed_on_gpu``) gives each instance a
buffer of its packed size and compacts the piece's records into those
buffers instead of a dense buffer, and nothing is shipped home: an eviction
later copies the region home, compressed, as it is.

With the granule cache (``IO/cache/native_cache.py``), an output in
``cached_outputs`` is a cached region: its home is carved as above, and at
the pack (a RAW output: at its compress) the tick's cached regions are
offered to the cache together.  The cache admits a prefix of each region's
granules; one copy kernel over absolute addresses (``batched_memcpy_abs``)
writes each piece's records into their slots up to the prefix and into the
dense buffer past it, and only the dense part goes home.  Then the cache's
cleaning targets are copied home on the writer, behind the tick's slot
fills.  A slab or dense buffer is a plain arena allocation: with the cache
attached, one that misses borrows the cache region's lowest blocks inside
the allocator (``memory/README.md``, "Lending"), which moves no granule
admitted this tick, so the slot addresses the pack took stay valid.  A
dense buffer that still does not fit is skipped, its records copied from
the slab straight to their homes (with or without a cache); a slab that
does not fit fails the pipe.

The compression searches keep seeing the output path in the timing they
read: the pipeline adds the pack's span on the compute stream to the tick's
``compress_out``, and the home writer's copies form the d2h lane
(``collect_lane``).  Under ``TQP_EXPERIMENT_TELEMETRY`` each shipment's
copies are a ``spill_bypass`` record of the movement ledger (a page there is
a region); a cached region's slot bytes are ``cache_fill`` (a page there is a
granule) and its refused part ``spill_rejected``, and cleaning copies are
``spill_mirror``.
"""
from __future__ import annotations

import os
from math import gcd
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import constants
from IO.cache import config as _cache_cfg
from TQPlib.tqpcache import split_at_granules as _native_split
from IO.codec.home_writer import HomeWriter, writer_stream
from IO.pages import abs_copy, device_regions
from IO.pages.record_table import RecordTable, RecordTableBuilder
from IO.pages.region import CACHED, DEVICE, HOST, Region
from IO.telemetry import gates, movement


_native = None


def _tqp():
    """The native compression module, imported once."""
    global _native
    if _native is None:
        import TQPlib.tqpcompress as native
        _native = native
    return _native


def _is_raw(algo: Optional[str]) -> bool:
    return not algo or algo == "RAW"


# Every record holds a power-of-two number of rows, the largest within both
# caps below, cut from the start of its row space (the scan's row 0) or
# producer slice (one tick's bucket), so the record boundaries of a narrower
# column are boundaries of every wider one.
# The raw-byte cap of a record, in MiB (``TQP_MGR_RECORD_MB``, fractions
# allowed); None: 4.
_env_mb = os.environ.get("TQP_MGR_RECORD_MB", "")
RECORD_MB: Optional[float] = float(_env_mb) if _env_mb else None
# The row cap of a record (None: no row cap), 2^20 rows, read with
# ``RECORD_MB`` on every use.  It keeps a narrow column's records (a 1-byte
# mask's) from growing with the byte cap, so a pipe that reads or writes a
# mask keeps its quantum at 2^20 rows (a tick rounds down by less than one
# such quantum) while columns of 4 bytes and wider get records up to the
# byte cap.
RECORD_ROWS_MAX: Optional[int] = 1 << 20


def record_cap_mb() -> float:
    """The raw-byte cap of a record, in MiB."""
    if RECORD_MB is not None:
        return RECORD_MB
    return 4


def record_rows(row_bytes: int) -> int:
    """Rows per record of a column of ``row_bytes`` (the last record of a
    stream may hold fewer): the largest power of two whose raw size fits
    the cap and that is at most ``RECORD_ROWS_MAX``, raised to the fewest
    rows that keep a record a multiple of 512 bytes (a power of two too,
    above the cap only for rows wider than 8 KiB at a 4 MiB cap)."""
    align = constants.COMPRESSION_ALIGNMENT
    align_rows = align // gcd(row_bytes, align)
    fit = int(record_cap_mb() * (1 << 20)) // row_bytes
    if RECORD_ROWS_MAX is not None:
        fit = min(fit, RECORD_ROWS_MAX)
    rows = 1 << (fit.bit_length() - 1) if fit else 1
    return max(rows, align_rows)


def record_cap_b(row_bytes: int) -> int:
    """The raw bytes of a full record: ``record_rows`` whole rows."""
    return record_rows(row_bytes) * row_bytes


def row_quantum(row_bytes: int = 1) -> int:
    """The rows every record boundary of a column of at most ``row_bytes``
    per row (and of a 1-byte mask) falls on a multiple of: the rows per
    record of a 1-byte column, or of the column itself when its alignment
    raises it higher (both powers of two).  A row space cut at multiples of
    it starts a record in every such column."""
    return max(record_rows(1), record_rows(row_bytes))


def _cut_records(starts: np.ndarray, nbytes: np.ndarray, cap: int):
    """Each non-empty bucket (its first byte ``starts``, its bytes
    ``nbytes``) cut into records of ``cap`` bytes from its start, the last
    shorter: each record's first byte, bytes and bucket."""
    if nbytes.max() <= cap:
        bucket = np.flatnonzero(nbytes)
        return starts[bucket], nbytes[bucket], bucket
    k = (nbytes + cap - 1) // cap
    first = np.zeros(len(k), dtype=np.int64)
    np.cumsum(k[:-1], out=first[1:])
    bucket = np.repeat(np.arange(len(k), dtype=np.int64), k)
    off = (np.arange(len(bucket), dtype=np.int64) - first[bucket]) * cap
    return (starts[bucket] + off,
            np.minimum(cap, nbytes[bucket] - off), bucket)


def slab_bytes_per_row(algo: Optional[str], elem_size: int,
                       row_bytes: int) -> int:
    """GPU bytes a compressed output holds per row from its compress until
    its pack: its worst-case slab, from the bound of one full record (0 for
    RAW, which has no slab)."""
    if _is_raw(algo):
        return 0
    from IO.codec.manager_compress import _codec_info, _worst_case_aligned
    codec_idx, chunk_b = _codec_info(algo)
    cap = record_cap_b(row_bytes)
    worst = _worst_case_aligned(codec_idx, elem_size, chunk_b, cap,
                                constants.COMPRESSION_ALIGNMENT)
    rows = cap // row_bytes
    return -(-worst // rows)


def _bucket_starts(tensor: torch.Tensor, counts: np.ndarray) -> np.ndarray:
    """Each bucket's first byte in ``tensor`` (2-D), in the padded layout
    the sort pipe uses (``batched_compress.aligned_partition_layout``: each
    non-empty bucket rounded up to whole ``lcm(row bytes, 512)`` units),
    where a compressed record's input must start.  A compact tensor (its
    buckets back to back) is that layout when every bucket but the last is
    whole units, one bucket included; a compact tensor with a bucket off its
    unit is refused."""
    rb = tensor.shape[1] * tensor.element_size()
    starts = np.zeros(len(counts), dtype=np.int64)
    align = constants.COMPRESSION_ALIGNMENT
    per_unit = (rb * align // gcd(rb, align)) // rb
    if tensor.shape[0] == int(counts.sum()):
        off = np.flatnonzero(counts[:-1] % per_unit)
        assert not len(off), (
            f"region producer: a compact tensor of {len(counts)} buckets "
            f"starts bucket {int(off[0]) + 1} off its {per_unit}-row unit "
            f"(bucket {int(off[0])} holds {int(counts[off[0]])} rows); lay "
            f"it out padded (aligned_partition_layout)")
        np.cumsum(counts[:-1], out=starts[1:])
        return starts * rb
    padded = (counts + per_unit - 1) // per_unit * per_unit
    np.cumsum(padded[:-1], out=starts[1:])
    assert tensor.shape[0] == int(padded.sum()), (
        f"region producer: {tensor.shape[0]} rows match neither the compact "
        f"({int(counts.sum())}) nor the padded ({int(padded.sum())}) layout")
    return starts * rb


class _Column:
    """One output instance: its buckets, row geometry, its regions and
    records so far (one block per tick, appended to a table builder when the
    pipe drains), and the pinned chunk its regions are carved from."""

    __slots__ = ("base", "B", "dtype", "width", "row_bytes",
                 "cap", "regions", "blocks", "chunk_ptr", "chunk",
                 "chunk_used", "home_ptrs", "written", "carved")

    def __init__(self, base: int, B: int):
        self.base = base
        self.B = B
        self.dtype = None
        self.width = None
        self.row_bytes = None
        self.cap = None             # raw bytes of a record
        self.regions: List[Region] = []
        # (region index, offsets, stored bytes, raw bytes, streams, codec
        # key or None, handles or None) per tick
        self.blocks: list = []
        self.chunk_ptr = None       # the open chunk's pool pointer
        self.chunk = None           # its bytes (uint8, pinned)
        self.chunk_used = 0
        self.home_ptrs: List[int] = []   # every chunk, for the table
        self.written = 0            # region bytes carved so far
        self.carved = 0             # regions carved so far

    def take(self, tensor: torch.Tensor, B: int) -> None:
        assert B == self.B, (
            f"output column {self.base} changed buckets: {self.B} -> {B}")
        rb = tensor.shape[1] * tensor.element_size()
        if self.row_bytes is None:
            self.dtype = tensor.dtype
            self.width = tensor.shape[1]
            self.row_bytes = rb
            self.cap = record_cap_b(rb)
        assert rb == self.row_bytes and tensor.dtype == self.dtype, (
            f"output column {self.base} changed row geometry: "
            f"{self.dtype} x {self.width} -> {tensor.dtype} x {tensor.shape[1]}")

    def carve(self, nbytes: int, pool, ticks_left,
              tag: str = "intermediate") -> Tuple[Region, int]:
        """A home of ``nbytes`` for the column's next region, carved from
        its open chunk, or from a new chunk when the open one is full:
        returns the region and the home's address.  A new chunk is sized
        for the column's expected remaining output (its mean region so far
        times ``ticks_left``, the ticks that still produce, this one
        included; ``_TICKS_LEFT_UNKNOWN`` when the caller does not know),
        at most ``_CHUNK_MAX_B`` and at least ``nbytes``, so regions share
        the pool's block rounding and the reservation is about what the
        column will still write.  When the pool has no run for such a chunk,
        the region alone is allocated.  ``tag`` is the chunk's pool tag."""
        align = constants.COMPRESSION_ALIGNMENT
        off = (self.chunk_used + align - 1) // align * align
        if self.chunk is None or off + nbytes > self.chunk.numel():
            self.close(pool)
            size = nbytes
            if self.carved:
                left = _TICKS_LEFT_UNKNOWN if ticks_left is None \
                    else int(ticks_left)
                expect = self.written // self.carved * left
                size = max(nbytes, min(expect, _CHUNK_MAX_B))
            got = pool.try_get_storage(size, tag) if size > nbytes else None
            # At the edge of the pool a chunk may not fit where the region
            # alone does: the region then gets an allocation of its own.
            self.chunk_ptr, self.chunk = (got if got is not None
                                          else pool.get_storage(nbytes, tag))
            self.home_ptrs.append(self.chunk_ptr)
            off = 0
        region = Region(nbytes, HOST, storage=self.chunk[off:off + nbytes])
        self.chunk_used = off + nbytes
        self.written += nbytes
        self.carved += 1
        return region, self.chunk.data_ptr() + off

    def close(self, pool) -> None:
        """Give the open chunk's unused blocks back to the pool."""
        if self.chunk is not None:
            pool.shrink(self.chunk_ptr, self.chunk_used)
            self.chunk = None
            self.chunk_ptr = None
            self.chunk_used = 0

    def freeze(self) -> RecordTable:
        """The column's table: its regions, its blocks appended in runs of
        one codec key, and its chunks (``RecordTable.host_blocks``)."""
        builder = RecordTableBuilder(self.dtype, self.width,
                                     num_streams=self.B)
        for region in self.regions:
            builder.add_region(region)
        i = 0
        while i < len(self.blocks):
            key = self.blocks[i][5]
            j = i
            while j < len(self.blocks) and self.blocks[j][5] == key:
                j += 1
            run = self.blocks[i:j]
            sizes = [len(b[1]) for b in run]
            builder.append(
                np.repeat(np.array([b[0] for b in run], dtype=np.int64),
                          sizes),
                np.concatenate([b[1] for b in run]),
                np.concatenate([b[2] for b in run]),
                np.concatenate([b[3] for b in run]),
                stream=np.concatenate([b[4] for b in run]),
                codec_key=key,
                handles=(None if key is None
                         else np.concatenate([b[6] for b in run])))
            i = j
        self.blocks = []
        table = builder.freeze()
        table.host_blocks = list(self.home_ptrs)
        self.home_ptrs = []
        return table


class _Piece:
    """One batched manager compress of one tick (one codec key, at most
    ``_MAX_OUT_BUF_BYTES`` of worst-case output): records ``lo:hi`` of the
    tick's arrays, compressed into the piece's own worst-case slab at
    ``slab_off`` (bytes from the slab's start)."""

    __slots__ = ("codec_key", "tid", "lo", "hi", "slab", "slab_off")

    def __init__(self, codec_key, tid, lo, hi, slab, slab_off):
        self.codec_key = codec_key
        self.tid = tid
        self.lo = lo                    # first record in the tick's arrays
        self.hi = hi
        self.slab = slab
        self.slab_off = slab_off


class _Tick:
    """One tick's compressed records, until their pack: its pieces (each
    with the slab its records were compressed into), their pinned actual
    sizes, and per record its worst-case bound, column base, bucket, raw
    bytes and config handle."""

    __slots__ = ("label", "pieces", "pin", "max_sizes",
                 "streams", "raw", "handles", "insts")

    def __init__(self, label, pieces, pin, max_sizes, streams, raw, handles,
                 insts):
        self.label = label              # the output tick, for the cap
        self.pieces = pieces
        self.pin = pin
        self.max_sizes = max_sizes
        self.streams = streams
        self.raw = raw
        self.handles = handles
        # The instances: (base, first record, end, codec key) in record
        # order.
        self.insts = insts


# Record sizes are rounded up to this before their worst-case bound is looked
# up, for cutting a tick's records into pieces: a few lookups per tick instead
# of one per record, at the price of pieces a little smaller than the limit.
_BOUND_GRAIN = 1 << 20


def _alloc_slab(nbytes: int, device) -> torch.Tensor:
    """A GPU byte buffer for a piece's worst-case slab or dense buffer (one
    place, so a test can watch every such allocation)."""
    return torch.empty(nbytes, dtype=torch.uint8, device=device)


def _out_of_memory(exc: BaseException) -> bool:
    """An allocation the arena could not serve (the arena's ``Pool out of
    memory`` or torch's ``OutOfMemoryError``)."""
    return (type(exc).__name__ == "OutOfMemoryError"
            or "out of memory" in str(exc).lower())


def _split_at_granules(off, size, G):
    """Byte ranges ``[off, off + size)`` (arrays) cut at multiples of ``G``:
    per part, its range index, granule and offset and size."""
    return _native_split(np.ascontiguousarray(off, dtype=np.int64),
                         np.ascontiguousarray(size, dtype=np.int64), int(G))

# The largest pinned chunk a column's regions are carved from.  Regions
# this large get an allocation of their own (the pool's block rounding is
# then under 2% of them), and no column reserves more than this ahead of
# what it has written.
_CHUNK_MAX_B = 16 << 20
# The ticks a column is assumed to still produce when its pipe does not say
# (``RegionProducer.ticks_left`` None): a chunk then holds this many of its
# mean regions, within the cap.
_TICKS_LEFT_UNKNOWN = 8


class RegionProducer:
    """One pipe's region producer (module docstring)."""

    def __init__(self, cpu_pool, pipe_name: str = "", cache=None,
                 chunk_b: Optional[int] = None, spills: bool = True,
                 pool_tag: str = "intermediate"):
        assert cpu_pool is not None, "regions need a pinned host pool"
        self._cpu_pool = cpu_pool
        # What the homes hold, for the pinned pool's residency counts
        # (``IO/pages/pinned_mem.py``): the scan names ``"scan"``.
        self.pool_tag = pool_tag
        self.pipe_name = pipe_name
        # The nvCOMP manager chunk of every codec (None: the codec map's);
        # only the codec sweep (``profiling/sweep.py``) sets it, to measure
        # its chunk grid through this path.
        self.chunk_b = chunk_b
        # Whether the copies home are spills in the movement ledger: the
        # scan's (``IO/data_scanner.py``) are not.
        self.spills = spills
        self.writer = HomeWriter()
        # The free-list baseline: outputs stay on the GPU as spillable
        # regions (module docstring); never the scan's or the sweep's.
        self.on_gpu = (device_regions.armed() and spills
                       and pool_tag == "intermediate")
        # The granule cache (``IO/cache/native_cache.py``) and, by base,
        # the outputs it may hold: (consumer position, expected reads).
        # Other outputs, and every output without a cache, are plain host
        # regions.
        self.cache = cache
        self.cached_outputs: Dict[int, Tuple[int, int]] = {}
        # Per-producer state: codec info by name, the device buffer of
        # actual sizes (reused: a tick's compress follows the last pack,
        # which read the sizes it held) and the pinned sizes' numpy view.
        self._codecs: Dict[str, Tuple[int, int]] = {}
        self._comp_dev = None
        self._size_np = None
        self._filled = torch.cuda.Event() if cache is not None else None
        # The events the next slot fill waits on, on the writer's stream:
        # the copies home of the last admission's victims (``_admit``).
        self._fill_waits: list = []
        self._cols: Dict[int, _Column] = {}
        self._tick: Optional[_Tick] = None
        self._tick_no = 0            # output ticks compressed so far
        # Ticks that will still produce output, this one included; the
        # pipeline sets it before each pack (None: unknown).
        self.ticks_left = None
        # The pipe's compute stream, set by the pipeline (None: whatever
        # stream is current at each call).  The pipeline calls compress,
        # pack and drain with it current.
        self.stream = None
        # Reused across ticks: the event after a tick's compress launches,
        # the event a shipment's copies wait for, and the pinned buffer of
        # the actual sizes (read back before its next use).  The pack's
        # copy kernels read their address arrays over UVA from pinned
        # buffers of ``IO.pages.abs_copy``.
        self._compressed = torch.cuda.Event()
        self._after = torch.cuda.Event()
        self._size_pin = None
        self._tables: Optional[Dict[int, RecordTable]] = None

    def _stream(self):
        """The stream the producer's work runs on: the pipe's compute
        stream when the pipeline set ``stream``, else the current one."""
        return self.stream if self.stream is not None \
            else torch.cuda.current_stream()

    def has_pending(self) -> bool:
        """Records compressed and not yet packed."""
        return self._tick is not None

    def _column(self, base: int, tensor: torch.Tensor, B: int) -> _Column:
        col = self._cols.get(base)
        if col is None:
            col = self._cols[base] = _Column(base, B)
        col.take(tensor, B)
        return col

    # -- tick t ----------------------------------------------------------

    def compress(self, items) -> Dict[int, int]:
        """Tick t, on the producer's stream (``stream``, current):
        ``items`` are ``(base, tensor, counts, algo)`` per output instance
        (``counts``: rows per bucket).  Compressed instances are launched;
        RAW ones are shipped.  Returns the stored bytes of the RAW
        instances, by base."""
        stream = self._stream()
        groups: Dict[tuple, list] = {}
        stored: Dict[int, int] = {}
        self._tick_no += 1
        for base, tensor, counts, algo in items:
            if tensor.dim() < 2:
                tensor = tensor.unsqueeze(1)
            counts = np.asarray(counts, dtype=np.int64)
            if not counts.any():
                continue
            col = self._column(base, tensor, len(counts))
            if _is_raw(algo):
                stored[base] = self._ship_raw(col, tensor, counts)
                continue
            key = (algo, tensor.dtype, tensor.shape[1])
            group = groups.get(key)
            if group is None:
                groups[key] = [(col, tensor, counts)]
            else:
                group.append((col, tensor, counts))
        if groups:
            self._launch(groups, stream)
        return stored

    @staticmethod
    def _col_records(col, tensor, counts):
        """One instance's records: absolute source pointers, raw bytes and
        buckets, bucket by bucket, each bucket's rows cut at the column's
        record cap."""
        rb = col.row_bytes
        cap = col.cap
        # One compact bucket: its one start is the tensor's.  A padded or
        # mismatched tensor takes ``_bucket_starts``, which checks its rows.
        if len(counts) == 1 and tensor.shape[0] == int(counts[0]):
            nb = int(counts[0]) * rb
            if nb <= cap:
                return (np.array([tensor.data_ptr()], dtype=np.int64),
                        np.array([nb], dtype=np.int64),
                        np.zeros(1, dtype=np.int64))
        nbytes = counts * rb
        starts = _bucket_starts(tensor, counts)
        starts += tensor.data_ptr()
        return _cut_records(starts, nbytes, cap)

    def _launch(self, groups, stream) -> None:
        """Every codec group's records, in pieces of at most
        ``_MAX_OUT_BUF_BYTES`` of worst-case output, each one batched manager
        compress into a worst-case slab of its own: no allocation of the
        tick is larger than one piece, and each instance's records are named
        as they are cut.  A failure on the way (an allocation that does not
        fit) finishes every ticket begun, so none is left in the manager's
        registry."""
        from IO.codec.batched_compress import _MAX_OUT_BUF_BYTES
        from IO.codec.manager_compress import _codec_info, _worst_case_aligned
        native = _tqp()
        align = constants.COMPRESSION_ALIGNMENT
        assert self._tick is None, "a tick compressed before the last pack"
        plan = []         # (codec key, lo, hi) per piece
        insts = []        # (base, lo, hi, codec key) per instance
        ptr_parts, size_parts, stream_parts = [], [], []
        n = 0
        device = None
        for (algo, _, _), cols in groups.items():
            info = self._codecs.get(algo)
            if info is None:
                info = self._codecs[algo] = _codec_info(algo, self.chunk_b)
            codec_idx, chunk_b = info
            elem_size = cols[0][1].element_size()
            device = cols[0][1].device
            key = (codec_idx, elem_size, chunk_b)
            g0 = n
            top = 0
            for col, tensor, counts in cols:
                ptrs, sizes, bucket = self._col_records(col, tensor, counts)
                k = len(sizes)
                insts.append((col.base, n, n + k, key))
                n += k
                ptr_parts.append(ptrs)
                size_parts.append(sizes)
                stream_parts.append(bucket)
                top = max(top, int(sizes.max()))
            m = n - g0
            top = (top + _BOUND_GRAIN - 1) // _BOUND_GRAIN
            if m * _worst_case_aligned(codec_idx, elem_size, chunk_b,
                                       top * _BOUND_GRAIN,
                                       align) <= _MAX_OUT_BUF_BYTES:
                plan.append((key, g0, n))
                continue
            gsizes = np.concatenate(size_parts[len(size_parts) - len(cols):])
            grains = (gsizes + _BOUND_GRAIN - 1) // _BOUND_GRAIN
            uniq, inv = np.unique(grains, return_inverse=True)
            cum = np.cumsum(np.array(
                [_worst_case_aligned(codec_idx, elem_size, chunk_b,
                                     int(g) * _BOUND_GRAIN, align)
                 for g in uniq.tolist()], dtype=np.int64)[inv])
            cuts = [0]
            while cuts[-1] < m:
                lo = cuts[-1]
                base_b = int(cum[lo - 1]) if lo else 0
                hi = int(np.searchsorted(cum, base_b + _MAX_OUT_BUF_BYTES,
                                         side="right"))
                cuts.append(max(hi, lo + 1))
            for lo, hi in zip(cuts[:-1], cuts[1:]):
                plan.append((key, g0 + lo, g0 + hi))
        one = len(size_parts) == 1
        ptrs = ptr_parts[0] if one else np.concatenate(ptr_parts)
        sizes = size_parts[0] if one else np.concatenate(size_parts)
        streams = stream_parts[0] if one else np.concatenate(stream_parts)
        if self._comp_dev is None or self._comp_dev.numel() < n:
            self._comp_dev = torch.empty(max(n, 1024), dtype=torch.int64,
                                         device=device)
        if self._size_pin is None or self._size_pin.numel() < n:
            self._size_pin = torch.empty(max(n, 1024), dtype=torch.int64,
                                         pin_memory=True)
            self._size_np = self._size_pin.numpy()
        comp_dev, pin = self._comp_dev, self._size_pin
        whole = len(plan) == 1
        max_sizes = None if whole else np.empty(n, dtype=np.int64)
        handles = None if whole else np.empty(n, dtype=np.int64)
        pieces = []
        begun = []
        try:
            for key, lo, hi in plan:
                tid, piece_max = native.mgr_compress_begin(
                    key[0], key[1], key[2],
                    torch.from_numpy(sizes if whole else sizes[lo:hi]))
                begun.append(tid)
                piece_max = piece_max.numpy()
                padded = piece_max + (align - 1)
                padded &= -align
                slab_off = np.empty(hi - lo, dtype=np.int64)
                slab_off[0] = 0
                np.cumsum(padded[:-1], out=slab_off[1:])
                slab = _alloc_slab(int(slab_off[-1] + padded[-1]), device)
                if whole:
                    h = native.mgr_compress_launch(
                        tid, torch.from_numpy(ptrs),
                        torch.from_numpy(slab_off + slab.data_ptr()),
                        comp_dev, pin)
                    max_sizes, handles = piece_max, h.numpy()
                else:
                    max_sizes[lo:hi] = piece_max
                    handles[lo:hi] = native.mgr_compress_launch(
                        tid, torch.from_numpy(ptrs[lo:hi]),
                        torch.from_numpy(slab_off + slab.data_ptr()),
                        comp_dev[lo:hi], pin[lo:hi]).numpy()
                pieces.append(_Piece(key, tid, lo, hi, slab, slab_off))
        except BaseException:
            stream.synchronize()
            for tid in begun:
                try:
                    native.mgr_compress_finish(tid)
                except Exception:
                    pass
            raise
        self._compressed.record(stream)
        self._tick = _Tick(self._tick_no, pieces, pin, max_sizes, streams,
                           sizes, handles, insts)

    def _ship_raw(self, col: _Column, tensor: torch.Tensor,
                  counts: np.ndarray) -> int:
        """A RAW instance's tick: the output tensor is its region's bytes,
        each bucket cut into records at the column's cap."""
        tensor = tensor.contiguous()
        rb = col.row_bytes
        starts = _bucket_starts(tensor, counts)
        offsets, sizes, streams = _cut_records(starts, counts * rb, col.cap)
        nbytes = tensor.numel() * tensor.element_size()
        if self.on_gpu:
            return self._keep_on_gpu(col, tensor, nbytes, offsets, sizes,
                                     streams)
        self.writer.wait_for_room(self._tick_no)
        # ``ticks_left`` counts the tick a pack is packing, the one before
        # this; a RAW output is this tick's.
        left = None if self.ticks_left is None else max(self.ticks_left - 1, 1)
        region, home = col.carve(nbytes, self._cpu_pool, left, self.pool_tag)
        stored = int(sizes.sum())
        streams = streams.astype(np.int64)
        info = self.cached_outputs.get(col.base) if self.cache is not None \
            else None
        if info is not None and self._under_floor(nbytes):
            info = None
        if info is not None:
            # A RAW record decodes to its own bytes.
            region = self._cache_region(region, home, offsets, sizes,
                                        streams, info, sizes)
        col.regions.append(region)
        col.blocks.append((len(col.regions) - 1, offsets, sizes, sizes,
                           streams, None, None))
        if info is None:
            self._after.record(self._stream())
            self.writer.ship([tensor.data_ptr()], [home], [nbytes],
                             keep=[tensor], after=self._after,
                             tick=self._tick_no)
            if gates.EXPERIMENT and self.spills:
                movement.record("spill_bypass", 1, stored, nbytes)
            return stored
        # A cached output: the admitted prefix into its slots, then the
        # rest home, both on the writer's stream (the slot copies behind the
        # copies home of the victims whose slots they reuse, so the
        # producer's stream never waits on those); the writer holds the
        # tensor until they have run (until the copies home end, when there
        # are some).
        cr = region.release[1]
        adm = self.cache.admit([cr])
        P = min(cr.admitted * self.cache.G, nbytes)
        waits = list(adm.events)
        if P:
            k = np.arange(cr.admitted, dtype=np.int64)
            parts = np.minimum(self.cache.G, nbytes - k * self.cache.G)
            srcs = tensor.data_ptr() + k * self.cache.G
            slots = self._slot_ptrs(cr, k)

        def fill():
            ws = torch.cuda.current_stream()
            for e in waits:
                ws.wait_event(e)
            if P:
                abs_copy.launch(srcs, slots, parts)
        self._after.record(self._stream())
        rest = P < nbytes
        self.writer.ship([tensor.data_ptr() + P] if rest else [],
                         [home + P] if rest else [],
                         [nbytes - P] if rest else [],
                         keep=[tensor] if rest else [], after=self._after,
                         tick=self._tick_no, fill=fill,
                         fill_keep=[] if rest else [tensor])
        if rest:
            self.cache.note_home_copy([cr], self._writer_event())
        if P:
            self._clean(self._tick_no)
        if gates.EXPERIMENT:
            in_slot = np.clip(P - offsets, 0, sizes)
            if P:
                movement.record("cache_fill", cr.admitted,
                                int(in_slot.sum()), P)
            if P < nbytes:
                movement.record("spill_rejected", 1,
                                stored - int(in_slot.sum()), nbytes - P)
        return stored

    def _keep_on_gpu(self, col: _Column, tensor: torch.Tensor, nbytes: int,
                     offsets, sizes, streams) -> int:
        """The free-list baseline's RAW tick: the output tensor's bytes are
        the region, on the GPU, with its home carved and not written; the
        region is registered as spillable (its bytes are queued on this
        stream; an eviction synchronises the device before copying)."""
        left = None if self.ticks_left is None else max(self.ticks_left - 1, 1)
        carved, _ = col.carve(nbytes, self._cpu_pool, left, self.pool_tag)
        flat = tensor.reshape(-1).view(torch.uint8)
        if (tensor.storage_offset() != 0
                or tensor.untyped_storage().nbytes() != nbytes):
            # A view into a larger allocation: the region gets its own, so
            # an eviction frees what it moves.
            flat = flat.clone()
        region = Region(nbytes, DEVICE, storage=flat, home=carved.storage)
        col.regions.append(region)
        col.blocks.append((len(col.regions) - 1, offsets, sizes, sizes,
                           streams.astype(np.int64), None, None))
        device_regions.register(region)
        return int(sizes.sum())

    def _keep_packed_on_gpu(self, tick, sizes, dst, at, stream
                            ) -> Dict[int, int]:
        """The free-list baseline's compressed pack: each instance's packed
        records are its region, on the GPU in a buffer of its own (so an
        eviction frees what it moves), with its home carved and not written.
        One copy kernel per piece compacts the piece's records from its slab
        into those buffers, and each region is registered as spillable once
        its copies are queued (its logical bytes, for the ledger, are its
        records' raw bytes)."""
        native = _tqp()
        raw = tick.raw
        n = len(raw)
        stored: Dict[int, int] = {}
        device = tick.pieces[0].slab.device
        # Per record: its region buffer's address less the instance's first
        # packed offset, so a record lands at that plus its packed offset.
        base_of = np.empty(n, dtype=np.int64)
        kept = []
        for base, lo, hi, key in tick.insts:
            col = self._cols[base]
            assert base not in stored, (
                f"output column {base} split across codec groups")
            d0 = at[lo]
            nbytes = at[hi] - d0
            carved, _ = col.carve(nbytes, self._cpu_pool, self.ticks_left,
                                  self.pool_tag)
            buf = _alloc_slab(nbytes, device)
            region = Region(nbytes, DEVICE, storage=buf, home=carved.storage)
            col.regions.append(region)
            col.blocks.append((len(col.regions) - 1, dst[lo:hi] - d0,
                               sizes[lo:hi], raw[lo:hi], tick.streams[lo:hi],
                               key, tick.handles[lo:hi]))
            base_of[lo:hi] = buf.data_ptr() - d0
            kept.append((region, int(raw[lo:hi].sum())))
            stored[base] = nbytes
        host, pin, staged = abs_copy.staging(3 * n)
        o = 0
        for p in tick.pieces:
            lo, hi = p.lo, p.hi
            k = hi - lo
            np.add(p.slab_off, p.slab.data_ptr(), out=pin[o:o + k])
            np.add(dst[lo:hi], base_of[lo:hi], out=pin[o + k:o + 2 * k])
            pin[o + 2 * k:o + 3 * k] = sizes[lo:hi]
            offs = host[o:o + 3 * k]
            o += 3 * k
            native.batched_memcpy_abs(offs[0:k], offs[k:2 * k], offs[2 * k:])
            # Its free is ordered on this stream, behind the copies.
            p.slab = None
        staged.record(stream)
        for region, logical_b in kept:
            device_regions.register(region, logical_b)
        return stored

    # -- the granule cache -----------------------------------------------

    def _under_floor(self, nbytes: int) -> bool:
        """Whether a cacheable region of ``nbytes`` is too small to cache
        (``IO.cache.config.FLOOR_GRANULES``): it then goes home as a plain
        region, the reads of it an H2D copy of that few bytes."""
        floor = _cache_cfg.FLOOR_GRANULES
        return bool(floor) and nbytes < floor * self.cache.G

    def _cache_region(self, home_region: Region, home: int, offsets, sizes,
                      streams, info, raw) -> Region:
        """The cached form of a region whose home ``home_region`` was just
        carved: registered with the cache (every granule evicted until the
        admission); ``raw``, its records' decoded sizes, goes to the cache
        under experiment telemetry, for the ledger's decoded bytes."""
        next_use_op, expected_reads = info
        region = Region(home_region.nbytes, CACHED,
                        storage=home_region.storage)
        cr = self.cache.register(home_region.nbytes, home, offsets, sizes,
                                 streams, next_use_op, expected_reads,
                                 raw=raw if gates.EXPERIMENT else None)
        region.release = (self.cache, cr)
        return region

    def _slot_ptrs(self, cr, k) -> np.ndarray:
        """Device addresses of the slots granules ``k`` of ``cr`` sit in."""
        return self.cache.slot_addrs(cr.first + np.asarray(k, dtype=np.int64))

    def _writer_event(self):
        """An event behind every copy issued so far on the writer's stream:
        reads of a home wait on it."""
        event = torch.cuda.Event()
        event.record(writer_stream())
        return event

    def _clean(self, label) -> None:
        """After a tick's admissions: copy the dirty granules at the front
        of the victim order home (``NativeGranuleCache.cleaning_targets``),
        behind the tick's slot fills.  Only with ``IO.cache.config.CLEANING``
        (off by default): without it a dirty granule goes home when it is
        evicted."""
        if not _cache_cfg.CLEANING:
            return
        targets = self.cache.cleaning_targets()
        if not len(targets):
            return
        self._filled.record(self._stream())
        self.writer.ship(targets.src.tolist(), targets.dst.tolist(),
                         targets.size.tolist(), keep=[], after=self._filled,
                         tick=label)
        self.cache.commit_cleaning(targets, self._writer_event())
        if gates.EXPERIMENT:
            movement.record("spill_mirror", len(targets),
                            *self.cache.ledger_bytes(targets.gids))

    # -- tick t+1 --------------------------------------------------------

    def pack(self) -> Dict[int, int]:
        """Tick t+1, on the producer's stream (the compute stream in a
        pipe), before the operator: pack the previous tick's records into
        one region per instance and ship them (module docstring).  Returns
        each instance's stored bytes, by base."""
        if self._tick is None:
            return {}
        return self._pack(self._stream())

    def _admit(self, crs) -> np.ndarray:
        """Offer a tick's cached regions (``crs``: per instance, its cached
        region or None) to the cache; the tick's slot fill, on the writer's
        stream, waits on the reused slots' copies (``_fill_waits``), so the
        producer's stream does not.  Returns each instance's bytes in slots
        (the admitted granules, a prefix)."""
        P = np.zeros(len(crs), dtype=np.int64)
        cached = [cr for cr in crs if cr is not None]
        if cached:
            adm = self.cache.admit(cached)
            self._fill_waits = list(adm.events)
            for j, cr in enumerate(crs):
                if cr is not None:
                    P[j] = min(cr.admitted * self.cache.G, cr.nbytes)
        return P

    def _issue_copies(self, tick, sizes, dst, edges, homes, crs,
                      P=None) -> None:
        """A tick's copies once its regions are carved: ``dst`` the packed
        offsets (``n + 1``), ``edges`` the instances' first records and
        the end, ``homes`` each instance's home address of tick offset 0,
        ``crs`` its cached region or None, ``P`` the bytes admitted to
        slots per instance (``_admit``; None: admitted here).  Writes each
        piece's slot parts and dense buffer by one kernel, ships the dense
        parts home, cleans, and counts the ledger."""
        raw = tick.raw
        n = len(sizes)
        m = len(edges) - 1
        align = constants.COMPRESSION_ALIGNMENT
        starts = np.asarray(edges[:-1], dtype=np.int64)
        d0s = dst[starts]
        ends = dst[np.asarray(edges[1:], dtype=np.int64)]
        # Bytes of each instance in slots: the admitted granules (a prefix).
        if P is None:
            P = self._admit(crs)
        stream = self._stream()
        cached = [cr for cr in crs if cr is not None]
        inst = np.repeat(np.arange(m, dtype=np.int64), np.diff(edges))
        roff = dst[:n] - d0s[inst]
        pre = np.clip(P[inst] - roff, 0, sizes)      # a record's bytes in slots

        # Per piece: its records' parts in slots (absolute addresses), and
        # its dense buffer, the piece's span of the tick with the slot
        # prefixes taken out, which keeps every part 512-aligned.
        plans = []
        total = 0
        G = self.cache.G if self.cache is not None else 1
        for p in tick.pieces:
            span0, span1 = int(dst[p.lo]), int(dst[p.hi])
            j0, j1 = int(inst[p.lo]), int(inst[p.hi - 1]) + 1
            in_piece = np.clip(np.minimum(d0s[j0:j1] + P[j0:j1], span1)
                               - np.maximum(d0s[j0:j1], span0), 0, None)
            removed = np.cumsum(in_piece)        # through each instance
            dense_b = (span1 - span0) - int(removed[-1])
            r = np.arange(p.lo, p.hi, dtype=np.int64)
            slab_src = p.slab.data_ptr() + p.slab_off
            # Slot parts, cut at granule boundaries.
            ri, k, g_lo, g_n = _split_at_granules(roff[r], pre[r], G)
            if len(ri):
                gids = np.array([crs[j].first for j in inst[r[ri]].tolist()],
                                dtype=np.int64) + k
                slot_dst = self.cache.slot_addrs(gids) + (g_lo - k * G)
                slot_src = slab_src[ri] + (g_lo - roff[r[ri]])
            else:
                slot_dst = slot_src = g_n
            # Dense parts: bytes [pre, size) of each record.
            di = np.flatnonzero(sizes[r] > pre[r])
            x0 = dst[r[di]] + pre[r[di]]
            d_rel = x0 - span0 - removed[inst[r[di]] - j0]
            d_src = slab_src[di] + pre[r[di]]
            d_n = sizes[r[di]] - pre[r[di]]
            # Home copies, one per instance with a refused part here.
            jj = np.arange(j0, j1, dtype=np.int64)
            h_x0 = np.maximum(d0s[jj] + P[jj], span0)
            h_x1 = np.minimum(ends[jj], span1)
            h = np.flatnonzero(h_x1 > h_x0)
            plans.append((p, dense_b, slot_src, slot_dst, g_n, d_rel, d_src,
                          d_n, homes[inst[r[di]]] + x0, jj[h],
                          h_x0[h] - span0 - removed[h], h_x0[h],
                          h_x1[h] - h_x0[h]))
            total += 3 * (len(g_n) + len(d_n))
        host, pin, staged = abs_copy.staging(total)
        from TQPlib.tqpcompress import batched_memcpy_abs
        copy_src, copy_dst, copy_b, keep = [], [], [], []
        launches, fill_keep = [], []
        at = 0
        for (p, dense_b, slot_src, slot_dst, g_n, d_rel, d_src, d_n, d_home,
             h_j, h_rel, h_x0, h_n) in plans:
            dense = None
            if len(d_n):
                dense = self._alloc_dense(dense_b, p.slab.device)
            if dense is not None:
                srcs = np.concatenate([slot_src, d_src])
                dsts = np.concatenate([slot_dst, dense.data_ptr() + d_rel])
                nbs = np.concatenate([g_n, d_n])
                copy_src += (dense.data_ptr() + h_rel).tolist()
                copy_dst += (homes[h_j] + h_x0).tolist()
                copy_b += h_n.tolist()
                keep.append(dense)
            else:
                # Only the slot parts go through the kernel; the refused
                # parts, if any (their dense buffer did not fit), are copied
                # from the slab straight to their homes.
                srcs, dsts, nbs = slot_src, slot_dst, g_n
                if len(d_n):
                    copy_src += d_src.tolist()
                    copy_dst += d_home.tolist()
                    copy_b += d_n.tolist()
                    keep.append(p.slab)
            k = len(nbs)
            if k:
                pin[at:at + k] = srcs
                pin[at + k:at + 2 * k] = dsts
                pin[at + 2 * k:at + 3 * k] = nbs
                offs = host[at:at + 3 * k]
                at += 3 * k
                launches.append((offs, k))
            # The kernel reads the slab on the writer's stream: held until
            # the fill has run (until the copies end when they read it too).
            if dense is not None or not len(d_n):
                fill_keep.append(p.slab)
            p.slab = None
        # The kernels on the writer's stream, behind the victims' copies
        # home, then the dense parts home.
        waits, self._fill_waits = self._fill_waits, []

        def fill():
            ws = torch.cuda.current_stream()
            for e in waits:
                ws.wait_event(e)
            for offs, k in launches:
                batched_memcpy_abs(offs[0:k], offs[k:2 * k], offs[2 * k:])
            staged.record(ws)
        self._after.record(stream)
        self.writer.ship(copy_src, copy_dst, copy_b, keep=keep,
                         after=self._after, tick=tick.label, fill=fill,
                         fill_keep=fill_keep)
        if cached:
            refused = [cr for j, cr in enumerate(crs)
                       if cr is not None and P[j] < cr.nbytes]
            if refused:
                self.cache.note_home_copy(refused, self._writer_event())
            if P.any():
                self._clean(tick.label)
        if gates.EXPERIMENT and self.spills:
            is_cached = np.array([cr is not None for cr in crs], dtype=bool)
            region_b = ends - d0s
            raw_inst = np.add.reduceat(raw, starts) if n else raw
            slot_raw = np.add.reduceat(
                raw * pre // np.maximum(sizes, 1), starts) if n else raw
            plain = ~is_cached
            if plain.any():
                movement.record("spill_bypass", int(plain.sum()),
                                int(raw_inst[plain].sum()),
                                int(region_b[plain].sum()))
            if is_cached.any():
                if P.any():
                    movement.record(
                        "cache_fill",
                        int(sum(cr.admitted for cr in crs if cr is not None)),
                        int(slot_raw[is_cached].sum()), int(P.sum()))
                rej = is_cached & (P < region_b)
                if rej.any():
                    movement.record(
                        "spill_rejected", int(rej.sum()),
                        int((raw_inst - slot_raw)[rej].sum()),
                        int((region_b - P)[rej].sum()))

    def _alloc_dense(self, dense_b: int, device):
        """A piece's dense buffer, or ``None`` when the arena cannot serve
        it (lending included)."""
        try:
            return _alloc_slab(dense_b, device)
        except RuntimeError as exc:
            if not _out_of_memory(exc) or dense_b == 0:
                raise
            return None

    def _pack(self, stream) -> Dict[int, int]:
        """``pack`` on the producer's stream.  A tick with no cached
        instance packs each piece by array arithmetic on the piece's
        records; one whose every instance is cached and admitted whole fills
        its slots by one kernel (``_fill_slots``); any other takes
        ``_issue_copies``."""
        tick = self._tick
        native = _tqp()
        align = constants.COMPRESSION_ALIGNMENT
        self.writer.wait_for_room(tick.label)
        self._tick = None
        if not self._compressed.query():
            self._compressed.synchronize()
        raw = tick.raw
        n = len(raw)
        sizes = self._size_np[:n].copy()
        if len(tick.pieces) == 1:
            statuses = native.mgr_compress_finish(tick.pieces[0].tid).numpy()
        else:
            statuses = np.concatenate(
                [native.mgr_compress_finish(p.tid).numpy()
                 for p in tick.pieces])
        if (statuses.any() or int(sizes.min()) <= 0
                or (sizes > tick.max_sizes).any()):
            bad = (statuses != 0) | (sizes <= 0) | (sizes > tick.max_sizes)
            idx = np.flatnonzero(bad)[:8]
            raise RuntimeError(
                f"region producer compress failed at records {idx.tolist()}: "
                f"statuses={statuses[idx].tolist()} S={sizes[idx].tolist()} "
                f"M={tick.max_sizes[idx].tolist()}")
        padded = sizes + (align - 1)
        padded &= -align
        dst = np.empty(n + 1, dtype=np.int64)
        dst[0] = 0
        np.cumsum(padded, out=dst[1:])
        at = dst.tolist()
        if self.on_gpu:
            return self._keep_packed_on_gpu(tick, sizes, dst, at, stream)
        stored: Dict[int, int] = {}
        cache = self.cache
        insts = tick.insts
        homes = []                 # home address of tick offset 0
        crs = None
        for j, (base, lo, hi, key) in enumerate(insts):
            col = self._cols[base]
            assert base not in stored, (
                f"output column {base} split across codec groups")
            d0 = at[lo]
            nbytes = at[hi] - d0
            region, home = col.carve(nbytes, self._cpu_pool, self.ticks_left,
                                     self.pool_tag)
            offs = dst[lo:hi] - d0
            if cache is not None:
                info = self.cached_outputs.get(base)
                if info is not None and self._under_floor(nbytes):
                    info = None
                if info is not None:
                    region = self._cache_region(
                        region, home, offs, sizes[lo:hi],
                        tick.streams[lo:hi], info, raw[lo:hi])
                    if crs is None:
                        crs = [None] * len(insts)
                    crs[j] = region.release[1]
            col.regions.append(region)
            col.blocks.append((len(col.regions) - 1, offs, sizes[lo:hi],
                               raw[lo:hi], tick.streams[lo:hi], key,
                               tick.handles[lo:hi]))
            homes.append(home - d0)
            stored[base] = nbytes
        if crs is not None:
            edges = [i[1] for i in insts] + [n]
            P = self._admit(crs)
            if all(cr is not None and P[j] == cr.nbytes
                   for j, cr in enumerate(crs)):
                self._fill_slots(tick, sizes, dst, edges, crs, stream)
            else:
                self._issue_copies(tick, sizes, dst, edges,
                                   np.array(homes, dtype=np.int64), crs, P)
            return stored
        host, pin, staged = abs_copy.staging(3 * n)
        copy_src, copy_dst, copy_b, keep = [], [], [], []
        o = 0
        for p in tick.pieces:
            lo, hi = p.lo, p.hi
            k = hi - lo
            span0, span1 = at[lo], at[hi]
            dense = self._alloc_dense(span1 - span0, p.slab.device)
            slab_ptr = p.slab.data_ptr()
            if dense is None:
                # The dense buffer did not fit: each record goes from the
                # slab straight to its home.
                for j, (_, ilo, ihi, _) in enumerate(insts):
                    if ihi <= lo or ilo >= hi:
                        continue
                    a, b = max(ilo, lo), min(ihi, hi)
                    copy_src += (p.slab_off[a - lo:b - lo]
                                 + slab_ptr).tolist()
                    copy_dst += (dst[a:b] + homes[j]).tolist()
                    copy_b += sizes[a:b].tolist()
                keep.append(p.slab)
                p.slab = None
                continue
            dptr = dense.data_ptr()
            np.add(p.slab_off, slab_ptr, out=pin[o:o + k])
            np.add(dst[lo:hi], dptr - span0, out=pin[o + k:o + 2 * k])
            pin[o + 2 * k:o + 3 * k] = sizes[lo:hi]
            offs = host[o:o + 3 * k]
            o += 3 * k
            native.batched_memcpy_abs(offs[0:k], offs[k:2 * k], offs[2 * k:])
            for j, (_, ilo, ihi, _) in enumerate(insts):
                x0, x1 = max(at[ilo], span0), min(at[ihi], span1)
                if x1 > x0:
                    copy_src.append(dptr + x0 - span0)
                    copy_dst.append(homes[j] + x0)
                    copy_b.append(x1 - x0)
            keep.append(dense)
            p.slab = None
        staged.record(stream)
        self._after.record(stream)
        if copy_b:
            self.writer.ship(copy_src, copy_dst, copy_b, keep=keep,
                             after=self._after, tick=tick.label)
        if gates.EXPERIMENT and self.spills:
            movement.record("spill_bypass", len(insts), int(raw.sum()),
                            at[n])
        return stored

    def _fill_slots(self, tick, sizes, dst, edges, crs, stream) -> None:
        """``_issue_copies`` for a tick whose every instance is cached and
        admitted whole: every record's bytes go to its granules' slots, by
        one kernel for the tick, and nothing goes home."""
        native = _tqp()
        cache = self.cache
        G = cache.G
        n = len(sizes)
        m = len(crs)
        counts = np.diff(np.asarray(edges, dtype=np.int64))
        inst = np.repeat(np.arange(m, dtype=np.int64), counts)
        d0s = dst[np.asarray(edges[:-1], dtype=np.int64)]
        roff = dst[:n] - d0s[inst]
        src = np.empty(n, dtype=np.int64)
        for p in tick.pieces:
            np.add(p.slab_off, p.slab.data_ptr(), out=src[p.lo:p.hi])
        ri, k, g_lo, g_n = _split_at_granules(roff, sizes, G)
        firsts = np.fromiter((cr.first for cr in crs), dtype=np.int64,
                             count=m)
        gids = firsts[inst[ri]] + k
        kn = len(g_n)
        if not kn:
            for p in tick.pieces:
                p.slab = None
            return
        host, pin, staged = abs_copy.staging(3 * kn)
        np.add(src[ri], g_lo - roff[ri], out=pin[:kn])
        np.add(cache.slot_addrs(gids), g_lo - k * G, out=pin[kn:2 * kn])
        pin[2 * kn:3 * kn] = g_n
        offs = host[:3 * kn]
        # On the writer's stream, behind the victims' copies home; the
        # writer holds the slabs until the fill has run.
        slabs = [p.slab for p in tick.pieces]
        for p in tick.pieces:
            p.slab = None
        waits, self._fill_waits = self._fill_waits, []

        def fill():
            ws = torch.cuda.current_stream()
            for e in waits:
                ws.wait_event(e)
            native.batched_memcpy_abs(offs[0:kn], offs[kn:2 * kn],
                                      offs[2 * kn:])
            staged.record(ws)
        self._after.record(stream)
        self.writer.ship([], [], [], keep=[], after=self._after,
                         tick=tick.label, fill=fill, fill_keep=slabs)
        self._clean(tick.label)
        if gates.EXPERIMENT:
            movement.record("cache_fill", int(sum(cr.admitted for cr in crs)),
                            int(tick.raw.sum()),
                            int(sum(cr.nbytes for cr in crs)))

    # -- the lane and the end of the pipe --------------------------------

    def wait_copies(self) -> None:
        """Wait for every copy home in flight and drop the buffers it read,
        so none of them is held through the caller's next allocation."""
        self.writer.wait_all()

    def collect_lane(self) -> Tuple[float, int]:
        """The span (ms) and bytes of the home writer's copies that ended
        since the last call."""
        return self.writer.take_lane()

    def drain(self) -> Dict[int, int]:
        """End of the pipe: pack the last tick's records, wait for every
        copy still in flight, give each column's unused chunk blocks back,
        and freeze the tables.  Returns the stored bytes packed here, by
        base."""
        self.ticks_left = 1
        stored = self.pack()
        # Every shipment has ended, the slot fills with them (they run on
        # the writer's stream): later pipes read the slots on other streams.
        self.writer.drain()
        self._size_pin = None
        self._size_np = self._comp_dev = None
        self._tables = {}
        for base, col in self._cols.items():
            col.close(self._cpu_pool)
            if col.regions:
                self._tables[base] = col.freeze()
        return stored

    def abort(self) -> None:
        """The pipe failed: wait for the copies in flight (so nothing they
        read or write is freed under them), drop the unpacked tick, and
        give the columns' chunks back to the pool."""
        self.writer.abort()
        if self._tick is not None:
            if not self._compressed.query():
                self._compressed.synchronize()
            for p in self._tick.pieces:
                try:
                    from TQPlib.tqpcompress import mgr_compress_finish
                    mgr_compress_finish(p.tid)
                except Exception:
                    pass
            self._tick = None
        self._comp_dev = None
        if self.cache is not None:
            # No work of a failed pack outlives the regions' drop (its slot
            # fills ran on the writer's stream, which ``writer.abort``
            # waited for).
            self._stream().synchronize()
        for col in self._cols.values():
            col.close(self._cpu_pool)
            for region in col.regions:
                if region.spillable:
                    # Its homes go back below; drop it from the GPU and
                    # from the eviction list.
                    region.free(self._cpu_pool, None)
                elif region.location == CACHED:
                    cr = region.release[1]
                    self.cache.drop(cr)
                    # A reclaim may still be copying its granules home.
                    for event in self.cache.home_events(cr):
                        event.synchronize()
            for ptr in col.home_ptrs:
                self._cpu_pool.free(ptr)
            col.home_ptrs = []
            col.regions = []
            col.blocks = []

    def tables(self) -> Dict[int, RecordTable]:
        """The frozen tables, by output base slot (instances that wrote no
        row have none); valid after ``drain``."""
        assert self._tables is not None, "tables are frozen by drain()"
        return self._tables
