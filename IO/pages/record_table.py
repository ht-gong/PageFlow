"""Record tables: the record index of one whole representation.

A ``RecordTable`` lists every record of one representation of a column
instance as numpy ``int64`` columns, one row per record:

  region     index into ``regions``: the region holding the record
  offset     bytes from the region's start (512-aligned for compressed
             records)
  nbytes     stored bytes (compressed size, or the raw size of a RAW record)
  raw_bytes  decoded bytes
  row_start  first row of the record within its stream
  rows       rows in the record
  codec      codec id (``compression_algorithms._ALGO_NAMES``; 0 = RAW)
  handle     nvCOMP decompression config handle (0 = none; always 0 for RAW)

Records are ordered by stream, then by rows. ``stream_ptr`` holds the CSR
offsets: stream ``s`` is records ``stream_ptr[s]:stream_ptr[s + 1]``. A
column has one stream per partition (B for a partitioned output: the join's
partition sort's or a partitioned aggregation's pass 1's; one otherwise).
The table also carries the constants every record shares: the stored dtype,
the width and the logical type, plus one manager codec key per codec id
present.

Rows are numbered two ways. ``row_start`` counts within a stream. The
table's row space concatenates the streams in order (stream 0's rows, then
stream 1's, ...), which is the order a partition-major column has today, and
every selection takes rows in that space. Since records are stream-major, a
contiguous range of rows is always a contiguous range of records.

A producer appends one block of records per step to a
``RecordTableBuilder`` (records of any streams, in any order) and freezes it
when its pipe ends: a stable sort by stream gives the CSR layout, and row
starts become per-stream prefix sums. Readers only see frozen tables.

Reading: a ``Selection`` is the records a reader needs plus the rows to trim
inside the first and the last of them. ``RecordTable.runs`` groups records
into runs, one contiguous byte range per run within one region (records with
no unselected record between them in the region's byte order), and lays the
runs out in one staging buffer, each run keeping its source alignment.

Byte order and row order. Runs are laid out in (region, byte) order, not
row order, so a reader that takes the staging buffer as the decoded rows
(RAW records copied straight to their output place) gets row order only if
every region stores each stream's records in row order and its streams in
stream order. The region producer's regions do, and so do the scan's.

Freeing: ``free`` returns every region's memory, then the pinned allocations
the regions were carved from. A slice of a column is a selection over the
column's table and frees nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import constants
from IO.pages.region import Region

_I64 = np.int64
_COLUMNS = ("region", "offset", "nbytes", "raw_bytes", "row_start", "rows",
            "codec", "handle")


def _as_i64(values, n: Optional[int] = None) -> np.ndarray:
    """``values`` as a 1-D int64 array, a scalar broadcast to length ``n``."""
    arr = np.asarray(values, dtype=_I64)
    if arr.ndim == 0:
        assert n is not None, "a scalar needs a length to broadcast to"
        return np.full(n, int(arr), dtype=_I64)
    assert arr.ndim == 1, f"expected a 1-D array, got shape {arr.shape}"
    return arr


def _exclusive_cumsum(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values) + 1, dtype=_I64)
    np.cumsum(values, out=out[1:])
    return out


class RecordTableBuilder:
    """Collects records block by block until ``freeze``."""

    def __init__(self, stored_dtype: torch.dtype, width: int,
                 logical_type: Optional[str] = None, num_streams: int = 1):
        assert num_streams >= 1, f"a table needs a stream, got {num_streams}"
        self.stored_dtype = stored_dtype
        self.width = int(width)
        self.logical_type = logical_type
        self.num_streams = int(num_streams)
        self.row_bytes = self.width * torch.empty(
            0, dtype=stored_dtype).element_size()
        self.regions: List[Region] = []
        self.codec_keys: Dict[int, Tuple[int, int, int]] = {}
        self._blocks: List[Tuple[np.ndarray, ...]] = []
        self._stream_blocks: List[np.ndarray] = []
        self._frozen = False

    def add_region(self, region: Region) -> int:
        """Register a region; returns its index for ``append``."""
        assert not self._frozen, "builder already frozen"
        self.regions.append(region)
        return len(self.regions) - 1

    def append(self, region, offset, nbytes, raw_bytes, stream=0,
               codec_key: Optional[Tuple[int, int, int]] = None,
               handles=None) -> None:
        """Append one block of records, all in region ``region``, or each in
        its own region when ``region`` is one index per record.

        ``offset``, ``nbytes`` and ``raw_bytes`` are per-record arrays (or
        scalars for one record); ``stream`` is a scalar or one value per
        record. ``codec_key`` is the manager key ``(codec id, element size,
        chunk bytes)`` of every record in the block, or ``None`` for RAW
        records. ``handles`` are the records' decompression config handles
        (``None``: no handle).
        """
        assert not self._frozen, "builder already frozen"
        offset = _as_i64(offset, 1)
        n = len(offset)
        if np.ndim(region) == 0:
            assert 0 <= region < len(self.regions), f"unknown region {region}"
            capacity = self.regions[region].nbytes
        else:
            region = _as_i64(region)
            assert len(region) == n, "one region per record"
            assert ((region >= 0) & (region < len(self.regions))).all(), (
                "unknown region")
            capacity = np.array([r.nbytes for r in self.regions],
                                dtype=_I64)[region]
        nbytes = _as_i64(nbytes, n)
        raw_bytes = _as_i64(raw_bytes, n)
        stream = _as_i64(stream, n)
        assert len(nbytes) == n and len(raw_bytes) == n and len(stream) == n, (
            "record block columns differ in length")
        if n == 0:
            return
        assert (raw_bytes > 0).all(), "a record holds at least one row"
        assert (raw_bytes % self.row_bytes == 0).all(), (
            f"record raw sizes must be whole rows of {self.row_bytes} B")
        rows = raw_bytes // self.row_bytes
        assert (stream >= 0).all() and (stream < self.num_streams).all(), (
            f"stream outside [0, {self.num_streams})")
        assert (offset >= 0).all() and (nbytes > 0).all()
        assert (offset + nbytes <= capacity).all(), (
            f"records overrun their region ({capacity} B)")
        if codec_key is None:
            codec = np.zeros(n, dtype=_I64)
            assert handles is None, "RAW records take no handles"
            handle = np.zeros(n, dtype=_I64)
        else:
            codec_key = tuple(int(v) for v in codec_key)
            cid = codec_key[0]
            assert cid > 0, "a compressed record needs a codec id above 0"
            known = self.codec_keys.setdefault(cid, codec_key)
            assert known == codec_key, (
                f"codec {cid} appended with key {codec_key}, table has {known}")
            align = constants.COMPRESSION_ALIGNMENT
            assert (offset % align == 0).all(), (
                f"compressed records start {align}-aligned in their region")
            codec = np.full(n, cid, dtype=_I64)
            handle = (np.zeros(n, dtype=_I64) if handles is None
                      else _as_i64(handles, n).copy())
            assert len(handle) == n, "one handle per record"
        self._blocks.append((_as_i64(region, n).copy(), offset.copy(),
                             nbytes.copy(), raw_bytes.copy(), rows, codec,
                             handle))
        self._stream_blocks.append(stream.copy())

    def freeze(self) -> "RecordTable":
        """Sort by stream, compute row starts and region references."""
        assert not self._frozen, "builder already frozen"
        self._frozen = True
        if self._blocks:
            cols = [np.concatenate(c) for c in zip(*self._blocks)]
            stream = np.concatenate(self._stream_blocks)
        else:
            cols = [np.zeros(0, dtype=_I64) for _ in range(7)]
            stream = np.zeros(0, dtype=_I64)
        order = np.argsort(stream, kind="stable")
        region, offset, nbytes, raw_bytes, rows, codec, handle = (
            c[order] for c in cols)
        stream = stream[order]
        stream_ptr = np.searchsorted(
            stream, np.arange(self.num_streams + 1, dtype=_I64),
            side="left").astype(_I64)
        global_start = _exclusive_cumsum(rows)
        row_start = global_start[:-1] - global_start[stream_ptr[stream]]
        return RecordTable(
            region=region, offset=offset, nbytes=nbytes, raw_bytes=raw_bytes,
            row_start=row_start, rows=rows, codec=codec, handle=handle,
            stream_ptr=stream_ptr, regions=self.regions,
            stored_dtype=self.stored_dtype, width=self.width,
            logical_type=self.logical_type, codec_keys=dict(self.codec_keys))


class RecordTable:
    """Frozen record index of one representation (module docstring)."""

    def __init__(self, *, region, offset, nbytes, raw_bytes, row_start, rows,
                 codec, handle, stream_ptr, regions, stored_dtype, width,
                 logical_type, codec_keys):
        self.region = region
        self.offset = offset
        self.nbytes = nbytes
        self.raw_bytes = raw_bytes
        self.row_start = row_start
        self.rows = rows
        self.codec = codec
        self.handle = handle
        self.stream_ptr = stream_ptr
        self.regions: List[Region] = regions
        self.stored_dtype = stored_dtype
        self.width = width
        self.logical_type = logical_type
        self.codec_keys: Dict[int, Tuple[int, int, int]] = codec_keys
        self.row_bytes = width * torch.empty(
            0, dtype=stored_dtype).element_size()

        n = len(rows)
        for name in _COLUMNS:
            col = getattr(self, name)
            assert col.dtype == _I64 and col.shape == (n,), name
        # Row space: the streams concatenated in order.
        self._global_start = _exclusive_cumsum(rows)
        self._row_end = self._global_start[1:]
        self._stream_row_ptr = self._global_start[stream_ptr]
        self._stream = np.repeat(np.arange(self.num_streams, dtype=_I64),
                                 np.diff(stream_ptr))
        # Byte order inside each region: records sorted by (region, offset).
        # ``_rank`` is a record's position in that order, so two records are
        # adjacent in a region exactly when their ranks differ by one.
        byte_order = np.lexsort((offset, region)).astype(_I64)
        self._rank = np.empty(n, dtype=_I64)
        self._rank[byte_order] = np.arange(n, dtype=_I64)
        ends = offset[byte_order] + nbytes[byte_order]
        same = region[byte_order][1:] == region[byte_order][:-1]
        assert (offset[byte_order][1:][same] >= ends[:-1][same]).all(), (
            "records overlap inside a region")
        # Pinned-pool allocations the table's regions are carved from (their
        # own ``release`` is then ``None``): returned by ``free`` once every
        # region is freed.  A producer that carves regions sets it.
        self.host_blocks: List[int] = []
        # True for a base table's representation (the scan's), whose reads
        # the movement ledger files as scans rather than restores.
        self.base = False
        # The zero-copy read's segments (``uva_segments``).
        self._uva = None

    # ------------------------------------------------------------------
    # Shape
    # ------------------------------------------------------------------

    @property
    def num_records(self) -> int:
        return len(self.rows)

    @property
    def num_streams(self) -> int:
        return len(self.stream_ptr) - 1

    @property
    def num_rows(self) -> int:
        return int(self._global_start[-1])

    def stream_rows(self) -> np.ndarray:
        """Rows per stream."""
        return np.diff(self._stream_row_ptr)

    def global_row_start(self) -> np.ndarray:
        """Each record's first row in the table's row space."""
        return self._global_start[:-1]

    def record_stream(self) -> np.ndarray:
        """Each record's stream."""
        return self._stream

    def stored_bytes(self) -> int:
        return int(self.nbytes.sum())

    @property
    def is_compressed(self) -> bool:
        return bool((self.codec != 0).any())

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_rows(self, start: int, stop: int) -> "Selection":
        """The records covering rows ``[start, stop)`` of the row space,
        with the rows to trim inside the first and last record."""
        start, stop = int(start), int(stop)
        assert 0 <= start <= stop <= self.num_rows, (
            f"rows [{start}, {stop}) outside [0, {self.num_rows})")
        if start == stop:
            return Selection(self, np.zeros(0, dtype=_I64), 0, 0)
        first = int(np.searchsorted(self._row_end, start, side="right"))
        last = int(np.searchsorted(self._row_end, stop, side="left"))
        records = np.arange(first, last + 1, dtype=_I64)
        return Selection(self, records, start - int(self._global_start[first]),
                         stop - start)

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def runs(self, records: np.ndarray,
             align: int = constants.COMPRESSION_ALIGNMENT) -> "Runs":
        """Group ``records`` into runs and lay the runs out back to back in
        one staging buffer.

        A run starts at the same offset modulo ``align`` in the buffer as in
        its region, after an ``align``-aligned boundary, so every record
        keeps its source alignment (a compressed record after an unaligned
        RAW record in the same run stays 512-aligned). Runs follow (region,
        byte) order, not row order: ``record_dst`` gives each record's
        place. Their bytes come out in row order only where each region holds
        its records in row order, as the scan's and the producer's do.
        """
        records = np.asarray(records, dtype=_I64)
        n = len(records)
        assert align >= 1
        if n == 0:
            empty = np.zeros(0, dtype=_I64)
            return Runs(empty, empty, empty, empty, 0, empty, empty)
        rank = self._rank[records]
        order = np.argsort(rank, kind="stable")
        recs = records[order]
        rank = rank[order]
        reg = self.region[recs]
        starts_new = np.ones(n, dtype=bool)
        starts_new[1:] = (np.diff(rank) != 1) | (np.diff(reg) != 0)
        first = np.flatnonzero(starts_new)
        last = np.r_[first[1:], n] - 1
        run_of = np.cumsum(starts_new) - 1
        run_region = reg[first]
        run_src = self.offset[recs[first]]
        run_bytes = self.offset[recs[last]] + self.nbytes[recs[last]] - run_src
        phase = run_src % align
        padded = (phase + run_bytes + align - 1) // align * align
        cursor = _exclusive_cumsum(padded)
        dst = cursor[:-1] + phase
        rec_dst = np.empty(n, dtype=_I64)
        rec_run = np.empty(n, dtype=_I64)
        rec_dst[order] = dst[run_of] + (self.offset[recs] - run_src[run_of])
        rec_run[order] = run_of
        return Runs(run_region, run_src, run_bytes, dst, int(cursor[-1]),
                    rec_dst, rec_run)

    # ------------------------------------------------------------------
    # Zero-copy reads
    # ------------------------------------------------------------------

    def uva_segments(self) -> np.ndarray:
        """What a zero-copy (UVA) read of the table's rows takes
        (``IO/zero_copy_reader.cu``): its RAW records as *segments*, each a
        stretch of records that follow each other both in the row space and
        in one region's bytes.  Returns one int64 array: the ``n``
        segments' device addresses (each region's ``device_ptr``, its own on
        the GPU or its device alias in host memory, plus the offset), then their ``n + 1`` row starts in the row space (the last
        is the end).  Built on the first read and kept with the table until
        an eviction moves a region (``IO/pages/device_regions.py``).  The
        reader copies it to the GPU per read."""
        from IO.pages import device_regions
        epoch = device_regions.epoch()
        if self._uva is None or self._uva[0] != epoch:
            self._uva = (epoch, self.record_segments(0, self.num_records))
        return self._uva[1]

    def record_segments(self, first: int, stop: int) -> np.ndarray:
        """``uva_segments`` of records ``[first, stop)`` alone, their rows
        counted from the first record's first row: a zero-copy read of
        whole records needs no row offset.  Built on every call."""
        n = stop - first
        assert n > 0, "a zero-copy read needs a record"
        sl = slice(first, stop)
        assert not self.codec[sl].any(), "a zero-copy read needs RAW records"
        # A segment's rows are read at whole-row steps, so a record joins
        # the one before only where that one's rows end.
        region, offset = self.region[sl], self.offset[sl]
        raw = self.raw_bytes[sl]
        new = np.ones(n, dtype=bool)
        new[1:] = ((region[1:] != region[:-1])
                   | (offset[1:] != offset[:-1] + raw[:-1]))
        head = np.flatnonzero(new)
        regs = region[head]
        bases = np.array([self.regions[k].device_ptr for k in regs.tolist()],
                         dtype=_I64)
        row0 = self._global_start[first]
        return np.concatenate([
            bases + offset[head],
            self._global_start[first + head] - row0,
            [self._global_start[stop] - row0]]).astype(_I64)

    # ------------------------------------------------------------------
    # Free
    # ------------------------------------------------------------------

    def free(self, cpu_pool, gpu_pool) -> None:
        """Return every region's memory, then the pinned allocations the
        regions were carved from (``host_blocks``). A second call does
        nothing."""
        for reg in self.regions:
            reg.free(cpu_pool, gpu_pool)
        for ptr in self.host_blocks:
            cpu_pool.free(ptr)
        self.host_blocks = []
        self._uva = None


@dataclass
class Runs:
    """Runs of a set of records and their layout in a staging buffer.

    Per run, in (region, byte) order: ``region``, ``src_offset`` (bytes from
    the region's start), ``nbytes`` and ``dst_offset`` (bytes from the
    buffer's start; equal to ``src_offset`` modulo the layout's alignment).
    ``total_bytes`` is the buffer size. Per record, in the order the records
    were given: ``record_dst`` (where the record lands in the buffer) and
    ``record_run``. The run order is the regions' byte order, which is row
    order only when each region stores its records in row order.
    """
    region: np.ndarray
    src_offset: np.ndarray
    nbytes: np.ndarray
    dst_offset: np.ndarray
    total_bytes: int
    record_dst: np.ndarray
    record_run: np.ndarray

    @property
    def num_runs(self) -> int:
        return len(self.region)


class Selection:
    """Records a reader needs from one table, and the rows it keeps.

    ``records`` are ascending record ids. Decoding them in order gives their
    rows back to back; the reader drops ``head_rows`` rows from the start and
    keeps the next ``rows``, so only the first and last record are trimmed.
    """

    __slots__ = ("table", "records", "head_rows", "rows")

    def __init__(self, table: RecordTable, records: np.ndarray,
                 head_rows: int, rows: int):
        self.table = table
        self.records = records
        self.head_rows = int(head_rows)
        self.rows = int(rows)
        assert self.head_rows + self.rows <= self.decoded_rows, (
            "selection keeps more rows than its records hold")

    @property
    def num_records(self) -> int:
        return len(self.records)

    @property
    def decoded_rows(self) -> int:
        """Rows the selected records decode to, before trimming."""
        return int(self.table.rows[self.records].sum())

    @property
    def tail_rows(self) -> int:
        """Rows dropped from the end of the last record."""
        return self.decoded_rows - self.head_rows - self.rows

    def stored_bytes(self) -> int:
        return int(self.table.nbytes[self.records].sum())
