import weakref
from dataclasses import dataclass, field

import numpy as np
import torch
import constants
import TQPlib.tqpcompress as tqpcompress
from IO.cache import config as _cache_cfg
from IO.pages import abs_copy
from IO.telemetry import gates, movement


# -----------------------------------------------------------------------------
# Compiled per-tick input geometry.
#
# compile_tick_layout(stages) builds the pointer-free aggregate-buffer and
# codec-group layout below.  bind_tick_sources(stages, layout) patches current
# host/slot/GPU source pointers and returns the launch plan:
#
#   agg_buf:  [ codec_group_0 runs | codec_group_1 runs | ... |
#               raw_group runs ]
#
# Each stage reads a selection of its column's record table
# (TableVariable.record_selection): the records its rows overlap.  The
# selected records of one region form one run, one contiguous byte range
# copied with one copy; over a producer's regions, which hold several
# streams, the selected records of one stream stretch are a run
# (_plan_table_read).  A stage's compressed runs sit
# back to back from a 512-aligned start, each run 512-aligned, so every
# record keeps the alignment its codec needs; its RAW records follow, dense
# and in row order, so a RAW stage's bytes are its rows.  A compressed
# record's index lives on the host (the table), so each codec group's
# decompress pointers are compiled from it (one manager template per group)
# and no metadata crosses PCIe.  Stages sharing a codec key share one
# decompressed output buffer, so one nvCOMP call per codec group decodes
# every selected record.  The operator's rows are the decoded rows: a tick
# the tick cutter cut in whole quanta selects whole records
# (``TableVariable.whole_records``); a tick cut by rows is trimmed inside its
# first and last record only.  A run of a region
# the granule cache holds is bound piece by piece (_bind_regions): its
# granules in slots by D2D, the rest from its home by H2D.
#
# Plan building is side-effect free; allocation and copies happen in
# plan_input_data_movement + launch_h2d + launch_d2d.
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class CompiledManagerTemplate:
    """Prepared manager launch arrays with launch-time base rebasing only.

    Relative arrays are immutable geometry.  The two absolute arrays and their
    CPU tensor views are stable reusable pybind arguments; launch overwrites
    every absolute pointer with ``np.add(..., out=...)``.  ``handles`` is
    ``None`` when a record arrived without a saved config handle; the launch
    then parses the group's record headers (cold, synchronizing).
    """

    codec_key: tuple
    compressed_relative: np.ndarray
    output_relative: np.ndarray
    compressed_absolute: np.ndarray
    output_absolute: np.ndarray
    compressed_ptrs: torch.Tensor
    output_ptrs: torch.Tensor
    handles: torch.Tensor | None
    launch_args: tuple | None = None


@dataclass(slots=True)
class CompiledTickLayout:
    """Tick-owned immutable geometry plus reusable dynamic pointer storage.

    ``pack_args`` / ``d2d_pack_args`` are the native packers' arguments,
    ``(source pointers, agg_buf offsets, bytes)`` with one entry per run.
    Both lanes share the offset and byte lists; binding overwrites only their
    source lists, where 0 means the other lane copies the run.
    """

    agg_buf_size: int
    codec_groups: tuple
    per_stage_views: tuple
    per_stage_bindings: tuple
    pack_args: tuple
    d2d_pack_args: tuple
    # The reads of the tick's fully cached stages as one plan
    # (``_TickCached``), else None.
    tick_cached: object = None
    _binding_active: bool = field(default=False, init=False, repr=False)


@dataclass(slots=True)
class BoundTickPlan:
    """Dynamic source binding over a :class:`CompiledTickLayout`, which
    keeps the geometry.  ``d2d_pack_pages`` and ``h2d_pack_pages`` count the
    runs each lane copies.
    """

    layout: CompiledTickLayout
    d2d_pack_pages: int
    h2d_pack_pages: int
    # The packers' arguments: the layout's, or, when a run of a cached
    # region was split into pieces, the layout's entries plus one per piece.
    pack_args: tuple
    d2d_pack_args: tuple
    # The slot pieces of cached regions' runs, ``(src, dst offset, bytes)``
    # int64 arrays, copied by one kernel (``launch_slot_pieces``), else
    # None.
    slot_pieces: tuple | None = None
    # The spillable GPU regions the bind took addresses of
    # (``IO/pages/device_regions.py``), pinned until the copies are queued
    # (``unpin``: the exchange's launch, or ``release``).
    pins: list = field(default_factory=list)
    # The events of copies into the homes the home pieces read, not yet
    # seen to end: the H2D lane waits on them before its copies
    # (``DirectIn.launch_h2d``).
    home_waits: list = field(default_factory=list)
    _released: bool = field(default=False, init=False, repr=False)

    def unpin(self) -> None:
        """The copies reading the bound regions are queued: an eviction
        (which synchronises the device first) may move them now."""
        if self.pins:
            from IO.pages import device_regions
            device_regions.unpin(self.pins)
            self.pins = []

    def release(self) -> None:
        """Allow an explicitly sequential rebind (tests/standalone tools).

        Production tick layouts are launched once and deliberately remain
        acquired while their manager template can still be consumed later on
        the compute stream.
        """
        self.unpin()
        if not self._released:
            self.layout._binding_active = False
            self._released = True


def _launch_mgr_group(ctx, shared_output):
    """Launch one prepared manager group after rebasing its two base pointers.

    RAW runs of mixed-codec stages (``mgr_raw_rows``) are copied
    data-region → shared-output first — the decompress template only covers
    the compressed records.
    """
    raw_rows = ctx.get('mgr_raw_rows')
    if raw_rows:
        # Plain slice copies, NOT batched_memcpy: its kernel does 8-byte
        # vectorized loads (base+offset must be 8-aligned), which holds for
        # every 512-aligned compressed-record caller but not for these
        # row-granular offsets (a 4-byte or 1-byte-wide column lands
        # anywhere).  Mixed stages are transient probe-window straddles
        # with a handful of RAW records — per-run launch cost is fine.
        agg_buf = ctx['agg_buf']
        for data_off, out_off, nbytes in raw_rows:
            shared_output[out_off:out_off + nbytes].copy_(
                agg_buf[data_off:data_off + nbytes], non_blocking=True)
    template = ctx.get('mgr_template')
    if template is None:
        return

    agg_ptr = ctx['agg_buf'].data_ptr()
    out_ptr = shared_output.data_ptr()
    np.add(template.compressed_relative, agg_ptr,
           out=template.compressed_absolute)
    np.add(template.output_relative, out_ptr,
           out=template.output_absolute)

    # In-run records normally carry saved config handles, making launch_args
    # stable from compile time.  Keep a fail-safe for imported pages that
    # lack them: the launch parses the group's record headers, for this
    # template only, so no parsed DecompressionConfig is shared with another
    # decompression.  It is cold and manager telemetry records the header
    # parse synchronizations.
    if template.launch_args is None:
        from TQPlib.tqpcompress import mgr_ingest_configs
        codec_idx, elem_size, chunk_b = template.codec_key
        template.handles = mgr_ingest_configs(
            codec_idx, elem_size, chunk_b, template.compressed_ptrs)
        template.launch_args = (
            codec_idx, elem_size, chunk_b,
            template.compressed_ptrs,
            template.output_ptrs,
            template.handles,
        )
    tqpcompress.mgr_decompress(*template.launch_args)


def _compile_manager_template(codec_key, compressed_relative, output_relative,
                              handles, orig_bytes):
    """Compile one codec group's manager launch from its records' relative
    compressed and output offsets and their config handles (0 = none)."""
    if not len(compressed_relative):
        return None
    if gates.VARCHK:
        assert (orig_bytes > 0).all() and (
            np.diff(output_relative) >= orig_bytes[:-1]).all(), (
            f"VARCHK mgr group: orig={orig_bytes} "
            f"outputs={output_relative}")
    compressed_relative = np.ascontiguousarray(compressed_relative,
                                               dtype=np.int64)
    output_relative = np.ascontiguousarray(output_relative, dtype=np.int64)
    compressed_relative.setflags(write=False)
    output_relative.setflags(write=False)
    compressed_absolute = np.empty_like(compressed_relative)
    output_absolute = np.empty_like(output_relative)
    compressed_ptrs = torch.from_numpy(compressed_absolute)
    output_ptrs = torch.from_numpy(output_absolute)
    handles_t = None
    launch_args = None
    if (handles != 0).all():
        handles_t = torch.from_numpy(np.array(handles, dtype=np.int64))
        codec_idx, elem_size, chunk_b = codec_key
        launch_args = (
            codec_idx, elem_size, chunk_b,
            compressed_ptrs, output_ptrs, handles_t,
        )
    template = CompiledManagerTemplate(
        codec_key=codec_key,
        compressed_relative=compressed_relative,
        output_relative=output_relative,
        compressed_absolute=compressed_absolute,
        output_absolute=output_absolute,
        compressed_ptrs=compressed_ptrs,
        output_ptrs=output_ptrs,
        handles=handles_t,
        launch_args=launch_args,
    )
    return template


def _exclusive_cumsum(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values) + 1, dtype=np.int64)
    np.cumsum(values, out=out[1:])
    return out


@dataclass(slots=True)
class _StageRead:
    """One stage's selection laid out relative to its own start in agg_buf
    and in its decoded output."""

    codec_key: tuple | None    # the compressed codec's manager key, or None
    is_raw: bool               # no compressed record selected
    data_bytes: int            # bytes of agg_buf the stage spans
    total_rows: int            # rows the selected records decode to
    row_range: slice | None    # the kept rows among them
    data_type: torch.dtype
    data_width: int
    row_bytes: int
    # (region, src_offset, nbytes, dst_offset, logical_b)
    # per run: compressed runs, then RAW runs.
    runs: list
    # per compressed record: table record id, offset in the stage's data,
    # offset in its output, config handle, decoded bytes
    table: object
    comp_ids: np.ndarray
    comp_data_off: np.ndarray
    comp_out_off: np.ndarray
    comp_handles: np.ndarray
    comp_raw_bytes: np.ndarray
    # RAW runs of a mixed stage: (data offset, output offset, bytes)
    raw_rows: list
    # A record table of regions (``_plan_table_stage``): the runs as numpy
    # arrays (region, src_offset, nbytes, dst_offset, logical_b) instead of
    # ``runs``.
    run_arr: tuple | None = None
    # The selection's last record when the stage's rows end inside it (the
    # rest of its rows are the next tick's), else -1: the cached read plan
    # leaves it out of the settle spans (``_cached_runs``).
    tail_record: int = -1


class _RegionIndex:
    """What the region planner needs of one record table, built once per
    table: whether its regions are page-shaped (records ascending by region,
    one codec per region, a region's records in ascending byte order with no
    overlap, and a region's RAW records back to back, each starting where
    the rows of the one before end), each region's first record, and
    exclusive prefix sums of rows and decoded bytes over the records."""

    __slots__ = ("ok", "reg_first", "reg_codec", "row_cum", "out_cum")

    def __init__(self, table):
        region, codec = table.region, table.codec
        n = len(region)
        same = region[1:] == region[:-1]
        self.ok = bool(
            n and (np.diff(region) >= 0).all()
            and not (codec[1:] != codec[:-1])[same].any()
            and not (table.offset[1:] < table.offset[:-1]
                     + table.nbytes[:-1])[same].any()
            and not ((codec[1:] == 0) & same
                     & (table.offset[1:] != table.offset[:-1]
                        + table.raw_bytes[:-1])).any())
        if not self.ok:
            return
        self.reg_first = np.searchsorted(
            region, np.arange(len(table.regions) + 1)).tolist()
        reg_codec = np.full(len(table.regions), -1, dtype=np.int64)
        reg_codec[region] = codec
        self.reg_codec = reg_codec.tolist()
        self.row_cum = _exclusive_cumsum(table.rows)
        self.out_cum = _exclusive_cumsum(table.raw_bytes)


_region_index = weakref.WeakKeyDictionary()


def _index_of(table) -> _RegionIndex:
    index = _region_index.get(table)
    if index is None:
        index = _region_index[table] = _RegionIndex(table)
    return index


def _plan_stage_read(stage) -> _StageRead:
    """Select, lay out and count one stage's records (module comment)."""
    return _plan_table_stage(stage.src_col.record_selection())


def _plan_by_region(sel, index) -> _StageRead:
    """Lay out a contiguous record range over page-shaped regions
    (``index``, its table's ``_RegionIndex``), region by region: Python per
    region, numpy over the records.  A compressed region's selected records
    are one run, decoded whole; a RAW region's selected rows are one run,
    copied by rows.  A stage may hold RAW regions besides the regions of its
    one compressed codec (a write flip): its RAW runs then follow the
    compressed runs in agg_buf and are copied into the decoded output at
    their row order (``raw_rows``)."""
    table = sel.table
    recs = sel.records
    n = len(recs)
    row_bytes = int(table.row_bytes)
    align = constants.COMPRESSION_ALIGNMENT
    empty = np.zeros(0, dtype=np.int64)
    runs = []
    raw_rows = []
    data_bytes = 0
    out = 0
    head = 0
    comp_codec = None
    comp = dict(comp_ids=empty, comp_data_off=empty, comp_out_off=empty,
                comp_handles=empty, comp_raw_bytes=empty)
    if n:
        r0, r1 = int(recs[0]), int(recs[-1]) + 1
        row_cum, out_cum = index.row_cum, index.out_cum
        reg_first, reg_codec = index.reg_first, index.reg_codec
        offset, nbytes = table.offset, table.nbytes
        first = int(row_cum[r0]) + sel.head_rows
        stop = first + sel.rows
        comp_runs = []    # (region, first record, end record, src, end, out)
        raw_runs = []     # (region, src, bytes, out)
        for k in range(int(table.region[r0]), int(table.region[r1 - 1]) + 1):
            a = max(r0, reg_first[k])
            b = min(r1, reg_first[k + 1])
            if a >= b:
                continue
            cid = reg_codec[k]
            if cid == 0:
                g_a, g_b = int(row_cum[a]), int(row_cum[b])
                lead = first - g_a if first > g_a else 0
                kept = min(g_b, stop) - max(g_a, first)
                nb = (kept if kept > 0 else 0) * row_bytes
                raw_runs.append((k, int(offset[a]) + lead * row_bytes, nb,
                                 out))
                out += nb
                continue
            assert comp_codec in (None, cid), (
                f"a DirectIn stage mixes codecs {comp_codec} and {cid}")
            comp_codec = cid
            if a == r0:
                head = sel.head_rows      # the first record decodes whole
            comp_runs.append((k, a, b, int(offset[a]),
                              int(offset[b - 1]) + int(nbytes[b - 1]), out))
            out += int(out_cum[b] - out_cum[a])
        deltas, counts, out_shift = [], [], []
        for k, a, b, src, end, o in comp_runs:
            dst = (data_bytes + align - 1) // align * align
            runs.append((k, src, end - src, dst,
                         int(out_cum[b] - out_cum[a])))
            deltas.append(dst - src)
            counts.append(b - a)
            out_shift.append(o - int(out_cum[a]))
            data_bytes = dst + end - src
        if comp_codec is not None and raw_runs:
            data_bytes = (data_bytes + align - 1) // align * align
        for k, src, nb, o in raw_runs:
            runs.append((k, src, nb, data_bytes, nb))
            if comp_codec is not None:
                raw_rows.append((data_bytes, o, nb))
            data_bytes += nb
        if comp_codec is not None:
            ids = np.arange(r0, r1, dtype=np.int64)
            ids = ids[table.codec[r0:r1] != 0]
            comp = dict(
                comp_ids=ids,
                comp_data_off=offset[ids] + np.repeat(
                    np.array(deltas, dtype=np.int64), counts),
                comp_out_off=out_cum[ids] + np.repeat(
                    np.array(out_shift, dtype=np.int64), counts),
                comp_handles=table.handle[ids],
                comp_raw_bytes=table.raw_bytes[ids])
    total_rows = out // row_bytes
    return _StageRead(
        codec_key=None if comp_codec is None else table.codec_keys[comp_codec],
        is_raw=comp_codec is None,
        data_bytes=data_bytes,
        total_rows=total_rows,
        row_range=(None if head == 0 and sel.rows == total_rows
                   else slice(head, head + sel.rows)),
        data_type=table.stored_dtype,
        data_width=int(table.width),
        row_bytes=row_bytes,
        runs=runs,
        table=table,
        raw_rows=raw_rows,
        **comp,
    )


def _plan_table_stage(sel) -> _StageRead:
    """A ``TableVariable`` stage's read: region by region
    (``_plan_by_region``, its runs as ``run_arr``) when its records are a
    contiguous range over page-shaped regions, as a scan's batches and a
    one-stream producer's regions are; else ``_plan_table_read``.  Both give
    the same read."""
    recs = sel.records
    n = len(recs)
    read = None
    if n and int(recs[-1]) - int(recs[0]) + 1 == n:
        index = _index_of(sel.table)
        if index.ok:
            read = _plan_by_region(sel, index)
            read.run_arr = tuple(np.array(c, dtype=np.int64)
                                 for c in zip(*read.runs))
            read.runs = []
    if read is None:
        read = _plan_table_read(sel)
    if n and sel.tail_rows > 0:
        read.tail_record = int(recs[-1])
    return read


def _plan_table_read(sel) -> _StageRead:
    """A record table whose regions hold several streams (a partition
    sort's, one region per tick holding every bucket), with numpy over the
    records and the runs and no Python per run: the runs come back as
    arrays (``_StageRead.run_arr``).  Compressed records are copied in runs
    of their regions' byte order (``RecordTable.runs``: one run per stretch
    of selected records with no unselected record between them in their
    region, so a range of buckets copies one run per region) and decoded to
    their row-order place; RAW records copy only their kept rows, in row
    order, one run per stretch adjacent both in the selection and in their
    region's bytes."""
    table = sel.table
    recs = sel.records
    n = len(recs)
    row_bytes = int(table.row_bytes)
    align = constants.COMPRESSION_ALIGNMENT
    rows = table.rows[recs]
    codec = table.codec[recs]
    gstart = table.global_row_start()[recs]
    first = int(gstart[0]) + sel.head_rows if n else 0
    stop = first + sel.rows
    lead = np.clip(first - gstart, 0, rows)
    kept = np.clip(np.minimum(gstart + rows, stop)
                   - np.maximum(gstart, first), 0, None)
    comp = codec != 0
    comp_codecs = np.unique(codec[comp])
    assert len(comp_codecs) <= 1, (
        f"a DirectIn stage mixes codecs {comp_codecs.tolist()}")
    comp_codec = int(comp_codecs[0]) if len(comp_codecs) else None
    # Compressed records decode whole (their leading rows are trimmed
    # after); RAW records copy only their kept rows.
    out_b = np.where(comp, rows, kept) * row_bytes
    out_off = _exclusive_cumsum(out_b)
    head = int(lead[comp].sum())

    empty = np.zeros(0, dtype=np.int64)
    parts = []          # (region, src, nbytes, dst, logical_b)
    data_bytes = 0
    ci = np.flatnonzero(comp)
    comp_ids = recs[ci]
    comp_data_off = empty
    if len(ci):
        lay = table.runs(comp_ids, align)
        parts.append((
            lay.region, lay.src_offset, lay.nbytes, lay.dst_offset,
            np.bincount(lay.record_run, weights=out_b[ci],
                        minlength=lay.num_runs).astype(np.int64)))
        comp_data_off = lay.record_dst
        data_bytes = int((lay.dst_offset + lay.nbytes).max())
    raw_rows = []
    ri = np.flatnonzero(~comp)
    if len(ri):
        if comp_codec is not None:
            data_bytes = (data_bytes + align - 1) // align * align
        raw_ids = recs[ri]
        region = table.region[raw_ids]
        src = table.offset[raw_ids] + lead[ri] * row_bytes
        nb = out_b[ri]
        brk = np.ones(len(ri), dtype=bool)
        brk[1:] = ((ri[1:] != ri[:-1] + 1) | (region[1:] != region[:-1])
                   | (src[1:] != src[:-1] + nb[:-1]))
        starts = np.flatnonzero(brk)
        run_nb = np.add.reduceat(nb, starts)
        run_dst = data_bytes + _exclusive_cumsum(run_nb)[:-1]
        parts.append((region[starts], src[starts], run_nb, run_dst, run_nb))
        if comp_codec is not None:
            raw_rows = list(zip(run_dst.tolist(),
                                out_off[ri[starts]].tolist(),
                                run_nb.tolist()))
        data_bytes = int(run_dst[-1] + run_nb[-1])
    if not parts:
        run_arr = (empty,) * 5
    elif len(parts) == 1:
        run_arr = parts[0]
    else:
        run_arr = tuple(np.concatenate(c) for c in zip(*parts))
    total_rows = int(out_off[-1]) // row_bytes
    return _StageRead(
        codec_key=(None if comp_codec is None
                   else table.codec_keys[comp_codec]),
        is_raw=comp_codec is None,
        data_bytes=data_bytes,
        total_rows=total_rows,
        row_range=(None if head == 0 and sel.rows == total_rows
                   else slice(head, head + sel.rows)),
        data_type=table.stored_dtype,
        data_width=int(table.width),
        row_bytes=row_bytes,
        runs=[],
        table=table,
        comp_ids=comp_ids,
        comp_data_off=comp_data_off.astype(np.int64, copy=False),
        comp_out_off=out_off[ci],
        comp_handles=table.handle[comp_ids],
        comp_raw_bytes=out_b[ci],
        raw_rows=raw_rows,
        run_arr=run_arr,
    )


@dataclass(slots=True)
class _TableRuns:
    """A ``TableVariable`` stage's runs in the pack arrays: entries
    ``[lo, hi)``, and per run its region, source offset, bytes and decoded
    bytes (the movement ledger's ``logical_b``)."""

    lo: int
    hi: int
    region: np.ndarray
    src_offset: np.ndarray
    nbytes: np.ndarray
    logical_b: np.ndarray
    # The runs of cached regions, planned at compile time
    # (``_CachedRuns``; None: no cached run, or the plan is off).
    cached: object = None
    # Every run cached and bound with the tick's other such stages
    # (``CompiledTickLayout.tick_cached``).
    in_tick: bool = False


@dataclass(slots=True)
class _CachedRuns:
    """A stage's reads of cached regions, derived from its runs when the
    layout is compiled (``IO.cache.config.PLAN_AT_COMPILE``): the cache;
    ``cmask``, which runs are of cached regions; ``ci``, those with bytes,
    and per one its region's index in the cache and its byte span
    (``pieces`` reads them); the settle spans, the same spans with the
    selection's tail record left out (``note_read`` counts them, so the
    tick that finishes the record counts its stream's end once); and the
    granule ids of the spans (the next tick's consume pins)."""

    cache: object
    cmask: np.ndarray
    ci: np.ndarray
    ri: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    s_ri: np.ndarray
    s_lo: np.ndarray
    s_hi: np.ndarray
    gids: np.ndarray


def _cached_runs(table, tail_record, region, src, nb):
    """``_CachedRuns`` of a stage's runs over ``table``, or None when none
    is of a cached region.  The settle spans drop the tail record: a run
    holding it alone goes, otherwise the run it ends is cut at its start
    (the records of a run are in byte order, and the tail record is the
    selection's last)."""
    cached = _cached_regions(table)
    if cached is None:
        return None
    cache, index = cached
    ri_all = index[region]
    cmask = ri_all >= 0
    if not cmask.any():
        return None
    ci = np.flatnonzero(cmask & (nb > 0))
    ri = ri_all[ci]
    lo = src[ci]
    hi = lo + nb[ci]
    s_ri, s_lo, s_hi = ri, lo, hi
    if tail_record >= 0 and len(ci):
        k = int(index[table.region[tail_record]])
        if k >= 0:
            o = int(table.offset[tail_record])
            j = np.flatnonzero((ri == k) & (lo <= o) & (o < hi))
            if len(j):
                j = int(j[0])
                if int(lo[j]) == o:
                    keep = np.ones(len(ri), dtype=bool)
                    keep[j] = False
                    s_ri, s_lo, s_hi = ri[keep], lo[keep], hi[keep]
                else:
                    s_hi = hi.copy()
                    s_hi[j] = o
    return _CachedRuns(cache, cmask, ci, ri, lo, hi, s_ri, s_lo, s_hi,
                       cache.granules(ri, lo, hi))


@dataclass(slots=True)
class _TickCached:
    """The cached runs of a tick's fully cached stages, concatenated at
    compile time: one ``pieces`` and one ``defer_read`` per tick instead of
    one per stage.  Per run: its region's index in the cache, its byte span, its agg_buf
    offset less the span's start (``d0``), its decoded and stored bytes (the
    ledger's); the settle spans and the granule ids, as ``_CachedRuns``."""

    cache: object
    ri: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    d0: np.ndarray
    lb: np.ndarray
    nb: np.ndarray
    s_ri: np.ndarray
    s_lo: np.ndarray
    s_hi: np.ndarray
    gids: np.ndarray


_NO_I64 = np.empty(0, dtype=np.int64)


def _tick_cached(bindings, dst_offsets):
    """``_TickCached`` over the stages whose runs are all of cached regions
    (marking their bindings ``in_tick``), or None when there is none."""
    parts = []
    cache = None
    for b in bindings:
        p = getattr(b, 'cached', None)
        if p is None or not len(p.ci) or not p.cmask.all():
            continue
        if cache is None:
            cache = p.cache
        elif p.cache is not cache:
            continue
        b.in_tick = True
        d0 = np.asarray(dst_offsets[b.lo:b.hi], dtype=np.int64)[p.ci] - p.lo
        parts.append((p, d0, b.logical_b[p.ci]))
    if not parts:
        return None
    if len(parts) == 1:
        p, d0, lb = parts[0]
        return _TickCached(cache, p.ri, p.lo, p.hi, d0, lb, p.hi - p.lo,
                           p.s_ri, p.s_lo, p.s_hi, p.gids)

    def cat(f):
        return np.concatenate([f(x) for x in parts])
    lo, hi = cat(lambda x: x[0].lo), cat(lambda x: x[0].hi)
    return _TickCached(cache, cat(lambda x: x[0].ri), lo, hi,
                       cat(lambda x: x[1]), cat(lambda x: x[2]), hi - lo,
                       cat(lambda x: x[0].s_ri), cat(lambda x: x[0].s_lo),
                       cat(lambda x: x[0].s_hi), cat(lambda x: x[0].gids))


def _bind_tick_cached(tc, reads, home_waits):
    """Bind a ``_TickCached``: one ``pieces`` call, its home pieces' copy
    events to ``home_waits``; returns ``(home pieces, slot pieces, six
    arrays)`` as ``_bind_regions`` does."""
    pcs = tc.cache.pieces(tc.ri, tc.lo, tc.hi)
    if pcs.events:
        home_waits.extend(pcs.events)
    dst = tc.d0[pcs.run] + pcs.offset
    on_gpu = pcs.on_gpu
    n_slot = int(on_gpu.sum())
    n_home = len(on_gpu) - n_slot
    if n_home == 0:
        ext = (_NO_I64, _NO_I64, _NO_I64, pcs.src, dst, pcs.size)
    else:
        home = ~on_gpu
        ext = (pcs.src[home], dst[home], pcs.size[home],
               pcs.src[on_gpu], dst[on_gpu], pcs.size[on_gpu])
    if reads is not None:
        for kind, m in (('cache_hit', on_gpu),
                        ('restore_evicted_h2d', ~on_gpu)):
            if m.any():
                b = np.bincount(pcs.run[m], weights=pcs.size[m],
                                minlength=len(tc.ri)).astype(np.int64)
                reads.append((kind, int(m.sum()),
                              int((tc.lb * b // tc.nb).sum()), int(b.sum())))
    return n_home, n_slot, ext


def compile_tick_layout(stages) -> CompiledTickLayout:
    """Compile pointer-free geometry for one tick's DirectIn stages."""
    reads = [None] * len(stages)
    codec_group_stages = {}    # manager codec_key -> [stage_idx]
    raw_group = []

    # Pass 1: each stage's selection, runs and codec.  In particular, do not
    # read tensor.is_cuda or any data_ptr here.
    for stage_idx, stage in enumerate(stages):
        if stage.src_col.reads_nothing():
            continue
        read = _plan_stage_read(stage)
        reads[stage_idx] = read
        if read.is_raw:
            raw_group.append(stage_idx)
        else:
            codec_group_stages.setdefault(read.codec_key, []).append(stage_idx)

    # Pass 2: aggregate-buffer and shared-output layout, compressed groups
    # first (one per manager identity), then the RAW group; one pack entry
    # per run, and each compressed group's manager launch template and its
    # mixed-codec RAW-run copies.
    codec_groups = []
    per_stage_views = [None] * len(stages)
    per_stage_bindings = [()] * len(stages)
    dst_offsets = []
    byte_sizes = []
    cursor = 0
    group_stage_lists = list(codec_group_stages.values())
    if raw_group:
        group_stage_lists.append(raw_group)
    for stage_indices in group_stage_lists:
        is_raw = stage_indices is raw_group
        align = 8 if is_raw else constants.COMPRESSION_ALIGNMENT
        out_cursor = 0
        comp_parts, out_parts, handle_parts, orig_parts = [], [], [], []
        raw_rows = []
        for si in stage_indices:
            read = reads[si]
            cursor = (cursor + align - 1) // align * align
            out_cursor = (out_cursor + 7) & ~7
            per_stage_views[si] = {
                'codec_group_idx': len(codec_groups),
                'is_raw': is_raw,
                'data_buf_slice': (cursor, cursor + read.data_bytes),
                'byte_offset_in_output': 0 if is_raw else out_cursor,
                'total_rows': read.total_rows,
                'row_range': read.row_range,
                'data_width': read.data_width,
                'data_type': read.data_type,
            }
            region, src, nb, dst, lb = read.run_arr
            lo = len(dst_offsets)
            dst_offsets.extend((dst + cursor).tolist())
            byte_sizes.extend(nb.tolist())
            per_stage_bindings[si] = _TableRuns(
                lo, len(dst_offsets), region, src, nb, lb)
            if _cache_cfg.PLAN_AT_COMPILE and len(region):
                per_stage_bindings[si].cached = _cached_runs(
                    read.table, read.tail_record, region, src, nb)
            if not is_raw:
                comp_parts.append(read.comp_data_off + cursor)
                out_parts.append(read.comp_out_off + out_cursor)
                handle_parts.append(read.comp_handles)
                orig_parts.append(read.comp_raw_bytes)
                raw_rows.extend((cursor + d, out_cursor + o, b)
                                for d, o, b in read.raw_rows)
                out_cursor += read.total_rows * read.row_bytes
            cursor += read.data_bytes
        group = {
            'is_raw': is_raw,
            'shared_output_size': 0 if is_raw else out_cursor,
            'mgr_template': None,
            'mgr_raw_rows': tuple(raw_rows),
        }
        if not is_raw:
            group['mgr_template'] = _compile_manager_template(
                reads[stage_indices[0]].codec_key,
                np.concatenate(comp_parts), np.concatenate(out_parts),
                np.concatenate(handle_parts), np.concatenate(orig_parts))
        codec_groups.append(group)
        cursor = (cursor + 7) & ~7
    agg_buf_size = cursor

    run_count = len(dst_offsets)
    tick_cached = None
    if _cache_cfg.PLAN_AT_COMPILE:
        # The in-tick stages' pack entries stay 0 (compiled here, never
        # bound): their pieces are entries of their own.
        tick_cached = _tick_cached(per_stage_bindings, dst_offsets)
    layout = CompiledTickLayout(
        agg_buf_size=agg_buf_size,
        codec_groups=tuple(codec_groups),
        per_stage_views=tuple(per_stage_views),
        per_stage_bindings=tuple(per_stage_bindings),
        pack_args=([0] * run_count, dst_offsets, byte_sizes),
        d2d_pack_args=([0] * run_count, dst_offsets, byte_sizes),
        tick_cached=tick_cached,
    )
    return layout


def bind_tick_sources(stages, layout: CompiledTickLayout) -> BoundTickPlan:
    """Bind current host/slot/GPU sources into compiled tick geometry.

    A stage's runs are bound by ``_bind_regions`` (a cached region's piece
    by piece, from its slots or its home).  The layout's geometry is
    trusted: a record table's regions and records never change.  The
    spillable GPU regions it reads are pinned (``BoundTickPlan.pins``): an
    eviction before the copies are queued would leave their addresses
    stale.
    """
    acquired = False
    try:
        if layout._binding_active:
            raise RuntimeError("CompiledTickLayout is already bound")
        layout._binding_active = True
        acquired = True

        h2d_src = layout.pack_args[0]
        d2d_src = layout.d2d_pack_args[0]
        d2d_pack_pages = 0
        h2d_pack_pages = 0
        # Movement ledger: the bind's reads, recorded once the bind has
        # succeeded.
        reads = [] if gates.EXPERIMENT else None
        # Pieces of cached regions' runs: six lists of arrays (home pieces'
        # src, dst offset, bytes; slot pieces' src, dst offset, bytes), the
        # home pieces appended to the layout's H2D entries and the slot
        # pieces copied by one kernel; and the stages reading cached
        # regions, whose reads the cache counts once the bind holds.
        pieces = None
        cached_stages = []
        on_gpu = []
        home_waits = []

        tc = layout.tick_cached
        for stage_idx, stage in enumerate(stages):
            if stage.src_col.reads_nothing():
                continue
            if tc is not None and layout.per_stage_bindings[stage_idx].in_tick:
                continue
            n_h2d, n_d2d, extra = _bind_regions(
                stage, layout.per_stage_bindings[stage_idx], h2d_src,
                d2d_src, reads, layout.pack_args[1], on_gpu, home_waits)
            h2d_pack_pages += n_h2d
            d2d_pack_pages += n_d2d
            if extra is not None:
                cached_stages.append((stage, stage_idx))
                if pieces is None:
                    pieces = ([], [], [], [], [], [])
                for acc, part in zip(pieces, extra):
                    acc.append(part)

        if tc is not None:
            n_h2d, n_d2d, extra = _bind_tick_cached(tc, reads, home_waits)
            h2d_pack_pages += n_h2d
            d2d_pack_pages += n_d2d
            if pieces is None:
                pieces = ([], [], [], [], [], [])
            for acc, part in zip(pieces, extra):
                acc.append(part)
        if reads is not None:
            for kind, pages, logical_b, nbytes in reads:
                movement.record(kind, pages, logical_b, nbytes)
        plan = BoundTickPlan(
            layout=layout,
            d2d_pack_pages=d2d_pack_pages,
            h2d_pack_pages=h2d_pack_pages,
            pack_args=layout.pack_args,
            d2d_pack_args=layout.d2d_pack_args,
            home_waits=home_waits,
        )
        if on_gpu:
            from IO.pages import device_regions
            plan.pins = device_regions.pin(on_gpu)
        for stage, stage_idx in cached_stages:
            cplan = layout.per_stage_bindings[stage_idx].cached
            if cplan is not None:
                if len(cplan.s_ri):
                    cplan.cache.defer_read(cplan.s_ri, cplan.s_lo, cplan.s_hi)
                continue
            spans = cached_read_ranges(stage.src_col, settle=True)
            if spans is not None:
                spans[0].defer_read(*spans[1:])
        if tc is not None and len(tc.s_ri):
            tc.cache.defer_read(tc.s_ri, tc.s_lo, tc.s_hi)
        if pieces is not None:
            h_src, h_dst, h_n, g_src, g_dst, g_n = (
                x[0] if len(x) == 1 else np.concatenate(x) for x in pieces)
            if len(h_n):
                dst = layout.pack_args[1] + h_dst.tolist()
                nbytes = layout.pack_args[2] + h_n.tolist()
                plan.pack_args = (h2d_src + h_src.tolist(), dst, nbytes)
                plan.d2d_pack_args = (d2d_src + [0] * len(h_n), dst, nbytes)
            if len(g_n):
                plan.slot_pieces = _coalesce(g_src, g_dst, g_n)
        return plan
    except Exception:
        if acquired:
            layout._binding_active = False
        raise



def _coalesce(src, dst, size):
    """Copies ``(src, dst, size)`` with each one that continues the previous
    in both source and destination merged into it."""
    if len(size) > 1:
        cont = (src[1:] == src[:-1] + size[:-1]) & (
            dst[1:] == dst[:-1] + size[:-1])
        if cont.any():
            first = np.concatenate(([0], np.flatnonzero(~cont) + 1))
            return src[first], dst[first], np.add.reduceat(size, first)
    return src, dst, size


def launch_slot_pieces(agg_buf, plan) -> None:
    """Copy a bound plan's slot pieces (``BoundTickPlan.slot_pieces``) into
    ``agg_buf`` with one copy kernel on the current stream, reading its
    address arrays over UVA (``IO.pages.abs_copy.launch``)."""
    pieces = plan.slot_pieces
    if pieces is None:
        return
    src, dst, size = pieces
    abs_copy.launch(src, dst, size, agg_buf.data_ptr())


def _region_bases(table):
    """Per region of ``table``: its first byte's address and whether it is
    on the GPU (numpy, built once per table and eviction epoch: only an
    eviction moves a region, ``IO/pages/device_regions.py``).  A cached
    region's address is its home's.
    """
    from IO.pages import device_regions
    epoch = device_regions.epoch()
    bases = getattr(table, '_region_bases', None)
    if bases is None or bases[2] != epoch:
        from IO.pages.region import DEVICE
        ptr = np.zeros(len(table.regions), dtype=np.int64)
        dev = np.zeros(len(table.regions), dtype=bool)
        for k, region in enumerate(table.regions):
            storage = region.storage
            if storage is not None:
                ptr[k] = storage.data_ptr()
                dev[k] = region.location == DEVICE
        bases = table._region_bases = (ptr, dev, epoch)
    return bases[0], bases[1]


def _cached_regions(table):
    """The granule cache holding ``table``'s cached regions and, per region,
    its index in that cache (-1: not cached); ``None`` when it holds none
    (built once per table)."""
    cached = getattr(table, '_cached_regions', False)
    if cached is False:
        from IO.pages.region import CACHED
        index = np.full(len(table.regions), -1, dtype=np.int64)
        cache = None
        for k, region in enumerate(table.regions):
            if region.location == CACHED:
                c, cr = region.release
                assert cache is None or c is cache, (
                    "a table's regions are cached by two caches")
                cache = c
                index[k] = cr.index
        cached = table._cached_regions = (
            (cache, index) if cache is not None else None)
    return cached


def cached_read_ranges(col, settle=False):
    """The cached regions a ``TableVariable``'s rows are read from, as
    ``(cache, regions, lo, hi)``: the regions' indices in the cache and the
    byte span of the selected records in each (``None`` when it reads
    none).  With ``settle``, a last record the selection ends inside (the
    rest of its rows are the next tick's) is left out, so the read of its
    stream's end is counted once, by the tick that finishes it."""
    if not getattr(col, 'table_backed', False):
        return None
    # A TableVariable's rows and its table's regions never change: the
    # spans are computed once per column and mode.
    memo = col.__dict__.setdefault('_cached_spans', {})
    if settle not in memo:
        memo[settle] = _cached_spans(col, settle)
    return memo[settle]


def _no_unpin() -> None:
    return None


def pin_cached_reads(col):
    """Pin the cached granules ``col``'s rows are read from, for a read
    outside a pipe's ticks (which pin their own, ``replace_consume_pins``):
    the read resolves its slot pieces at the bind and then allocates its
    buffers, and any arena allocation may lend the region's lowest blocks,
    evicting or moving a granule that is not pinned.  Returns the unpin, to
    call once the copies are issued (a lend synchronises the device before
    it moves anything); a no-op for a column with no cached region."""
    gids = cached_read_granules(col)
    if gids is None or not len(gids):
        return _no_unpin
    cache = _cached_regions(col.record_table())[0]
    cache.pin(gids)
    return lambda: cache.unpin(gids)


def cached_read_granules(col):
    """The ids of the cached granules a ``TableVariable``'s rows are read
    from (``NativeGranuleCache.granules`` over ``cached_read_ranges``), once per
    range; ``None`` when it reads none.  Memoised like the spans: a
    region's granule ids never change."""
    if not getattr(col, 'table_backed', False):
        return None
    memo = col.__dict__.setdefault('_cached_spans', {})
    if 'granules' not in memo:
        spans = cached_read_ranges(col)
        memo['granules'] = (None if spans is None
                            else spans[0].granules(*spans[1:]))
    return memo['granules']


def _cached_spans(col, settle):
    table = col.record_table()
    cached = _cached_regions(table)
    if cached is None:
        return None
    cache, index = cached
    sel = col.record_selection()
    recs = sel.records
    if settle and len(recs) and sel.tail_rows > 0:
        recs = recs[:-1]
    if not len(recs):
        return None
    reg = index[table.region[recs]]
    keep = reg >= 0
    if not keep.any():
        return None
    reg = reg[keep]
    lo = table.offset[recs[keep]]
    hi = lo + table.nbytes[recs[keep]]
    if (reg[1:] >= reg[:-1]).all():
        # Each region's records are one stretch of the selection.
        first = np.concatenate(([0], np.flatnonzero(reg[1:] != reg[:-1]) + 1))
        return (cache, reg[first], np.minimum.reduceat(lo, first),
                np.maximum.reduceat(hi, first))
    uniq, inv = np.unique(reg, return_inverse=True)
    r_lo = np.full(len(uniq), np.iinfo(np.int64).max, dtype=np.int64)
    r_hi = np.zeros(len(uniq), dtype=np.int64)
    np.minimum.at(r_lo, inv, lo)
    np.maximum.at(r_hi, inv, hi)
    return cache, uniq, r_lo, r_hi


def _bind_regions(stage, runs, h2d_src, d2d_src, reads, dst_offsets,
                  on_gpu=None, home_waits=None):
    """Bind a ``TableVariable`` stage's runs (``_TableRuns``): each from its
    region, a host home or scan batch (H2D) or a GPU buffer (D2D), plus the
    run's offset.  A run of a cached region is read piece by piece
    (``NativeGranuleCache.pieces``): its granules in slots by D2D, the rest
    from its home by H2D, after the copies into the home (their events go to
    ``home_waits``, which the H2D lane waits on; with none given, the
    current stream waits on them); its layout entry is skipped and its
    pieces are returned as six arrays, ``(home src, home dst offset, home bytes, slot src, slot
    dst offset, slot bytes)`` (``None`` when there is none): the home pieces
    become layout entries of their own, the slot pieces one kernel's copies
    (``launch_slot_pieces``).  Appends the movement ledger's counts to
    ``reads`` (``None``: not recorded).  Appends the spillable GPU regions
    it reads to ``on_gpu`` (for the bind to pin).  Returns the runs each
    lane copies and the pieces."""
    table = stage.src_col.record_table()
    if not len(runs.region):
        return 0, 0, None
    ptr, dev = _region_bases(table)
    assert not any(table.regions[k].freed
                   for k in np.unique(runs.region).tolist()), (
        "a region was freed before its read")
    src = ptr[runs.region] + runs.src_offset
    on_dev = dev[runs.region]
    if on_gpu is not None and on_dev.any():
        for k in np.unique(runs.region[on_dev]).tolist():
            if table.regions[k].spillable:
                on_gpu.append(table.regions[k])
    lo, hi = runs.lo, runs.hi
    plan = runs.cached
    if plan is not None:
        cmask = plan.cmask
    elif _cache_cfg.PLAN_AT_COMPILE:
        cmask = None          # compiled with the plan: no cached run
    else:
        cached = _cached_regions(table)
        cmask = cached[1][runs.region] >= 0 if cached is not None else None
    if cmask is None or not cmask.any():
        if not on_dev.any():
            h2d_src[lo:hi] = src.tolist()
            d2d_src[lo:hi] = [0] * (hi - lo)
        else:
            h2d_src[lo:hi] = np.where(on_dev, 0, src).tolist()
            d2d_src[lo:hi] = np.where(on_dev, src, 0).tolist()
        n_h2d = int(((runs.nbytes > 0) & ~on_dev).sum())
        n_d2d = int(on_dev.sum())
        if reads is not None:
            for kind, m in ((movement.table_read_kind(table), ~on_dev),
                            ('resident_d2d', on_dev)):
                if m.any():
                    reads.append((kind, int(m.sum()),
                                  int(runs.logical_b[m].sum()),
                                  int(runs.nbytes[m].sum())))
        return n_h2d, n_d2d, None
    plain = ~cmask
    h2d_src[lo:hi] = np.where(plain & ~on_dev, src, 0).tolist()
    d2d_src[lo:hi] = np.where(plain & on_dev, src, 0).tolist()
    n_h2d = int(((runs.nbytes > 0) & plain & ~on_dev).sum())
    n_d2d = int((plain & on_dev).sum())
    if reads is not None:
        for kind, m in ((movement.table_read_kind(table), plain & ~on_dev),
                        ('resident_d2d', plain & on_dev)):
            if m.any():
                reads.append((kind, int(m.sum()),
                              int(runs.logical_b[m].sum()),
                              int(runs.nbytes[m].sum())))
    if plan is not None:
        ci, so, nb = plan.ci, plan.lo, plan.hi - plan.lo
        if not len(ci):
            return n_h2d, n_d2d, None
        cache = plan.cache
        pcs = cache.pieces(plan.ri, so, so + nb)
    else:
        ci = np.flatnonzero(cmask & (runs.nbytes > 0))
        if not len(ci):
            return n_h2d, n_d2d, None
        cache, index = cached
        so = runs.src_offset[ci]
        nb = runs.nbytes[ci]
        pcs = cache.pieces(index[runs.region[ci]], so, so + nb)
    if pcs.events:
        if home_waits is not None:
            # The bind runs on whatever stream is current (the legacy
            # default stream in a pipe), and the exchange's streams are
            # non-blocking: a wait there would order none of their copies.
            home_waits.extend(pcs.events)
        else:
            stream = torch.cuda.current_stream()
            for e in pcs.events:
                stream.wait_event(e)
    on_gpu = pcs.on_gpu
    d0 = np.asarray(dst_offsets[lo:hi], dtype=np.int64)[ci] - so
    dst = d0[pcs.run] + pcs.offset
    home = ~on_gpu
    ext = (pcs.src[home], dst[home], pcs.size[home],
           pcs.src[on_gpu], dst[on_gpu], pcs.size[on_gpu])
    n_d2d += int(on_gpu.sum())
    n_h2d += int((~on_gpu).sum())
    if reads is not None:
        # A home piece holds granules the cache evicted or never admitted.
        lb = runs.logical_b[ci]
        for kind, m in (('cache_hit', on_gpu),
                        ('restore_evicted_h2d', ~on_gpu)):
            if m.any():
                b = np.bincount(pcs.run[m], weights=pcs.size[m],
                                minlength=len(ci)).astype(np.int64)
                reads.append((kind, int(m.sum()), int((lb * b // nb).sum()),
                              int(b.sum())))
    return n_h2d, n_d2d, ext
