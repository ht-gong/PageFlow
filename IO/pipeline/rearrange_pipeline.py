from __future__ import annotations
import time
import torch
from utility.logger import perf_logger, message_logger
from variable import TableVariable, Variable
from adaptive.instage_optimization import (
    get_fixed_compression, _find_all_cpu_rep,
)
from IO.pages.combined_column_rep import CombinedColumnRep
from adaptive.adaptive_driver import (
    make_gather_sizing_session, has_gather_sizing_factory,
    GatherIterSample,
)
import sizing_resolver
import TQPlib.tqpmemory as cmp
from IO.pipeline.common import _WindowPeakTracker, col_to_gpu
from IO.telemetry import gates, movement
from IO.pages import device_regions


# ---------------------------------------------------------------------------
# Rearrange (join materialization): a gather per source column.
#
# For each column ``c`` and an int64 output index ``outIdx`` (length n_out):
#     result[c] = gather(c, outIdx)
# written back to the host (RAW or compressed per the applier's plan) as
# regions; the granule cache does not hold them.  The output streams over
# output-row windows; single-shot is just the loop with one window.
#
# Each source is read one of two ways, the one that moves fewer bytes
# (``_read_plan``):
#
# - over UVA: the gather kernel (``zero_copy_rearrange``) reads only the
#   indexed rows from a RAW rep (``TableVariable.uva_source``), 32-byte
#   sectors at a time, window by window (``uva_bytes``);
# - streamed: the source is loaded whole onto the GPU through a tick pipe
#   (its records copied in on the exchange streams, compressed when its rep
#   is, decoded on the GPU, the ticks overlapped), and each window gathers
#   from there with ``index_select``; it moves the rep's stored bytes.
#
# UVA needs a RAW rep a kernel can address: a scan keeps one, so a scanned
# source takes either way; a producer's column is written in its policy
# codec, so it is streamed; a RAW source on the GPU (a free-list baseline's
# RAW intermediate) is read in place.  A source streamed but too large for the GPU is read
# over UVA from its RAW rep, or, with none, first decoded home RAW
# (``_raw_home``).
# ---------------------------------------------------------------------------

# Output-row budget per chunk.  The gather argsorts the chunk's index, whose
# scratch (perm + sort temporaries ≈ 3 int64 per row) coexists with the
# output buffer, so we charge it against the budget to keep peak GPU use
# bounded — this is the OOM the chunking exists to avoid.  The window is read
# live from the uniform chunk budget, constants.get_chunk_size_b() (CLI:
# --chunk_size_mb), so it can be swept; there is no rearrange-specific knob.
# Because of the argsort charge, an 8-byte column turns only ~25% of the
# budget into transfer — the same number buys a smaller window here than in
# a tick pipe, by design.
_ARGSORT_OVERHEAD_PER_ROW = 3 * 8      # ~3 int64 scratch tensors per output row


def _event_pair():
    """A fresh (start, end) pair of timing-enabled CUDA events."""
    return (torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True))


def _uva_rep(ccr: CombinedColumnRep):
    """The RAW rep a UVA gather can read: one whose records are all in
    pinned host memory, else one whose RAW regions are on the GPU (the
    free-list baseline's intermediates, or some evicted home), read by
    address like host ones; None when there is neither."""
    rep = _find_all_cpu_rep(ccr)
    if rep is None:
        rep = next((r for r in ccr.reps
                    if r.table_backed and r.is_uva_readable()), None)
    return rep


def _gather(src, idx_chunk):
    """Zero-copy UVA gather of rows ``idx_chunk`` of ``src`` (a
    ``variable.UvaSource``; indices count from its ``row_start``) into a
    fresh GPU buffer.

    Returns a 2-D ``(rows, width)`` tensor.  Outer-join semantics are handled
    inside the kernel: any ``idx_chunk`` entry equal to ``constants.null``
    (-64) is written back as ``constants.null`` instead of being dereferenced
    (``zero_copy_reader.cu`` CONST_NULL branch), so right-outer needs no
    separate Python path — index_with_null's semantics live in the gather.
    """
    import TQPlib.zero_copy_reader.zero_copy_reader as zero_copy_reader
    return zero_copy_reader.zero_copy_rearrange(
        src.seg_ptrs, src.seg_rows, idx_chunk, src.row_start, src.row_stop,
        src.dtype, src.width)


def _window(col, out_alg):
    """``(row_bytes, overhead, quantum, rows)`` of a gather writing ``col``'s
    rows in ``out_alg``: the output row bytes; the GPU bytes per output row
    a window holds besides them, the argsort scratch and the previous
    window's worst-case slab, which lives until its pack after this
    window's gather (a RAW window, and the previous window's dense buffer,
    are copied home before the next gather: ``wait_copies``); the output's
    record quantum (windows are whole quanta of the output's records, so
    the output's records nest as a scan's do; a window smaller than one
    quantum is cut by rows); and the window's rows at the uniform chunk
    budget."""
    from IO.codec import region_producer
    from IO.codec.region_producer import slab_bytes_per_row
    from adaptive.sizing_controller import window_rows
    import constants
    row_bytes = col.col_width() * col.orig_elem_size()
    overhead = _ARGSORT_OVERHEAD_PER_ROW + slab_bytes_per_row(
        out_alg, col.orig_elem_size(), row_bytes)
    quantum = region_producer.record_rows(row_bytes)
    rows = window_rows(constants.get_chunk_size_b(), row_bytes + overhead,
                       quantum)
    return row_bytes, overhead, quantum, rows


def uva_bytes(src_rows, out_rows, window_rows, row_bytes):
    """The bytes a UVA gather of ``out_rows`` rows of a ``src_rows``-row
    source reads, as 32-byte sectors, ``window_rows`` rows per window.
    A window's rows are taken as random rows of the source: ``k`` rows of
    up to 32 bytes read ``S (1 - exp(-k / S))`` of the source's ``S``
    sectors, so a sparse window reads a sector per row and a dense one at
    most every sector; a wider row reads its own sectors, two more at
    most.  At SF100 it is within 0.89-1.29x of the sectors the gathers'
    sorted windows read, for 80% of the gathers."""
    import math
    if row_bytes > 32:
        return out_rows * (row_bytes // 32 + 2) * 32
    sectors = max(1.0, src_rows * row_bytes / 32)
    full, last = divmod(out_rows, max(1, window_rows))

    def window(k):
        return sectors * -math.expm1(-k / sectors)
    return int(32 * (full * window(window_rows) + window(last)))


def _streamed_rep(src, applier):
    """The rep a streamed read loads: the one in the source's codec (the
    applier's resolution, a consumer pipe's), else the first; None when it
    is not a record table."""
    from adaptive.instage_optimization import find_rep_by_algorithm
    codec = (applier.codec(src) if applier is not None
             else get_fixed_compression())
    rep = find_rep_by_algorithm(src.ccr, codec)
    return rep if getattr(rep, "table_backed", False) else None


def _fits_on_gpu(nbytes):
    """Whether the operator space can hold a ``nbytes`` buffer and, beside
    it, the streamed load's three ticks in flight and the gather's windows
    (three chunk budgets): its free bytes and its largest free run, with
    what the cache region could lend now (an allocation that misses
    borrows it)."""
    import constants
    try:
        # Under the free-list baseline its spillable regions count free.
        arena = (device_regions.arena_stats() if device_regions.armed()
                 else cmp.get_arena_stats())
    except (AttributeError, RuntimeError):
        return False
    lend = int(arena.get("lendable_b") or 0)
    run = max(int(arena["largest_free_run_b"]),
              int(arena.get("top_free_run_b") or 0) + lend)
    return (nbytes <= run and nbytes + 3 * constants.get_chunk_size_b()
            <= int(arena["free_b"]) + lend)


def _read_plan(src, n_out, out_alg, applier):
    """``("uva", rep)`` or ``("streamed", rep)``: how to read ``src`` for a
    gather of ``n_out`` rows written in ``out_alg``.  UVA when the source
    has a RAW rep a kernel can address and either it is on the GPU or its
    sectors (``uva_bytes``) are fewer bytes than the streamed rep's stored
    bytes; streamed otherwise, when the source fits on the GPU beside the
    gather (``_fits_on_gpu``); else UVA, from a RAW home made for it when
    it has no RAW rep (``("raw_home", rep)``)."""
    uva = _uva_rep(src.ccr)
    streamed = _streamed_rep(src, applier)
    assert uva is not None or streamed is not None, (
        f"gather source {src.name} has no rep a gather can read")
    if uva is not None:
        if streamed is None or not uva.is_host_raw():
            return "uva", uva
        row_bytes, _, _, rows = _window(uva, out_alg)
        if (uva_bytes(src.ccr.col_length(), n_out, rows, row_bytes)
                < streamed.record_table().stored_bytes()):
            return "uva", uva
    table = streamed.record_table()
    if _fits_on_gpu(src.ccr.col_length() * table.row_bytes):
        return "streamed", streamed
    return ("uva", uva) if uva is not None else ("raw_home", streamed)


def _stream_in(src, rep, name, cpu_pool, applier):
    """Source ``src`` whole on the GPU, ``(rows, width)``: its rep ``rep``
    read through a tick pipe, each tick's decoded rows copied into one
    buffer allocated first, on the current stream.  The pipe has a node of
    its own with no output and no cache: the gather node advanced the cache
    already, and a gather's sources are host-resident."""
    from types import SimpleNamespace
    from IO.pipeline.vortex_pipeline import OutputLayout, VortexPipeline
    table = rep.record_table()
    n = src.ccr.col_length()
    buf = torch.empty((n, int(table.width)), dtype=table.stored_dtype,
                      device="cuda")
    filled = [0]

    def load(columns):
        t = columns[0]
        t = t.reshape(t.shape[0], -1)
        buf[filled[0]:filled[0] + t.shape[0]].copy_(t)
        filled[0] += t.shape[0]
        return []

    pipe = VortexPipeline(
        node=SimpleNamespace(name=f"{name} load {src.local}", outputs=[]),
        inputs=[(0, src)], layout=OutputLayout.flat(0),
        output_columns_type=[], operator=load, cpu_mem_pool=cpu_pool,
        applier=applier)
    pipe.do_exchange()
    pipe.get_result()
    assert filled[0] == n, (name, src.name, filled[0], n)
    return buf


def _raw_home(src, rep, name, cpu_pool, applier):
    """A RAW rep of ``src`` in pinned host memory, for a UVA gather of a
    compressed source too large to stream onto the GPU: a tick pipe decodes
    its rep ``rep`` (the read plan's, which the pipe selects through the
    applier as ``_streamed_rep`` does) and writes it home RAW."""
    from types import SimpleNamespace
    from IO.pipeline.vortex_pipeline import OutputLayout, VortexPipeline
    out = SimpleNamespace(next_use_op=0, expected_reads=1,
                          host_resident=True, ccr=None)
    pipe = VortexPipeline(
        node=SimpleNamespace(name=f"{name} raw home {src.local}",
                             outputs=[out]),
        inputs=[(0, src)], layout=OutputLayout.flat(1),
        output_columns_type=[rep.record_table().stored_dtype],
        operator=lambda columns: [columns[0]], cpu_mem_pool=cpu_pool,
        applier=applier, output_compression_algs=["RAW"])
    pipe.do_exchange()
    pipe.get_result()
    var = out.ccr.reps[0]
    var.tensor_type = rep.tensor_type
    return var


def _hbm_gather(buf, has_null):
    """The window gather of a streamed source: rows ``idx`` of the
    GPU-resident ``buf``; with ``has_null``, a ``constants.null`` index (a
    right-outer join's, or a lineage's through one) gives a null row, as
    the UVA kernel writes it."""
    import constants

    def gather(idx_chunk):
        if not has_null:
            return buf.index_select(0, idx_chunk)
        out = buf.index_select(0, idx_chunk.clamp(min=0))
        out.masked_fill_((idx_chunk < 0).unsqueeze(1), constants.null)
        return out
    return gather


def _stream_gather(col, idx, out_alg, name, cpu_pool,
                   peak_tracker=None, sizing_instance=None, gather=None):
    """Gather one column by ``idx`` and write it to the host: over UVA from
    its RAW rep ``col``, or with ``gather`` (``_hbm_gather``), from the
    source streamed onto the GPU (``col`` its rep, for the row's shape).

    Streams over output-row chunks: each iteration gathers a chunk and
    writes it out as regions (``out_alg``; RAW is a no-op codec;
    ``IO/codec/region_producer.py``): chunk k is compressed right after its
    gather and not waited for; chunk k+1's gather sync sees that compress
    end, so chunk k is then packed into one region and the home writer
    copies it home in the background; the drain packs the last chunk and
    waits for every copy.  Each gather first waits for the copies home
    still in flight, so the only earlier window it allocates beside is the
    previous window's worst-case slab, which the window's rows are charged
    for (``_window``).  The gather, the compress and the pack are
    timed by CUDA events only when a gather-sizing session reads the times;
    the iteration then also waits for the chunk's compress, its compress
    time holds its compress and the previous chunk's pack, and its d2h time
    the copies home that have ended.

    Sizing plane (gather-window leg): when a gather-sizing session is live the
    output-row budget is re-resolved from ``sizing_resolver`` **between**
    iterations (each iteration is internally synchronized before the next
    resolve), each iteration's sample — attributed to the exact window that
    built it — feeds the session, and the per-iteration arena peak is
    bracketed through ``peak_tracker``.  With no session (``--sizing_search
    off``, warmup, or a stream too small to search) the loop runs today's
    single up-front window, byte-identically.
    """
    from IO.codec.region_producer import RegionProducer
    from adaptive.sizing_controller import window_rows

    row_bytes, overhead, quantum, chunk_rows = _window(col, out_alg)
    # Under the gather-window search, its class floor: a window whose whole
    # quanta fall under it is rounded up a quantum, as the search's are.
    floor_b = (sizing_resolver.bounds("rearrange")[0]
               if has_gather_sizing_factory() else 0)

    n_out = idx.numel()
    session = None
    if has_gather_sizing_factory():
        session = make_gather_sizing_session(
            pipe_name=name, instance=sizing_instance or name,
            row_bytes=row_bytes, row_overhead=overhead,
            total_rows=n_out, codec_epoch=out_alg or "RAW",
            quantum_rows=quantum)
        if session is None:
            # Search armed but this stream is too small to host one: still
            # honor the resolver's static/config layer with a single
            # up-front resolve (the factory-off path above stays
            # byte-identical to today).
            chunk_rows = window_rows(sizing_resolver.resolve(
                "rearrange", sizing_instance or name).chunk_b,
                row_bytes + overhead, quantum, floor_b)
    if session is not None and peak_tracker is None:
        peak_tracker = _WindowPeakTracker()

    # The gather's, the compress's and the pack's times: the session's
    # samples read them, so without one they are not taken.
    timed = session is not None
    h2d_evt, cmp_evt, pack_evt = ((_event_pair(), _event_pair(),
                                   _event_pair()) if timed
                                  else (None, None, None))
    producer = RegionProducer(cpu_pool, name)
    try:
        lo = 0
        while lo < n_out:
            iter_t0 = time.perf_counter()
            if session is not None:
                # The window is live per iteration — the resolver is
                # the only budget channel; the session moved its override
                # between iterations (observe() below), never mid-flight.
                window_b = sizing_resolver.resolve(
                    "rearrange", session.instance).chunk_b
                chunk_rows = window_rows(window_b, row_bytes + overhead,
                                         quantum, floor_b)
                live_at_entry_b = peak_tracker.iter_begin()
            idx_chunk = idx[lo:lo + chunk_rows]

            # 1) Gather: the window's rows into a fresh GPU buffer.
            #    The previous windows' copies home end first, so the gather
            #    allocates beside one worst-case slab at most.
            producer.wait_copies()
            if timed:
                h2d_evt[0].record()
            if gather is not None:
                gpu_chunk = gather(idx_chunk)
            else:
                # The source's GPU regions stay put while the gather holds
                # their addresses (its output allocation may evict).
                with device_regions.pinned(col.record_table().regions):
                    gpu_chunk = _gather(col.uva_source(), idx_chunk)
            if timed:
                h2d_evt[1].record()
            torch.cuda.synchronize()
            if gpu_chunk.dim() == 1:
                gpu_chunk = gpu_chunk.unsqueeze(1)
            if gates.EXPERIMENT and gather is None:
                movement.record_uva_read(col, idx_chunk.numel() * row_bytes)

            # 2) The previous chunk's region: its compress has ended (the
            #    sync above), so it is packed and shipped home.  The
            #    chunks still to write a region, the packed one
            #    included.
            producer.ticks_left = 1 + -(-(n_out - lo) // chunk_rows)
            packed = producer.has_pending()
            if packed:
                if timed:
                    pack_evt[0].record()
                producer.pack()
                if timed:
                    pack_evt[1].record()
            # 3) This chunk's compress, not waited for (RAW: shipped as
            #    it is).
            if timed:
                cmp_evt[0].record()
            producer.compress(
                [(0, gpu_chunk, [gpu_chunk.shape[0]], out_alg)])
            if timed:
                cmp_evt[1].record()
            del gpu_chunk
            if timed:
                # The iteration ends synchronized: the session's sizing
                # boundary, and the events are readable.
                torch.cuda.synchronize()
                h2d_ms = h2d_evt[0].elapsed_time(h2d_evt[1])
                compress_ms = cmp_evt[0].elapsed_time(cmp_evt[1])
                if packed:
                    compress_ms += pack_evt[0].elapsed_time(pack_evt[1])
                d2h_ms = producer.collect_lane()[0]
            if session is not None:
                # The iteration is fully synchronized above — the sizing
                # boundary: the session may move its override and
                # request/release leases here and nowhere else.
                # iter_ms is a HOST bracket of the synchronized iteration:
                # unlike the CUDA legs it includes the per-iteration fixed
                # cost (sync latency, bookkeeping) that a
                # larger window amortizes.  In-run, like-for-like use only
                # (acceptance compares full windows within one stream) —
                # never a cross-run number (AGENTS.md timing rule; note it
                # inflates under --torch_profile like any host time).
                session.observe(GatherIterSample(
                    window_b=window_b,
                    rows=idx_chunk.numel(),
                    row_bytes=row_bytes,
                    chunk_bytes=idx_chunk.numel() * row_bytes,
                    h2d_ms=h2d_ms,
                    compress_ms=compress_ms,
                    d2h_ms=d2h_ms,
                    iter_peak_b=peak_tracker.iter_end(),
                    live_at_entry_b=live_at_entry_b,
                    iter_ms=(time.perf_counter() - iter_t0) * 1e3,
                ))
            lo += chunk_rows
        # The last chunk's pack and every copy home.
        producer.drain()
        table = producer.tables().get(0)
        if table is not None:
            return CombinedColumnRep([TableVariable(table, col.tensor_type)])
    except BaseException:
        # Wait for the copies in flight before their buffers and homes go.
        producer.abort()
        raise
    finally:
        if session is not None:
            session.close()

    return CombinedColumnRep([Variable(tensor_type=col.tensor_type)])


def rearrange_pipeline(
    node,
    cpu_mem_pool,
    type,
    *,
    applier = None,
    page_cache = None,
):
    """Run a gather node: a join's rearrange, a lineage compose or a deferred
    gather.  The node's inputs are its *sources* (every input but the last)
    and its *index* (the last); each source is gathered by the index and the
    result is written to the node's output instance at the same position.

    One gather per column: its source read over UVA or streamed onto the
    GPU, whichever moves fewer bytes (``_read_plan``), its output streamed
    in chunks, and output compression a per-column parameter (RAW or a
    codec) — not a code path.  Each output's codec comes off its instance
    through the applier (the fixed codec without an applier).

    The node always advances the cache's position, even when nothing is
    gathered: the graph holds exactly the pipelines that run.  A zero-length
    source is not gathered; its output is the source itself.  The lifetime
    rule: the gather frees its sources (it is their last consumer, which the
    builder asserts), and frees the index when it is the index's last
    consumer.
    """
    name = node.name
    perf_logger().start(name)
    # Bracket this rearrange's GPU-arena peak (index H2D + per-column streaming
    # gather/compress/spill).  Independent of the global OOM-diagnostic peak.
    cmp.reset_window_peak()
    if page_cache is not None:
        page_cache.advance(node)

    # 'type' is validated for early failure on an unexpected join kind.  A
    # right-outer index carries constants.null sentinels (unmatched rows),
    # and so does a lineage composed through one: the UVA kernel writes them
    # back as nulls (see _gather), a streamed source's gather too
    # (_hbm_gather), so no type-specific branch is needed here.
    assert type in ("inner", "right-semi", "right-anti", "right-outer"), (
        f"rearrange_pipeline: unexpected join type {type!r}"
    )
    *sources, index = node.inputs
    assert len(sources) == len(node.outputs), (name, len(sources), len(node.outputs))
    assert index.ccr is not None, f"{name}: index {index.name} has no payload"

    to_gather = []
    for src, out in zip(sources, node.outputs):
        assert src.ccr is not None, f"{name}: source {src.name} has no payload"
        if src.ccr.col_length() == 0:
            out.ccr = src.ccr
        else:
            to_gather.append((src, out))

    if to_gather:
        # Each column's chunks as host regions: the granule cache does not
        # hold a gather's outputs.
        # Decompress + H2D the index once; the gather requires a 1-D int64
        # tensor (it argsorts along the last dim).
        idx = col_to_gpu(index.ccr).long().reshape(-1)
        has_null = None
        # One peak tracker for the whole pipe: per-iteration brackets (gather
        # sizing feedback) and the streamed loads compose into the
        # pipe-level peak_gpu_bytes record.
        peak_tracker = _WindowPeakTracker()
        for src, out in to_gather:
            out_alg = applier.codec(out) if applier is not None else get_fixed_compression()
            how, rep = _read_plan(src, idx.numel(), out_alg, applier)
            message_logger().debug(
                "[gather] %s %s: %s %s", name, src.name, how,
                rep.first_codec_name() or "RAW")
            gather = None
            if how == "streamed":
                if has_null is None:
                    has_null = bool((idx < 0).any())
                peak_tracker.iter_begin()
                gather = _hbm_gather(
                    _stream_in(src, rep, name, cpu_mem_pool, applier),
                    has_null)
            elif how == "raw_home":
                peak_tracker.iter_begin()
                rep = _raw_home(src, rep, name, cpu_mem_pool, applier)
            result = _stream_gather(
                rep, idx, out_alg, name, cpu_mem_pool,
                peak_tracker=peak_tracker, sizing_instance=f"{name}:{out.local}",
                gather=gather)
            del gather
            if how == "raw_home":
                rep.free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=None)
            # Free the source's record tables (every rep) now that the gather
            # has consumed them; the pinned pool only reclaims via an
            # explicit free (no __del__).
            src.ccr.free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=None)
            out.ccr = result
        # final_peak() == cmp.get_window_peak_bytes() when no sizing session
        # ran (the tracker was never advanced) — the pre-sizing read.
        if gates.EXPERIMENT:
            perf_logger().record(name, peak_gpu_bytes=peak_tracker.final_peak())
    else:
        message_logger().debug("No cols for rearrange")

    perf_logger().stop(name)

    # The lifetime rule: the index is freed after its last consumer.
    if index.consumers and index.consumers[-1] is node:
        index.ccr.free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=None)
