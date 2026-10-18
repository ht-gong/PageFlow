"""Regions for the direct pipes, the gathers and the zero-copy stages.

The join core, the aggregation passes and the project write their outputs
through the region producer (``IO/codec/region_producer.py``), and the
gathers write each chunk as a region (``IO/pipeline/rearrange_pipeline.
py``; over UVA here, ``test_gather_reads.py`` covers the streamed read).
The regions must give the rows: pipes shaped like each direct pipe
(per-tick outputs, a ``div`` geometry with fan-out, outputs only at the
last tick, a RAW output that is a view of the tick's input), gathers over a
scan table and over a producer's table, compressed and RAW.  A gather chunk
waits once, for its gather.  The zero-copy stages emit their output as a GPU region, and such a region stays
held while a copy home reads it.  ``compute-sanitizer`` runs this file for
illegal accesses (it sees the arena as one allocation, so the data checks
catch a buffer reused too early)."""
from __future__ import annotations

import pytest
import torch

import constants
from IO.codec import region_producer
from IO.pages.combined_column_rep import CombinedColumnRep
from utility.logger import set_message_logger, set_perf_logger
from variable import TableVariable, Variable

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")


# ── GPU: pipes, gathers and zero-copy stages ─────────────────────────────

@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


@pytest.fixture
def engine(monkeypatch):
    from IO.transfer.data_exchange import set_exchange_to_naive
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()


def _pinned_column(pool, raw_2d, tensor_type="int"):
    """A scanned column: the scan's RAW record table of ``raw_2d`` in
    ``pool``."""
    from tests.scan_table import scan_table
    return CombinedColumnRep([scan_table(
        raw_2d.contiguous(), tensor_type, pool, "RAW")])


def _node(kind, name, cols, n_out):
    """A plan with a scan producing ``cols`` and one ``kind`` node reading
    them, with ``n_out`` output instances."""
    from dataflow.dataflow_gen import ColumnInstance, PipeNode, PlanGraph
    plan = PlanGraph(0)
    scan = PipeNode(op_id=0, kind="scan", name="scan 0", plan=plan)
    node = PipeNode(op_id=1, kind=kind, name=name, plan=plan)
    plan.nodes.extend([scan, node])
    for key, ccr in cols.items():
        inst = ColumnInstance(producer=scan, local=key, ccr=ccr)
        scan.outputs.append(inst)
        node.inputs.append(inst)
        node.input_keys.append(key)
        inst.consumers.append(node)
    for i in range(n_out):
        node.outputs.append(ColumnInstance(producer=node, local=100 + i))
    plan.assign_positions()
    return node


def _run_pipe(pool, node, op, out_types, algs, **kw):
    from IO.pipeline.vortex_pipeline import VortexPipeline
    pipe = VortexPipeline(
        node=node, inputs=node.input_pairs(), output_columns_type=out_types,
        operator=op, cpu_mem_pool=pool,
        output_compression_algs=algs, **kw)
    pipe.do_exchange()
    pipe.get_result()
    return pipe


def _host(var):
    """The column's rows, decoded, on the host."""
    from IO.pipeline.common import col_to_gpu
    return col_to_gpu(CombinedColumnRep([var])).cpu()


def _late_copies(producer, monkeypatch, ms=200):
    """Every copy home of ``producer`` waits ``ms`` of GPU time after its
    real event, so the copies are in flight while the pipe goes on."""
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
def test_a_project_writes_regions(cuda_allocator, pool, engine, monkeypatch):
    """Per-tick outputs, one compressed and one RAW, over many ticks."""
    rows = 300_000
    x = torch.arange(rows, dtype=constants.int_dtype).unsqueeze(1) % 1000
    node = _node("project", "project 1", {0: _pinned_column(pool, x)}, 2)
    pipe = _run_pipe(
        pool, node,
        lambda d: [d[0] * 3 + 1, d[0].to(constants.float_dtype) / 2],
        [constants.int_dtype, constants.float_dtype], ["BITCOMP", "RAW"],
        chunk_size=256 * 1024)
    assert pipe.stages > 6
    a = pipe.output_variable(0, "int")
    b = pipe.output_variable(1, "float")
    assert a.table_backed and b.table_backed
    assert node.outputs[0].ccr.reps[0] is a       # published, reused
    assert a.record_table().is_compressed
    assert not b.record_table().is_compressed
    assert torch.equal(_host(a), (x * 3 + 1).squeeze(1))
    assert torch.equal(_host(b), (x.double() / 2).to(
        constants.float_dtype).squeeze(1))
    a.free_underlying_mem(pool, None)
    b.free_underlying_mem(pool, None)


@requires_cuda
def test_a_join_core_geometry_writes_regions(cuda_allocator, pool, engine,
                                             monkeypatch):
    """A ``div`` geometry of uneven ticks whose outputs fan out (each row
    twice), an index column RAW and a value column compressed."""
    rows = 200_000
    x = (torch.arange(rows, dtype=constants.int_dtype) * 7 % 501).unsqueeze(1)
    bounds = [10_000, 70_000, 71_000, 150_000, rows]
    node = _node("core", "join 1 core pipeline", {0: _pinned_column(pool, x)}, 2)

    def core(d):
        v = d[0]
        return [torch.repeat_interleave(v, 2),
                torch.repeat_interleave(
                    torch.arange(v.numel(), device=v.device), 2)]
    pipe = _run_pipe(pool, node, core,
                     [constants.int_dtype, constants.int_dtype],
                     ["CASCADED", "RAW"], div=[bounds])
    vals = pipe.output_variable(0, "int")
    idx = pipe.output_variable(1, "int")
    assert vals.table_backed and idx.table_backed
    assert len(vals.record_table().regions) == len(bounds)
    want_idx = torch.cat([torch.arange(hi - lo).repeat_interleave(2)
                          for lo, hi in zip([0] + bounds[:-1], bounds)])
    assert torch.equal(_host(vals), torch.repeat_interleave(x.squeeze(1), 2))
    assert torch.equal(_host(idx), want_idx)


@requires_cuda
def test_aggregation_passes_write_regions(cuda_allocator, pool, engine,
                                          monkeypatch):
    """A pass-1 shape (per-tick partials, a width-2 packed state) and a
    pass-2 shape (nothing until the last tick, then the state), as the
    aggregation passes emit them."""
    rows = 250_000
    x = (torch.arange(rows, dtype=constants.int_dtype) % 97).unsqueeze(1)
    node = _node("pass1", "aggregation 2 pass1", {0: _pinned_column(pool, x)}, 2)

    def pass1(d):
        v = d[0]
        u, c = torch.unique(v, return_counts=True)
        return [u, torch.stack([u * 2, c], 1)]
    p1 = _run_pipe(pool, node, pass1,
                   [constants.int_dtype, constants.int_dtype],
                   ["BITCOMP", "ANS"], chunk_size=512 * 1024)
    keys = p1.output_variable(0, "int")
    state = p1.output_variable(1, "int")
    assert keys.table_backed and state.table_backed
    per = (512 * 1024) // 8          # rows per tick: the chunk / row bytes
    want_k, want_s = [], []
    for lo in range(0, rows, per):
        u, c = torch.unique(x[lo:lo + per].squeeze(1), return_counts=True)
        want_k.append(u)
        want_s.append(torch.stack([u * 2, c], 1))
    assert torch.equal(_host(keys), torch.cat(want_k))
    assert torch.equal(_host(state), torch.cat(want_s))

    # Pass 2 reads pass 1's columns (tables with regions) and emits only at
    # its last tick.
    node2 = _node("pass2", "aggregation 2 pass2",
                  {0: CombinedColumnRep([keys])}, 1)
    seen = []

    def pass2(d):
        seen.append(d[0])
        if len(seen) < p2_ticks[0]:
            return []
        return [torch.unique(torch.cat(seen))]
    p2_ticks = [0]
    from IO.pipeline.vortex_pipeline import VortexPipeline
    p2 = VortexPipeline(
        node=node2, inputs=node2.input_pairs(),
        output_columns_type=[constants.int_dtype], operator=pass2,
        cpu_mem_pool=pool,
        output_compression_algs=["BITCOMP"], chunk_size=1024)
    p2_ticks[0] = p2.stages - 2
    assert p2_ticks[0] > 2
    p2.do_exchange()
    p2.get_result()
    out = p2.output_variable(0, "int")
    assert out.table_backed
    assert torch.equal(_host(out), torch.arange(97))


@requires_cuda
def test_a_raw_view_of_the_input_is_held_while_its_copy_waits(
        cuda_allocator, pool, engine, monkeypatch):
    """An operator that returns its input (a view of the tick's input
    buffer) as a RAW output: the copy home holds the buffer, so the next
    ticks' inputs, allocated while every copy waits, do not write over it."""
    rows = 400_000
    x = (torch.arange(rows, dtype=constants.int_dtype) * 13 % 7919) \
        .unsqueeze(1)
    node = _node("project", "project 2", {0: _pinned_column(pool, x)}, 1)
    from IO.pipeline.vortex_pipeline import VortexPipeline
    pipe = VortexPipeline(
        node=node, inputs=node.input_pairs(),
        output_columns_type=[constants.int_dtype], operator=lambda d: [d[0]],
        cpu_mem_pool=pool,
        output_compression_algs=["RAW"], chunk_size=512 * 1024)
    _late_copies(pipe._VortexPipeline__producer, monkeypatch)
    pipe.do_exchange()
    pipe.get_result()
    out = pipe.output_variable(0, "int")
    assert out.table_backed
    assert torch.equal(_host(out), x.squeeze(1))


def _gather_node(src_ccr, idx_ccr, name="gather 3"):
    from dataflow.dataflow_gen import ColumnInstance, PipeNode, PlanGraph
    plan = PlanGraph(0)
    scan = PipeNode(op_id=0, kind="scan", name="scan 0", plan=plan)
    core = PipeNode(op_id=1, kind="core", name="core 1", plan=plan)
    node = PipeNode(op_id=2, kind="rearrange", name=name, plan=plan)
    plan.nodes.extend([scan, core, node])
    src = ColumnInstance(producer=scan, local=7, ccr=src_ccr)
    index = ColumnInstance(producer=core, local="out_index_left",
                           ccr=idx_ccr)
    for inst, producer in ((src, scan), (index, core)):
        producer.outputs.append(inst)
        node.inputs.append(inst)
        node.input_keys.append(inst.local)
        inst.consumers.append(node)
    node.outputs.append(ColumnInstance(producer=node, local=7))
    plan.assign_positions()
    return node


@requires_cuda
@pytest.mark.parametrize("codec", ["BITCOMP", "RAW"])
def test_gather_chunks_write_regions(cuda_allocator, pool, engine,
                                     monkeypatch, codec):
    """A gather over several chunks, then a gather of its output (a table
    when the first wrote regions): both give the gathered rows, and a
    region chunk waits for its gather only."""
    from IO.pipeline.rearrange_pipeline import rearrange_pipeline
    from IO.pipeline import rearrange_pipeline as rp_mod
    monkeypatch.setattr(constants, "get_chunk_size_b", lambda: 1 << 20)
    rows, n_out = 1 << 17, 300_000
    g = torch.Generator().manual_seed(11)
    data = torch.randint(0, 1000, (rows, 1), dtype=torch.int64, generator=g)
    idx = torch.randint(0, rows, (n_out, 1), dtype=torch.int64, generator=g)
    perm = torch.randperm(n_out, generator=g).unsqueeze(1)
    gathers, syncs = [0], [0]
    real_gather, real_sync = rp_mod._gather, torch.cuda.synchronize
    producers = []

    class Producer(region_producer.RegionProducer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            producers.append(self)
    monkeypatch.setattr(region_producer, "RegionProducer", Producer)

    def gather(*a):
        gathers[0] += 1
        # No copy home in flight: a gather allocates beside the previous
        # window's slab at most.
        assert all(not p.writer._ships for p in producers)
        return real_gather(*a)

    def sync(*a, **k):
        syncs[0] += 1
        return real_sync(*a, **k)
    monkeypatch.setattr(rp_mod, "_gather", gather)
    monkeypatch.setattr(torch.cuda, "synchronize", sync)

    # The gathers read over UVA (a scan source takes the cheaper read; this
    # one's windows are dense enough to be streamed otherwise).
    monkeypatch.setattr(rp_mod, "uva_bytes", lambda *a: 0)
    applier = _FixedCodec(codec)
    first = _gather_node(_pinned_column(pool, data), _pinned_column(pool, idx))
    rearrange_pipeline(first, pool, "inner", applier=applier)
    out = first.outputs[0].ccr.reps[0]
    assert gathers[0] > 3
    # The window: the chunk budget over the row, its argsort scratch and,
    # with regions, the previous window's worst-case slab.
    from IO.pipeline.rearrange_pipeline import _ARGSORT_OVERHEAD_PER_ROW
    per_row = 8 + _ARGSORT_OVERHEAD_PER_ROW + \
        region_producer.slab_bytes_per_row(codec, 8, 8)
    assert gathers[0] == -(-n_out // ((1 << 20) // per_row))
    assert out.table_backed
    assert len(out.record_table().regions) == gathers[0]
    # One sync per chunk (its gather's) and one for the index load.
    assert syncs[0] == gathers[0] + 1
    assert torch.equal(_host(out), data[idx.squeeze(1)].squeeze(1))
    # The output gathered again (RAW, so UVA-readable): from its table's
    # regions when it has one.
    gathers[0] = 0
    second = _gather_node(first.outputs[0].ccr, _pinned_column(pool, perm),
                          name="gather 4")
    if codec == "RAW":
        rearrange_pipeline(second, pool, "inner", applier=_FixedCodec("RAW"))
        got = _host(second.outputs[0].ccr.reps[0])
        assert torch.equal(got, data[idx.squeeze(1)][perm.squeeze(1)]
                           .squeeze(1))


def _no_mask():
    """A mask column with nothing to read: the test sets the stage's GPU
    mask itself."""
    return Variable(tensor_type="bool")


class _FixedCodec:
    """An applier that gives every output instance one codec."""

    def __init__(self, codec):
        self._codec = codec

    def codec(self, inst):
        return self._codec


@requires_cuda
def test_a_zero_copy_mask_stage_reads_a_table_and_emits_a_gpu_region(
        cuda_allocator, pool, engine, monkeypatch):
    """The mask stage reads a table's RAW host records over UVA (a tick
    slice of it) and emits its rows as one RAW record of a GPU region."""
    from IO.codec.region_producer import RegionProducer
    from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
    rows = 120_000
    x = (torch.arange(rows, dtype=constants.int_dtype) * 3 % 1013) \
        .unsqueeze(1)
    producer = RegionProducer(pool, "t 0 project")
    for lo in range(0, rows, 40_000):
        if producer.has_pending():
            producer.pack()
        chunk = x[lo:lo + 40_000].to("cuda:0")
        producer.compress([(0, chunk, [chunk.shape[0]], "RAW")])
        torch.cuda.synchronize()
    producer.drain()
    src = TableVariable(producer.tables()[0], "int")
    tick = src.get_sliced_rep(slice(30_000, 90_000))
    mask = (torch.arange(rows) % 5 == 0)
    stage = ZeroCopyMaskIn(tick, _no_mask(), 1)
    stage.mask_gpu = mask[30_000:90_000].to("cuda:0")
    stage.mask_count = int(stage.mask_gpu.sum())
    stage.execute_zero_copy()
    col = stage.gpu_col
    assert col.table_backed
    region = col.record_table().regions[0]
    assert region.location == "device" and col.col_length() == 12_000
    want = x[30_000:90_000][mask[30_000:90_000]].squeeze(1)
    assert torch.equal(col.data().cpu(), want)


@requires_cuda
def test_a_zero_copy_region_is_held_while_its_copy_home_waits(
        cuda_allocator, pool, engine, monkeypatch):
    """A zero-copy stage's GPU region shipped RAW (an operator passing it
    through): the stage drops its column, GPU memory is allocated and
    written while the copy waits, and the home still gets the rows."""
    from IO.codec.region_producer import RegionProducer
    from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
    rows = 1 << 18
    x = (torch.arange(rows, dtype=constants.int_dtype) * 11 % 4099) \
        .unsqueeze(1)
    src = _pinned_column(pool, x).reps[0]
    producer = RegionProducer(pool, "t 1 project")
    _late_copies(producer, monkeypatch, ms=300)
    want = []
    for lo in range(0, rows, rows // 4):
        tick = src.get_sliced_rep(slice(lo, lo + rows // 4))
        mask = torch.ones(rows // 4, dtype=torch.bool)
        mask[::3] = False
        stage = ZeroCopyMaskIn(tick, _no_mask(), 1)
        stage.mask_gpu = mask.to("cuda:0")
        stage.mask_count = int(mask.sum())
        stage.execute_zero_copy()
        want.append(x[lo:lo + rows // 4][mask].squeeze(1))
        if producer.has_pending():
            producer.pack()
        out = stage.gpu_col.data()
        producer.compress([(0, out, [out.shape[0]], "RAW")])
        stage.gpu_col = None
        del out
        for _ in range(4):
            scribble = torch.full((8 << 20,), 0x5A, dtype=torch.uint8,
                                  device="cuda:0")
            del scribble
        torch.cuda.current_stream().synchronize()
    producer.drain()
    got = TableVariable(producer.tables()[0], "int").col_to_gpu().cpu()
    assert torch.equal(got, torch.cat(want))
