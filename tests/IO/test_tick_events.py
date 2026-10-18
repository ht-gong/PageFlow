"""Event-ordered ticks against the tick barrier.

``VortexPipeline`` ends every tick at the tick barrier (a host sync of the
compute and exchange streams) while a search of the pipe is live (a tick
observer, until it settles); otherwise, with or without a granule cache,
its ticks are ordered by events: the host issues tick i+1 while the GPU
still computes tick i, and the buffers the compute stream reads stay on the
tick's hold list until the tick that computed them has ended
(``IO/pipeline/vortex_pipeline.py``, ``_event_ticks``).  A pipe whose
searches settle mid-pipe hands its remaining ticks over to events.  Bare
pipelines over synthetic tables run under the three paths (the barrier one
built with an explicit observer, the hand-over one with an installed
observer that settles after a few ticks): a filter with a DirectIn mask
(RAW or compressed), a zero-copy mask, a folded mask (``fold_mask``: the
operator gets the rows uncompacted and the mask's count), compressed and
RAW outputs, a ``div`` pipe with an empty tick.  Their outputs decode to the same rows (the compressed bytes
differ run to run).  An operator that first sleeps on the compute stream
keeps the host ahead of the GPU, so every free of a held buffer is
exercised.  ``compute-sanitizer`` runs this file for illegal accesses (it
sees the arena as one allocation, so the data checks catch a buffer reused
too early)."""
from __future__ import annotations

import weakref

import pytest
import torch

import constants
from IO.codec import region_producer
from IO.pages.combined_column_rep import CombinedColumnRep
from utility.logger import set_message_logger, set_perf_logger

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")

ARMS = pytest.mark.parametrize("mode", ["barrier", "events", "handover"])

# The hand-over arm's observer settles after this many views: the barrier
# runs the prologue, tick 1 and a steady tick before the events take over.
SETTLE_AFTER = 3


@pytest.fixture
def pool():
    from IO.pages.pinned_mem import PinnedMemory
    return PinnedMemory(capacity_gb=1, block_size_mb=0.5)


@pytest.fixture
def engine(monkeypatch):
    """64 KiB records (8192 rows of an 8-byte column, 65536 of a mask), so
    a pipe of a few hundred thousand rows has many whole-record ticks; a
    mask's zero-copy threshold at 0.1, as ``main.py`` sets it."""
    from adaptive import instage_optimization
    from IO import data_scanner
    from IO.transfer.data_exchange import set_exchange_to_naive
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()
    monkeypatch.setattr(data_scanner, "COMPRESSION_BATCH_SIZE_B", 1 << 20)
    monkeypatch.setattr(region_producer, "RECORD_MB", 1 / 16)
    monkeypatch.setattr(instage_optimization, "_global_mask_selectivity",
                        0.1)


def _column(pool, x, tensor_type, codecs):
    """A scanned column with one representation per codec."""
    from tests.scan_table import scan_table
    return CombinedColumnRep([
        scan_table(x.contiguous(), tensor_type, pool, c)
        for c in codecs])


def _mask(pool, m, codecs):
    """A mask column (``m``, 1-D bool) with its selectivity."""
    ccr = _column(pool, m.unsqueeze(1), "bool", codecs)
    for rep in ccr.reps:
        rep.selectivity = float(m.float().mean())
    return ccr


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


def _settling_factory(after):
    """An observer factory whose observers return no picks and settle
    after ``after`` views."""
    def factory(pipe_name, menu):
        views = [0]

        def observer(view):
            views[0] += 1
            return None
        observer.settled = lambda: views[0] >= after
        return observer
    return factory


def _pipe(pool, node, op, out_types, algs, mode, **kw):
    """``mode``: "barrier" (an explicit observer keeps the barrier on every
    tick), "events" (none), "handover" (an installed observer that settles
    after ``SETTLE_AFTER`` views; the factory is read at construction)."""
    from IO.pipeline.vortex_pipeline import VortexPipeline
    from adaptive import adaptive_driver
    with pytest.MonkeyPatch.context() as mp:
        if mode == "barrier":
            kw["tick_observer"] = lambda view: None
        elif mode == "handover":
            mp.setattr(adaptive_driver, "_tick_observer_factory",
                       _settling_factory(SETTLE_AFTER))
        pipe = VortexPipeline(
            node=node, inputs=node.input_pairs(),
            output_columns_type=out_types, operator=op, cpu_mem_pool=pool,
            output_compression_algs=algs, **kw)
    return pipe


def _run(pipe):
    pipe.do_exchange()
    pipe.get_result()
    return pipe


def _events(pipe):
    return pipe._VortexPipeline__events


def _host(var):
    """The column's rows, decoded, on the host."""
    from IO.pipeline.common import col_to_gpu
    return col_to_gpu(CombinedColumnRep([var])).cpu()


def _filter_inputs(pool, rows, mask_codecs, data_codecs=("RAW", "ANS")):
    """Data ``x`` and a mask ``m`` whose density changes every 65536 rows
    (a stale mask count gives the wrong rows)."""
    r = torch.arange(rows)
    x = ((r * 7) % 1009).to(constants.int_dtype).unsqueeze(1)
    m = ((r % 3) == 0) | (((r // 65536) % 2) == 0)
    return x, m, {0: _column(pool, x, "int", data_codecs),
                  1: _mask(pool, m, mask_codecs)}


@requires_cuda
@ARMS
@pytest.mark.parametrize("mask_codecs", [("RAW",), ("RAW", "ANS")],
                         ids=["raw-mask", "ans-mask"])
def test_a_filter_with_a_direct_mask(cuda_allocator, pool, engine,
                                     monkeypatch, mode, mask_codecs):
    """A DirectIn mask (summed on the h2d stream; a compressed one decoded
    there too) over compressed data; one compressed and one RAW output."""
    from adaptive import instage_optimization
    monkeypatch.setattr(instage_optimization, "_global_fixed_compression",
                        "ANS")
    rows = 400_000
    x, m, cols = _filter_inputs(pool, rows, mask_codecs)
    node = _node("filter", "filter 1 filter pipe", cols, 2)
    pipe = _pipe(pool, node, lambda d: [d[0] * 2 + 1, d[0]],
                 [constants.int_dtype, constants.int_dtype],
                 ["BITCOMP", "RAW"], mode,
                 mask_map={0: 1}, chunk_size=1 << 20)
    assert _events(pipe) == (mode == "events")
    _run(pipe)
    assert _events(pipe) == (mode != "barrier")
    assert pipe.stages > 6
    want = x[m].squeeze(1)
    assert torch.equal(_host(pipe.output_variable(0, "int")), want * 2 + 1)
    assert torch.equal(_host(pipe.output_variable(1, "int")), want)


@requires_cuda
@ARMS
def test_a_zero_copy_mask(cuda_allocator, pool, engine, monkeypatch,
                          mode):
    """RAW host data under a sparse mask: the zero-copy kernel reads the
    rows over UVA with the mask decoded on the h2d stream."""
    rows = 400_000
    r = torch.arange(rows)
    x = ((r * 11) % 4099).to(constants.int_dtype).unsqueeze(1)
    m = ((r % 13) == 0) & (((r // 65536) % 3) != 1)
    cols = {0: _column(pool, x, "int", ("RAW",)), 1: _mask(pool, m, ("RAW",))}
    node = _node("filter", "filter 2 filter pipe", cols, 2)
    pipe = _pipe(pool, node, lambda d: [d[0] + 5, d[0]],
                 [constants.int_dtype, constants.int_dtype],
                 ["BITCOMP", "RAW"], mode,
                 mask_map={0: 1}, chunk_size=1 << 20)
    from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
    assert isinstance(pipe._VortexPipeline__tick_stages[0].in_stage[0],
                      ZeroCopyMaskIn)
    _run(pipe)
    assert pipe.stages > 6
    want = x[m].squeeze(1)
    assert torch.equal(_host(pipe.output_variable(0, "int")), want + 5)
    assert torch.equal(_host(pipe.output_variable(1, "int")), want)


@requires_cuda
@ARMS
@pytest.mark.parametrize("mask_codecs", [("RAW",), ("RAW", "ANS")],
                         ids=["raw-mask", "ans-mask"])
@pytest.mark.parametrize("keys", [(0, 1), (20, 10)], ids=["keys-0-1",
                                                          "keys-20-10"])
def test_a_folded_mask_reaches_the_operator_raw(cuda_allocator, pool, engine,
                                                monkeypatch, mode,
                                                mask_codecs, keys):
    """``fold_mask``: a DirectIn column reaches the operator uncompacted,
    with the mask and the mask's kept rows, and the operator applies the
    mask itself.  The counts go by the mask's key, which need not be its
    position (keys 20 and 10: the mask comes first)."""
    from adaptive import instage_optimization
    monkeypatch.setattr(instage_optimization, "_global_fixed_compression",
                        "ANS")
    rows = 400_000
    x, m, by_pos = _filter_inputs(pool, rows, mask_codecs)
    dk, mk = keys
    cols = {dk: by_pos[0], mk: by_pos[1]}
    cols = dict(sorted(cols.items(), key=lambda kv: kv[0] != min(keys)))
    seen = []

    def op(d, counts):
        data, mask = d[dk], d[mk]
        seen.append(data.shape[0] == mask.shape[0]
                    and set(counts) == {mk} and counts[mk] == int(mask.sum()))
        kept = data[mask]
        return [kept * 2 + 1, kept]

    node = _node("filter", "filter 3 filter pipe", cols, 2)
    pipe = _pipe(pool, node, op, [constants.int_dtype, constants.int_dtype],
                 ["BITCOMP", "RAW"], mode, mask_map={dk: mk},
                 pass_mask=True, fold_mask=True, chunk_size=1 << 20)
    _run(pipe)
    assert pipe.stages > 6
    assert len(seen) == pipe.stages - 2 and all(seen)
    want = x[m].squeeze(1)
    assert torch.equal(_host(pipe.output_variable(0, "int")), want * 2 + 1)
    assert torch.equal(_host(pipe.output_variable(1, "int")), want)


@requires_cuda
@ARMS
def test_a_folded_mask_leaves_a_zero_copy_read_compacted(
        cuda_allocator, pool, engine, monkeypatch, mode):
    """Under ``fold_mask`` a zero-copy column still arrives compacted (its
    read applies the mask), holding the mask's kept rows."""
    rows = 400_000
    r = torch.arange(rows)
    x = ((r * 11) % 4099).to(constants.int_dtype).unsqueeze(1)
    m = ((r % 13) == 0) & (((r // 65536) % 3) != 1)
    cols = {0: _column(pool, x, "int", ("RAW",)), 1: _mask(pool, m, ("RAW",))}
    seen = []

    def op(d, counts):
        seen.append(d[0].shape[0] == counts[1] == int(d[1].sum()))
        return [d[0] + 5, d[0]]

    node = _node("filter", "filter 4 filter pipe", cols, 2)
    pipe = _pipe(pool, node, op, [constants.int_dtype, constants.int_dtype],
                 ["BITCOMP", "RAW"], mode, mask_map={0: 1}, pass_mask=True,
                 fold_mask=True, chunk_size=1 << 20)
    from IO.transfer.zero_copy_stage import ZeroCopyMaskIn
    assert isinstance(pipe._VortexPipeline__tick_stages[0].in_stage[0],
                      ZeroCopyMaskIn)
    _run(pipe)
    assert len(seen) == pipe.stages - 2 and all(seen)
    want = x[m].squeeze(1)
    assert torch.equal(_host(pipe.output_variable(0, "int")), want + 5)
    assert torch.equal(_host(pipe.output_variable(1, "int")), want)


@requires_cuda
@ARMS
def test_a_div_pipe_with_an_empty_tick(cuda_allocator, pool, engine,
                                       monkeypatch, mode):
    """A ``div`` geometry of uneven buckets, one of them empty (the
    operator's short-circuit, which still packs in events mode), fanning
    each row out twice; a compressed and a RAW output."""
    rows = 200_000
    x = (torch.arange(rows, dtype=constants.int_dtype) * 7 % 501) \
        .unsqueeze(1)
    bounds = [10_000, 70_000, 70_000, 150_000, rows]
    node = _node("core", "join 1 core pipeline",
                 {0: _column(pool, x, "int", ("RAW",))}, 2)
    calls = []

    def core(d):
        v = d[0]
        calls.append(v.numel())
        return [torch.repeat_interleave(v, 2),
                torch.repeat_interleave(
                    torch.arange(v.numel(), device=v.device), 2)]
    pipe = _run(_pipe(pool, node, core,
                      [constants.int_dtype, constants.int_dtype],
                      ["CASCADED", "RAW"], mode,
                      div=[bounds]))
    assert 0 not in calls and len(calls) == len(bounds) - 1
    want_idx = torch.cat([torch.arange(hi - lo).repeat_interleave(2)
                          for lo, hi in zip([0] + bounds[:-1], bounds)])
    assert torch.equal(_host(pipe.output_variable(0, "int")),
                       torch.repeat_interleave(x.squeeze(1), 2))
    assert torch.equal(_host(pipe.output_variable(1, "int")), want_idx)


@requires_cuda
@pytest.mark.parametrize("out_algs", [("RAW", "RAW"), ("BITCOMP", "ANS")],
                         ids=["raw-out", "compressed-out"])
def test_the_host_ahead_frees_held_buffers_after_their_tick(
        cuda_allocator, pool, engine, monkeypatch, out_algs):
    """An operator that first sleeps on the compute stream keeps the GPU
    behind the host, which then waits on the tick event two ticks old.
    The rows equal the barrier arm's; every held buffer dies only once the
    tick that computed its chunk has ended; at most three ticks' holds are
    alive at once; each mask count is read once its tick's input issue has
    ended."""
    from adaptive import instage_optimization
    from IO.pipeline.vortex_pipeline import VortexPipeline
    from IO.transfer.data_exchange import NaiveExchange
    monkeypatch.setattr(instage_optimization, "_global_fixed_compression",
                        "ANS")
    rows = 600_000
    x, m, cols = _filter_inputs(pool, rows, ("RAW", "ANS"))

    def slow(d):
        torch.cuda._sleep(int(30 * 1.5e6))
        v = d[0]
        return [v * 3, v - 1]

    results = {}
    for barrier in (True, False):
        mp = pytest.MonkeyPatch()
        node = _node("filter", f"filter {3 + barrier} filter pipe",
                     dict(cols), 2)
        pipe = _pipe(pool, node, slow,
                     [constants.int_dtype, constants.int_dtype],
                     list(out_algs), "barrier" if barrier else "events",
                     mask_map={0: 1}, chunk_size=1 << 20)
        alive, deaths, peaks, unfired, counts_read = {}, [], [], [], []
        if not barrier:
            t_events = pipe._VortexPipeline__t_events
            h_events = pipe._VortexPipeline__h_events
            real_launch = NaiveExchange.launch
            issued = [0]

            def died(k):
                # Chunk k is computed at tick k + 1.
                deaths.append((k, t_events[(k + 1) % 3].query()))
                alive[k] -= 1

            def launch(self, stages, bound=None, holds=None):
                out = real_launch(self, stages, bound, holds)
                if holds is not None:
                    k = issued[0]
                    issued[0] += 1
                    alive[k] = len(holds)
                    for t in holds:
                        weakref.finalize(t, died, k)
                    peaks.append(sum(1 for v in alive.values() if v > 0))
                return out
            mp.setattr(NaiveExchange, "launch", launch)

            real_wait = VortexPipeline._tick_wait

            def tick_wait(self, i, released):
                unfired.append(not t_events[(i - 2) % 3].query())
                return real_wait(self, i, released)
            mp.setattr(VortexPipeline, "_tick_wait", tick_wait)

            real_read = VortexPipeline._read_mask_counts

            def read_counts(self, tick):
                real_read(self, tick)
                counts_read.append(h_events[(tick - 1) % 3].query())
            mp.setattr(VortexPipeline, "_read_mask_counts", read_counts)
        assert _events(pipe) == (not barrier)
        try:
            _run(pipe)
        finally:
            mp.undo()
        assert pipe.stages > 8
        results[barrier] = [_host(pipe.output_variable(i, "int"))
                            for i in range(2)]
        if not barrier:
            assert alive and all(n == 0 for n in alive.values())
            assert deaths and all(fired for _, fired in deaths)
            assert max(peaks) <= 3
            assert counts_read and all(counts_read)
            if out_algs[0] == "RAW":
                # Nothing to pack: only the tick event holds the host.
                assert any(unfired)
    want = x[m].squeeze(1)
    assert torch.equal(results[True][0], want * 3)
    assert torch.equal(results[True][1], want - 1)
    for a, b in zip(results[True], results[False]):
        assert torch.equal(a, b)


def _count_syncs(monkeypatch):
    from IO.transfer.data_exchange import NaiveExchange
    n = [0]
    real = NaiveExchange.sync

    def sync(self):
        n[0] += 1
        return real(self)
    monkeypatch.setattr(NaiveExchange, "sync", sync)
    return n


@requires_cuda
def test_a_sizing_search_alone_leaves_a_pipe_on_events(
        cuda_allocator, pool, engine, monkeypatch):
    """A tick sizing factory installed: a pipe without a session (here none:
    it has no sizing key) and without an observer orders its ticks by
    events and syncs the exchange once."""
    from adaptive import adaptive_driver
    monkeypatch.setattr(adaptive_driver, "_tick_sizing_factory",
                        lambda *a, **k: None)
    rows = 300_000
    x, m, cols = _filter_inputs(pool, rows, ("RAW",), ("RAW",))
    node = _node("project", "project 5 project pipeline", {0: cols[0]}, 1)
    pipe = _pipe(pool, node, lambda d: [d[0] + 1], [constants.int_dtype],
                 ["BITCOMP"], "events", chunk_size=1 << 20)
    assert _events(pipe)
    syncs = _count_syncs(monkeypatch)
    _run(pipe)
    assert pipe.stages > 4 and syncs[0] == 1
    assert torch.equal(_host(pipe.output_variable(0, "int")),
                       x.squeeze(1) + 1)


@requires_cuda
@pytest.mark.parametrize("settles", [None, SETTLE_AFTER, 10 ** 6],
                         ids=["no-settled", "settles", "never"])
def test_an_observer_keeps_the_barrier_until_it_settles(
        cuda_allocator, pool, engine, monkeypatch, settles):
    """An installed observer: the pipe syncs the exchange every tick until
    the observer settles (prologue, tick 1, then each steady tick up to the
    hand-over), then once at its end.  An observer without ``settled``, or
    one that never settles, keeps the barrier on every tick."""
    from adaptive import adaptive_driver
    if settles is None:
        monkeypatch.setattr(adaptive_driver, "_tick_observer_factory",
                            lambda name, menu: (lambda view: None))
    else:
        monkeypatch.setattr(adaptive_driver, "_tick_observer_factory",
                            _settling_factory(settles))
    rows = 1_000_000
    x, m, cols = _filter_inputs(pool, rows, ("RAW",), ("RAW",))
    node = _node("project", "project 9 project pipeline", {0: cols[0]}, 1)
    pipe = _pipe(pool, node, lambda d: [d[0] + 3], [constants.int_dtype],
                 ["BITCOMP"], "events", chunk_size=1 << 20)
    assert not _events(pipe)
    syncs = _count_syncs(monkeypatch)
    _run(pipe)
    assert pipe.stages > 6
    if settles == SETTLE_AFTER:
        # The views of ticks 0, 1 and 2 settle it: ticks 0-2 under the
        # barrier, one sync at the end of the events.
        assert _events(pipe) and syncs[0] == SETTLE_AFTER + 1
    else:
        assert not _events(pipe) and syncs[0] >= pipe.stages
    assert torch.equal(_host(pipe.output_variable(0, "int")),
                       x.squeeze(1) + 3)


@requires_cuda
def test_a_granule_cache_orders_its_ticks_by_events(cuda_allocator, pool,
                                                     engine, monkeypatch):
    """A pipe built with a granule cache orders its ticks by events, as one
    without does; that one syncs the exchange once, at its end.  (Pipes
    that run with a cache are tests/page_cache and tests/e2e's.)"""
    rows = 300_000
    x, m, cols = _filter_inputs(pool, rows, ("RAW",), ("RAW",))
    node = _node("project", "project 6 project pipeline", {0: cols[0]}, 1)
    cached = _pipe(pool, node, lambda d: [d[0]], [constants.int_dtype],
                   ["RAW"], "events", chunk_size=1 << 20,
                   page_cache=object())
    assert _events(cached)
    pipe = _pipe(pool, node, lambda d: [d[0] + 2], [constants.int_dtype],
                 ["RAW"], "events", chunk_size=1 << 20)
    assert _events(pipe)
    syncs = _count_syncs(monkeypatch)
    _run(pipe)
    assert pipe.stages > 4 and syncs[0] == 1
    assert torch.equal(_host(pipe.output_variable(0, "int")),
                       x.squeeze(1) + 2)


@requires_cuda
def test_a_failure_leaves_the_next_pipe_intact(cuda_allocator, pool, engine,
                                               monkeypatch):
    """An operator that raises mid-pipe in events mode: the streams are
    waited for before the holds go, and the next pipe's rows are right."""
    rows = 400_000
    x, m, cols = _filter_inputs(pool, rows, ("RAW",), ("RAW", "ANS"))
    from adaptive import instage_optimization
    monkeypatch.setattr(instage_optimization, "_global_fixed_compression",
                        "ANS")
    seen = []

    def fails(d):
        seen.append(1)
        torch.cuda._sleep(int(10 * 1.5e6))
        if len(seen) == 4:
            raise RuntimeError("operator failed")
        return [d[0] * 5]
    node = _node("filter", "filter 7 filter pipe", dict(cols), 1)
    pipe = _pipe(pool, node, fails, [constants.int_dtype], ["BITCOMP"],
                 "events", mask_map={0: 1}, chunk_size=1 << 20)
    assert _events(pipe)
    with pytest.raises(RuntimeError, match="operator failed"):
        pipe.do_exchange()
    assert all(not t.holds for t in pipe._VortexPipeline__tick_stages)
    node2 = _node("filter", "filter 8 filter pipe", dict(cols), 1)
    pipe2 = _run(_pipe(pool, node2, lambda d: [d[0] * 5],
                       [constants.int_dtype], ["BITCOMP"], "events",
                       mask_map={0: 1}, chunk_size=1 << 20))
    assert torch.equal(_host(pipe2.output_variable(0, "int")),
                       x[m].squeeze(1) * 5)


class _RecutSession:
    """A tick sizing session (``sizing_controller.TickJumpSession``'s
    shape) that asks for a re-cut at half the budget at its 2nd chunk and
    back at its 6th, and records what it is fed."""

    k = 2

    def __init__(self, chunk_b):
        self.seen, self.chunk_b = [], chunk_b
        self.closed = False

    def observe(self, built_b, nbytes, q_ms, peak_b, remaining_b):
        self.seen.append((built_b, nbytes, q_ms, peak_b, remaining_b))
        if len(self.seen) == 2:
            return self.chunk_b // 2
        if len(self.seen) == 6:
            return self.chunk_b
        return None

    def recut_refused(self):
        pass

    def close(self):
        self.closed = True


@requires_cuda
@pytest.mark.parametrize("mask_codecs", [("RAW",), ("RAW", "ANS")],
                         ids=["raw-mask", "ans-mask"])
def test_a_session_recuts_event_ordered_ticks(cuda_allocator, pool, engine,
                                              monkeypatch, mask_codecs):
    """The tick sizing session re-cuts a pipe on event-ordered ticks from
    its next unissued tick, twice: every chunk after the first is fed once,
    in order, with the budget it was built at, its tick time and readback
    and the bytes still to issue, and the rows come out whole."""
    from adaptive import adaptive_driver, instage_optimization
    monkeypatch.setattr(instage_optimization, "_global_fixed_compression",
                        "ANS")
    sessions = []

    def factory(pipe_name, pipe_class, instance, chunk_b, n_ticks, input_b,
                sectionable=None, quantum_b=0):
        assert input_b == rows * 9           # 8-byte data, 1-byte mask
        sessions.append(_RecutSession(chunk_b))
        return sessions[-1]
    monkeypatch.setattr(adaptive_driver, "_tick_sizing_factory", factory)
    rows = 1_200_000
    x, m, cols = _filter_inputs(pool, rows, mask_codecs)
    node = _node("filter", "filter 12 filter pipe", cols, 2)
    pipe = _pipe(pool, node, lambda d: [d[0] * 2 + 1, d[0]],
                 [constants.int_dtype, constants.int_dtype],
                 ["BITCOMP", "RAW"], "events", mask_map={0: 1},
                 chunk_size=1 << 20, sizing_key=("filter", "12"))
    assert _events(pipe)
    _run(pipe)
    (s,) = sessions
    assert s.closed and len(s.seen) >= 6
    assert {b for b, *_ in s.seen} == {1 << 20, 1 << 19}
    assert all(q > 0 for _, _, q, _, _ in s.seen)
    # The bytes still to issue only fall, to none at the end.
    left = [r for *_, r in s.seen]
    assert left == sorted(left, reverse=True) and left[-1] == 0
    want = x[m].squeeze(1)
    assert torch.equal(_host(pipe.output_variable(0, "int")), want * 2 + 1)
    assert torch.equal(_host(pipe.output_variable(1, "int")), want)
