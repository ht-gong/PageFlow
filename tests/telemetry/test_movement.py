"""The movement ledger (``IO/telemetry/movement.py``) and the scripts'
reader of it (``experiments/paper_scripts/helpers/perf_json.py``).

Without a GPU: the record/report contract, the output phase and
``perf_json``.
With one, real pipelines on synthetic columns: a cache-less pipe reads its
base column H2D (``scan_h2d``) and spills its output regions
(``spill_bypass``), and a second pipe reading that output back restores it
(``restore_h2d``); the rearrange gather reads a sparse gather's source
zero-copy (``scan_uva``) and loads a dense one's (``scan_h2d``); a cached
region's evicted granules are copied home in
their records' decoded bytes and read back from their homes
(``restore_evicted_h2d``); with the switch off no hook runs.
The last tests run SF1 Q6 through ``main.exec``: both ways (the ledger, the
output step and the residency section only when on), and twice on (the
same bytes per kind).
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

import constants
from IO.pages.combined_column_rep import CombinedColumnRep
from IO.telemetry import movement
from utility.logger import set_message_logger, set_perf_logger

_SCRIPTS = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir,
                        "experiments", "paper_scripts", "helpers")
sys.path.insert(0, os.path.abspath(_SCRIPTS))
import perf_json  # noqa: E402

try:
    import TQPlib.zero_copy_reader.zero_copy_reader  # noqa: F401
    _HAS_ZC = True
except Exception:
    _HAS_ZC = False

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_ZC,
    reason="CUDA + custom allocator + zero_copy_reader extension required")


@pytest.fixture
def ledger(telemetry):
    """The switch on and the ledger empty."""
    telemetry(experiment=True)
    movement.reset()
    yield
    movement.reset()


# ── record / report ──────────────────────────────────────────────────────

def test_report_lists_every_kind_and_sums_records(ledger):
    movement.record("spill_bypass", 2, 100, 40)
    movement.record("spill_bypass", 1, 50, 20)
    report = movement.report()
    assert list(report) == ["kinds", "output"]
    kinds = report["kinds"]
    assert list(kinds) == list(report["output"]) == list(movement.KINDS)
    assert kinds["spill_bypass"] == {"pages": 3, "logical_b": 150,
                                     "wire_b": 60}
    assert kinds["scan_h2d"] == {"pages": 0, "logical_b": 0, "wire_b": 0}
    movement.reset()
    assert movement.report()["kinds"]["spill_bypass"]["pages"] == 0


def test_the_output_step_is_its_own_phase(ledger):
    movement.record("restore_h2d", 1, 64, 32)
    movement.set_phase("output")
    movement.record("restore_h2d", 1, 16, 8)
    movement.record("result_d2h", 2, 16, 16)
    movement.set_phase("query")
    movement.record("restore_h2d", 1, 64, 32)
    report = movement.report()
    assert report["kinds"]["restore_h2d"] == {"pages": 2, "logical_b": 128,
                                              "wire_b": 64}
    assert report["kinds"]["result_d2h"]["pages"] == 0
    assert report["output"]["restore_h2d"]["wire_b"] == 8
    assert report["output"]["result_d2h"] == {"pages": 2, "logical_b": 16,
                                              "wire_b": 16}
    # A failed query can leave the output phase on: the next query's reset
    # goes back to the query phase.
    movement.set_phase("output")
    movement.reset()
    movement.record("scan_h2d", 1, 8, 8)
    assert movement.report()["kinds"]["scan_h2d"]["pages"] == 1
    assert movement.report()["output"]["scan_h2d"]["pages"] == 0


def test_perf_json_kinds_match_the_ledger():
    assert perf_json.KINDS == movement.KINDS
    assert perf_json.SPILL_KINDS == movement.SPILL_KINDS
    assert set(perf_json.OUT_KINDS) <= set(movement.KINDS)


def test_perf_json_reads_the_ledger_strictly(ledger, tmp_path):
    movement.record("cache_fill", 2, 80, 40)
    movement.record("spill_bypass", 1, 30, 10)
    movement.record("spill_mirror", 1, 8, 5)
    perf = {"wall": {"approx_time": 1.5}, "movement": movement.report()}
    assert perf_json.approx_time(perf) == 1.5
    assert perf_json.born(perf) == 110
    assert perf_json.spill_total(perf) == 15
    assert perf_json.mv(perf, "result_d2h", "pages", phase="output") == 0
    with pytest.raises(KeyError):
        perf_json.mv(perf, "no_such_kind", "pages")
    # A report from before the later kinds: they read 0, the others still
    # raise, and so does the output step it never had.
    old = {"movement": {"kinds": {k: v for k, v in perf["movement"]["kinds"].items()
                                  if k not in perf_json.LATER_KINDS}}}
    assert perf_json.mv(old, "restore_evicted_h2d", "wire_b") == 0
    assert perf_json.mv_sum(old, ("cache_fill", "result_d2h"), "pages") == 2
    del old["movement"]["kinds"]["cache_hit"]
    with pytest.raises(KeyError):
        perf_json.mv(old, "cache_hit", "pages")
    with pytest.raises(KeyError):
        perf_json.mv(old, "result_d2h", "pages", phase="output")

    (tmp_path / "SF1_3.json").write_text(json.dumps({"wall": {}}))
    with pytest.raises(perf_json.PerfDataMissing):
        perf_json.load(str(tmp_path), 1, 3, movement=True)
    with pytest.raises(perf_json.PerfDataMissing):
        perf_json.born(perf_json.load(str(tmp_path), 1, 3))
    with pytest.raises(FileNotFoundError):
        perf_json.load(str(tmp_path), 1, 4)


# ── the hooks, on real pipelines ─────────────────────────────────────────

def _pinned_column(raw_2d, tensor_type="int", pool=None):
    """A scanned column: the scan's RAW record table of ``raw_2d``, in
    ``pool`` (the consumer's, when it frees the column), else a pinned pool
    of its own."""
    from tests.scan_table import scan_table
    from IO.pages.pinned_mem import PinnedMemory
    if pool is None:
        pool = PinnedMemory(capacity_gb=0.25, block_size_mb=0.5)
    return CombinedColumnRep([scan_table(
        raw_2d.contiguous(), tensor_type, pool, "RAW")])


def _identity_pipe(col, name, **kw):
    from IO.pages.pinned_mem import PinnedMemory
    from IO.pipeline.vortex_pipeline import VortexPipeline
    return VortexPipeline(
        input_columns={0: col},
        output_columns_type=[constants.int_dtype],
        operator=lambda d: [d[0]],
        cpu_mem_pool=PinnedMemory(capacity_gb=1, block_size_mb=1),
        chunk_size=256 * 1024,             # four ticks over a 1 MiB column
        name=name, **kw)


@pytest.fixture
def pipes():
    from IO.transfer.data_exchange import set_exchange_to_naive
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()


@requires_cuda
def test_pipeline_scans_spills_and_restores(ledger, pipes):
    rows = 1 << 17                                     # 1 MiB of int64
    data = torch.arange(rows, dtype=constants.int_dtype).unsqueeze(1)
    first = _identity_pipe(_pinned_column(data), "producer")
    first.do_exchange()
    first.get_result()
    table = first.result_tables()[0]
    kinds = movement.report()["kinds"]
    # One 1 MiB RAW record, read by each of the four ticks for its
    # quarter: each tick copies only its own rows of it.
    assert kinds["scan_h2d"]["pages"] == 4
    assert kinds["scan_h2d"]["logical_b"] == kinds["scan_h2d"]["wire_b"] \
        == rows * 8
    # RAW outputs: one region per tick, shipped home as it is.
    spill = kinds["spill_bypass"]
    assert spill["pages"] == len(table.regions) == 4
    assert spill["wire_b"] == sum(r.nbytes for r in table.regions)

    from variable import TableVariable
    movement.reset()
    second = _identity_pipe(
        CombinedColumnRep([TableVariable(table, "int")]), "consumer")
    second.do_exchange()
    kinds = movement.report()["kinds"]
    assert kinds["restore_h2d"]["logical_b"] >= rows * 8
    assert kinds["scan_h2d"]["pages"] == 0


@requires_cuda
@pytest.mark.parametrize("n_out,kind", [(1 << 12, "scan_uva"),
                                        (1 << 18, "scan_h2d")])
def test_rearrange_gather_reads(ledger, pipes, monkeypatch, n_out, kind):
    """A sparse gather reads its source's rows zero-copy, its windows' rows
    times the row's bytes; a dense one, over several windows, loads the
    whole source H2D.  Either way the index is read H2D."""
    monkeypatch.setattr(constants, "get_chunk_size_b", lambda: 1 << 20)
    from IO.pages.pinned_mem import PinnedMemory
    from IO.pipeline.rearrange_pipeline import rearrange_pipeline
    from dataflow.dataflow_gen import ColumnInstance, PipeNode, PlanGraph
    rows = 1 << 16
    g = torch.Generator().manual_seed(5)
    data = torch.randint(0, 1000, (rows, 1), dtype=torch.int64, generator=g)
    idx = torch.randint(0, rows, (n_out, 1), dtype=torch.int64, generator=g)
    pool = PinnedMemory(capacity_gb=1, block_size_mb=0.5)
    plan = PlanGraph(0)
    scan = PipeNode(op_id=0, kind="scan", name="scan 0", plan=plan)
    core = PipeNode(op_id=0, kind="core", name="core 0", plan=plan)
    node = PipeNode(op_id=0, kind="rearrange", name="gather 0", plan=plan)
    plan.nodes.extend([scan, core, node])
    src = ColumnInstance(producer=scan, local=7,
                         ccr=_pinned_column(data, pool=pool))
    index = ColumnInstance(producer=core, local="out_index_left",
                           ccr=_pinned_column(idx, pool=pool))
    for inst, producer in ((src, scan), (index, core)):
        producer.outputs.append(inst)
        node.inputs.append(inst)
        node.input_keys.append(inst.local)
        inst.consumers.append(node)
    node.outputs.append(ColumnInstance(producer=node, local=7))
    plan.assign_positions()

    rearrange_pipeline(node, pool, "inner")
    kinds = movement.report()["kinds"]
    if kind == "scan_uva":
        uva = kinds["scan_uva"]
        assert uva["pages"] == 0
        assert uva["logical_b"] == uva["wire_b"] == n_out * 8
        assert kinds["scan_h2d"]["logical_b"] == n_out * 8
    else:
        assert kinds["scan_uva"]["logical_b"] == 0
        assert kinds["scan_h2d"]["logical_b"] == rows * 8 + n_out * 8
    assert kinds["spill_bypass"]["logical_b"] == n_out * 8


@requires_cuda
def test_regions_are_spilled_per_tick_and_restored(ledger, pipes):
    from variable import TableVariable
    rows = 1 << 17
    data = (torch.arange(rows, dtype=constants.int_dtype) % 7).unsqueeze(1)
    pipe = _identity_pipe(_pinned_column(data), "t filter pipe",
                          output_compression_algs=["BITCOMP"])
    pipe.do_exchange()
    pipe.get_result()
    table = pipe.result_tables()[0]
    spill = movement.report()["kinds"]["spill_bypass"]
    # One region per tick, four ticks; each region's copy home is one
    # spill of its stored bytes.
    assert spill["pages"] == len(table.regions) == 4
    assert spill["wire_b"] == sum(r.nbytes for r in table.regions)
    assert spill["logical_b"] == rows * 8

    movement.reset()
    second = _identity_pipe(
        CombinedColumnRep([TableVariable(table, "int")]), "consumer")
    second.do_exchange()
    kinds = movement.report()["kinds"]
    assert kinds["restore_h2d"]["logical_b"] >= rows * 8
    assert kinds["restore_h2d"]["wire_b"] == table.stored_bytes()
    assert kinds["scan_h2d"]["pages"] == 0


@requires_cuda
def test_evicted_granules_are_read_back_from_their_homes(ledger, pipes,
                                                         monkeypatch):
    """Three ticks through a six-slot cache: each tick's region takes the
    slots and evicts the one before (its dirty granules copied home), and
    the rest of each region is refused.  Reading the table back gets the
    last region's slots D2D and every other byte from the homes:
    ``restore_evicted_h2d``, never ``restore_h2d``."""
    import numpy as np
    pytest.importorskip("TQPlib.tqpcache")
    from IO.cache.native_cache import NativeGranuleCache
    from IO.codec.home_writer import copy_home
    from IO.codec.region_producer import RegionProducer
    from IO.pages.pinned_mem import PinnedMemory
    from tests.page_cache.regions import TensorRegion
    from variable import TableVariable
    monkeypatch.setattr("IO.codec.region_producer.RECORD_MB", 1)
    G = 16 << 10
    cache = NativeGranuleCache(TensorRegion(6, G, device="cuda:0"), "lru",
                               copy_home=copy_home)
    pool = PinnedMemory(capacity_gb=0.25, block_size_mb=0.5)
    producer = RegionProducer(pool, "t 0 sort pipeline", cache=cache)
    producer.cached_outputs = {0: (5, 1)}
    rng = np.random.default_rng(11)
    # Three ticks of three buckets, laid out as the sort pipe lays them out,
    # as tests/page_cache/test_granule_producer.py.
    from IO.codec.batched_compress import aligned_partition_layout
    ticks = [[torch.from_numpy(rng.integers(0, 50, 150_000).astype(np.int64))
              .unsqueeze(1) for _ in range(3)] for _ in range(3)]
    for t in ticks:
        cache.advance_tick()
        if producer.has_pending():
            producer.pack()
        counts = [b.shape[0] for b in t]
        _, starts, total = aligned_partition_layout(counts, 8, 512)
        out = torch.full((total, 1), -1, dtype=torch.int64, device="cuda:0")
        for b, s in zip(t, starts):
            out[s // 8:s // 8 + b.shape[0]] = b.to("cuda:0")
        producer.compress([(0, out, counts, "BITCOMP")])
        torch.cuda.current_stream().synchronize()
    cache.advance_tick()
    producer.drain()
    table = producer.tables()[0]
    assert [r.release[1].admitted for r in table.regions] == [6, 6, 6]
    written = movement.report()["kinds"]
    assert written["spill_evict"]["pages"] > 0
    assert written["spill_rejected"]["pages"] == 3
    assert written["spill_bypass"]["pages"] == 0
    # Decoded bytes like the fill's: every granule filled was copied home
    # by a reclaim, released, or is still in its slot (the last region's),
    # and BITCOMP stored the evicted ones in fewer bytes.
    last = table.regions[-1].release[1]
    resident = np.arange(last.first, last.first + last.admitted)
    held = cache.ledger_bytes(resident)
    gone = [sum(written[k][f] for k in ("spill_evict", "avoided_clean",
                                        "avoided_dead"))
            for f in ("logical_b", "wire_b")]
    fill = written["cache_fill"]
    assert gone[1] + held[1] == fill["wire_b"]
    assert abs(gone[0] + held[0] - fill["logical_b"]) <= 3 * len(table.regions)
    assert written["spill_evict"]["logical_b"] > written["spill_evict"]["wire_b"]

    movement.reset()
    var = TableVariable(table, "int")
    assert torch.equal(var.col_to_gpu().cpu(), torch.cat(
        [t[b] for b in range(3) for t in ticks]).squeeze(1))
    kinds = movement.report()["kinds"]
    assert kinds["restore_h2d"]["pages"] == 0
    assert kinds["restore_evicted_h2d"]["pages"] > 0
    assert kinds["cache_hit"]["pages"] > 0
    # Every record once, with the padding between them in a region.
    assert table.stored_bytes() <= (
        kinds["restore_evicted_h2d"]["wire_b"] + kinds["cache_hit"]["wire_b"]
    ) <= sum(r.nbytes for r in table.regions)
    var.free_underlying_mem(pool, None)


@requires_cuda
def test_no_hook_runs_with_the_switch_off(pipes, monkeypatch, telemetry):
    calls = []
    monkeypatch.setattr(movement, "record", lambda *a: calls.append(a))
    telemetry(experiment=False)
    rows = 1 << 17
    data = (torch.arange(rows, dtype=constants.int_dtype) % 7).unsqueeze(1)
    pipe = _identity_pipe(_pinned_column(data), "t filter pipe",
                          output_compression_algs=["BITCOMP"])
    pipe.do_exchange()
    pipe.get_result()
    assert calls == []


# ── the perf report, end to end ──────────────────────────────────────────

def _run_q6(perf_dir):
    import main
    from adaptive.instage_optimization import set_adaptive_compression
    from IO.cache.config import BufferStrategy, set_global_buffer_strategy
    from IO.transfer.data_exchange import set_exchange_to_naive
    set_message_logger(None, False)
    set_global_buffer_strategy(BufferStrategy.CPU)
    set_exchange_to_naive()
    set_adaptive_compression(True)
    perf_dir.mkdir(parents=True)
    try:
        main.exec(1, [6], str(perf_dir), None, True, None)
    finally:
        set_adaptive_compression(False)
    with open(perf_dir / "SF1_6.json", encoding="utf-8") as f:
        return json.load(f)


def test_perf_report_carries_the_ledger_only_when_on(sf1_parquet, tmp_path,
                                                     monkeypatch, telemetry):
    from IO.telemetry import residency
    calls = []
    monkeypatch.setattr(movement, "record", lambda *a: calls.append(a))
    monkeypatch.setattr(movement, "set_phase", lambda *a: calls.append(a))
    monkeypatch.setattr(residency, "begin", lambda *a: calls.append(a))
    telemetry(experiment=False)
    off = _run_q6(tmp_path / "off")
    assert calls == []
    assert set(off) == {"wall"}
    monkeypatch.undo()

    telemetry(experiment=True)
    on = _run_q6(tmp_path / "on")
    assert {"wall", "movement", "residency", "sizing_model",
            "compression"} <= set(on)
    kinds = on["movement"]["kinds"]
    assert list(kinds) == list(movement.KINDS)
    # With no cache, Q6 scans its base columns, writes every intermediate
    # D2H and reads each one back.
    assert kinds["scan_h2d"]["pages"] > 0
    assert kinds["spill_bypass"]["pages"] > 0
    assert kinds["restore_h2d"]["pages"] > 0
    assert kinds["cache_hit"]["pages"] == kinds["cache_fill"]["pages"] == 0
    assert kinds["spill_rejected"]["pages"] == 0
    # The output step is apart: its one column's copy to the host (8
    # bytes, a sum), never in the query's kinds.
    out = on["movement"]["output"]
    assert out["result_d2h"] == {"pages": 1, "logical_b": 8, "wire_b": 8}
    assert kinds["result_d2h"]["pages"] == 0
    # Every pipe's arena peak is reported, and the query's arena peak (the
    # timed run's) covers each of them.
    res = on["residency"]
    assert res["timed"] and res["slot_pool"] is None
    assert res["gpu_arena"]["peak_b"] >= perf_json.peak_gpu_bytes(on) > 0
    host = res["host_pool"]
    assert host["at_start_b"]["scan"] > 0
    assert 0 < host["peak_b"]["intermediate"] <= host["peak_total_b"] \
        <= host["capacity_b"]
