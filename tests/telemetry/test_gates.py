"""The telemetry and debug switches (``IO/telemetry/gates.py``).

Without a GPU: their parsing, each switch reading its own variable once at
import, and main.py refusing the experiment switch without
``--perf_report``.  With one, the runtime inputs the engine's decisions read
are there with the switch off: each tick's transfer lanes and compute time
reach the tick observer, a sized pipe's tick sizing session gets each
chunk's tick time and arena peak, and a gather session's
``GatherIterSample`` its times.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import pytest
import torch

import constants
from IO.pages.combined_column_rep import CombinedColumnRep
from IO.telemetry import gates

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir))

try:
    import TQPlib.zero_copy_reader.zero_copy_reader  # noqa: F401
    _HAS_ZC = True
except Exception:
    _HAS_ZC = False

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or not _HAS_ZC,
    reason="CUDA + custom allocator + zero_copy_reader extension required")


@pytest.mark.parametrize("value, on", [
    ("1", True), ("true", True), (" ON ", True), ("yes", True),
    ("0", False), ("false", False), ("", False), ("2", False)])
def test_switch_parses_on_words(monkeypatch, value, on):
    monkeypatch.setenv("TQP_EXPERIMENT_TELEMETRY", value)
    try:
        assert importlib.reload(gates).EXPERIMENT is on
    finally:
        monkeypatch.delenv("TQP_EXPERIMENT_TELEMETRY")
        importlib.reload(gates)


SWITCHES = {
    "EXPERIMENT": "TQP_EXPERIMENT_TELEMETRY",
    "VARCHK": "TQP_VARCHK",
    "PICK_DEBUG": "TQP_PICK_DEBUG",
    "TIMED_GC_VALIDATE": "TQP_TIMED_GC_VALIDATE",
}


def test_every_switch_is_listed():
    assert {name for name, value in vars(gates).items()
            if name.isupper() and isinstance(value, bool)} == set(SWITCHES)


@pytest.mark.parametrize("attr, var", sorted(SWITCHES.items()))
def test_each_switch_reads_its_variable_once_at_import(monkeypatch, attr, var):
    monkeypatch.setenv(var, "on")
    try:
        importlib.reload(gates)
        assert getattr(gates, attr) is True
        assert not any(getattr(gates, other) for other in SWITCHES
                       if other != attr)
        # Read once: a later change to the variable is not seen ...
        monkeypatch.setenv(var, "0")
        assert getattr(gates, attr) is True
        # ... until the next import, where "0" is off.
        assert getattr(importlib.reload(gates), attr) is False
    finally:
        monkeypatch.delenv(var)
        importlib.reload(gates)


def test_experiment_telemetry_without_perf_report_stops_main():
    env = dict(os.environ, TQP_EXPERIMENT_TELEMETRY="1")
    proc = subprocess.run(
        [sys.executable, "main.py", "--SF", "1", "--q", "6", "--no_warmup"],
        cwd=_REPO_ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 2
    assert "TQP_EXPERIMENT_TELEMETRY" in proc.stderr


# ── runtime inputs, with the switch off ──────────────────────────────────

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


@pytest.fixture
def engine():
    from IO.transfer.data_exchange import set_exchange_to_naive
    from utility.logger import set_message_logger, set_perf_logger
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()


@requires_cuda
def test_tick_inputs_and_sizing_samples_flow_with_every_switch_off(
        engine, telemetry):
    from adaptive import adaptive_driver
    from IO.pages.pinned_mem import PinnedMemory
    from IO.pipeline.vortex_pipeline import VortexPipeline
    telemetry(experiment=False)
    ticks = []

    def observer(view):
        ticks.append((view.last_transfer, view.last_compute))
        return None

    rows = 1 << 17                                     # 1 MiB of int64
    data = torch.arange(rows, dtype=constants.int_dtype).unsqueeze(1)
    pool = PinnedMemory(capacity_gb=1, block_size_mb=1)
    pipe = VortexPipeline(
        input_columns={0: _pinned_column(data)},
        output_columns_type=[constants.int_dtype],
        operator=lambda d: [d[0]],
        cpu_mem_pool=pool,
        chunk_size=256 * 1024,                         # four ticks
        name="observed", tick_observer=observer)
    pipe.do_exchange()
    from variable import TableVariable
    pipe.get_result()
    out = TableVariable(pipe.result_tables()[0], "int").col_to_gpu()
    assert torch.equal(out.cpu().reshape(-1), data.reshape(-1))

    # Prologue + four compute ticks + the drain tick: H2D for the four
    # chunks, compute from tick 1; each chunk's copy home (RAW, shipped at
    # its compute) shows in the d2h lane once it has ended.
    assert len(ticks) == 6
    assert all(t.h2d_ms > 0 for t, _ in ticks[:4])
    assert all(c is not None and c.compute > 0 for _, c in ticks[1:5])
    assert ticks[0][0].d2h_ms == 0
    assert any(t.d2h_ms > 0 for t, _ in ticks)

    # A sized pipe on events: its session gets each chunk after the first
    # with its tick time and the arena's peak while it computed.
    seen = []

    class _Session:
        k = 2

        def observe(self, built_b, nbytes, q_ms, peak_b, remaining_b):
            seen.append((built_b, nbytes, q_ms, peak_b, remaining_b))
            return None

        def recut_refused(self):
            pass

        def close(self):
            pass

    adaptive_driver.set_tick_sizing_factory(lambda *a, **k: _Session())
    try:
        sized = VortexPipeline(
            input_columns={0: _pinned_column(data)},
            output_columns_type=[constants.int_dtype],
            operator=lambda d: [d[0]], cpu_mem_pool=pool,
            chunk_size=128 * 1024,                     # eight ticks
            name="sized", sizing_key=("filter", "7"))
        sized.do_exchange()
    finally:
        adaptive_driver.set_tick_sizing_factory(None)
    assert len(seen) == 7                  # chunks 1..7, not the empty 8th
    assert all(b == 128 * 1024 and n == 128 * 1024 for b, n, *_ in seen)
    assert all(q > 0 and peak > 0 for _, _, q, peak, _ in seen)


@requires_cuda
def test_gather_samples_flow_with_every_switch_off(engine, telemetry):
    from adaptive import adaptive_driver
    from dataflow.dataflow_gen import ColumnInstance, PipeNode, PlanGraph
    from IO.pages.pinned_mem import PinnedMemory
    from IO.pipeline.rearrange_pipeline import rearrange_pipeline
    telemetry(experiment=False)
    sessions = []

    def factory(pipe_name, instance, row_bytes, row_overhead, total_rows,
                codec_epoch, quantum_rows=0):
        session = types.SimpleNamespace(instance=instance, samples=[],
                                        closed=False)
        session.observe = session.samples.append

        def close():
            session.closed = True
        session.close = close
        sessions.append(session)
        return session

    rows, n_out = 1 << 16, 1 << 15
    g = torch.Generator().manual_seed(5)
    data = torch.randint(0, 1_000_000, (rows, 1), dtype=torch.int64,
                         generator=g)
    idx = torch.randint(0, rows, (n_out,), dtype=torch.int64, generator=g)
    pool = PinnedMemory(capacity_gb=1, block_size_mb=0.5)
    plan = PlanGraph(0)
    scan = PipeNode(op_id=0, kind="scan", name="scan 0", plan=plan)
    core = PipeNode(op_id=0, kind="core", name="core 0", plan=plan)
    node = PipeNode(op_id=0, kind="rearrange", name="gather 0", plan=plan)
    plan.nodes.extend([scan, core, node])
    src = ColumnInstance(producer=scan, local=7,
                         ccr=_pinned_column(data, pool=pool))
    index = ColumnInstance(producer=core, local="out_index_left",
                           ccr=_pinned_column(idx.unsqueeze(1), pool=pool))
    for inst, producer in ((src, scan), (index, core)):
        producer.outputs.append(inst)
        node.inputs.append(inst)
        node.input_keys.append(inst.local)
        inst.consumers.append(node)
    node.outputs.append(ColumnInstance(producer=node, local=7))
    plan.assign_positions()

    adaptive_driver.set_gather_sizing_factory(factory)
    try:
        rearrange_pipeline(node, pool, "inner")
    finally:
        adaptive_driver.set_gather_sizing_factory(None)
    out = node.outputs[0].ccr.reps[0].col_to_gpu()
    assert torch.equal(out.cpu(), data[idx].squeeze(1))
    (session,) = sessions
    assert session.closed and session.samples
    for s in session.samples:
        assert s.h2d_ms > 0 and s.d2h_ms > 0 and s.compress_ms >= 0
        assert s.iter_ms > 0 and s.iter_peak_b > 0
    assert sum(s.rows for s in session.samples) == n_out
