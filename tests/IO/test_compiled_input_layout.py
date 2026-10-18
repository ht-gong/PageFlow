"""Invariants for compiled DirectIn H2D/D2D geometry (CPU-only, one CUDA
test)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from IO.transfer.in_stage import DirectIn
from IO.transfer.tick_layout import bind_tick_sources, compile_tick_layout
from IO.pages.record_table import RecordTableBuilder
from IO.pages.region import DEVICE, HOST, Region
from IO.telemetry import gates, movement
from variable import TableVariable


def _raw_column(nbytes=64, device="cpu"):
    """One RAW record of ``nbytes`` uint8 rows in one region, on the host or
    the GPU."""
    storage = torch.arange(nbytes, dtype=torch.uint8, device=device)
    builder = RecordTableBuilder(torch.uint8, 1)
    k = builder.add_region(Region(nbytes, DEVICE if storage.is_cuda else HOST,
                                  storage))
    builder.append(k, 0, nbytes, nbytes)
    return TableVariable(builder.freeze(), "int")


def test_compile_captures_no_absolute_source_and_bind_fills_it():
    col = _raw_column()
    stage = DirectIn(col)
    layout = compile_tick_layout([stage])

    assert layout.pack_args[0] == [0]
    plan = bind_tick_sources([stage], layout)
    region = col.record_table().regions[0]
    assert plan.pack_args[0] == [region.data_ptr()]
    assert plan.d2d_pack_args[0] == [0]
    assert plan.d2d_pack_pages == 0
    # One run per record; both lanes share the compiled destination offsets
    # and sizes.
    assert plan.pack_args[1] == [0]
    assert plan.pack_args[2] == [64]
    for h2d_geometry, d2d_geometry in zip(
            plan.pack_args[1:], plan.d2d_pack_args[1:]):
        assert h2d_geometry is d2d_geometry


def test_sequential_rebind_reuses_pointer_storage():
    stages = [DirectIn(_raw_column()), DirectIn(_raw_column())]
    layout = compile_tick_layout(stages)
    pack_args_id = id(layout.pack_args)
    source_list_id = id(layout.pack_args[0])
    d2d_args_id = id(layout.d2d_pack_args)
    d2d_source_list_id = id(layout.d2d_pack_args[0])

    first_plan = bind_tick_sources(stages, layout)
    first_ptrs = tuple(first_plan.pack_args[0])
    first_plan.release()

    second_plan = bind_tick_sources(stages, layout)
    assert id(second_plan.pack_args) == pack_args_id
    assert id(second_plan.pack_args[0]) == source_list_id
    assert id(second_plan.d2d_pack_args) == d2d_args_id
    assert id(second_plan.d2d_pack_args[0]) == d2d_source_list_id
    assert tuple(second_plan.pack_args[0]) == first_ptrs


@pytest.fixture
def ledger(telemetry):
    """The movement ledger on and empty; the switch restored afterwards."""
    telemetry(experiment=True)
    movement.reset()
    yield
    movement.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_a_gpu_region_binds_d2d_and_a_host_region_h2d(ledger):
    gpu, host = _raw_column(device="cuda:0"), _raw_column()
    stages = [DirectIn(gpu), DirectIn(host)]
    layout = compile_tick_layout(stages)

    plan = bind_tick_sources(stages, layout)
    gpu_ptr = gpu.record_table().regions[0].data_ptr()
    host_ptr = host.record_table().regions[0].data_ptr()
    assert sorted(plan.pack_args[0]) == [0, host_ptr]
    assert sorted(plan.d2d_pack_args[0]) == [0, gpu_ptr]
    assert (plan.h2d_pack_pages, plan.d2d_pack_pages) == (1, 1)
    plan.release()

    kinds = movement.report()["kinds"]
    assert (kinds["resident_d2d"]["pages"],
            kinds["resident_d2d"]["wire_b"]) == (1, 64)
    assert (kinds["restore_h2d"]["pages"],
            kinds["restore_h2d"]["wire_b"]) == (1, 64)


def _two_records(region_b, sizes):
    """A table of one host region holding two compressed int64 records at
    offsets 0 and 512 (``sizes`` stored bytes; 1 and 3 rows), with
    synthetic decompression handles."""
    builder = RecordTableBuilder(torch.int64, 1, "int")
    k = builder.add_region(Region(region_b, HOST,
                                  torch.zeros(region_b, dtype=torch.uint8)))
    builder.append(k, [0, 512], sizes, [8, 24], codec_key=(1, 8, 1024),
                   handles=[101, 102])
    return builder.freeze()


def test_manager_template_compiles_relative_prefixes_and_handle_vector():
    col = TableVariable(_two_records(524, [12, 12]), "int")
    layout = compile_tick_layout([DirectIn(col)])
    template = layout.codec_groups[0]['mgr_template']
    assert template.compressed_relative.tolist() == [0, 512]
    assert template.output_relative.tolist() == [0, 8]
    assert template.handles.tolist() == [101, 102]
    assert template.launch_args[3] is template.compressed_ptrs
    assert template.launch_args[4] is template.output_ptrs
    assert template.launch_args[5] is template.handles


def test_overlapping_records_are_rejected_whatever_varchk(monkeypatch):
    # The second record starts inside the first one's 600 compressed bytes.
    # The record table checks this whenever it is built, with TQP_VARCHK
    # off as well as on.
    for varchk in (False, True):
        monkeypatch.setattr(gates, "VARCHK", varchk)
        with pytest.raises(AssertionError, match="records overlap"):
            _two_records(1024, [600, 12])


def test_launch_d2d_passes_compiled_pack_tuple_without_rebasing(monkeypatch):
    agg_buf = object()
    d2d_pack_args = ([101, 0], [0, 64], [64, 64])
    calls = []

    def record(*args):
        calls.append(args)

    monkeypatch.setattr(
        "IO.transfer.in_stage.tqpcompress.batched_d2d_pack", record)
    DirectIn.launch_d2d(agg_buf, SimpleNamespace(
        d2d_pack_pages=1, d2d_pack_args=d2d_pack_args, slot_pieces=None))

    assert calls == [(agg_buf, *d2d_pack_args)]


def test_launch_d2d_skips_native_crossing_when_all_sources_are_h2d(monkeypatch):
    def unexpected(*_args):
        raise AssertionError("empty D2D lane crossed into native code")

    monkeypatch.setattr(
        "IO.transfer.in_stage.tqpcompress.batched_d2d_pack", unexpected)
    DirectIn.launch_d2d(object(), SimpleNamespace(
        d2d_pack_pages=0, d2d_pack_args=(), slot_pieces=None))
