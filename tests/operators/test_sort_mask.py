"""The Sort operator applies the mask of a filter right below it.

A HAVING filter (Q11's) emits only its mask, so the Sort receives its input
columns unfiltered and must sort exactly the rows the mask keeps; DISTINCT and
LIMIT apply to those rows, and an all-false mask gives an empty result that
the output step still renders.  Runs the real ``tqp_sort`` on tiny
scanned columns; its outputs are GPU regions.  Requires CUDA; the shared
custom allocator is bootstrapped in tests/conftest.py.
"""
from __future__ import annotations

import pytest
import torch

from IO.pages.combined_column_rep import CombinedColumnRep
from IO.transfer.data_exchange import set_exchange_to_naive

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA + custom allocator required")

KEY, ROW, NAME, MASK = 1, 2, 3, 10000


@pytest.fixture(autouse=True)
def _engine_setup():
    from utility.logger import set_message_logger, set_perf_logger
    set_message_logger(None, False)
    set_perf_logger(None, True)
    set_exchange_to_naive()


@pytest.fixture(autouse=True, params=["payload-whole", "payload-windowed"])
def _payload(request, monkeypatch):
    """Every test twice: the columns that only follow the order decoded
    whole (each under the default chunk budget), and decoded window by
    window (over a 1 KiB chunk budget: windows of 128 rows of 8 B, cut
    inside the scan's one record)."""
    import constants
    if request.param == "payload-windowed":
        monkeypatch.setattr(constants, "get_chunk_size_b", lambda: 1 << 10)


def _pinned_col(values: torch.Tensor, tensor_type: str) -> CombinedColumnRep:
    """A scanned column: the scan's RAW record table, in a pinned pool of
    its own."""
    from tests.scan_table import scan_table
    from IO.pages.pinned_mem import PinnedMemory
    raw = (values.reshape(-1, 1) if values.dim() == 1 else values).contiguous()
    pool = PinnedMemory(capacity_gb=0.25, block_size_mb=0.5)
    return CombinedColumnRep([scan_table(
        raw, tensor_type, pool, "RAW")])


def _sort(n, seed, mask=None, **args):
    """Sort KEY (int), ROW (the row id) and NAME (a string column) under a
    mask; returns the inputs, the mask, tqp_sort's row count and the
    tensor groups."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    key = torch.randint(0, 50, (n,), dtype=torch.int64, generator=g)
    row = torch.arange(n, dtype=torch.int64)
    name = torch.randint(65, 67, (n, 4), dtype=torch.int8, generator=g)
    if mask is None:
        mask = torch.rand(n, generator=g) < 0.3
    tensor_groups = {KEY: _pinned_col(key, "int"), ROW: _pinned_col(row, "int"),
                     NAME: _pinned_col(name, "string"), MASK: _pinned_col(mask, "bool")}
    sort_args = {"input ids": [KEY, ROW, NAME], "key ids": [KEY], "key orders": [True]}
    sort_args.update(args)
    sort_args["mask map"] = {c: MASK for c in sort_args["input ids"]}
    from operators.sort import tqp_sort
    rows = tqp_sort(tensor_groups, sort_args, with_limit="limit" in sort_args,
                    name="t_sort_mask")
    return (key, row, name), mask, rows, tensor_groups


def _out(tensor_groups, cid):
    return tensor_groups[cid].reps[0].col_to_gpu().cpu()


@requires_cuda
def test_sort_keeps_only_the_masked_rows():
    (key, row, name), mask, rows, tg = _sort(5000, 1)
    kept = mask.nonzero().squeeze(1)
    expect = kept[torch.sort(key[kept], stable=True, descending=True).indices]
    assert rows == kept.numel() > 0
    assert torch.equal(_out(tg, ROW), expect)
    assert torch.equal(_out(tg, KEY), key[expect])
    assert torch.equal(_out(tg, NAME), name[expect])


@requires_cuda
def test_limit_applies_to_the_masked_rows():
    (key, row, name), mask, rows, tg = _sort(5000, 2, limit=10)
    kept = mask.nonzero().squeeze(1)
    expect = kept[torch.sort(key[kept], stable=True, descending=True).indices][:10]
    assert rows == kept.numel()  # the count before the limit
    assert torch.equal(_out(tg, ROW), expect)


@requires_cuda
def test_distinct_applies_to_the_masked_rows():
    (key, row, name), mask, rows, tg = _sort(
        5000, 3, **{"input ids": [KEY, NAME], "distinct": True})
    kept = mask.nonzero().squeeze(1).tolist()
    tuples = {(int(key[i]), tuple(name[i].tolist())) for i in kept}
    expect = sorted(tuples, key=lambda t: (-t[0], t[1]))
    got = list(zip(_out(tg, KEY).tolist(), map(tuple, _out(tg, NAME).tolist())))
    assert rows == len(expect) < len(kept)
    assert got == expect


@requires_cuda
@pytest.mark.parametrize("args", [
    {},
    {"key ids": [NAME], "key orders": [False], "distinct": True},
    {"limit": 5},
], ids=["int-key", "string-key-distinct", "limit"])
def test_an_all_false_mask_gives_an_empty_result(args):
    _, _, rows, tg = _sort(3000, 4, mask=torch.zeros(3000, dtype=torch.bool), **args)
    assert rows == 0
    for cid in (KEY, ROW, NAME):
        assert _out(tg, cid).numel() == 0
    from operators.output import tqp_output
    frame = tqp_output(tg, {"output ids": [KEY, NAME], "output names": ["key", "name"]})
    assert len(frame) == 0 and list(frame.columns) == ["key", "name"]
