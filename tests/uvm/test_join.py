"""The UVM executor's join (``uvm/join.py``) and gathers (``uvm/gather.py``)
against the streaming ``tqp_join`` on SF1 customer and orders columns.

Each case builds one Join's dataflow nodes as the graph builder builds them
(its two sort nodes, its core and its gathers), then runs the streaming
join (over the scan's record tables in a pinned pool,
``tests/scan_table.py``, as ``test_basic_ops.py``'s parity cases do) and the
UVM join on the same columns, and compares the join's outputs (the keys the
core emits and the payloads its rearranges gather) as sets of rows: the two
modes partition and call the core over different row groups, so they emit
pairs in different orders.  Every value is gathered, never computed, so the
rows are compared exactly.  Hand-picked cases, at B = 1 (the default
budget) or B = 16 (a partition budget the estimate turns into 16, at a
4 MiB chunk, so each side is partitioned in several morsels and the core
runs in several calls):

* inner on the unique-hash core, B = 16: customer (proven unique on its
  key) with orders under a mask, a string payload among the gathered ones;
* inner on the sort/search core with a residual condition across the two
  sides (no uniqueness proof), B = 16;
* semi (``LeftSemi``) at B = 1 and anti (``LeftAnti``) at B = 16: orders
  with or without a customer of a masked nation, on the hash-set core;
* right outer (``LeftOuter``), B = 16: every customer with its orders under
  a mask, or one null row (``constants.null`` in every gathered column, a
  string's every byte included) when it has none.

Each case also checks that both modes took the case's core, and that the
UVM join left only its outputs in the tensor group.  Two more cases cover
the gather nodes alone: a null index entry gives a null row of a 2-D
column, a source with no row is passed through, and the lifetime rule
(sources dropped, the index only by its last consumer).

Requires CUDA and the SF1 dataset.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import constants
from operators import hashjoin, join_core
from operators.partitions import max_load
from tensor_group import TensorGroup
from tests.uvm.conftest import MIB, requires_cuda
from tests.uvm.helpers import (C_CUSTKEY, C_MASK, JOIN_COLUMNS, O_CUSTKEY,
                               O_ORDERDATE, O_ORDERKEY, O_PRIORITY,
                               both_modes, join_args, join_graph, sf1_columns,
                               sorted_rows, streaming_out)
from uvm import executor
from uvm.column import Payloads, UColumn
from uvm.gather import gather_rows, has_null, run_gather

@pytest.fixture
def cores(monkeypatch):
    """Counts of the core calls each path made: the unique-hash core, the
    hash-set semi/anti core and the sort/search core."""
    calls = {'unique_hash': 0, 'semi_anti': 0, 'sort_search': 0}

    def spy(key, fn):
        def wrapped(*a, **kw):
            calls[key] += 1
            return fn(*a, **kw)
        return wrapped

    monkeypatch.setattr(join_core, 'unique_hash_pairs',
                        spy('unique_hash', join_core.unique_hash_pairs))
    monkeypatch.setattr(join_core, 'semi_anti',
                        spy('semi_anti', join_core.semi_anti))
    monkeypatch.setattr(hashjoin, '_sort_search_pairs',
                        spy('sort_search', hashjoin._sort_search_pairs))
    return calls


def _budget_for(est_sides, B):
    """A partition budget the estimate turns into ``B``."""
    return math.ceil(sum(rows / keys * max_load(keys, B) * row_b
                         for rows, row_b, keys in est_sides if rows and keys))


_CASES = [('inner_unique_hash', 16), ('inner_sort_search', 16),
          ('semi', 1), ('anti', 16), ('right_outer', 16)]
_CORE_OF = {'inner_unique_hash': 'unique_hash',
            'inner_sort_search': 'sort_search', 'semi': 'semi_anti',
            'anti': 'semi_anti', 'right_outer': 'sort_search'}


@requires_cuda
@pytest.mark.parametrize("kind,B", _CASES,
                         ids=[f"{k}-B{b}" for k, b in _CASES])
def test_join_equals_the_streaming_join(sf1_parquet, budgets, cores, kind, B):
    from IO.pages.pinned_mem import PinnedMemory

    cols = sf1_columns(sf1_parquet, JOIN_COLUMNS)
    args, masks = join_args(kind, cols)
    col_ids = args['input ids']
    inputs = {cid: cols[cid] for cid in col_ids}
    for mid, m in masks.items():
        inputs[mid] = UColumn(m, 'bool', selectivity=int(m.sum()) / m.numel())

    if B == 1:
        budgets(256 * MIB, 512 * MIB)
    else:
        budgets(4 * MIB, 512 * MIB)
        keys = hashjoin.join_call_args(args, inputs)[:2]
        _, _, est_sides, _ = hashjoin.join_partition_count(args, keys, inputs)
        budgets(4 * MIB, _budget_for(est_sides, B))
    keys = hashjoin.join_call_args(args, inputs)[:2]
    assert hashjoin.join_partition_count(args, keys, inputs)[0] == B

    insts = join_graph(args, col_ids + sorted(masks))
    pool = PinnedMemory(capacity_gb=2, block_size_mb=0.5)
    streaming, uvm, pl = both_modes(inputs, insts, pool)

    lk, rk, conds, join_type = hashjoin.join_call_args(args, streaming)
    s_args = dict(args, **{'mask map': dict(args['mask map'])})
    hashjoin.tqp_join(streaming, s_args, lk, rk, conds, join_type,
                      cpu_mem_pool=pool, cache_region=None, name="t_join")
    streaming_cores = dict(cores)
    assert streaming_cores[_CORE_OF[kind]] > 0, streaming_cores
    cores.update({k: 0 for k in cores})

    executor.run('Join', 5, args, uvm, None, None, pl)
    assert cores[_CORE_OF[kind]] > 0, cores
    if B > 1:
        assert cores[_CORE_OF[kind]] > 1, "one core call: the case packs no calls"

    out_ids = args['materialize_now']
    assert sorted(uvm.keys()) == sorted(out_ids)
    got = [uvm[c].tensor for c in out_ids]
    expect = [streaming_out(streaming, c) for c in out_ids]
    assert len({t.shape[0] for t in got}) == 1
    assert got[0].shape[0] == expect[0].shape[0] > 0
    for c, t in zip(out_ids, got):
        assert uvm[c].type_name == cols[c].type_name, c
        assert t.dtype == cols[c].tensor.dtype, c
    np.testing.assert_array_equal(sorted_rows(got), sorted_rows(expect))
    if kind == 'right_outer':
        # Customers with no order under the mask: one null row each.
        null = uvm[O_ORDERKEY].tensor == constants.null
        assert 0 < int(null.sum()) < null.numel()
        assert bool((uvm[O_PRIORITY].tensor[null] == constants.null).all())
        assert bool((uvm[O_ORDERDATE].tensor[null] == constants.null).all())
    if kind in ('semi', 'anti'):
        kept = torch.isin(uvm[O_CUSTKEY].tensor,
                          cols[C_CUSTKEY].tensor[masks[C_MASK]])
        assert bool(kept.all() if kind == 'semi' else (~kept).all())
    # Every input but the outputs is gone, from the tensor group and from
    # the payloads.
    for cid, inst in insts.items():
        assert pl.get(inst) is None, cid


@requires_cuda
def test_gather_rows_with_null_rows_of_a_string(budgets):
    src = torch.arange(40, dtype=torch.int8, device='cuda').reshape(10, 4)
    idx = torch.tensor([3, constants.null, 0, 9, constants.null],
                       device='cuda')
    budgets(2 * (4 + 17))
    out = gather_rows(src, idx, has_null(idx))
    assert out.tolist() == [src[3].tolist(), [constants.null] * 4,
                            src[0].tolist(), src[9].tolist(),
                            [constants.null] * 4]
    f = torch.tensor([1.5, 2.5], dtype=torch.float64, device='cuda')
    idx = torch.tensor([1, constants.null], device='cuda')
    out = gather_rows(f, idx, has_null(idx))
    assert out.tolist() == [2.5, float(constants.null)]


@requires_cuda
def test_gather_node_lifetimes():
    from dataflow.dataflow_gen import ColumnInstance, PipeNode, PlanGraph
    plan = PlanGraph(0)
    producer = PipeNode(0, "core", "core", plan)
    first = PipeNode(1, "rearrange", "rearrange", plan)
    second = PipeNode(2, "deferred", "deferred", plan)
    pl = Payloads()

    def inst(local, col, consumers):
        i = ColumnInstance(producer=producer, local=local)
        i.consumers = consumers
        pl.publish(i, col)
        return i

    index = inst("out_index_left", UColumn(
        torch.tensor([2, 0, constants.null], device='cuda'), 'int'),
        [first, second])
    a = inst(7, UColumn(torch.tensor([10, 11, 12], device='cuda'), 'int'),
             [first])
    empty_col = UColumn(torch.empty(0, 3, dtype=torch.int8, device='cuda'),
                        'string')
    empty = inst(8, empty_col, [first])
    first.inputs = [a, empty, index]
    first.outputs = [ColumnInstance(producer=first, local=7),
                     ColumnInstance(producer=first, local=8)]
    tg = TensorGroup()
    tg[7], tg[8] = pl.get(a), empty_col
    run_gather(first, tg, pl)
    assert tg[7].tensor.tolist() == [12, 10, constants.null]
    assert pl.get(first.outputs[0]) is tg[7] and pl.get(a) is None
    # A source with no row is the output itself.
    assert tg[8] is empty_col and pl.get(first.outputs[1]) is empty_col
    # The index has a later consumer.
    assert pl.get(index) is not None

    b = inst(9, UColumn(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64,
                                     device='cuda'), 'float'), [second])
    second.inputs = [b, index]
    second.outputs = [ColumnInstance(producer=second, local=9)]
    run_gather(second, tg, pl)
    assert tg[9].tensor.tolist() == [3.0, 1.0, float(constants.null)]
    assert pl.get(b) is None and pl.get(index) is None
