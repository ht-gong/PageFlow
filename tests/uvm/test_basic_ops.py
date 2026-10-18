"""The UVM executor's operators (``uvm/executor.py``) on ordinary CUDA
tensors.

The kernels cannot tell managed memory from device memory, so these run in
the pytest process on its device arena; ``uvm.scan.scan_to_managed`` reads
the SF1 columns without its prefetch to the host node, which needs managed
memory.  Hand-picked cases, each against a direct torch computation, the
streaming operator or the SF1 reference:

* Q6 through the executor at a 1 MiB chunk (morsels of about 37,000
  lineitem rows): the filter's mask equal to one evaluation over the whole
  columns, its selectivity, the revenue against torch and the reference at
  10 significant figures, and every input dropped, its payloads
  included;
* Q22's subquery plan (a substring/IN filter on a string column, a masked
  AVG over several morsels) against torch's mean of the same rows;
* a project with a rename and two computed columns, a float and a 2-D
  substring, across morsels;
* a filter, a project (a rename, a float and a 2-D substring output) and a
  masked global aggregation (SUM, COUNT, AVG) on SF1 lineitem columns,
  bit-equal to the streaming ``tqp_filter``, ``tqp_project`` and
  ``tqp_hash_aggregate`` (one tick and one morsel each, so the same
  summation order);
* a sort under a mask with a LIMIT, equal to the streaming ``tqp_sort``;
* a masked global aggregation (SUM, COUNT, AVG) merged across many
  morsels, one whose mask keeps no row and one over no rows;
* the morsel sizes.

Requires CUDA; the shared custom allocator is bootstrapped in
tests/conftest.py.
"""
from __future__ import annotations

import math
import pathlib

import pandas as pd
import pytest
import torch

from tensor_group import TensorGroup
from tests.uvm.conftest import MIB, requires_cuda
from tests.uvm.helpers import (Node, both_modes, column, lineitem,
                               query_plans, read_by, scan_instances,
                               streaming_out)
from uvm import executor
from uvm.column import Payloads, UColumn
from uvm.morsel import MORSEL_ROWS_MAX, morsel_rows, morsels

_REFERENCE = (pathlib.Path(__file__).resolve().parents[2] / "tests"
              / "verification" / "SF1_reference.out")


def _run(plan, schema, root, stop_after=None):
    """Run ``plan`` through the executor on the SF1 data under ``root``, up
    to its output (or up to operator ``stop_after``).  Returns the tensor
    group, the output frame (None when stopped early) and the payloads."""
    from uvm.scan import scan_to_managed
    scan_group = scan_to_managed({'tables path': root}, schema, 1,
                                 plan.disk_read_cols)
    tensor_group, payloads = TensorGroup(), Payloads()
    for u in plan.ops_idx:
        args = plan.ops_dict[u]
        if args['name'] == 'AdaptiveSparkPlan':
            return tensor_group, executor.output(tensor_group, args), payloads
        executor.run(args['name'], u, args, tensor_group, scan_group, None,
                     payloads)
        if u == stop_after:
            return tensor_group, None, payloads
    raise AssertionError("the plan has no output")


@requires_cuda
def test_q6_matches_the_reference_over_many_morsels(sf1_parquet, budgets):
    from operators.operator_utils import evaluate_filter
    from tests.sigfig_compare import (SIGFIGS_THRESHOLD, compare_results,
                                      results_of)
    budgets(MIB)
    plans, schema = query_plans(6)
    (plan,) = plans
    filter_u = next(u for u in plan.ops_idx
                    if plan.ops_dict[u]['name'] == 'Filter')
    filter_args = plan.ops_dict[filter_u]
    tg, _, _ = _run(plan, schema, sf1_parquet, stop_after=filter_u)
    mask = tg[filter_args['out mask id']]
    whole = {c: tg[c].tensor for c in filter_args['materialize_in']}
    rows = mask.col_length()
    assert morsel_rows(sum(tg[c].row_bytes() for c in whole)) * 100 < rows
    expect = evaluate_filter('cuda', whole, filter_args['filter tree'], rows, None)
    assert torch.equal(mask.tensor, expect)
    assert mask.selectivity == int(expect.sum()) / rows

    tg, frame, payloads = _run(plan, schema, sf1_parquet)
    scanned = plan.ops_dict[plan.ops_idx[0]]['nodes']['scan'].outputs
    # Only the result is left, and no payload holds an input any more.
    assert list(tg.keys()) == [122]
    assert all(payloads.get(inst) is None for inst in scanned)

    pd.options.display.float_format = '{:.6f}'.format
    compare_results(frame.to_string() + "\n",
                    results_of(_REFERENCE.read_text(encoding="utf-8"), [6]),
                    SIGFIGS_THRESHOLD)
    from uvm.scan import scan_to_managed
    cols = scan_to_managed({'tables path': sf1_parquet}, schema, 1,
                           plan.disk_read_cols)['LINEITEM']
    price = cols['L_EXTENDEDPRICE'][0].tensor[expect]
    disc = cols['L_DISCOUNT'][0].tensor[expect]
    revenue = float((price * disc).sum())
    assert math.isclose(frame['revenue'][0], revenue, rel_tol=1e-12)


@requires_cuda
def test_q22_subquery_avg_over_a_string_filter(sf1_parquet, budgets):
    budgets(MIB)
    plans, schema = query_plans(22)
    plan = plans[0]
    _, frame, _ = _run(plan, schema, sf1_parquet)
    from uvm.scan import scan_to_managed
    cols = scan_to_managed({'tables path': sf1_parquet}, schema, 1,
                           plan.disk_read_cols)['CUSTOMER']
    phone = cols['C_PHONE'][0].tensor
    bal = cols['C_ACCTBAL'][0].tensor
    assert morsel_rows(phone.shape[1] + 8) * 3 < phone.shape[0]
    code = torch.zeros(phone.shape[0], dtype=torch.bool, device='cuda')
    for cc in ('13', '31', '23', '29', '30', '18', '17'):
        code |= (phone[:, 0] == ord(cc[0])) & (phone[:, 1] == ord(cc[1]))
    keep = code & (bal > 0.0)
    assert 0 < int(keep.sum()) < keep.numel()
    assert math.isclose(frame.iloc[0, 0], float(bal[keep].mean()), rel_tol=1e-12)


@requires_cuda
def test_project_with_a_rename_across_morsels(budgets):
    budgets(0.01 * MIB)
    n = 5000
    g = torch.Generator(device='cpu').manual_seed(7)
    price = (torch.rand(n, generator=g, dtype=torch.float64) * 1000).cuda()
    disc = (torch.rand(n, generator=g, dtype=torch.float64) / 10).cuda()
    phone = torch.randint(48, 58, (n, 15), dtype=torch.int8, generator=g).cuda()
    tg, pl = TensorGroup(), Payloads()
    insts = [column(tg, pl, 1, price, 'float'),
             column(tg, pl, 2, disc, 'float'),
             column(tg, pl, 3, phone, 'string')]
    node = Node(insts, [20, 21])
    args = {'name': 'Project', 'rename map': {3: 30},
            'used expr': [[['p#1', '*', ['1.0', '-', 'd#2']], 'AS', 'x#20'],
                          [['substring', ['ph#3', '1', '2']], 'AS', 'cc#21'],
                          ['p#1']],
            'materialize_in': [1, 2, 3], 'materialize_now': [20, 21],
            'input ids': [1, 2, 3], 'output ids': [1, 20, 21, 30],
            'nodes': {'pipe': node}}
    assert morsel_rows(8 + 8 + 15) * 10 < n
    executor.run('Project', 3, args, tg, None, None, pl)
    assert torch.equal(tg[20].tensor, price * (1.0 - disc))
    assert tg[20].type_name == 'float'
    assert torch.equal(tg[21].tensor, phone[:, 0:2])
    assert tg[21].type_name == 'string' and tg[21].col_width() == 2
    assert pl.get(node.outputs[0]) is tg[20]
    assert pl.get(node.outputs[1]) is tg[21]
    # The renamed column stays (under its new id, and as its instance's
    # payload); the dropped one's payload goes.
    assert sorted(tg.keys()) == [1, 20, 21, 30]
    assert tg[30].tensor is phone and pl.get(insts[2]) is tg[30]
    assert pl.get(insts[1]) is None and pl.get(insts[0]) is tg[1]


@requires_cuda
def test_sort_with_a_mask_and_limit_equals_the_streaming_sort():
    from operators.sort import tqp_sort
    from IO.pages.pinned_mem import PinnedMemory
    from tests.scan_table import pinned_column
    KEY, ROW, NAME, MASK = 1, 2, 3, 10000
    n = 5000
    g = torch.Generator(device='cpu').manual_seed(11)
    key = torch.randint(0, 50, (n,), dtype=torch.int64, generator=g)
    row = torch.arange(n, dtype=torch.int64)
    name = torch.randint(65, 67, (n, 4), dtype=torch.int8, generator=g)
    mask = torch.rand(n, generator=g) < 0.3
    args = {"input ids": [KEY, ROW, NAME], "key ids": [KEY, NAME],
            "key orders": [True, False], "limit": 10,
            "mask map": {KEY: MASK, ROW: MASK, NAME: MASK}}
    pool = PinnedMemory(capacity_gb=0.25, block_size_mb=0.5)
    streaming = {cid: pinned_column(t, typ, pool)
                 for cid, t, typ in ((KEY, key, "int"), (ROW, row, "int"),
                                     (NAME, name, "string"),
                                     (MASK, mask, "bool"))}
    rows = tqp_sort(streaming, args, with_limit=True, name="t_sort")

    tg, pl = TensorGroup(), Payloads()
    for cid, t, typ in ((KEY, key, 'int'), (ROW, row, 'int'),
                        (NAME, name, 'string'), (MASK, mask, 'bool')):
        column(tg, pl, cid, t.cuda(), typ)
    assert executor.sort(tg, args, pl, with_limit=True) == rows == int(mask.sum())
    # The sorted inputs and the mask are dropped, their payloads included.
    assert MASK not in tg and len(pl) == 0
    for cid in (KEY, ROW, NAME):
        got = tg[cid].tensor.cpu()
        expect = streaming[cid].reps[0].col_to_gpu().cpu()
        assert got.shape[0] == 10
        assert torch.equal(got.reshape(expect.shape), expect)
    assert tg[NAME].type_name == 'string' and tg[NAME].col_width() == 4


def _global_args(node):
    return {'name': 'Aggregate', 'keys': None, 'input ids': [1],
            'mask map': {1: 2},
            'results': [[[[['sum', ['x#1']], '#', '11'], 'AS', 's#10']],
                        [[[['count', ['1']], '#', '13L'], 'AS', 'c#12L']],
                        [[[['avg', ['x#1']], '#', '15'], 'AS', 'a#14']]],
            'out agg ids': [10, 12, 14], 'agg ops': ['sum', 'count', 'avg'],
            'output ids': [10, 12, 14], 'nodes': {'pipe': node}}


@requires_cuda
@pytest.mark.parametrize("rows", [0, 3000], ids=["no-rows", "no-row-kept"])
def test_global_aggregate_of_no_rows(budgets, rows):
    budgets(0.001 * MIB)
    tg, pl = TensorGroup(), Payloads()
    insts = [column(tg, pl, 1, torch.arange(rows, dtype=torch.int64, device='cuda'), 'int'),
             column(tg, pl, 2, torch.zeros(rows, dtype=torch.bool, device='cuda'),
                     'bool', selectivity=0.0)]
    node = Node(insts, [10, 12, 14])
    executor.run('Aggregate', 6, _global_args(node), tg, None, None, pl)
    assert sorted(tg.keys()) == [10, 12, 14]
    assert tg[10].tensor.tolist() == [0] and tg[12].tensor.tolist() == [0]
    assert math.isnan(float(tg[14].tensor[0]))
    assert all(pl.get(inst) is None for inst in insts)


@requires_cuda
def test_global_aggregate_merges_its_morsels(budgets):
    budgets(0.001 * MIB)
    n = 3000
    g = torch.Generator(device='cpu').manual_seed(5)
    # No -64: COUNT reads it as null.
    x = torch.randint(0, 1000, (n,), dtype=torch.int64, generator=g)
    keep = torch.rand(n, generator=g) < 0.4
    tg, pl = TensorGroup(), Payloads()
    insts = [column(tg, pl, 1, x.cuda(), 'int'),
             column(tg, pl, 2, keep.cuda(), 'bool')]
    node = Node(insts, [10, 12, 14])
    assert len(list(morsels(n, 9))) > 10
    executor.run('Aggregate', 6, _global_args(node), tg, None, None, pl)
    kept = x[keep]
    assert tg[10].tensor.tolist() == [int(kept.sum())]
    assert tg[12].tensor.tolist() == [kept.numel()]
    assert math.isclose(float(tg[14].tensor[0]),
                        float(kept.sum()) / kept.numel(), rel_tol=1e-12)
    assert [tg[c].type_name for c in (10, 12, 14)] == ['int', 'int', 'float']


# ── Against the streaming operators, on SF1 lineitem columns ─────────────
#
# Each case runs the streaming operator (a VortexPipeline over the scan's
# record tables in a pinned pool, ``tests/scan_table.py``) and
# the executor on the same columns, at the default 256 MiB chunk: lineitem's
# 6M rows are one tick of the streaming pipe and one morsel of the
# executor, so both evaluate the same rows in the same order and the
# results are bit-equal.

_Q6_FILTER = [[[[[[[[['isnotnull', ['l_shipdate#102']], 'AND',
                     ['isnotnull', ['l_discount#98']]], 'AND',
                    ['isnotnull', ['l_quantity#96']]], 'AND',
                   ['l_shipdate#102', '>=', '1994-01-01']], 'AND',
                  ['l_shipdate#102', '<', '1995-01-01']], 'AND',
                 ['l_discount#98', '>=', '0.05']], 'AND',
                ['l_discount#98', '<=', '0.07']], 'AND',
               ['l_quantity#96', '<', '24']]]


def _one_node(columns, kind, outputs):
    """``columns`` (``{col id: UColumn}``) as both modes' inputs to one
    pipeline node of ``kind`` (``helpers.both_modes``): the streaming
    tensor group, whose pinned pool also takes the operator's outputs, the
    UVM one, the node, the pool and the payloads."""
    from dataflow.dataflow_gen import ColumnInstance, PipeNode, PlanGraph
    from IO.pages.pinned_mem import PinnedMemory
    plan = PlanGraph(0)
    insts = scan_instances(plan, columns)
    node = PipeNode(1, kind, f"{kind} 1 {kind} pipe", plan, position=0)
    plan.nodes.append(node)
    read_by(node, insts.values())
    node.outputs = [ColumnInstance(producer=node, local=c) for c in outputs]
    pool = PinnedMemory(capacity_gb=1, block_size_mb=0.5)
    streaming, uvm, payloads = both_modes(columns, insts, pool)
    return streaming, uvm, node, pool, payloads


@requires_cuda
def test_filter_equals_the_streaming_filter(sf1_parquet):
    from operators.filter import tqp_filter
    cols = lineitem(sf1_parquet, ['l_quantity', 'l_discount', 'l_shipdate'])
    streaming, uvm, node, pool, pl = _one_node(
        {96: cols['l_quantity'], 98: cols['l_discount'],
         102: cols['l_shipdate']}, "filter", [10000])
    args = {'name': 'Filter', 'input ids': [96, 98, 102],
            'filter tree': _Q6_FILTER, 'output ids': [96, 98, 102],
            'materialize_in': [96, 98, 102], 'out mask id': 10000,
            'nodes': {'pipe': node}}
    tqp_filter(streaming, dict(args), None, cpu_mem_pool=pool, cache_region=None,
               name="t_filter")
    executor.run('Filter', 2, dict(args), uvm, None, None, pl)
    expect = streaming_out(streaming, 10000)
    assert expect.dtype == torch.bool and 0 < int(expect.sum()) < expect.numel()
    assert torch.equal(uvm[10000].tensor, expect)
    assert uvm[10000].selectivity == streaming[10000].reps[0].selectivity


@requires_cuda
def test_project_equals_the_streaming_project(sf1_parquet):
    from operators.project import tqp_project
    cols = lineitem(sf1_parquet, ['l_extendedprice', 'l_discount',
                                   'l_shipmode'])
    streaming, uvm, node, pool, pl = _one_node(
        {97: cols['l_extendedprice'], 98: cols['l_discount'],
         105: cols['l_shipmode']}, "project", [200, 201])

    def args():
        return {'name': 'Project', 'rename map': {97: 300},
                'used expr': [
                    [['l_extendedprice#97', '*', ['1.0', '-', 'l_discount#98']],
                     'AS', 'disc_price#200'],
                    [['substring', ['l_shipmode#105', '1', '3']], 'AS', 'm#201'],
                    ['l_extendedprice#97']],
                'materialize_in': [97, 98, 105], 'materialize_now': [200, 201],
                'input ids': [97, 98, 105], 'output ids': [200, 201, 300],
                'nodes': {'pipe': node}}

    tqp_project(streaming, args(), cpu_mem_pool=pool, cache_region=None,
                name="t_project")
    executor.run('Project', 3, args(), uvm, None, None, pl)
    assert sorted(uvm.keys()) == sorted(streaming.keys()) == [200, 201, 300]
    for cid in (200, 201):
        expect = streaming_out(streaming, cid)
        assert torch.equal(uvm[cid].tensor, expect), cid
        assert uvm[cid].type_name == streaming[cid].reps[0].tensor_type
    assert uvm[201].col_width() == 3
    assert torch.equal(uvm[300].tensor, cols['l_extendedprice'].tensor)


@requires_cuda
def test_global_aggregate_equals_the_streaming_aggregate(sf1_parquet):
    from operators.aggregate import tqp_hash_aggregate
    from operators.operator_utils import evaluate_filter
    cols = lineitem(sf1_parquet, ['l_quantity', 'l_extendedprice',
                                   'l_discount', 'l_shipdate'])
    mask = evaluate_filter(
        'cuda', {96: cols['l_quantity'].tensor, 98: cols['l_discount'].tensor,
                 102: cols['l_shipdate'].tensor},
        _Q6_FILTER, cols['l_quantity'].col_length(), None)
    streaming, uvm, node, pool, pl = _one_node(
        {97: cols['l_extendedprice'], 98: cols['l_discount'],
         10000: UColumn(mask, 'bool',
                        selectivity=int(mask.sum()) / mask.numel())},
        "agg", [122, 123, 124])

    def args():
        return {'name': 'Aggregate', 'keys': None, 'input ids': [97, 98],
                'mask map': {97: 10000, 98: 10000},
                'results': [
                    [[[['sum', ['l_extendedprice#97', '*', 'l_discount#98']],
                       '#', '125'], 'AS', 'revenue#122']],
                    [[[['count', ['1']], '#', '126L'], 'AS', 'n#123L']],
                    [[[['avg', ['l_discount#98']], '#', '127'], 'AS',
                      'avg_disc#124']]],
                'out agg ids': [122, 123, 124],
                'agg ops': ['sum', 'count', 'avg'],
                'output ids': [122, 123, 124], 'nodes': {'pipe': node}}

    assert morsel_rows(8 + 8 + 1) > mask.numel()
    tqp_hash_aggregate(streaming, args(), cache_region=None, cpu_mem_pool=pool,
                       name="t_agg")
    executor.run('Aggregate', 6, args(), uvm, None, None, pl)
    for cid in (122, 123, 124):
        expect = streaming_out(streaming, cid)
        assert torch.equal(uvm[cid].tensor, expect), (
            cid, uvm[cid].tensor, expect)
        assert uvm[cid].type_name == streaming[cid].reps[0].tensor_type
    assert uvm[123].tensor.tolist() == [int(mask.sum())]


def test_morsel_rows():
    assert morsel_rows(28, 256 * MIB) == 256 * MIB // 28
    assert morsel_rows(1, 1 << 40) == MORSEL_ROWS_MAX == 1 << 30
    assert morsel_rows(1 << 30, MIB) == 1
    assert list(morsels(10, 3, chunk_b=12)) == [(0, 4), (4, 8), (8, 10)]
    assert list(morsels(0, 3, chunk_b=12)) == []
