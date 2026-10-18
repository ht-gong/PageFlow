"""The row bounds of the UVM executor's kernel calls.

Every kernel call of the executor sees at most one morsel
(``uvm.morsel.MORSEL_ROWS_MAX`` = 2^30 rows) or a bound the code checks
with an error naming the operator.  Hand-picked cases:

* real inputs above 2^31 rows of 1-byte columns: a filter over a string
  column of 2^31 + 7 rows at the 2^30-row morsel cap; the mask -> kept
  rows path of a keyed aggregation (its masked rows compacted per morsel)
  and of a join side (each morsel of pre-mask rows partitioned with its
  first row as ``goff``, at the cap), each against a direct computation of
  the few kept rows;
* the cap lowered (``uvm.morsel.MORSEL_ROWS_MAX`` monkeypatched, the chunk
  budget out of the way) for a join on SF1 customer and orders columns
  (several partition morsels a side, payloads gathered by ``out_index``
  morsel by morsel), a keyed aggregation on SF1 lineitem columns and a
  gather with null entries, each equal to the one-morsel result;
* a sort of 2^31 - 1 input rows and a sort/search core call whose
  residual conditions would take more pairs than ``nonzero`` counts raise
  an error naming the operator.

Requires CUDA; the cases on SF1 columns also need the SF1 dataset.  The
large cases hold about 10 GiB of the test arena at their peak.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

import constants
from operators import hashjoin
from operators.operator_utils import INT_ROWS_MAX
from operators.sort import sort_tensors
from tensor_group import TensorGroup
from tests.uvm.conftest import MIB, requires_cuda
from tests.uvm.helpers import (DISC, JOIN_COLUMNS, MASK, O_CUSTKEY, PRICE,
                               SUPP, Node, agg_args, agg_nodes,
                               assert_same_groups, column, join_args,
                               join_graph, lineitem, quantity_mask,
                               sf1_columns, sorted_rows)
from uvm import executor, morsel
from uvm import join as uvm_join
from uvm.buckets import split_buckets
from uvm.column import Payloads, UColumn
from uvm.gather import gather_rows, has_null

GIB = 1024 ** 3
N_LARGE = (1 << 31) + 7


@pytest.fixture
def partitions(monkeypatch):
    """The ``(rows, goff)`` of every partition call the UVM join makes, one
    per morsel of a side."""
    from operators.partition_sort import partition
    calls = []

    def spy(cols, key_pos, m, goff, B, n_valid=None):
        calls.append((cols[0].shape[0], goff))
        return partition(cols, key_pos, m, goff, B, n_valid=n_valid)
    monkeypatch.setattr(uvm_join, 'partition', spy)
    return calls


def _kept_rows(mask: torch.Tensor):
    """The positions of ``mask``'s True rows, one 2^30-row slice at a time
    (``nonzero`` takes fewer than 2^31 elements)."""
    out = []
    step = 1 << 30
    for a in range(0, mask.shape[0], step):
        out += (torch.nonzero(mask[a:a + step]).view(-1) + a).tolist()
    return out


# Positions of the kept rows: each side of every 2^30 morsel boundary, the
# int32 boundary and the last row.
_POSITIONS = [0, (1 << 30) - 1, 1 << 30, (1 << 31) - 1, 1 << 31, N_LARGE - 1]


@requires_cuda
def test_filter_over_more_than_2_31_rows(budgets):
    budgets(4 * GIB)
    assert morsel.morsel_rows(1) == morsel.MORSEL_ROWS_MAX == 1 << 30
    flag = torch.full((N_LARGE,), ord('A'), dtype=torch.int8, device='cuda')
    flag[_POSITIONS] = ord('R')
    tg, pl = TensorGroup(), Payloads()
    insts = [column(tg, pl, 1, flag, 'string')]
    node = Node(insts, [50])
    args = {'name': 'Filter', 'filter tree': ['f#1', '=', 'R'],
            'materialize_in': [1], 'input ids': [1], 'output ids': [1],
            'out mask id': 50, 'nodes': {'pipe': node}}
    executor.run('Filter', 2, args, tg, None, None, pl)
    mask = tg[50].tensor
    assert mask.dtype == torch.bool and mask.shape[0] == N_LARGE
    assert _kept_rows(mask) == _POSITIONS
    assert tg[50].selectivity == len(_POSITIONS) / N_LARGE


@requires_cuda
def test_keyed_aggregation_mask_over_more_than_2_31_rows(budgets):
    from dataflow.dataflow_gen import PlanGraph
    budgets(64 * MIB)
    KEY, CNT = 1, 200
    key = torch.full((N_LARGE,), ord('A'), dtype=torch.int8, device='cuda')
    mask = torch.zeros(N_LARGE, dtype=torch.bool, device='cuda')
    # Kept rows of three keys, and a key on rows the mask drops.
    letters = [ord(c) for c in "BCBDCB"]
    key[_POSITIONS] = torch.tensor(letters, dtype=torch.int8, device='cuda')
    key[1 << 29] = ord('Z')
    mask[_POSITIONS] = True

    tg, pl = TensorGroup(), Payloads()
    insts = [column(tg, pl, KEY, key, 'string'),
             column(tg, pl, MASK, mask, 'bool',
                    selectivity=len(_POSITIONS) / N_LARGE)]
    nodes = agg_nodes(PlanGraph(0), insts, 1, 1, [KEY, CNT])
    args = agg_args([('k', KEY)], [('count', '1', CNT)], [KEY], MASK)
    assert len(list(morsel.morsels(N_LARGE, 2))) > 64
    executor.run('Aggregate', 1, dict(args, nodes=nodes), tg, None, None, pl)
    got = dict(zip(tg[KEY].tensor.tolist(), tg[CNT].tensor.tolist()))
    expect = {k: letters.count(k) for k in set(letters)}
    assert got == expect


@requires_cuda
def test_join_side_partition_over_more_than_2_31_rows(budgets, partitions):
    budgets(16 * GIB)
    KEY, MID = 1, 2
    # One key value for every row (a stride-0 view: each morsel's slice is
    # made contiguous by the partition, 8 GiB at the cap).
    key = torch.full((1,), 42, dtype=torch.int64,
                     device='cuda').expand(N_LARGE)
    mask = torch.zeros(N_LARGE, dtype=torch.bool, device='cuda')
    mask[_POSITIONS] = True
    inputs = {KEY: UColumn(key, 'int'), MID: UColumn(mask, 'bool')}
    parts = uvm_join._partition_side(inputs, [KEY], 0, MID, 4, True, "t_join")
    # Three morsels at the cap, each with a kept row.
    assert partitions == [(1 << 30, 0), (1 << 30, 1 << 30), (7, 1 << 31)]
    rows = sum(counts for _, counts, _ in parts)
    assert rows.sum() == len(_POSITIONS) and np.count_nonzero(rows) == 1
    (keys, gidx), = split_buckets(parts, [0, 4], bucket_major=True)
    assert keys.tolist() == [42] * len(_POSITIONS)
    assert gidx.tolist() == _POSITIONS


# ── The cap lowered: several morsels against one ────────────────────────


@requires_cuda
def test_join_over_lowered_morsel_cap_equals_one_morsel(sf1_parquet,
                                                        partitions,
                                                        monkeypatch):
    """A right outer join (orders under a mask, its payloads gathered with
    null rows) with the cap at 2^16 rows: orders' 1.5M rows are 23
    partition morsels, each with its first row as ``goff``, and each
    gather runs per 2^16 rows of its index."""
    def run():
        cols = sf1_columns(sf1_parquet, JOIN_COLUMNS)
        args, masks = join_args('right_outer', cols)
        tg, pl = TensorGroup(), Payloads()
        for cid in args['input ids']:
            tg[cid] = cols[cid]
        for mid, m in masks.items():
            tg[mid] = UColumn(m, 'bool', selectivity=int(m.sum()) / m.numel())
        insts = join_graph(args, args['input ids'] + sorted(masks))
        for cid, inst in insts.items():
            tg[cid] = pl.publish(inst, tg[cid])
        executor.run('Join', 5, args, tg, None, None, pl)
        out = args['materialize_now']
        return [tg[c].tensor for c in out], cols[O_CUSTKEY].col_length()

    one, orders = run()
    assert len(partitions) == 2
    partitions.clear()
    monkeypatch.setattr(morsel, 'MORSEL_ROWS_MAX', 1 << 16)
    many, _ = run()
    starts = list(range(0, orders, 1 << 16))
    assert len(starts) > 20
    # The orders side's morsels keep a row each, then customer's three.
    assert [goff for _, goff in partitions] == starts + [0, 1 << 16, 2 << 16]
    assert many[0].shape[0] > 1 << 16
    np.testing.assert_array_equal(sorted_rows(many), sorted_rows(one))


@requires_cuda
def test_keyed_aggregation_over_lowered_morsel_cap_equals_one_morsel(
        sf1_parquet, budgets, monkeypatch):
    """SUM, COUNT and AVG by ``l_suppkey`` under a mask with the cap at
    2^20 rows (six morsels, so pass 2 merges each group's six partials)
    against one morsel, at a chunk budget that never binds."""
    from dataflow.dataflow_gen import PlanGraph
    budgets(4 * GIB)
    cols = lineitem(sf1_parquet, ['l_suppkey', 'l_quantity',
                                   'l_extendedprice', 'l_discount'])
    mask = quantity_mask(cols)
    outputs = [SUPP, 200, 201, 202]

    def run():
        tg, pl = TensorGroup(), Payloads()
        insts = [column(tg, pl, cid, col.tensor, col.type_name,
                        selectivity=col.selectivity)
                 for cid, col in ((SUPP, cols['l_suppkey']),
                                  (PRICE, cols['l_extendedprice']),
                                  (DISC, cols['l_discount']), (MASK, mask))]
        nodes = agg_nodes(PlanGraph(0), insts, 1, 3, outputs)
        args = agg_args([('l_suppkey', SUPP)],
                         [('sum', f'l_extendedprice#{PRICE}', 200),
                          ('count', '1', 201),
                          ('avg', f'l_discount#{DISC}', 202)],
                         [SUPP, PRICE, DISC], MASK)
        executor.run('Aggregate', 1, dict(args, nodes=nodes), tg, None,
                     None, pl)
        return {c: tg[c].tensor for c in outputs}

    rows = mask.col_length()
    assert len(list(morsel.morsels(rows, 8 + 8 + 8 + 1))) == 1
    one = run()
    monkeypatch.setattr(morsel, 'MORSEL_ROWS_MAX', 1 << 20)
    assert len(list(morsel.morsels(rows, 8 + 8 + 8 + 1))) == 6
    many = run()
    assert_same_groups(one, many, [SUPP], [200, 201, 202])
    assert int(many[201].sum()) == int(mask.tensor.sum())


@requires_cuda
def test_gather_over_lowered_morsel_cap_equals_one_morsel(monkeypatch):
    g = torch.Generator(device='cpu').manual_seed(3)
    src = torch.randint(-50, 100, (1000, 5), dtype=torch.int8,
                        generator=g).cuda()
    idx = torch.randint(0, 1000, (3001,), generator=g)
    idx[torch.rand(3001, generator=g) < 0.1] = constants.null
    idx = idx.cuda()
    one = gather_rows(src, idx, has_null(idx))
    monkeypatch.setattr(morsel, 'MORSEL_ROWS_MAX', 100)
    assert len(list(morsel.morsels(idx.shape[0], 5 + 17))) == 31
    many = gather_rows(src, idx, has_null(idx))
    assert torch.equal(one, many)
    null = idx == constants.null
    assert bool((many[null] == constants.null).all())
    assert torch.equal(many[~null], src[idx[~null]])


# ── Bounds checked before the kernel ────────────────────────────────────


@requires_cuda
def test_sort_over_its_row_bound_raises():
    rows = INT_ROWS_MAX + 1
    # Stride-0 views: the bound is checked on the row count, before any
    # kernel reads the rows.
    key = torch.zeros(1, dtype=torch.int8, device='cuda').expand(rows)
    mask = torch.ones(1, dtype=torch.bool, device='cuda').expand(rows)
    args = {"input ids": [1], "key ids": [1], "key orders": [False],
            "limit": 10}
    with pytest.raises(RuntimeError,
                       match=r"sort lim 9: 2147483647 input rows are over "
                             r"the 2147483646 rows one sort takes"):
        sort_tensors({1: key}, mask, args, True, "sort lim 9")


@requires_cuda
def test_sort_search_pairs_over_their_bound_raise(monkeypatch):
    """A many-to-many call with a residual condition: its pairs over the
    bound raise before the conditions run (the bound lowered to 99 here,
    2^31 - 2 in the engine), at the bound they are filtered."""
    left = torch.zeros(10, dtype=torch.int64, device='cuda')
    right = torch.zeros(10, dtype=torch.int64, device='cuda')
    lv = torch.arange(10, dtype=torch.float64, device='cuda')
    rv = torch.arange(10, dtype=torch.float64, device='cuda')
    tl, tr = {1: left, 2: lv}, {11: right, 12: rv}
    gl = torch.arange(10, dtype=torch.int64, device='cuda')
    cond = [['a#2', '<', 'b#12']]

    def call():
        return hashjoin.join_kernel_new(
            tl, tr, 1, 11, cond, [1], True, 'inner', gl, gl.clone(),
            name="join 7")

    monkeypatch.setattr(hashjoin, 'INT_ROWS_MAX', 99)
    with pytest.raises(RuntimeError,
                       match=r"join 7: a core call of 10 left and 10 right "
                             r"rows makes 100 pairs, over the 99"):
        call()
    monkeypatch.setattr(hashjoin, 'INT_ROWS_MAX', 100)
    keys, out_left, out_right = call()
    assert keys.numel() == out_left.numel() == 45
    assert bool((lv[out_left] < rv[out_right]).all())
