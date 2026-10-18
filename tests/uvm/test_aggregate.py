"""The UVM executor's keyed aggregation (``uvm/aggregate.py``) against the
streaming ``tqp_hash_aggregate`` on SF1 lineitem columns.

Each case runs both on the same columns (the streaming one over the scan's
record tables in a pinned pool, ``tests/scan_table.py``, as
``test_basic_ops.py``'s parity cases do) and compares the groups as sets,
ordered by key: the two emit them in different orders (the UVM pass 2 by
hash within each round, the streaming one by its route's run).  Hand-picked
cases:

* an int key (``l_suppkey``) with SUM, COUNT, MAX, MIN and a compound SUM;
* a string key (``l_shipmode``, 10 bytes) under a mask;
* a composite key (``l_suppkey``, ``l_returnflag``), and the same groups
  with ``l_suppkey`` not emitted;
* AVG only over an int key (``l_partkey``; a dense shape in the streaming
  engine);
* COUNT over a column with nulls (-64), next to COUNT(1);
* B forced to 16 (``AGGREGATION_PARTITIONS``) at a 1 MiB chunk and a
  256 KiB partition budget, so the UVM pass 1 runs many morsels and its
  pass 2 several rounds;
* a mask that keeps no row, and no row at all: empty outputs of the right
  dtypes and widths (the UVM aggregation alone).

Keys, integer values, MAX and MIN are compared exactly.  Float sums are
compared at 10 significant figures: a group's sum is a scatter-add, whose
atomic additions land in an order that varies from run to run even over
the same rows (at the default 256 MiB chunk lineitem's 6M rows are one tick
of the streaming pass 1 and one morsel of the UVM one), and in the forced-B
case the morsels are not the streaming ticks (a tick is whole 2^20-row
quanta), so the partials differ as well.

Requires CUDA and the SF1 dataset.
"""
from __future__ import annotations

import numpy as np
import torch

import constants
from operators import aggregate
from operators.aggregate import KeyedAggregate
from operators.sorted_run_merge import reduce_tick_slice
from tensor_group import TensorGroup
from tests.uvm.conftest import MIB, requires_cuda
from tests.uvm.helpers import (DISC, FLAG, MASK, MODE, NULLABLE, PART,
                               PRICE, QTY, SUPP, agg_args, agg_nodes,
                               assert_same_groups, both_modes, column,
                               lineitem, quantity_mask, scan_instances,
                               streaming_out)
from uvm import aggregate as uvm_aggregate
from uvm import executor
from uvm.column import Payloads, UColumn
from uvm.morsel import morsels
from variable import torch_to_type

def _run_both(columns, args_fn):
    """Run ``args_fn()``'s aggregation in both modes on ``columns``; returns
    the streaming outputs and the UVM ones (``{out id: tensor}``) and the
    UVM tensor group."""
    from dataflow.dataflow_gen import PlanGraph
    from IO.pages.pinned_mem import PinnedMemory
    from operators.aggregate import tqp_hash_aggregate
    probe = args_fn()
    n_keys, n_aggs = len(probe['out key ids']), len(probe['out agg ids'])
    outputs = probe['out key ids'] + probe['out agg ids']
    plan = PlanGraph(0)
    insts = scan_instances(plan, columns)
    nodes = agg_nodes(plan, insts.values(), n_keys, n_aggs, outputs)
    pool = PinnedMemory(capacity_gb=2, block_size_mb=0.5)
    streaming, uvm, pl = both_modes(columns, insts, pool)
    s_args = dict(args_fn(), nodes=nodes)
    tqp_hash_aggregate(streaming, s_args, cache_region=None, cpu_mem_pool=pool,
                       name="t_agg")
    u_args = dict(args_fn(), nodes=nodes)
    executor.run('Aggregate', 1, u_args, uvm, None, None, pl)
    assert sorted(uvm.keys()) == sorted(outputs)
    for i, cid in enumerate(outputs):
        assert pl.get(nodes['pass2'].outputs[i]) is uvm[cid]
        assert uvm[cid].type_name == streaming[cid].reps[0].tensor_type, cid
    got_s = {cid: streaming_out(streaming, cid) for cid in outputs}
    got_u = {cid: uvm[cid].tensor for cid in outputs}
    return got_s, got_u, uvm


@requires_cuda
def test_int_key_equals_streaming(sf1_parquet):
    cols = lineitem(sf1_parquet, ['l_suppkey', 'l_quantity',
                                   'l_extendedprice', 'l_discount'])

    def args():
        return agg_args([('l_suppkey', SUPP)],
                     [('sum', f'l_quantity#{QTY}', 200),
                      ('count', '1', 201),
                      ('max', f'l_extendedprice#{PRICE}', 202),
                      ('min', f'l_discount#{DISC}', 203),
                      ('sum', [f'l_extendedprice#{PRICE}', '*',
                               ['1.0', '-', f'l_discount#{DISC}']], 204)],
                     [SUPP, QTY, PRICE, DISC], None)
    got_s, got_u, _ = _run_both(
        {SUPP: cols['l_suppkey'], QTY: cols['l_quantity'],
         PRICE: cols['l_extendedprice'], DISC: cols['l_discount']}, args)
    assert_same_groups(got_s, got_u, [SUPP], [200, 201, 202, 203, 204])
    assert got_u[SUPP].numel() == 10_000
    assert int(got_u[201].sum()) == cols['l_suppkey'].col_length()


@requires_cuda
def test_string_key_under_a_mask_equals_streaming(sf1_parquet):
    cols = lineitem(sf1_parquet, ['l_shipmode', 'l_quantity',
                                   'l_extendedprice'])

    def args():
        return agg_args([('l_shipmode', MODE)],
                     [('sum', f'l_extendedprice#{PRICE}', 200),
                      ('count', '1', 201)],
                     [MODE, PRICE], MASK)
    mask = quantity_mask(cols)
    got_s, got_u, _ = _run_both(
        {MODE: cols['l_shipmode'], PRICE: cols['l_extendedprice'],
         MASK: mask}, args)
    assert_same_groups(got_s, got_u, [MODE], [200, 201])
    assert tuple(got_u[MODE].shape) == (7, 10)
    assert int(got_u[201].sum()) == int(mask.tensor.sum())


@requires_cuda
def test_composite_key_equals_streaming(sf1_parquet):
    cols = lineitem(sf1_parquet, ['l_suppkey', 'l_returnflag',
                                   'l_extendedprice'])

    def args():
        return agg_args([('l_suppkey', SUPP), ('l_returnflag', FLAG)],
                     [('sum', f'l_extendedprice#{PRICE}', 200),
                      ('count', '1', 201)],
                     [SUPP, FLAG, PRICE], None)
    got_s, got_u, _ = _run_both(
        {SUPP: cols['l_suppkey'], FLAG: cols['l_returnflag'],
         PRICE: cols['l_extendedprice']}, args)
    assert_same_groups(got_s, got_u, [SUPP, FLAG], [200, 201])
    assert got_u[SUPP].numel() == 30_000


@requires_cuda
def test_grouping_by_a_key_it_does_not_emit_equals_streaming(sf1_parquet):
    """GROUP BY (l_returnflag, l_suppkey), emitting l_returnflag only: the
    emitted keys repeat, so the groups are compared as a multiset of rows
    (all exact: the key, COUNT and MAX)."""
    cols = lineitem(sf1_parquet, ['l_suppkey', 'l_returnflag',
                                   'l_extendedprice'])

    def args():
        return agg_args([('l_returnflag', FLAG)],
                     [('count', '1', 200),
                      ('max', f'l_extendedprice#{PRICE}', 201)],
                     [SUPP, FLAG, PRICE], None, hidden=[('l_suppkey', SUPP)])
    got_s, got_u, _ = _run_both(
        {SUPP: cols['l_suppkey'], FLAG: cols['l_returnflag'],
         PRICE: cols['l_extendedprice']}, args)

    def rows(got):
        mat = np.stack([got[c].cpu().reshape(-1).to(torch.float64).numpy()
                        for c in (FLAG, 200, 201)])
        return mat[:, np.lexsort(mat[::-1])]
    assert got_u[FLAG].numel() == 30_000
    assert np.array_equal(rows(got_s), rows(got_u))
    assert int(got_u[200].sum()) == cols['l_suppkey'].col_length()


@requires_cuda
def test_avg_equals_streaming(sf1_parquet):
    cols = lineitem(sf1_parquet, ['l_partkey', 'l_quantity', 'l_discount'])

    def args():
        return agg_args([('l_partkey', PART)],
                     [('avg', f'l_quantity#{QTY}', 200),
                      ('avg', f'l_discount#{DISC}', 201)],
                     [PART, QTY, DISC], None)
    got_s, got_u, uvm = _run_both(
        {PART: cols['l_partkey'], QTY: cols['l_quantity'],
         DISC: cols['l_discount']}, args)
    assert_same_groups(got_s, got_u, [PART], [200, 201])
    assert got_u[200].dtype == constants.float_dtype
    assert uvm[200].type_name == 'float'


@requires_cuda
def test_count_skips_nulls_equals_streaming(sf1_parquet):
    cols = lineitem(sf1_parquet, ['l_returnflag', 'l_partkey'])
    part = cols['l_partkey'].tensor.clone()
    part[::5] = -64
    nullable = UColumn(part, 'int')

    def args():
        return agg_args([('l_returnflag', FLAG)],
                     [('count', f'x#{NULLABLE}', 200),
                      ('count', '1', 201)],
                     [FLAG, NULLABLE], None)
    got_s, got_u, _ = _run_both(
        {FLAG: cols['l_returnflag'], NULLABLE: nullable}, args)
    assert_same_groups(got_s, got_u, [FLAG], [200, 201])
    nulls = int((part == -64).sum())
    assert int(got_u[201].sum()) - int(got_u[200].sum()) == nulls > 0


@requires_cuda
def test_forced_b_over_many_morsels_and_rounds(sf1_parquet, budgets,
                                               monkeypatch):
    monkeypatch.setattr(aggregate, "AGGREGATION_PARTITIONS", 16)
    budgets(1 * MIB, 256 * 1024)
    rounds = []

    def counting(keys, vals, ops):
        rounds.append(keys.shape[0])
        return reduce_tick_slice(keys, vals, ops)
    monkeypatch.setattr(uvm_aggregate, "reduce_tick_slice", counting)

    cols = lineitem(sf1_parquet, ['l_suppkey', 'l_quantity',
                                   'l_extendedprice', 'l_discount'])

    def args():
        return agg_args([('l_suppkey', SUPP)],
                     [('sum', f'l_extendedprice#{PRICE}', 200),
                      ('count', '1', 201),
                      ('avg', f'l_discount#{DISC}', 202)],
                     [SUPP, PRICE, DISC], MASK)
    mask = quantity_mask(cols)
    inputs = {SUPP: cols['l_suppkey'], PRICE: cols['l_extendedprice'],
              DISC: cols['l_discount'], MASK: mask}
    n = mask.col_length()
    assert len(list(morsels(n, sum(c.row_bytes() for c in inputs.values())))) > 50
    got_s, got_u, _ = _run_both(inputs, args)
    assert_same_groups(got_s, got_u, [SUPP], [200, 201, 202])
    assert len(rounds) > 1, rounds
    assert sum(rounds) > got_u[SUPP].numel()
    assert int(got_u[201].sum()) == int(mask.tensor.sum())


@requires_cuda
def test_no_kept_row_gives_empty_outputs(sf1_parquet):
    """A mask that keeps no row of SF1 lineitem, then no input row at all:
    empty outputs of the streaming output types (the string key 2-D at its
    width, COUNT int64, AVG float, SUM as its result type).  The UVM aggregation alone: the
    streaming pass 1 is not given a tick that keeps no row."""
    from dataflow.dataflow_gen import PlanGraph
    cols = lineitem(sf1_parquet, ['l_shipmode', 'l_extendedprice'])
    rows = cols['l_shipmode'].col_length()

    def args():
        return agg_args([('l_shipmode', MODE)],
                     [('sum', f'l_extendedprice#{PRICE}', 200),
                      ('count', '1', 201),
                      ('avg', f'l_extendedprice#{PRICE}', 202)],
                     [MODE, PRICE], MASK)

    for n in (rows, 0):
        tg, pl = TensorGroup(), Payloads()
        insts = [column(tg, pl, cid, t, typ, **attrs)
                 for cid, t, typ, attrs in (
                     (MODE, cols['l_shipmode'].tensor[:n], 'string', {}),
                     (PRICE, cols['l_extendedprice'].tensor[:n], 'float', {}),
                     (MASK, torch.zeros(n, dtype=torch.bool, device='cuda'),
                      'bool', {'selectivity': 0.0}))]
        nodes = agg_nodes(PlanGraph(0), insts, 1, 3, [MODE, 200, 201, 202])
        executor.run('Aggregate', 1, dict(args(), nodes=nodes), tg, None,
                     None, pl)
        assert sorted(tg.keys()) == sorted([MODE, 200, 201, 202])
        assert tuple(tg[MODE].tensor.shape) == (0, 10)
        assert tg[MODE].tensor.dtype == torch.int8
        assert [tuple(tg[c].tensor.shape) for c in (200, 201, 202)] == [(0,)] * 3
        # The result types the streaming engine gives them (its empty
        # outputs take these).
        spec = KeyedAggregate(args(), {MODE: cols['l_shipmode'],
                                       PRICE: cols['l_extendedprice']},
                              [MODE, PRICE, MASK],
                              [torch.int8, torch.float64, torch.bool])
        assert [tg[c].tensor.dtype for c in (200, 201, 202)] == spec.agg_types
        assert spec.agg_types[1:] == [torch.int64, torch.float64]
        assert [tg[c].type_name for c in (MODE, 200, 201, 202)] == [
            'string'] + [torch_to_type(t) for t in spec.agg_types]
