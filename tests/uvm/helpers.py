"""Builders the UVM executor tests share: the SF1 columns and plans, the
stand-in and dataflow nodes the executor reads, one node's inputs in both
engine modes (the streaming one over the scan's record tables in a pinned
pool, ``tests/scan_table.py``), and the keyed aggregations' and joins'
plan args and comparisons."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from tensor_group import TensorGroup
from uvm.column import Payloads, UColumn

_SCANNED = {}


def query_plans(qid):
    """Query ``qid``'s prepared plans with their dataflow nodes, as
    ``main.exec`` builds them, and the schema."""
    from dataflow.dataflow_gen import build_query_dataflow, load_schema
    from tests.dataflow.plan_graphs import prepared_plans
    plans, schema = prepared_plans(qid), load_schema()
    build_query_dataflow(plans, schema, qid)
    return plans, schema


def sf1_columns(root, names):
    """SF1 columns ``names`` (``{key: (TABLE, COLUMN)}``) as fresh
    ``UColumn`` objects by key, each column scanned once per session."""
    missing = [tc for tc in names.values() if (root, tc) not in _SCANNED]
    if missing:
        from dataflow.dataflow_gen import load_schema
        from uvm.scan import scan_to_managed
        to_scan = [SimpleNamespace(table=t, name=n) for t, n in set(missing)]
        cols = scan_to_managed({'tables path': root}, load_schema(), 1, to_scan)
        for t, n in missing:
            _SCANNED[(root, (t, n))] = cols[t][n][0]
    return {key: UColumn(_SCANNED[(root, tc)].tensor,
                         _SCANNED[(root, tc)].type_name)
            for key, tc in names.items()}


def lineitem(root, names):
    """SF1 lineitem's columns ``names`` (lower case) as fresh ``UColumn``
    objects, by name."""
    return sf1_columns(root, {n: ('LINEITEM', n.upper()) for n in names})


class Node:
    """A pipeline node as the executor reads it: its input instances and
    an output instance per column id of ``outputs``."""

    name = "test node"

    def __init__(self, inputs, outputs):
        from dataflow.dataflow_gen import ColumnInstance
        self.inputs = inputs
        self.outputs = [ColumnInstance(producer=None, local=c)
                        for c in outputs]

    def input_pairs(self):
        return [(inst.local, inst) for inst in self.inputs]


def column(tensor_group, payloads, col_id, tensor, type_name, **attrs):
    """``tensor`` as column ``col_id`` of the tensor group, the payload of
    an instance of its own (read by no node yet)."""
    from dataflow.dataflow_gen import ColumnInstance
    inst = ColumnInstance(producer=None, local=col_id)
    tensor_group[col_id] = payloads.publish(
        inst, UColumn(tensor, type_name, **attrs))
    return inst


def scan_instances(plan, col_ids):
    """A scan node of ``plan`` producing ``col_ids``; its instances by
    column id."""
    from dataflow.dataflow_gen import ColumnInstance, PipeNode
    scan = PipeNode(0, "scan", "scan 0", plan)
    plan.nodes.append(scan)
    insts = {}
    for cid in col_ids:
        insts[cid] = ColumnInstance(producer=scan, local=cid)
        scan.outputs.append(insts[cid])
    return insts


def read_by(node, insts):
    """Make ``insts`` (instances) inputs of ``node``, its last reader."""
    for inst in insts:
        node.inputs.append(inst)
        node.input_keys.append(inst.local)
        inst.consumers.append(node)


def agg_nodes(plan, inputs, n_keys, n_aggs, outputs):
    """A keyed aggregation's two nodes in ``plan``, as the graph builder
    builds them: pass 1 reads ``inputs`` (instances) and writes the hash,
    ``n_keys`` keys and ``n_aggs`` partials, which pass 2 reads to write
    ``outputs`` (column ids).  Returns the plan args' ``nodes``."""
    from dataflow.dataflow_gen import ColumnInstance, PipeNode
    pass1 = PipeNode(1, "pass1", "aggregation 1 two-pass aggregation pass1",
                     plan, position=0)
    pass2 = PipeNode(1, "pass2", "aggregation 1 two-pass aggregation pass2",
                     plan, position=1)
    plan.nodes += [pass1, pass2]
    read_by(pass1, inputs)
    labels = (["inter_hash"] + [f"inter_key_{i}" for i in range(n_keys)]
              + [f"inter_val_{i}" for i in range(n_aggs)])
    pass1.outputs = [ColumnInstance(producer=pass1, local=label)
                     for label in labels]
    read_by(pass2, pass1.outputs)
    pass2.outputs = [ColumnInstance(producer=pass2, local=c) for c in outputs]
    return {'pass1': pass1, 'pass2': pass2}


def both_modes(columns, insts, pool):
    """``columns`` (``{col id: UColumn}``) as both modes' inputs, each the
    payload of its instance ``insts[col id]``: a streaming tensor group of
    scanned columns in ``pool``, a UVM one of the same tensors, and the
    payloads."""
    from tests.scan_table import pinned_column
    streaming, uvm, payloads = TensorGroup(), TensorGroup(), Payloads()
    for cid, col in columns.items():
        insts[cid].ccr = streaming[cid] = pinned_column(
            col.tensor, col.type_name, pool, col.selectivity)
        uvm[cid] = payloads.publish(insts[cid], col)
    return streaming, uvm, payloads


def streaming_out(tensor_group, cid):
    """Column ``cid`` of a streaming tensor group, on the GPU."""
    from IO.pipeline.common import col_to_gpu
    return col_to_gpu(tensor_group[cid])


# ── Keyed aggregations on SF1 lineitem columns ──────────────────────────

# lineitem's column ids in the plans; a nullable stand-in and a mask.
QTY, PRICE, DISC, PART, SUPP, FLAG, DATE, MODE = 96, 97, 98, 90, 91, 100, 102, 105
NULLABLE = 500
MASK = 10000


def _agg_result(op, arg, out):
    return [[[[op, [arg]], '#', '9001'], 'AS', out]]


def agg_args(keys, aggs, input_ids, mask, hidden=()):
    """A keyed Aggregate's plan args: GROUP BY ``keys`` (``(name, id)``,
    every one emitted), the results ``aggs`` (``(op, argument, out id)``;
    the argument an expression tree), over ``input ids``, every input
    under ``mask`` (None: none); ``hidden`` keys (``(name, id)``) group
    after them without being emitted."""
    results = [[f"{n}#{c}"] for n, c in keys]
    results += [_agg_result(op, arg, f"out#{out}") for op, arg, out in aggs]
    key_ids = [c for _, c in keys]
    agg_ids = [out for _, _, out in aggs]
    args = {'name': 'Aggregate',
            'keys': [f"{n}#{c}" for n, c in list(keys) + list(hidden)],
            'input ids': list(input_ids), 'results': results,
            'out key ids': key_ids, 'out agg ids': agg_ids,
            'agg ops': [op for op, _, _ in aggs],
            'output ids': key_ids + agg_ids}
    if mask is not None:
        args['mask map'] = {c: mask for c in input_ids}
    return args


def quantity_mask(cols):
    """``l_quantity < 24`` over ``cols`` as a mask column."""
    mask = cols['l_quantity'].tensor < 24
    return UColumn(mask, 'bool', selectivity=int(mask.sum()) / mask.numel())


def sorted_groups(got, key_ids, agg_ids):
    """The groups of ``got`` ordered by their keys: the key matrix (each
    key's elements as int64 columns) and each value, on the host."""
    keys = [got[k].cpu().reshape(got[k].shape[0], -1).to(torch.int64)
            for k in key_ids]
    mat = torch.cat(keys, dim=1).numpy()
    order = np.lexsort(mat.T[::-1])
    return mat[order], [got[a].cpu().numpy()[order] for a in agg_ids]


def assert_same_groups(got_s, got_u, key_ids, agg_ids, sigfigs=10):
    """Both modes' groups equal: the keys and integer values exactly, float
    values at ``sigfigs`` significant figures (None: bit-equal)."""
    for cid in key_ids:
        assert got_u[cid].dtype == got_s[cid].dtype, cid
        assert got_u[cid].dim() == (1 if got_s[cid].dim() == 1
                                    or got_s[cid].shape[1] == 1 else 2), cid
    keys_s, vals_s = sorted_groups(got_s, key_ids, agg_ids)
    keys_u, vals_u = sorted_groups(got_u, key_ids, agg_ids)
    assert keys_s.shape[0] > 0
    assert np.array_equal(keys_s, keys_u)
    for aid, vs, vu in zip(agg_ids, vals_s, vals_u):
        vs, vu = vs.reshape(vu.shape), vu
        if sigfigs is None or not np.issubdtype(vu.dtype, np.floating):
            assert np.array_equal(vs, vu), aid
        else:
            np.testing.assert_allclose(vu, vs, rtol=0.5 * 10.0 ** (1 - sigfigs),
                                       atol=0, err_msg=str(aid))


# ── Joins on SF1 customer and orders columns ────────────────────────────

# Column ids of the join cases.
C_CUSTKEY, C_NATIONKEY, C_ACCTBAL = 1, 2, 3
O_ORDERKEY, O_CUSTKEY, O_TOTALPRICE, O_ORDERDATE, O_PRIORITY = 11, 12, 13, 14, 15
C_MASK, O_MASK = 10000, 10001
JOIN_COLUMNS = {C_CUSTKEY: ('CUSTOMER', 'C_CUSTKEY'),
                C_NATIONKEY: ('CUSTOMER', 'C_NATIONKEY'),
                C_ACCTBAL: ('CUSTOMER', 'C_ACCTBAL'),
                O_ORDERKEY: ('ORDERS', 'O_ORDERKEY'),
                O_CUSTKEY: ('ORDERS', 'O_CUSTKEY'),
                O_TOTALPRICE: ('ORDERS', 'O_TOTALPRICE'),
                O_ORDERDATE: ('ORDERS', 'O_ORDERDATE'),
                O_PRIORITY: ('ORDERS', 'O_ORDERPRIORITY')}


def join_args(kind, cols):
    """The plan args of a Join of ``kind`` (``inner_unique_hash``,
    ``inner_sort_search``, ``semi``, ``anti``, ``right_outer``) over
    ``cols`` (``JOIN_COLUMNS``), and its masks (``{mask id: bool
    tensor}``)."""
    price = cols[O_TOTALPRICE].tensor
    nation = cols[C_NATIONKEY].tensor
    if kind in ('inner_unique_hash', 'inner_sort_search'):
        masks = {O_MASK: price < 150000.0}
        args = {'join type': 'Inner', 'left keys': [C_CUSTKEY],
                'right keys': [O_CUSTKEY],
                'left group': [C_CUSTKEY, C_NATIONKEY, C_ACCTBAL],
                'right group': [O_CUSTKEY, O_TOTALPRICE, O_PRIORITY],
                'join in': [C_CUSTKEY, O_CUSTKEY],
                'materialize_now': [O_CUSTKEY, C_NATIONKEY, O_TOTALPRICE,
                                    O_PRIORITY],
                'mask map': {O_CUSTKEY: O_MASK},
                'left unique keys': [[C_CUSTKEY]], 'right unique keys': []}
        if kind == 'inner_sort_search':
            args['join condition'] = [[['c_acctbal#3', '*', '50.0'], '>',
                                       'o_totalprice#13']]
            args['join in'] = [C_CUSTKEY, C_ACCTBAL, O_CUSTKEY, O_TOTALPRICE]
            args['materialize_now'] = [O_CUSTKEY, C_ACCTBAL, O_TOTALPRICE,
                                       C_NATIONKEY, O_PRIORITY]
            args['left unique keys'] = []
    elif kind in ('semi', 'anti'):
        masks = {C_MASK: nation < 5}
        args = {'join type': 'LeftSemi' if kind == 'semi' else 'LeftAnti',
                'left keys': [C_CUSTKEY], 'right keys': [O_CUSTKEY],
                'left group': [C_CUSTKEY],
                'right group': [O_CUSTKEY, O_TOTALPRICE, O_PRIORITY],
                'join in': [C_CUSTKEY, O_CUSTKEY],
                'materialize_now': [O_CUSTKEY, O_TOTALPRICE, O_PRIORITY],
                'mask map': {C_CUSTKEY: C_MASK},
                'left unique keys': [[C_CUSTKEY]], 'right unique keys': []}
    else:
        assert kind == 'right_outer'
        masks = {O_MASK: price < 20000.0}
        args = {'join type': 'LeftOuter', 'left keys': [O_CUSTKEY],
                'right keys': [C_CUSTKEY],
                'left group': [O_ORDERKEY, O_CUSTKEY, O_ORDERDATE, O_PRIORITY],
                'right group': [C_CUSTKEY, C_NATIONKEY],
                'join in': [O_CUSTKEY, C_CUSTKEY],
                'materialize_now': [C_CUSTKEY, O_ORDERKEY, O_ORDERDATE,
                                    O_PRIORITY, C_NATIONKEY],
                'mask map': {O_CUSTKEY: O_MASK},
                'left unique keys': [[O_ORDERKEY]],
                'right unique keys': [[C_CUSTKEY]]}
    ids = args['left group'] + args['right group']
    args.update({'name': 'Join', 'build': 'right', 'input ids': ids,
                 'output ids': list(args['materialize_now']),
                 'materialize_deferred': [],
                 'materialize_in': list(ids)})
    return args, masks


def join_graph(args, col_ids):
    """The Join's nodes, as the graph builder builds them over scanned
    columns ``col_ids``; returns the scan's instances by column id."""
    from dataflow.dataflow_gen import PlanGraph, _Builder, _mark_gather_roles
    plan = PlanGraph(0)
    builder = _Builder(plan)
    scan = builder._node(0, "scan", "scan 0")
    insts = {cid: builder._produce(scan, cid) for cid in col_ids}
    builder._handle_join(5, args)
    plan.assign_positions()
    _mark_gather_roles(plan)
    return insts


def sorted_rows(tensors):
    """Columns of equal length as one int64 matrix, a row per output row,
    sorted: floats by their bits, strings a column per byte."""
    parts = []
    for t in tensors:
        t = t.cpu()
        if t.dim() == 1:
            t = t.unsqueeze(1)
        if t.dtype == torch.float64:
            t = t.view(torch.int64)
        parts.append(t.long())
    m = torch.cat(parts, dim=1).numpy()
    return m[np.lexsort(m.T[::-1])]
