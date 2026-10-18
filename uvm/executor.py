"""The UVM executor: one function per operator, over managed tensors.

``run`` dispatches one plan operator (``main.exec`` hands it every operator
but the output under ``--uvm``; ``output`` builds the result frame).  Each
operator reads its columns as the streaming operator does, a pipeline's
inputs from the payloads of its dataflow node's input instances (the
occurrence's ``Payloads``, keyed by the node's input keys) and a sort's
from the tensor group, and calls the streaming engine's leaf functions
over morsels of plain tensor slices (``uvm/morsel.py``).  Its outputs are
``UColumn`` objects, published on the node's output instances and set in
the tensor group.

Masks stay in pre-mask row space: a filter writes one bool tensor for the
whole input, an aggregation compacts its inputs per morsel
(``uvm.aggregate.compact_morsel``) and a sort keeps its ``nonzero``.  A
payload is dropped once the node that reads it last has run, and an
operator drops its inputs from the tensor group as the streaming one frees
them, with their payloads, so a tensor goes back to the arena with its
last reference.

A keyed aggregation runs in ``uvm/aggregate.py``, a join (its gathers
included) in ``uvm/join.py``, and the deferred gathers an op reads, before
it, in ``uvm/gather.py``.
"""
from __future__ import annotations

import contextlib

import torch

from operators.aggregate import GlobalAggregate, agg_transfer_set
from operators.filter import filter_chunk
from operators.operator_utils import infer_project_output_types
from operators.output import output_frame, result_to_host
from operators.project import project_chunk, project_exprs
from operators.sort import sort_inputs, sort_tensors
from utility.logger import torch_profiler
from variable import torch_to_type
from uvm.aggregate import aggregate_keyed, compact_morsel
from uvm.column import (DEVICE, UColumn, concat_or_empty, drop_inputs,
                        row_bytes_of, rows_of)
from uvm.gather import run_gather
from uvm.join import join
from uvm.morsel import morsels

# The perf-tag prefix of the operators ``--torch_profile`` wraps, as in the
# streaming engine.
_PROFILED_OPS = {'Filter': 'filter', 'Aggregate': 'aggregation',
                 'Join': 'join'}


def run(name, u, args, tensor_group, scan_group, subquery_result, payloads):
    """Run plan operator ``u`` (``name``, its plan args ``args``) on the
    query's ``tensor_group``; ``scan_group`` is ``scan_to_managed``'s and
    ``payloads`` the occurrence's ``Payloads``.  A filter, an aggregation
    and a join run under the torch profiler."""
    # Payload deferral: the columns deferred to this op are gathered by
    # their lineage first, so the op reads them as any other column.
    for node in (args.get('nodes') or {}).get('deferred') or []:
        run_gather(node, tensor_group, payloads)
    tag = _PROFILED_OPS.get(name)
    with (torch_profiler().profile(f"{tag} {u}") if tag
          else contextlib.nullcontext()):
        _dispatch(name, u, args, tensor_group, scan_group, subquery_result,
                  payloads)


def _dispatch(name, u, args, tensor_group, scan_group, subquery_result,
              payloads):
    if name == 'Scan':
        scan(scan_group, tensor_group, args, payloads)
    elif name == 'Filter':
        filter_(tensor_group, args, subquery_result, payloads)
    elif name == 'Project':
        project(tensor_group, args, payloads)
    elif name == 'Aggregate':
        if args.get('keys'):
            aggregate_keyed(tensor_group, args, payloads,
                            name=f"aggregation {u}")
        else:
            aggregate_global(tensor_group, args, payloads)
    elif name == 'Sort':
        sort(tensor_group, args, payloads, with_limit=False, name=f"sort {u}")
    elif name == 'take ordered':
        sort(tensor_group, args, payloads, with_limit=True,
             name=f"sort lim {u}")
    elif name == 'Join':
        join(tensor_group, args, payloads, name=f"join {u}")
    elif name.endswith('Exchange'):
        assert False, "Exchange is eliminated"
    else:
        raise Exception(f"Unsupported Operator: {name}")


def scan(scan_group, tensor_group, args, payloads):
    """The scanned columns, as ``tqp_scan`` takes them; the scan's instances
    carry them."""
    table_name = args['table name']
    for col_name, col_id in zip(args['output names'], args['output ids']):
        tensor_group[col_id] = scan_group[table_name][col_name.upper()].pop()
    for inst in args['nodes']['scan'].outputs:
        payloads.publish(inst, tensor_group[inst.local])


def filter_(tensor_group, args, subquery_result, payloads):
    """The filter's mask over its whole input, one morsel at a time
    (``tqp_filter``)."""
    assert 'mask map' not in args, "Filter not accepting masks"
    if len(args['materialize_in']) == 0:
        return
    node = args['nodes']['pipe']
    tree = args['filter tree']
    inputs = payloads.inputs(node)
    rows = rows_of(inputs)
    mask = torch.empty(rows, dtype=torch.bool, device=DEVICE)
    # The kept rows, counted per morsel: a sum over the whole mask would
    # take an 8-byte temporary per row.
    kept = torch.zeros((), dtype=torch.int64, device=DEVICE)
    for a, b in morsels(rows, row_bytes_of(inputs)):
        cols = {key: col.tensor[a:b] for key, col in inputs.items()}
        mask[a:b] = filter_chunk(cols, tree, subquery_result)
        kept += mask[a:b].sum()
        del cols
    del inputs
    payloads.consumed(node)
    kept = int(kept.item())
    selectivity = kept / rows if rows > 0 else 1.0
    tensor_group[args['out mask id']] = payloads.publish(
        node.outputs[0], UColumn(mask, 'bool', selectivity=selectivity))
    drop_inputs(tensor_group, payloads, args['input ids'], args['output ids'])


def project(tensor_group, args, payloads):
    """The project's computed columns, one morsel at a time; renames are
    tensor-group aliases (``tqp_project``)."""
    for orig, new in args['rename map'].items():
        tensor_group.alias(orig, new)

    if len(args['used expr']) > 0:
        node = args['nodes']['pipe']
        input_type_map = {col_id: tensor_group[col_id].get_type_torch()
                          for col_id in args['materialize_in']}
        out_types = infer_project_output_types(args['used expr'], input_type_map)
        exprs = project_exprs(args['used expr'])
        out_ids = list(args['materialize_now'])
        assert [inst.local for inst in node.outputs] == out_ids
        assert len(exprs) == len(out_ids), (
            f"{len(exprs)} computed expressions for {len(out_ids)} outputs")
        assert 'mask map' not in args

        inputs = payloads.inputs(node)
        rows = rows_of(inputs)
        # Each output's one tensor, allocated at its first morsel.
        outs = [[] for _ in exprs]
        for a, b in morsels(rows, row_bytes_of(inputs)):
            cols = {key: col.tensor[a:b] for key, col in inputs.items()}
            for i, result in enumerate(project_chunk(cols, exprs, out_types)):
                if not outs[i]:
                    outs[i].append(torch.empty((rows, *result.shape[1:]),
                                               dtype=result.dtype,
                                               device=DEVICE))
                outs[i][0][a:b] = result
            del cols
        del inputs
        payloads.consumed(node)
        for i, col_id in enumerate(out_ids):
            out = concat_or_empty(outs[i], out_types[i])
            tensor_group[col_id] = payloads.publish(
                node.outputs[i], UColumn(out, torch_to_type(out_types[i])))

    # An aliased input shares its column with the renamed output id: the
    # stale key goes, the column stays (``drop_inputs`` keeps it while the
    # tensor group holds it).
    drop_inputs(tensor_group, payloads, args['input ids'], args['output ids'])


def aggregate_global(tensor_group, args, payloads):
    """A global (no-key) aggregation, one morsel at a time: masked rows
    compacted, each morsel's partials merged into the running results as
    the streaming engine merges its ticks (``GlobalAggregate``), AVG
    finalized once at the end."""
    node = args['nodes']['pipe']
    transfer_set, transfer_types, agg_mask_map = agg_transfer_set(
        args, tensor_group)
    spec = GlobalAggregate(args, transfer_set, transfer_types)
    assert [inst.local for inst in node.outputs] == spec.agg_ids

    inputs = payloads.inputs(node)
    for mid in set(agg_mask_map.values()):
        assert mid in inputs, f"mask {mid} not among the aggregation's inputs"
    rows = rows_of(inputs)
    # No rows: one empty morsel, so every result still has its value.
    spans = list(morsels(rows, row_bytes_of(inputs))) or [(0, 0)]
    results = None
    for a, b in spans:
        cols, _ = compact_morsel(inputs, agg_mask_map, a, b)
        partials = spec.partials(DEVICE, cols)
        del cols
        if results is None:
            results = partials
        else:
            spec.merge(results, partials)
    del inputs
    payloads.consumed(node)

    for i, value in enumerate(spec.finalize(results)):
        tensor_group[spec.agg_ids[i]] = payloads.publish(
            node.outputs[i], UColumn(value, torch_to_type(spec.agg_types[i])))

    # The input ids include the masks (agg_transfer_set).
    output_ids = set(args.get('output ids', spec.agg_ids))
    drop_inputs(tensor_group, payloads, args['input ids'], output_ids)


def sort(tensor_group, args, payloads, with_limit, name="sort"):
    """``tqp_sort`` over the columns' tensors (``sort_tensors``): each input
    column is replaced by its sorted rows, and the replaced columns and the
    mask are dropped.  Returns the count it does."""
    ids, mask_id = sort_inputs(args, name)
    replaced = [tensor_group[cid] for cid in ids]
    tensors = {cid: tensor_group[cid].tensor.contiguous() for cid in ids}
    mask = None if mask_id is None else tensor_group[mask_id].tensor
    outs, count = sort_tensors(tensors, mask, args, with_limit, name)
    del tensors, mask
    for col_id, out in outs.items():
        tensor_group[col_id] = UColumn(out.contiguous(),
                                       tensor_group[col_id].type_name)
    del outs
    payloads.drop_columns(replaced, tensor_group)
    if mask_id is not None:
        drop_inputs(tensor_group, payloads, [mask_id],
                    set(args.get('output ids', ())))
    return count


def output(tensor_group, args):
    """The query's result frame (``tqp_output``)."""
    gpu_tensors, out_types = {}, {}
    for output_id in args['output ids']:
        gpu_tensors[output_id] = tensor_group[output_id].tensor
        out_types[output_id] = tensor_group[output_id].type_name
    return output_frame(gpu_tensors, out_types, args)


def output_to_host(tensor_group, args):
    """The result's columns copied to the host: (host tensors, type names)
    by output id, for ``operators.output.host_frame``."""
    gpu_tensors, out_types = {}, {}
    for output_id in args['output ids']:
        gpu_tensors[output_id] = tensor_group[output_id].tensor
        out_types[output_id] = tensor_group[output_id].type_name
    return result_to_host(gpu_tensors, args['output ids']), out_types
