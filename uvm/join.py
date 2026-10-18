"""The UVM executor's join, partitioned by hash.

The join's arguments (the partition keys, the residual conditions, the
core's join type) and its B come from the streaming engine's functions
(``hashjoin.join_call_args``, ``hashjoin.join_partition_count``), so both
modes take the same B for the same inputs.

Per side, the transfer set is the join's ``join in`` columns of the side,
sorted (the order of the side's sort node), with the side's one mask.  Each
morsel of pre-mask rows is partitioned by the fused kernel
(``operators.partition_sort.partition``): the mask folded, the kept rows
bucketed by the key's hash, each bucket's rows in source order, and the
gidx (the morsel's first row plus the row's pre-mask position) written
beside them when a payload is gathered, in the kernel's padded layout
(each non-empty bucket starts on a whole ``lcm(row bytes, 512)``-byte
unit).  Each morsel's outputs and per-bucket counts are kept; the side's
inputs and mask are dropped before the other side is partitioned.

Core calls pack whole buckets to half the chunk budget at the bytes per
row the core reads (``hashjoin.pack_core_calls``, as the streaming core
pipe packs its ticks).  A call's columns are its buckets' rows, bucket by
bucket and, within a bucket, morsel by morsel (``uvm.buckets``), so
``join_kernel_new`` sees each bucket's rows contiguous with its per-bucket
counts; each morsel's outputs are dropped once copied into the calls'
columns, and each call's columns once the call is done.  A call over
``join_core.MAX_ROWS`` rows on a side fails the query.  The calls' outputs
(the key columns the join emits, then the two ``out_index`` columns when a
payload is gathered) are concatenated in call order, column by column.  Every ``out_index``
value is a pre-mask row position of its side (or ``constants.null``), the
row space the gathers read the payloads in (``uvm/gather.py``).

Then the side's gather nodes run, left side then right side, in the order
the graph lists them (the rearrange, then the lineage composes), and the
join's inputs that are not outputs are dropped.  No size here depends on
how full the arena is.
"""
from __future__ import annotations

import numpy as np
import torch

import constants
from operators import join_core
from operators.hashjoin import (
    join_call_args,
    join_kernel_kwargs,
    join_kernel_new,
    join_output_keys,
    join_partition_count,
    join_side_inputs,
    log_join_partition_count,
    pack_core_calls,
)
from operators.partition_sort import partition
from utility.logger import message_logger
from utility.tensor_utils import row_bytes
from uvm.buckets import split_buckets
from uvm.column import (DEVICE, UColumn, concat_or_empty, drop_inputs,
                        rows_of)
from uvm.gather import run_gather
from uvm.morsel import morsels

LEFT, RIGHT = 0, 1


def _partition_side(inputs, transfer, key_pos, mask_id, B, with_gidx, name):
    """Partition one side, one morsel of pre-mask rows at a time.  Returns,
    per morsel with a kept row, its outputs (the transfer set's columns,
    then the gidx when ``with_gidx``), its rows per bucket and each
    output's bucket starts, as ``split_buckets`` takes them."""
    cols = [inputs[c].tensor for c in transfer]
    mask = inputs[mask_id].tensor if mask_id is not None else None
    rows = rows_of({c: inputs[c] for c in transfer})
    if mask is not None:
        assert mask.shape[0] == rows, (
            f"{name}: mask {mask_id} has {mask.shape[0]} rows, its columns "
            f"{rows}")
    row_b = sum(row_bytes(t) for t in cols) + (1 if mask is not None else 0)
    spans = list(morsels(rows, row_b))
    # Every morsel's kept rows, with one host sync.
    kept = ([None] * len(spans) if mask is None or not spans else
            torch.stack([mask[a:b].sum() for a, b in spans]).tolist())
    parts = []
    for (a, b), n_valid in zip(spans, kept):
        if n_valid == 0:
            continue
        m = mask[a:b] if mask is not None else None
        outs = partition([t[a:b] for t in cols], key_pos, m, a, B,
                         n_valid=n_valid)
        if not with_gidx:
            outs = outs[:-1]
        parts.append(([o.tensor for o in outs],
                      np.asarray(outs[0].partition_row_counts, dtype=np.int64),
                      [o.bucket_starts() for o in outs]))
    return parts


def join(tensor_group, args, payloads, name="join"):
    """A plan Join (``tqp_join``'s sort, core and gathers): each side
    partitioned per morsel, the core per call of whole buckets, then the
    gather nodes.  The emitted keys are published on the core's output
    instances and set in the tensor group, the ``out_index`` columns on
    theirs; the inputs that are not outputs are dropped."""
    left_key, right_key, condition_list, join_type = join_call_args(
        args, tensor_group)
    key_ids = [left_key, right_key]
    sort_nodes = args['nodes']['sort']
    core = args['nodes']['core']

    B, pred_b, est_sides, col_bytes = join_partition_count(
        args, key_ids, tensor_group)
    log_join_partition_count(name, B, pred_b, est_sides)

    # The keys the core emits, and whether it emits the out_index columns.
    out_keys, rearrange_required = join_output_keys(args)
    assert [inst.local for inst in core.outputs] == out_keys + (
        ['out_index_left', 'out_index_right'] if rearrange_required else []), (
        [inst.local for inst in core.outputs], out_keys)
    out_types = {k: tensor_group[k].type_name for k in out_keys}
    out_templates = {k: (tensor_group[k].tensor.dtype,
                         tuple(tensor_group[k].tensor.shape[1:]))
                     for k in out_keys}

    transfers, templates, parts = [], [], []
    for side in (LEFT, RIGHT):
        transfer, mask_id = join_side_inputs(args, side)
        inputs = payloads.inputs(sort_nodes[side])
        for c in transfer + ([mask_id] if mask_id is not None else []):
            assert c in inputs and inputs[c] is not None, (
                f"{name}: side {side} reads no column {c}")
        # (dtype, trailing shape) of each output, for a call with no row on
        # the side.
        templates.append(
            [(inputs[c].tensor.dtype, tuple(inputs[c].tensor.shape[1:]))
             for c in transfer]
            + ([(torch.int64, ())] if rearrange_required else []))
        parts.append(_partition_side(inputs, transfer,
                                     transfer.index(key_ids[side]), mask_id,
                                     B, rearrange_required, name))
        transfers.append(transfer)
        # The side's inputs and mask are done with.
        del inputs
        payloads.consumed(sort_nodes[side])
        drop_inputs(tensor_group, payloads,
                    transfer + ([mask_id] if mask_id is not None else []),
                    set())

    # Whole buckets per core call, up to half the chunk budget at the bytes
    # per row the core reads.
    side_row_bytes = [sum(col_bytes[c] for c in t) + 8 for t in transfers]
    rows = [np.sum([counts for _, counts, _ in p], axis=0, dtype=np.int64)
            if p else np.zeros(B, dtype=np.int64) for p in parts]
    ends, pair_b = pack_core_calls(rows[LEFT], rows[RIGHT], side_row_bytes,
                                   constants.get_chunk_size_b() // 2)
    message_logger().info(
        "[%s] partitioned join: B=%d calls=%d morsels=%d/%d largest pair "
        "%.1f MB", name, B, len(ends), len(parts[LEFT]), len(parts[RIGHT]),
        int(pair_b.max()) / (1 << 20))
    bounds = [0] + ends
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        for side_rows, side_name in ((rows[LEFT], "left"),
                                     (rows[RIGHT], "right")):
            n = int(side_rows[lo:hi].sum())
            if n > join_core.MAX_ROWS:
                raise RuntimeError(
                    f"{name}: a core call of {n} {side_name} rows (buckets "
                    f"{lo}-{hi - 1} of B={B}) is over the {join_core.MAX_ROWS} "
                    f"rows a call takes")
    calls = [split_buckets(p, bounds, bucket_major=True) for p in parts]
    del parts

    results = []
    for j, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        call = [calls[LEFT][j], calls[RIGHT][j]]
        calls[LEFT][j] = calls[RIGHT][j] = None
        if call[LEFT] is None and call[RIGHT] is None:
            continue
        tensors, gidx = [], []
        for side in (LEFT, RIGHT):
            cols = call[side] or [concat_or_empty([], dtype, shape)
                                  for dtype, shape in templates[side]]
            n = len(transfers[side])
            tensors.append(dict(zip(transfers[side], cols[:n])))
            gidx.append(cols[n] if rearrange_required else
                        torch.empty(0, dtype=torch.int64, device=DEVICE))
            del cols
        del call
        results.append(join_kernel_new(
            tensors[LEFT], tensors[RIGHT], left_key, right_key,
            condition_list, out_keys, rearrange_required, join_type,
            gidx[LEFT], gidx[RIGHT],
            bucket_counts=(rows[LEFT][lo:hi].tolist(),
                           rows[RIGHT][lo:hi].tolist()),
            name=name, **join_kernel_kwargs(args)))
        del tensors, gidx
    del calls

    # Each output concatenated from the calls, column by column, each
    # call's column dropped as soon as it is copied.  An out_index nothing
    # reads is not published.
    for i, inst in enumerate(core.outputs):
        dtype, shape = (out_templates[out_keys[i]] if i < len(out_keys)
                        else (torch.int64, ()))
        out = concat_or_empty([r[i] for r in results], dtype, shape)
        for r in results:
            r[i] = None
        if i < len(out_keys):
            k = out_keys[i]
            tensor_group[k] = payloads.publish(inst, UColumn(out, out_types[k]))
        elif inst.consumers:
            payloads.publish(inst, UColumn(out, 'int'))
        del out
    del results

    if rearrange_required:
        # Per side, the gathers in the order the graph lists them: the
        # side's rearrange, then one lineage compose per deferred group
        # composed here.  A seed join's out_index is the lineage, kept
        # until the compose or deferred gather that last reads it.
        for side_label in ("left", "right"):
            for gnode in args['nodes']['gathers'][side_label]:
                run_gather(gnode, tensor_group, payloads)

    drop_inputs(tensor_group, payloads, args['input ids'],
                set(args['output ids']))
