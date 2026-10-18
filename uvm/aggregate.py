"""The UVM executor's keyed aggregation (GROUP BY), partitioned by hash.

B, the number of hash buckets, comes from the partition budget
(``operators.aggregate.agg_partition_count``) with the streaming engine's
inputs: P, the input rows times the selectivity the filter recorded on the
mask; T, the morsels; G, the static bound on the groups the GROUP BY keys'
dtypes give; and w, the hash route's run bytes per row
(``operators.aggregate.AGGREGATION_PARTITIONS`` > 0 forces B).  The
streaming engine sends some shapes to a dense or key route with a B of its
own (1 for a dense shape); here every shape takes the hash route's B.

Pass 1, per morsel: the masked rows compacted (``compact_morsel``), then
the streaming engine's
per-chunk work (``KeyedAggregate.pass1``): the grouping hash, one row per
group with its emitted keys and each result's partial (an AVG as its
``[sum, count]`` state), and each group's bucket.  A morsel's partials are
kept bucket by bucket (a stable sort on the bucket) with each bucket's row
offsets, so no kernel ever covers more than one morsel's partials.

Pass 2, per round: whole buckets packed to the partition budget at w
(``pass2_rounds``, the streaming engine's packing).  A round's rows are
its buckets' ranges of every morsel's partials, concatenated; one
unique-and-reduce over them on the hash (``reduce_tick_slice``) merges each
group's partials, the emitted keys by ``amax`` (equal hashes carry equal
keys), an AVG state by SUM and every other result by its expression's first
op, as the streaming engine merges ticks.  The rounds' outputs, AVG
finalized, are concatenated in round order.

The aggregation's inputs are dropped once pass 1 is done.  The partials
are held once: when the rounds are known, each morsel's partials are
copied into the rounds' columns and dropped (``uvm.buckets``), and each
round's columns are dropped as soon as the round is reduced; each output
column is concatenated from the rounds' and their copies dropped column by
column.

No size here depends on how full the arena is: a morsel is the chunk
budget, a round the partition budget, and a round over ``MORSEL_ROWS_MAX``
rows (a skewed bucket) fails the query.
"""
from __future__ import annotations

import numpy as np
import torch

import constants
from operators.aggregate import (KeyedAggregate, agg_partition_count,
                                 agg_transfer_set, log_agg_partition_count,
                                 pass2_rounds)
from operators.sorted_run_merge import reduce_tick_slice
from utility.logger import message_logger
from variable import torch_to_type
from uvm.buckets import split_buckets
from uvm.column import (DEVICE, UColumn, concat_or_empty, drop_inputs,
                        row_bytes_of, rows_of)
from uvm import morsel
from uvm.morsel import morsels


def compact_morsel(inputs, agg_mask_map, a, b):
    """An aggregation's data columns over rows ``[a, b)`` (``inputs`` less
    the masks), each masked column cut to its mask's kept rows: per mask,
    the kept row ids taken once (a morsel is at most ``MORSEL_ROWS_MAX``
    rows, within ``nonzero``'s bounds) and each column it masks gathered
    by them.  Returns the columns and whether a mask keeps no row."""
    keep = {mid: inputs[mid].tensor[a:b].nonzero().squeeze(1)
            for mid in set(agg_mask_map.values())}
    cols = {}
    for key, col in inputs.items():
        if key in keep:
            continue
        t = col.tensor[a:b]
        if key in agg_mask_map:
            t = t.index_select(0, keep[agg_mask_map[key]])
        cols[key] = t
    return cols, any(ids.numel() == 0 for ids in keep.values())


def _pass1(spec, inputs, agg_mask_map, spans, B):
    """Each morsel's partials (``[hash] + keys + values``) in bucket order,
    as ``split_buckets`` takes them (its buckets' rows and starts); morsels
    with no group are left out."""
    parts = []
    for a, b in spans:
        cols, empty = compact_morsel(inputs, agg_mask_map, a, b)
        if empty:
            continue
        outs, buckets = spec.pass1(DEVICE, cols, B)
        del cols
        groups = outs[0].shape[0]
        if groups == 0:
            continue
        if buckets is None:
            counts = [groups]
        else:
            bucket, counts, _ = buckets
            order = torch.sort(bucket, stable=True).indices
            outs = [t.index_select(0, order) for t in outs]
            del bucket, order
        counts = np.asarray(counts, dtype=np.int64)
        starts = np.cumsum(counts) - counts
        parts.append((outs, counts, [starts] * len(outs)))
    return parts


def _pass2(spec, parts, B, row_b, name):
    """The merged groups, round by round: ``keys + values`` (AVG states
    still packed) per round.  ``parts`` is emptied as the partials are
    copied into the rounds' columns, and each round's columns are dropped
    once it is reduced."""
    K = len(spec.key_ids)
    n_parts = len(parts)
    bucket_rows = np.zeros(B, dtype=np.int64)
    for _, counts, _ in parts:
        bucket_rows += counts
    points = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(bucket_rows, out=points[1:])
    ends = pass2_rounds(bucket_rows, row_b,
                        constants.get_partition_budget_b())
    # Each round's first bucket, and B after the last.
    bounds = [0] + ends
    for j in range(len(bounds) - 1):
        rows = int(points[bounds[j + 1]] - points[bounds[j]])
        if rows > morsel.MORSEL_ROWS_MAX:
            raise RuntimeError(
                f"{name}: a pass-2 round of {rows} rows (B={B}) is over "
                f"the {morsel.MORSEL_ROWS_MAX} rows one kernel call takes")
    rounds_in = split_buckets(parts, bounds, bucket_major=False)
    ops = ["amax"] * K + list(spec.merge_ops)
    rounds = []
    largest = 0
    for j in range(len(rounds_in)):
        cols, rounds_in[j] = rounds_in[j], None
        if cols is None:
            continue
        largest = max(largest, int(cols[0].shape[0]))
        _, merged = reduce_tick_slice(cols[0], cols[1:], ops)
        del cols
        rounds.append(merged)
    if B > 1:
        message_logger().info(
            "[%s] partitioned pass 2: B=%d rounds=%d route=hash run_row=%d B "
            "largest round=%d rows, %d morsels with groups",
            name, B, len(ends), row_b, largest, n_parts)
    return rounds


def aggregate_keyed(tensor_group, args, payloads, name="aggregation"):
    """A keyed aggregation (``tqp_hash_aggregate``'s two passes): pass 1
    per morsel, pass 2 per round of whole buckets.  Its inputs are dropped
    after pass 1; its outputs, the emitted keys then the results, are
    published on pass 2's output instances and set in the tensor group."""
    pass1_node = args['nodes']['pass1']
    pass2_node = args['nodes']['pass2']
    transfer_set, transfer_types, agg_mask_map = agg_transfer_set(
        args, tensor_group)
    spec = KeyedAggregate(args, tensor_group, transfer_set, transfer_types)
    K, N = len(spec.key_ids), len(spec.agg_ids)
    assert spec.merge_ops_ok, (
        f"{name}: results merge by {spec.merge_ops}; pass 2 merges only by "
        f"sum, amax and amin")
    assert [inst.local for inst in pass2_node.outputs] == \
        spec.key_ids_new + spec.agg_ids, (
            [inst.local for inst in pass2_node.outputs],
            spec.key_ids_new, spec.agg_ids)

    inputs = payloads.inputs(pass1_node)
    mask_ids = set(agg_mask_map.values())
    for mid in mask_ids:
        assert mid in inputs, f"mask {mid} not among the aggregation's inputs"
    rows = rows_of(inputs)
    spans = list(morsels(rows, row_bytes_of(inputs)))
    selectivity = (inputs[next(iter(mask_ids))].selectivity
                   if len(mask_ids) == 1 else None)
    part = agg_partition_count(
        spec, rows if selectivity is None else int(round(rows * selectivity)),
        max(1, len(spans)), "hash")
    log_agg_partition_count(name, part, unit="morsels")
    B, row_b = part.B, part.row_b

    parts = _pass1(spec, inputs, agg_mask_map, spans, B)
    # Pass 2 reads only the partials: the inputs go now.  The input ids
    # include the masks (agg_transfer_set).
    del inputs
    payloads.consumed(pass1_node)
    output_ids = set(args.get('output ids', spec.key_ids_new + spec.agg_ids))
    drop_inputs(tensor_group, payloads, args['input ids'], output_ids)
    rounds = _pass2(spec, parts, B, row_b, name)

    # Each output concatenated from the rounds, column by column, each
    # round's column dropped as soon as it is copied.
    outs = []
    for i in range(K + N):
        if i < K:
            width = spec.key_widths[i]
            dtype, shape = spec.key_types[i], (width,) if width > 1 else ()
        else:
            state_width = spec.state_width[i - K]
            dtype = spec.agg_types[i - K]
            shape = (state_width,) if state_width else ()
        outs.append(concat_or_empty([r[i] for r in rounds], dtype, shape))
        for r in rounds:
            r[i] = None
    del rounds
    keys, values = outs[:K], spec.finalize(outs[K:])

    for i, kid in enumerate(spec.key_ids_new):
        tensor_group[kid] = payloads.publish(
            pass2_node.outputs[i], UColumn(keys[i], spec.key_types_name[i]))
    for i, aid in enumerate(spec.agg_ids):
        tensor_group[aid] = payloads.publish(
            pass2_node.outputs[K + i],
            UColumn(values[i], torch_to_type(spec.agg_types[i])))
