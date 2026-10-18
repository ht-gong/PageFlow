from typing import List, Literal, Dict
from IO.pages.combined_column_rep import CombinedColumnRep
from IO.pipeline.vortex_pipeline import OutputLayout, VortexPipeline
from IO.pipeline.rearrange_pipeline import rearrange_pipeline
from IO.pipeline import tick_cutter
from conversion import index_with_null
from operators.operator_utils import (INT_ROWS_MAX, evaluate_filter,
                                      normalize_expr_tree)
from parsing import column_ref
from utility.logger import perf_logger, message_logger
from variable import TableVariable, Variable
import numpy as np
import sizing_resolver
import torch

import constants
from utility.tensor_utils import dtype_size
from operators.partition_sort import partition
from operators import join_core
from operators.partitions import JOIN_B_MAX, partition_count

LEFT = 0
RIGHT = 1

# Number of hash partitions per side per join, B.  Top log2(B) bits of
# mix64(key) pick the bucket; mix64 decorrelates from low-bit key skew
# (TPC-H orderkey is sparse, low bits are biased).  Each input chunk
# produces (transfer_set + 1) * B output Variables — one per (logical
# column, partition) pair — and the core pipe iterates partitions
# instead of sort runs.  A power of two in [1, JOIN_B_MAX] (512, the
# partition kernel's limit, operators/kernels/partition_sort.cu; at 1 every
# row is in bucket 0).  Each join derives its own B from its input sizes and
# the partition budget (operators/partitions.py, partition_count).


# The unique-build hash core's gate: ``legacy`` sends inner joins to the
# sort/search core too.  Only tests switch it (``set_join_core_mode``); the
# engine runs ``unique_hash``.
_JOIN_CORE_MODES = {"legacy", "unique_hash"}
_join_core_mode = "unique_hash"
# Measured on the Triton core this replaced: isolated A100 event timing found
# hashing beneficial through a build/probe row ratio of 8 (1.02x at 8), but
# slower after its power-of-two table's next capacity jump (0.57x at 10).  The
# first full SF300 A/B confirmed that the bound must be independent of
# physical orientation: Q12's proven-left 48:1 build raised core compute
# 0.218 -> 0.458 s, and Q18 exposed the same shape on a smaller subset.  The
# CUDA core's table has exactly twice the build rows; the bound is not
# re-measured.
_UNIQUE_HASH_MAX_BUILD_PROBE_RATIO = 8
# SF300 showed that the 256 MiB input-chunk setting is not the join core's
# scratch budget: unchanged pair outputs alone can exceed it, while measured
# legacy core peaks retained more than 10 GiB of allocator headroom.  Bound the
# table, partner and probe-sized outputs (join_core.unique_hash_peak_bytes) at
# 2 GiB instead.
_UNIQUE_HASH_SCRATCH_LIMIT_BYTES = 2 * 1024 ** 3


def set_join_core_mode(mode: str) -> None:
    global _join_core_mode
    if mode not in _JOIN_CORE_MODES:
        raise ValueError(
            f"join core mode must be one of {sorted(_JOIN_CORE_MODES)}, got {mode!r}"
        )
    _join_core_mode = mode


def _join_core_fallback(reason: str):
    """None: the caller falls back to the sort/search core (``reason`` names
    the guard that refused the unique-hash core)."""
    return None


def _proof_is_covered(unique_key_sets, selected_ids) -> bool:
    selected = set(selected_ids)
    return any(set(unique_key) <= selected for unique_key in (unique_key_sets or []))


def _valid_hash_key(tensor: torch.Tensor) -> bool:
    return (
        tensor.is_cuda
        and tensor.dtype == torch.int64
        and tensor.dim() == 1
        and tensor.is_contiguous()
    )


def _try_unique_hash_pairs(
    tensors_left,
    tensors_right,
    left_key,
    right_key,
    join_type,
    equality_keys,
    unique_key_sets,
    scratch_limit_bytes,
):
    """``(left_idx, right_idx, hashed_pairs)`` from the unique-build hash
    core, or ``None`` for the sort/search core."""
    if _join_core_mode == "legacy":
        return _join_core_fallback("gate_legacy")
    if join_type != "inner":
        return _join_core_fallback("join_type")
    if equality_keys is None or unique_key_sets is None:
        return _join_core_fallback("missing_proof_metadata")

    left_eq, right_eq = equality_keys
    if len(left_eq) != len(right_eq) or not left_eq:
        return _join_core_fallback("unaligned_equality_keys")

    selected = []
    for left_id, right_id in zip(left_eq, right_eq):
        if left_id not in tensors_left or right_id not in tensors_right:
            continue
        left_tensor, right_tensor = tensors_left[left_id], tensors_right[right_id]
        if _valid_hash_key(left_tensor) and _valid_hash_key(right_tensor):
            selected.append((left_id, right_id))

    # The upstream partition key must be part of the exact hash tuple; otherwise
    # the selected rows are not guaranteed to be co-located in a streaming tick.
    if (left_key, right_key) not in selected:
        return _join_core_fallback("primary_key_layout_or_dtype")
    if len(selected) not in (1, 2):
        return _join_core_fallback("unsupported_key_count")

    left_ids = [pair[0] for pair in selected]
    right_ids = [pair[1] for pair in selected]
    left_unique, right_unique = unique_key_sets
    can_build_left = _proof_is_covered(left_unique, left_ids)
    can_build_right = _proof_is_covered(right_unique, right_ids)
    if not can_build_left and not can_build_right:
        return _join_core_fallback("selected_keys_not_proven_unique")

    left_rows = tensors_left[left_ids[0]].numel()
    right_rows = tensors_right[right_ids[0]].numel()
    # The first Q20 SF300 A/B settled the policy: when both sides are proven
    # unique, build the smaller per-tick side.  Ties stay left so existing
    # order is preserved when size gives no benefit.
    build_left = can_build_left and (not can_build_right or left_rows <= right_rows)
    if build_left:
        build_ids, probe_ids = left_ids, right_ids
        build_tensors, probe_tensors = tensors_left, tensors_right
    else:
        build_ids, probe_ids = right_ids, left_ids
        build_tensors, probe_tensors = tensors_right, tensors_left

    n_build = build_tensors[build_ids[0]].numel()
    n_probe = probe_tensors[probe_ids[0]].numel()
    if (n_probe > 0
            and n_build > _UNIQUE_HASH_MAX_BUILD_PROBE_RATIO * n_probe):
        return _join_core_fallback("build_probe_ratio_guard")
    estimated = join_core.unique_hash_peak_bytes(n_build, n_probe)
    if scratch_limit_bytes is not None and estimated > scratch_limit_bytes:
        return _join_core_fallback("scratch_guard")

    build_idx, probe_idx = join_core.unique_hash_pairs(
        [build_tensors[key] for key in build_ids],
        [probe_tensors[key] for key in probe_ids],
    )

    hashed_pairs = frozenset(selected)
    if build_left:
        return build_idx, probe_idx, hashed_pairs
    return probe_idx, build_idx, hashed_pairs


# A comparison under NOT, as the comparison it is (Q21 lowers
# ``l2.l_suppkey <> l1.l_suppkey`` as ``NOT (l_suppkey = l_suppkey)``).
_NEGATED_OP = {'=': '<>', '<': '>=', '<=': '>', '>': '<=', '>=': '<'}
# The operator with its operands swapped.
_FLIPPED_OP = {'<>': '<>', '<': '>', '<=': '>=', '>': '<', '>=': '<='}


def _summary_column(token):
    """The column id of a plain column operand (``name#id``), else None."""
    token = normalize_expr_tree(token)
    if not isinstance(token, str) or ' ' in token:
        return None
    return column_ref(token)


def _residual_summary(condition, tensors_left, tensors_right):
    """``(build column, probe column, op)`` when ``condition`` compares one
    build (left) column with one probe (right) column, both int64, by an
    operator the per-key summary answers (``join_core.SUMMARY_OPS``, oriented
    build op probe); else None."""
    tree = normalize_expr_tree(condition)
    negated = False
    while isinstance(tree, list) and len(tree) == 2 and tree[0] == 'NOT':
        negated = not negated
        tree = normalize_expr_tree(tree[1])
    if not (isinstance(tree, list) and len(tree) == 3
            and isinstance(tree[1], str)):
        return None
    first, op, second = (_summary_column(tree[0]), tree[1],
                         _summary_column(tree[2]))
    if negated:
        op = _NEGATED_OP.get(op)
    # evaluate_join reads an id the left side holds from the left side.
    if any(c in tensors_left and c in tensors_right for c in (first, second)):
        return None
    if first in tensors_left and second in tensors_right:
        build_id, probe_id = first, second
    elif first in tensors_right and second in tensors_left:
        build_id, probe_id, op = second, first, _FLIPPED_OP.get(op)
    else:
        return None
    if op not in join_core.SUMMARY_OPS:
        return None
    build, probe = tensors_left[build_id], tensors_right[probe_id]
    if not (_valid_hash_key(build) and _valid_hash_key(probe)):
        return None
    return build, probe, op


def _try_hash_semi_anti(tensors_left, tensors_right, left_key, right_key,
                        condition_list, anti):
    """The kept probe (right) rows of a semi or anti join on a hash set of
    the build (left) keys, or None for the sort/search path: a key that is
    not one int64 column a side, a residual the summary cannot answer, or a
    build key equal to int64 min (the table's empty-slot value)."""
    build, probe = tensors_left[left_key], tensors_right[right_key]
    if not (_valid_hash_key(build) and _valid_hash_key(probe)):
        return None
    summary = None
    if condition_list:
        if len(condition_list) != 1:
            return None
        summary = _residual_summary(condition_list[0], tensors_left,
                                    tensors_right)
        if summary is None:
            return None
    return join_core.semi_anti(build, probe, anti, summary)


def evaluate_join(device_name, tensorsa, tensorsb, tree, leftidx, rightidx):
    def leaf(id: int):
        if id in tensorsa.keys():
            return tensorsa[id][leftidx].squeeze()
        else:
            return tensorsb[id][rightidx].squeeze()
    # Shared evaluator is centralized in operator_utils.
    return evaluate_filter(device_name, {}, tree, leftidx.size(0), None, leaf)
  
def _condition_col_id(token):
    if not isinstance(token, str) or '#' not in token:
        return None
    try:
        return int(token.rsplit('#', 1)[1])
    except ValueError:
        return None


def _is_hashed_equality_condition(condition, hashed_pairs) -> bool:
    """True only for an exact ``#left = #right`` pair hashed by the core."""
    if not isinstance(condition, (list, tuple)) or len(condition) != 3:
        return False
    if condition[1] != '=':
        return False
    first = _condition_col_id(condition[0])
    second = _condition_col_id(condition[2])
    return (first, second) in hashed_pairs or (second, first) in hashed_pairs


def _apply_fast_path_conditions(tensors_left, tensors_right, condition_list,
                                left_matched, right_matched, device,
                                hashed_pairs):
    """Resume the existing residual-condition semantics after pair lookup."""
    # A secondary integer equality included in the exact hash tuple is already
    # true for every pair, including after a fingerprint collision (the probe
    # compares the complete tuple).  Re-evaluating that tautology introduced a
    # second dynamic nonzero in the first composite A/B.  Only remove this
    # exact syntactic case; float, non-equi, and compound residuals stay frozen.
    residual_conditions = [
        condition for condition in condition_list
        if not _is_hashed_equality_condition(condition, hashed_pairs)
    ]
    if not residual_conditions or left_matched.numel() == 0:
        return left_matched, right_matched

    match_mask = torch.ones(left_matched.numel(), dtype=torch.bool, device=device)
    for cond in residual_conditions:
        match_mask &= evaluate_join(
            device.type, tensors_left, tensors_right, cond,
            left_matched, right_matched,
        )
    matched_positions = torch.nonzero(match_mask, as_tuple=False).view(-1)
    if matched_positions.numel() == 0:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty.clone()
    return left_matched[matched_positions], right_matched[matched_positions]


def _finish_join_result(tensors_left, tensors_right, join_output_keys,
                        return_global_index, join_type, left_matched,
                        right_matched, left_global_idx, right_global_idx,
                        n_right, device):
    """Unchanged join-type, gather, global-index, and null-sentinel tail."""
    if join_type == 'inner':
        leftOutIdx = left_matched
        rightOutIdx = right_matched
    elif join_type == 'right-semi':
        # Bool-scatter dedup: torch.unique sorts internally (~5-6 GB transient
        # on 339M int64 at SF300/ANS).  nR-byte bool + nonzero is far cheaper.
        if right_matched.numel() > 0:
            mask_semi = torch.zeros(n_right, dtype=torch.bool, device=device)
            mask_semi.index_fill_(0, right_matched, True)
            rightOutIdx = torch.nonzero(mask_semi, as_tuple=False).view(-1)
            mask_semi = None
        else:
            rightOutIdx = torch.tensor([], dtype=torch.int64, device=device)
        leftOutIdx = torch.tensor([], dtype=torch.int64, device=device)
    elif join_type == 'right-anti':
        # right rows that never matched — bool-scatter the matched set, take complement.
        mask_all = torch.ones(n_right, dtype=torch.bool, device=device)
        if right_matched.numel() > 0:
            # idempotent on duplicates; index_fill_ copies no host scalar
            mask_all.index_fill_(0, right_matched, False)
        rightOutIdx = torch.nonzero(mask_all, as_tuple=False).view(-1)
        mask_all = None
        leftOutIdx = torch.tensor([], dtype=torch.int64, device=device)
    elif join_type == 'right-outer':
        # matched pairs plus unmatched rights appended as nulls on left side
        mask_all = torch.ones(n_right, dtype=torch.bool, device=device)
        if right_matched.numel() > 0:
            # idempotent on duplicates; index_fill_ copies no host scalar
            mask_all.index_fill_(0, right_matched, False)
        unmatched_rights = torch.nonzero(mask_all, as_tuple=False).view(-1)
        mask_all = None
        leftOutIdx = left_matched
        rightOutIdx = right_matched
        if unmatched_rights.numel() > 0:
            rightOutIdx = torch.cat([rightOutIdx, unmatched_rights])
            nulls = unmatched_rights.size(0)
        else:
            nulls = 0
    else:
        raise NotImplementedError("unhandled join type")

    select_function = _index_select
    # The unmatched right rows' left side is null, also when the call matched
    # nothing (an empty left_matched): both out_index columns keep one row
    # per output row.
    if join_type == 'right-outer' and nulls > 0:
        leftOutIdx = torch.cat(
        (leftOutIdx,
        torch.full((nulls, ), constants.null, dtype=torch.int64, device=device)
        ), dim=0)
        select_function = index_with_null
    return _gather_join_result(tensors_left, tensors_right, join_output_keys,
                               return_global_index, leftOutIdx, rightOutIdx,
                               left_global_idx, right_global_idx,
                               select_function)


def _index_select(x, idx):
    return torch.index_select(x, 0, idx)


def _gather_join_result(tensors_left, tensors_right, join_output_keys,
                        return_global_index, leftOutIdx, rightOutIdx,
                        left_global_idx, right_global_idx,
                        select_function=_index_select):
    """The join's output columns at the output rows of each side, then the
    two global indexes when the join returns them."""
    result = []
    for key in join_output_keys:
        if key in tensors_left:
            result.append(select_function(tensors_left[key], leftOutIdx))
        else:
            result.append(torch.index_select(tensors_right[key], 0, rightOutIdx))

    if return_global_index:
        result.extend([
            select_function(left_global_idx, leftOutIdx),
            torch.index_select(right_global_idx, 0, rightOutIdx),
        ])
    return result


def _sort_search_pairs(tensors_left, tensors_right, left, right,
                       condition_list, bucket_counts, device, name="join"):
    """The sort/search core's matching ``(left rows, right rows)``.

    Per partition bucket, the left keys are sorted and each right row's run
    of equal keys is found with two ``searchsorted`` (a key lives in one
    bucket, so runs never cross buckets); every right row's run, empty or
    not, is expanded into pairs by one kernel, which waits for the pair
    count; the residual conditions are evaluated on every pair.  The pairs
    are in right-row order and, within a right row, sorted-key order.
    ``bucket_counts``: the tick's per-bucket row counts of each side, or
    None for one bucket.  A call whose pairs the residual conditions would
    filter is at most ``INT_ROWS_MAX`` pairs (``nonzero`` over their match
    mask; a many-to-many key's runs can make more); over it the call
    raises, naming the join ``name``, before the pairs are made.
    """
    nL, nR = left.size(0), right.size(0)
    left_counts, right_counts = (bucket_counts if bucket_counts is not None
                                 else ([nL], [nR]))
    assert len(left_counts) == len(right_counts), (
        f"bucket_counts sides disagree: {len(left_counts)} vs "
        f"{len(right_counts)} buckets")
    assert sum(left_counts) == nL and sum(right_counts) == nR, (
        f"bucket_counts misaligned with tick tensors: left "
        f"{sum(left_counts)} != {nL} or right {sum(right_counts)} != {nR}")
    sorted_ids = torch.empty(nL, dtype=torch.int64, device=device)
    start = torch.zeros(nR, dtype=torch.int64, device=device)
    length = torch.zeros(nR, dtype=torch.int64, device=device)
    ls = rs = 0
    for lcnt, rcnt in zip(left_counts, right_counts):
        le, re_ = ls + lcnt, rs + rcnt
        if lcnt > 0:
            sk_b, perm_b = torch.sort(left[ls:le])
            # Left ids are the identity, so the within-bucket perm offset by
            # the bucket start is the tick-local row.
            sorted_ids[ls:le] = perm_b.add_(ls)
            if rcnt > 0:
                rk_b = right[rs:re_]
                lo = torch.searchsorted(sk_b, rk_b)
                length[rs:re_] = torch.searchsorted(sk_b, rk_b, right=True) - lo
                start[rs:re_] = lo.add_(ls)
        ls, rs = le, re_
    if condition_list:
        n_pairs = int(length.sum().item())
        if n_pairs > INT_ROWS_MAX:
            raise RuntimeError(
                f"{name}: a core call of {nL} left and {nR} right rows makes "
                f"{n_pairs} pairs, over the {INT_ROWS_MAX} its residual "
                f"conditions take")
    left_matched, right_matched = join_core.expand(sorted_ids, start, length)
    if condition_list and left_matched.numel() > 0:
        match_mask = torch.ones(left_matched.numel(), dtype=torch.bool,
                                device=device)
        for cond in condition_list:
            match_mask &= evaluate_join(device.type, tensors_left,
                                        tensors_right, cond, left_matched,
                                        right_matched)
        kept = torch.nonzero(match_mask, as_tuple=False).view(-1)
        match_mask = None
        left_matched = left_matched[kept]
        right_matched = right_matched[kept]
    return left_matched, right_matched


def join_kernel_new(tensors_left, tensors_right, left_key, right_key, condition_list, join_output_keys,
                return_global_index, type: Literal['inner', 'right-semi', 'right-anti', 'right-outer'],
                left_global_idx, right_global_idx, bucket_counts=None,
                equality_keys=None, unique_key_sets=None,
                hash_scratch_limit_bytes=None, name="join"):
    assert type in {'inner', 'right-semi', 'right-anti', 'right-outer'}, "Invalid Join type."
    device_name = 'cuda'
    device = torch.device(device_name)
  
    left = tensors_left[left_key]
    right = tensors_right[right_key]
    nR = right.size(0)

    fast_pairs = _try_unique_hash_pairs(
        tensors_left, tensors_right, left_key, right_key, type,
        equality_keys, unique_key_sets,
        (_UNIQUE_HASH_SCRATCH_LIMIT_BYTES
         if hash_scratch_limit_bytes is None else hash_scratch_limit_bytes),
    )
    if fast_pairs is not None:
        left_matched, right_matched, hashed_pairs = fast_pairs
        left_matched, right_matched = _apply_fast_path_conditions(
            tensors_left, tensors_right, condition_list,
            left_matched, right_matched, device, hashed_pairs,
        )
        return _finish_join_result(
            tensors_left, tensors_right, join_output_keys,
            return_global_index, type, left_matched, right_matched,
            left_global_idx, right_global_idx, nR, device,
        )

    if type in ('right-semi', 'right-anti'):
        kept = _try_hash_semi_anti(tensors_left, tensors_right, left_key,
                                   right_key, condition_list,
                                   anti=type == 'right-anti')
        if kept is not None:
            return _gather_join_result(
                tensors_left, tensors_right, join_output_keys,
                return_global_index,
                torch.empty(0, dtype=torch.int64, device=device), kept,
                left_global_idx, right_global_idx)

    left_matched, right_matched = _sort_search_pairs(
        tensors_left, tensors_right, left, right, condition_list,
        bucket_counts, device, name)
    left = right = None  # free local aliases; the dicts hold the tensors
    return _finish_join_result(
        tensors_left, tensors_right, join_output_keys,
        return_global_index, type, left_matched, right_matched,
        left_global_idx, right_global_idx, nR, device,
    )


# The plan's join types, as the join core names them.
JOIN_TYPES = {
    'Inner': 'inner',
    'LeftSemi': 'right-semi',
    'LeftOuter': 'right-outer',
    'LeftAnti': 'right-anti',
}


def join_call_args(args, tensor_group):
    """``(left key, right key, condition list, join type)`` of a plan Join:
    the partition key of each side, the residual conditions the core
    evaluates on each pair, and the core's join type.  A two-key join
    partitions on its second key when the first is a float (and on its
    first otherwise); the other key's equality joins the conditions."""
    left_keys, right_keys = args['left keys'], args['right keys']
    condition_list = []
    if 'join condition' in args.keys():
        condition_list.append(args['join condition'])
    join_left_key, join_right_key = left_keys[0], right_keys[0]
    if len(left_keys) == 2:
        if tensor_group[left_keys[0]].is_floating_point():
            assert not tensor_group[left_keys[1]].is_floating_point(), \
                "Do not support join on two float-type keys"
            join_left_key, join_right_key = left_keys[1], right_keys[1]
            condition_list.append([f'#{left_keys[0]}', '=', f'#{right_keys[0]}'])
        else:
            condition_list.append([f'#{left_keys[1]}', '=', f'#{right_keys[1]}'])
    assert args['join type'] in JOIN_TYPES, \
        f"Unsupport Join Type {args['join type']}"
    return (join_left_key, join_right_key, condition_list,
            JOIN_TYPES[args['join type']])


def join_side_inputs(args, side):
    """One side's transfer set, the join's ``join in`` columns of the side,
    sorted (the order of the side's sort node's outputs), and the id of the
    one mask over them in the join's ``mask map`` (None for none)."""
    group = (args['left group'], args['right group'])[side]
    transfer = sorted(c for c in args['join in'] if c in group)
    mask_map = args.get('mask map') or {}
    masks = {mask_map[c] for c in transfer if c in mask_map}
    assert len(masks) <= 1, (
        f"one mask per join side, got {sorted(masks)} on side {side}")
    return transfer, next(iter(masks), None)


def join_output_keys(args):
    """The keys the join's core emits and whether it emits the two
    ``out_index`` columns too.  The join's full output is the columns
    gathered now plus the deferred ones; keys are never deferred, so the
    core emits them, and it emits both ``out_index`` columns whenever any
    payload exists, deferred or not."""
    full_output = (list(args['materialize_now'])
                   + list(args.get('materialize_deferred') or []))
    keys = [k for k in full_output if k in args['join in']]
    return keys, len(keys) != len(full_output)


def join_kernel_kwargs(args):
    """``join_kernel_new``'s keyword arguments a plan Join gives: its
    equality keys, the key sets each side is proven unique on, and the
    unique-build hash core's scratch limit."""
    return dict(
        equality_keys=(args.get('left keys', []), args.get('right keys', [])),
        unique_key_sets=(args.get('left unique keys', []),
                         args.get('right unique keys', [])),
        hash_scratch_limit_bytes=_UNIQUE_HASH_SCRATCH_LIMIT_BYTES)


def pack_core_calls(rows_left, rows_right, side_row_bytes, budget_b):
    """Whole buckets per core call: each bucket pair's bytes (its rows on
    each side, ``rows_left[b]`` and ``rows_right[b]``, at the side's
    ``side_row_bytes``) packed up to ``budget_b`` by the tick cutter
    (``IO/pipeline/tick_cutter.py``), at least one bucket a call; matches
    are confined to a bucket pair.  Returns each call's end bucket
    (exclusive) and the pairs' bytes."""
    pair_b = (np.asarray(rows_left, dtype=np.int64) * side_row_bytes[LEFT]
              + np.asarray(rows_right, dtype=np.int64) * side_row_bytes[RIGHT])
    ends = tick_cutter.pack(np.concatenate(([0], np.cumsum(pair_b))),
                            budget_b).tolist()
    return ends, pair_b


def join_partition_count(args, join_key_index, tensor_group):
    """A join's hash-partition count B, the same for both sides, from the
    partition budget.  Each side's estimate is what its sort keeps:

    * rows: the transfer set's rows, times the selectivity the filter
      recorded on the mask the sort folds (exact), if it folds one;
    * row bytes: the bytes per row the core reads, each transfer column's
      element size times its width (a string is int8 [rows, width]) plus
      the int64 gidx;
    * keys: an estimate of its distinct join keys.  A side proven unique on
      the partition key has one row per key.  The other side of such a join
      is taken to draw its keys from the unique side's rows before any mask:
      for a scanned unique side, the table's key domain (referential
      integrity); for a join output, only its matched rows, which can be far
      fewer than the other side's keys (an under-count, which only raises
      the predicted pair).  Otherwise every row is taken as its own key.

    ``join_key_index``: the two partition keys; ``tensor_group``: the
    columns (their types, widths and lengths) and masks (the selectivity
    each recorded, None if unknown).  Returns ``(B, predicted largest pair
    bytes, the two sides' (rows, row bytes, keys), {transfer column: bytes
    per row})``."""
    unique_key_sets = (args.get('left unique keys') or [],
                       args.get('right unique keys') or [])
    unique_on_key = [any(k and set(k) <= {join_key_index[s]}
                         for k in unique_key_sets[s]) for s in (LEFT, RIGHT)]
    col_bytes = {}
    side_rows, domain_rows = [], []
    for side in [LEFT, RIGHT]:
        cols, mask_id = join_side_inputs(args, side)
        assert cols, f"side {side} of the join sorts no column"
        for c in cols:
            col_bytes[c] = (dtype_size(tensor_group[c].get_type_torch())
                            * max(1, tensor_group[c].col_width()))
        rows = tensor_group[cols[0]].col_length()
        domain_rows.append(rows)
        if mask_id is not None:
            sel = tensor_group[mask_id].selectivity
            if sel is not None:
                rows = int(round(rows * sel))
        side_rows.append((rows, sum(col_bytes[c] for c in cols)
                          + dtype_size(torch.int64)))
    est_sides = []
    for side in [LEFT, RIGHT]:
        rows, row_bytes = side_rows[side]
        keys = rows
        if (not unique_on_key[side] and unique_on_key[1 - side]
                and domain_rows[1 - side] > 0):
            keys = min(rows, domain_rows[1 - side])
        est_sides.append((rows, row_bytes, keys))
    B, pred_b = partition_count(est_sides, constants.get_partition_budget_b(),
                                JOIN_B_MAX)
    return B, pred_b, est_sides, col_bytes


def log_join_partition_count(name, B, pred_b, est_sides):
    """The ``[join_b]`` line: B, each side's estimate and the predicted
    largest pair against the budget."""
    partition_budget_b = constants.get_partition_budget_b()
    est_b = sum(rows * row_b for rows, row_b, _ in est_sides)
    message_logger().info(
        "[join_b] Q%s %s: B=%d for %.1f MB (left %d rows x %d B over %d "
        "keys, right %d rows x %d B over %d keys), largest pair predicted "
        "%.1f MB against %.1f MB%s",
        constants.get_query_id(), name, B, est_b / (1 << 20),
        *est_sides[LEFT], *est_sides[RIGHT], pred_b / (1 << 20),
        partition_budget_b / (1 << 20),
        " OVER" if pred_b > partition_budget_b else "")
    return est_b


def tqp_join(tensor_group: Dict[int, CombinedColumnRep], args, left_key_col, right_key_col, condition_list, type,
                cpu_mem_pool, cache_region, name, applier=None, page_cache=None):
    perf_logger().start(name)

    # Chunk budgets are per pipe through the sizing resolver:
    # the sort pipes resolve per side (class 'sort') at their construction,
    # the core pipe (class 'core') at the bin-pack below.  With
    # --sizing_search off every resolve returns the uniform
    # constants.get_chunk_size_b().

    join_key_index = [left_key_col, right_key_col]

    # Per-call hash-partition count, derived ONCE here (before the for-side
    # loop) so LEFT and RIGHT use the same B by construction, from each
    # side's estimate of what its sort keeps (join_partition_count).  The
    # nested closures (sort_kernel, _partition_rows,
    # _per_col_boundaries_packed, join_wrapper) and the bin-pack body close
    # over this local via Python LEGB scoping.
    B_PARTITIONS, pred_b, est_sides, col_bytes = join_partition_count(
        args, join_key_index, tensor_group)
    partition_budget_b = constants.get_partition_budget_b()
    est_b = log_join_partition_count(name, B_PARTITIONS, pred_b, est_sides)

    column_view, key_view, global_index_view = [{}, {}], [None, None], [None, None]
    sort_nodes = args['nodes']['sort']
    cur_remapped_id = 0
    join_remapped_ids, global_index_remapped_ids = {}, [None, None]
    used_row_size = [0, 0]
    # Per-side B-length list of per-partition row counts, populated when
    # the sort pipe's per-(col, partition) page sub-lists are collapsed
    # into one partition-major CCR per logical column. Consumed by the
    # bin-pack (_partition_rows) and the packed per-tick boundary builder.
    partition_row_counts_per_side: List[List[int]] = [[], []]
    # Persistent bucket_counts (per-chunk cumsum lists) is gone.  With
    # B_PARTITIONS-fold partitioning, bucket boundaries are encoded by
    # variable identity (one Variable per partition); the cumsum is a
    # per-chunk transient inside sort_kernel.

    for side in [LEFT, RIGHT]:
        # Sort chunk K (governs page fill K/B and sort-pipe peak; not
        # correctness).  Resolved per side below (class 'sort'), once the
        # side's pipe name exists (exactly one resolve per pipe).
        # Phase 3 keeps `args['join in']` (key + condition cols) in the
        # sort-pipe scatter — condition cols ride the core pipeline alongside
        # the key, so condition_list evaluates per-tick exactly as in Phase 2.
        # What changes in Phase 3 is the *output* shape: the core pipe emits
        # only matched gidx pairs (no materialized columns), and downstream
        # rearrange gathers the join's payload from the original
        # tensor_group cols via the matched gidx.
        # Sorted, as the graph lists the side's transfer set (the sort node's
        # outputs are in this order); one mask at most.
        transfer_set, _ = join_side_inputs(args, side)

        sizes = {tensor_group[col].col_length() for col in transfer_set}
        assert len(sizes) == 1, f"Columns in join have different numbers of rows: {sizes}"

        key_index = transfer_set.index(join_key_index[side])
        sort_mask_map = {}
        data_size = next(iter(sizes))

        if 'mask map' in args:
            for k in list(args['mask map'].keys()):
                if k in transfer_set:
                    mask_id = args['mask map'].pop(k)
                    message_logger().critical("mask len %s width %s", tensor_group[mask_id].col_length(), tensor_group[mask_id].col_width())
                    if tensor_group[mask_id].col_length() != data_size:
                        raise AssertionError(f"Mask col {mask_id} size {tensor_group[mask_id].col_length()} != data size {data_size}")
                    sort_mask_map[k] = mask_id
        # sort_kernel applies one mask to the whole side (join_side_inputs
        # checked there is one at most), and a folding pipe leaves every
        # masked column raw for it.

        current_global_offset = 0

        # The side's partition, by the fused kernel (operators/partition_sort.py).
        # Under a mask the pipe folds it: the direct columns arrive raw and
        # mask_counts holds the mask's kept rows.
        def sort_kernel(columns, mask_counts=None):
            nonlocal current_global_offset

            mask_id = next(iter(sort_mask_map.values()), None)
            mask = columns[mask_id] if mask_id is not None else None
            n_valid = mask_counts[mask_id] if mask_counts is not None else None
            cols = [columns[i] for i in transfer_set]
            goff = current_global_offset
            current_global_offset += (mask.shape[0] if mask is not None
                                      else cols[key_index].shape[0])
            return partition(cols, key_index, mask, goff, B_PARTITIONS,
                             n_valid=n_valid)

        # Output type list: same dtype repeated B_PARTITIONS times per logical column.
        sort_output_columns_type = (
            [tensor_group[i].get_type_torch()
             for i in transfer_set for _ in range(B_PARTITIONS)]
            + [torch.int64 for _ in range(B_PARTITIONS)]
        )

        # The side's sort node: its inputs are the transfer set (plus the
        # side's mask ids) as current instances, its outputs one instance per
        # sorted column and one for global_index.  The physical slots are B
        # partitions per logical column, in the same order; the layout maps
        # every slot onto its instance and carries the partition as the
        # cache's use-order hint.
        sort_node = sort_nodes[side]
        sort_pipe_name = sort_node.name
        sort_chunk_size_b = sizing_resolver.resolve(
            "sort", sort_pipe_name).chunk_b
        assert [inst.local for inst in sort_node.outputs] == (
            [f"sorted_col_{k}" for k in transfer_set] + ["global_index"]), (
            [inst.local for inst in sort_node.outputs], transfer_set)
        sort_layout = OutputLayout.partitioned(len(transfer_set) + 1, B_PARTITIONS)

        sort_pipe = VortexPipeline(node=sort_node,
                                inputs=sort_node.input_pairs(),
                                layout=sort_layout,
                                output_columns_type=sort_output_columns_type,
                                operator=sort_kernel,
                                mask_map=sort_mask_map,
                                cpu_mem_pool=cpu_mem_pool,
                                pass_mask=True,
                                fold_mask=bool(sort_mask_map),
                                chunk_size=sort_chunk_size_b,
                                applier=applier,
                                page_cache=page_cache,
                                sizing_key=("sort", sort_pipe_name))
        
        sort_pipe.do_exchange()
        sort_pipe.get_result()
        # The sort's outputs are record tables of regions, one stream per
        # partition, by output instance (none for an instance with no row).
        sort_tables = sort_pipe.result_tables()
        ## Free sort input — release pinned backing memory for columns consumed by the sort pipe.
        ## The sorted output (sort_tables) replaces these entirely for the join.
        ## (Phase 3 attempted to keep transfer_set ∩ payload cols alive
        ## for a post-pipe rearrange-from-original gather, but at SF100 the
        ## big keys (e.g. l_orderkey, 4.8 GB) staying live across multiple
        ## chained joins blew the pool.  Reverted to Phase 2's eager free.)
        for col_id in transfer_set + list(sort_mask_map.values()):
            tensor_group[col_id].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=cache_region)

        # Wrap the sort's outputs into one CCR per logical column.
        # sort_tables[c] for c in [0, len(transfer_set)) is transfer_set[c];
        # sort_tables[len(transfer_set)] is the global index.
        from variable import torch_to_type

        # tensor_type per logical col for an output with no row (an empty
        # Variable cannot infer it; '' would map to string_dtype=int8 —
        # wrong for int64 cols and int64 gidx).  Derive from the declared
        # output_columns_type.
        tensor_type_per_col = [
            torch_to_type(tensor_group[k].get_type_torch()) for k in transfer_set
        ] + [torch_to_type(torch.int64)]  # gidx

        join_mask_map = {}
        # One partition-major CCR per logical column: a table's row space is
        # its B streams (partitions) in order; the core pipe slices it per
        # tick via the row-count cumsum below.  An output with no row is an
        # empty column of B empty partitions.
        for c, k in enumerate(transfer_set):
            table = sort_tables.get(c)
            if table is not None:
                counts = table.stream_rows().tolist()
                ccr = CombinedColumnRep([TableVariable(
                    table, torch_to_type(table.stored_dtype))])
            else:
                counts = [0] * B_PARTITIONS
                ccr = CombinedColumnRep([Variable(
                    tensor_type=tensor_type_per_col[c])])
            label = f"sorted_col_{k}"
            sort_node.outputs[c].ccr = ccr
            column_view[side][k] = ccr
            join_remapped_ids[k] = cur_remapped_id
            cur_remapped_id += 1

            if c == 0:
                partition_row_counts_per_side[side] = counts
            else:
                assert counts == partition_row_counts_per_side[side], (
                    f"partition counts diverged across cols on side {side}: "
                    f"col {k} got {counts}, expected {partition_row_counts_per_side[side]}"
                )

            if c == key_index:
                key_view[side] = ccr
            used_row_size[side] += col_bytes[k]

            if 'mask map' in args:
                if k in list(args['mask map'].keys()):
                    raise RuntimeError(
                        "hashjoin mask_map for sort-pipe outputs is not yet "
                        "wired through the partitioned pipeline."
                    )

        # global_index — same partition-major collapse.
        gidx_table = sort_tables.get(len(transfer_set))
        gidx_ccr = CombinedColumnRep([
            TableVariable(gidx_table, torch_to_type(gidx_table.stored_dtype))
            if gidx_table is not None
            else Variable(tensor_type=tensor_type_per_col[-1])])
        sort_node.outputs[-1].ccr = gidx_ccr
        global_index_view[side] = gidx_ccr
        global_index_remapped_ids[side] = cur_remapped_id
        cur_remapped_id += 1

    # Greedy bin-pack of partition pairs into ticks.  Each tick processes
    # K partitions on both sides; partition `b` always goes to the same
    # tick on both sides because hash partitioning means key matches are
    # confined to within-bucket pairs.  Budget halved because the pipeline
    # double-buffers (H2D tick i+1 while compute tick i runs).
    def _partition_rows(side: int, b: int) -> int:
        if not column_view[side]:
            return 0
        return partition_row_counts_per_side[side][b]

    # Per-row bytes per side: data cols (used_row_size, each column's element
    # size times its width) + int64 gidx.  Uncompressed size; compression
    # typically improves headroom, so this is a conservative budget.
    side_row_bytes = [used_row_size[s] + dtype_size(torch.int64) for s in (LEFT, RIGHT)]
    # Core budget K (class 'core'): resolved once here — the bin-pack and the
    # pipe construction below both derive from this single value.
    core_chunk_size_b = sizing_resolver.resolve(
        "core", f'{name} core pipeline').chunk_b
    budget = core_chunk_size_b // 2

    # Whole buckets up to the budget, at least one per tick
    # (``pack_core_calls``); bucket boundaries are record boundaries of
    # every column the core reads.
    tick_ends, pair_b = pack_core_calls(
        [_partition_rows(LEFT, b) for b in range(B_PARTITIONS)],
        [_partition_rows(RIGHT, b) for b in range(B_PARTITIONS)],
        side_row_bytes, budget)
    tick_partitions: List[List[int]] = [
        list(range(a, e)) for a, e in zip([0] + tick_ends[:-1], tick_ends)]
    num_ticks = len(tick_partitions)
    message_logger().critical(
        "[%s] partitioned join: B=%d num_ticks=%d budget=%.1f MB | partition "
        "budget %.1f MB: estimated %.1f MB, partitioned %.1f MB, largest "
        "pair %.1f MB", name, B_PARTITIONS, num_ticks, budget / (1 << 20),
        partition_budget_b / (1 << 20), est_b / (1 << 20),
        int(pair_b.sum()) / (1 << 20), int(pair_b.max()) / (1 << 20))
    message_logger().debug(
        "[%s] bucket rows: left %s right %s", name,
        [_partition_rows(LEFT, b) for b in range(B_PARTITIONS)],
        [_partition_rows(RIGHT, b) for b in range(B_PARTITIONS)])

    # ── Per-bucket sort schedule ─────────────────────────────────────────
    # join_wrapper receives only the column dict from VortexPipeline, so
    # the current tick's bucket boundaries are threaded through closure
    # state via a nonlocal invocation counter (same pattern as
    # sort_kernel's nonlocal current_global_offset above).  One schedule
    # entry per *operator invocation*: _execute_pre_comp_and_operator
    # early-returns without calling the operator when every input chunk
    # is empty (all raw_numel() == 0) — exactly the ticks whose assigned
    # partitions hold zero rows on BOTH sides — and the two epilogue
    # stages are empty by construction, so all-empty ticks are excluded
    # here to keep the counter aligned with real compute ticks.
    join_tick_bucket_counts: List = []
    for parts_in_tick in tick_partitions:
        lc = [partition_row_counts_per_side[LEFT][b] for b in parts_in_tick]
        rc = [partition_row_counts_per_side[RIGHT][b] for b in parts_in_tick]
        if sum(lc) + sum(rc) == 0:
            continue  # all-empty tick: the operator is never invoked for it
        join_tick_bucket_counts.append((lc, rc))
    join_invocation_no = 0

    # The keys the core emits, and whether it emits the out_index columns.
    out_keys, rearrange_required = join_output_keys(args)
    # The core node's inputs are the two sort nodes' outputs, in order (left
    # sorted cols, left global_index, right sorted cols, right global_index):
    # the remapped ids assigned above pair with them positionally.
    core = args['nodes']['core']
    core_name = core.name
    core_input_pairs = []
    join_div: List[List[int]] = []
    join_out_types = []

    def _per_col_boundaries_packed(counts_b: List[int]) -> List[int]:
        """Cumulative per-tick row boundaries for one packed (partition-major)
        logical-column input. Each tick consumes the rows for its assigned
        partitions; the merged Variable's row layout is partition 0..B-1 in
        order, and ``tick_partitions`` itself is ordered, so the slice for
        tick t spans `[sum(counts[b] for b in tick_partitions[0..t-1]),
        sum(counts[b] for b in tick_partitions[0..t]))` — a contiguous,
        page-aligned page sub-range.
        """
        cum: List[int] = []
        running = 0
        for parts_in_tick in tick_partitions:
            running += sum(counts_b[b] for b in parts_in_tick)
            cum.append(running)
        return cum

    for side in [LEFT, RIGHT]:
        boundaries = _per_col_boundaries_packed(partition_row_counts_per_side[side])
        for c, k in enumerate(column_view[side]):
            core_input_pairs.append((join_remapped_ids[k], sort_nodes[side].outputs[c]))
            join_div.append(boundaries)
        core_input_pairs.append((global_index_remapped_ids[side], sort_nodes[side].outputs[-1]))
        join_div.append(boundaries)
    assert [inst for _, inst in core_input_pairs] == core.inputs, (
        f"[{name}] the core's inputs are not the sort nodes' outputs")

    join_out_types.extend([tensor_group[k].get_type_torch() for k in out_keys])
    
    if rearrange_required:
        join_out_types.extend([torch.int64, torch.int64])
    assert [inst.local for inst in core.outputs] == out_keys + (
        ['out_index_left', 'out_index_right'] if rearrange_required else []), (
        [inst.local for inst in core.outputs], out_keys)
     
    def join_wrapper(columns):
        # VortexPipeline's H2D + decompress already produced one fat
        # per-tick tensor per logical column covering all partitions in
        # tick_partitions[t] (in partition-major row order).  The tick's
        # per-bucket row counts are passed down so the sort/search core sorts
        # and probes per partition bucket (matches are confined to a
        # bucket); the hash cores take the tick whole.
        nonlocal join_invocation_no
        tick_no = join_invocation_no
        join_invocation_no += 1
        assert tick_no < len(join_tick_bucket_counts), (
            f"[{name}] join_wrapper invoked {tick_no + 1} times but only "
            f"{len(join_tick_bucket_counts)} non-empty ticks were scheduled"
        )
        bucket_counts = join_tick_bucket_counts[tick_no]
        columns = list(columns.values())
        tensor_inputs = [{}, {}]
        global_idx = [None, None]

        for side in [LEFT, RIGHT]:
            for k in column_view[side]:
                tensor_inputs[side][k] = columns[join_remapped_ids[k]]
            global_idx[side] = columns[global_index_remapped_ids[side]]

        return join_kernel_new(tensor_inputs[LEFT], tensor_inputs[RIGHT], join_key_index[LEFT], join_key_index[RIGHT],
                            condition_list, out_keys, rearrange_required, type, global_idx[LEFT], global_idx[RIGHT],
                            bucket_counts=bucket_counts, name=name,
                            **join_kernel_kwargs(args))
    
    # Codecs and host residency come off the core's output instances: a key
    # a downstream rearrange gathers is RAW and host-resident, the seed
    # join's lineage out_index is RAW, and every out_index is host-resident
    # (it exists only to drive a UVA gather).
    pipe = VortexPipeline(
        node=core,
        inputs=core_input_pairs,
        output_columns_type=join_out_types,
        mask_map=join_mask_map,
        operator=join_wrapper,
        div=join_div,
        cpu_mem_pool=cpu_mem_pool,
        chunk_size=core_chunk_size_b,
        sizing_key=("core", core_name),
        applier=applier,
        page_cache=page_cache,
    )

    pipe.do_exchange()
    pipe.get_result()

    materialized_columns = {}
    for i, k in enumerate(out_keys):
        # A key keeps its column's own type name: re-wrap the instance's
        # payload (the pipeline typed it from the torch dtype).
        materialized_columns[k] = CombinedColumnRep([pipe.output_variable(
            i, tensor_group[k].tensor_type())])
        core.outputs[i].ccr = materialized_columns[k]

    if rearrange_required:
        n_out = len(join_out_types)
        assert len(materialized_columns) + 2 == n_out
        out_index = [CombinedColumnRep([pipe.output_variable(n_out - 2, 'int')]),
                     CombinedColumnRep([pipe.output_variable(n_out - 1, 'int')])]
        core.outputs[-2].ccr = out_index[0]
        core.outputs[-1].ccr = out_index[1]

        del pipe
        message_logger().debug("hashjoin: after del pipe")
        for side in [LEFT, RIGHT]:
            for k, ccr in column_view[side].items():
                ccr.free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=cache_region)
            global_index_view[side].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=cache_region)
        del column_view, key_view, global_index_view
        message_logger().debug("hashjoin: after freeing sort/join intermediates")
        for side in [LEFT, RIGHT]:
            side_label = "left" if side == LEFT else "right"
            # The side's gather nodes, in the order the graph lists them: a
            # rearrange of the side's payload (materialize_now minus the
            # keys) when the side has any, then one lineage compose per
            # deferred group composed here — the group's current lineage
            # gathered by this side's out_index; the gather kernel preserves
            # CONST_NULL, so outer-join nulls propagate through composition.
            # A seed join has no compose: its out_index is the lineage, kept
            # alive by the lifetime rule until the compose or deferred gather
            # that reads it.  Deferred columns stay as their base reps (they
            # are in 'output ids', so the end-of-join free loop won't drop
            # them) and are gathered once, before the op that first reads
            # them.  Each node reads its sources and index off its instances
            # and writes its outputs to theirs.
            for gnode in args['nodes']['gathers'][side_label]:
                rearrange_pipeline(gnode, cpu_mem_pool, type,
                                   applier=applier, page_cache=page_cache)
                if gnode.kind == 'rearrange':
                    for inst in gnode.outputs:
                        materialized_columns[inst.local] = inst.ccr
            message_logger().debug("hashjoin: after gathers side=%d", side)

    for m in materialized_columns:
        tensor_group[m] = materialized_columns[m]
    # for k in args['left group']:
    #     message_logger().debug("ROWS -- %s", tensor_group[k].tensor.size(0))
    #     message_logger().debug("ROWS -- %s %s", k, tensor_group[k].tensor[0:50])
    # for k in args['right group']:
    #     message_logger().debug("ROWS ++ %s", tensor_group[k].tensor.size(0))
    #     message_logger().debug("ROWS ++ %s %s", k, tensor_group[k].tensor[0:50])

    for input_col in args['input ids']:
        if input_col in tensor_group and input_col not in args['output ids']:
          tensor_group[input_col].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=cache_region)
          del tensor_group[input_col]

    perf_logger().stop(name)
    return 0
