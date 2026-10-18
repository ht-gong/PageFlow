import os
from math import gcd

import numpy as np
import torch

from variable import TableVariable, Variable, torch_to_type, type_to_torch
from parsing import get_id
from IO.codec.batched_compress import aligned_partition_layout
from IO.pipeline import tick_cutter
from IO.pipeline.common import PartitionedTensor
from IO.pipeline.vortex_pipeline import OutputLayout, VortexPipeline
from IO.pages.combined_column_rep import CombinedColumnRep
from utility.logger import perf_logger, message_logger
from utility.tensor_utils import dtype_size
from operators.partitions import agg_partitions
from operators.operator_utils import (
  evaluate_aggregate,
  get_aggregation_op,
  get_result_type,
  is_group_key_expr,
  normalize_expr_tree,
  right_most_expr,
)
from operators.sorted_run_merge import (
  MERGEABLE_OPS,
  merge_sorted_unique_runs_inplace,
  reduce_tick_slice,
)
from operators.aggregation_slab import (
  DenseAggregationSlab,
  FixedColumnSlab,
)
from typing import Dict, List, NamedTuple, Optional
import constants
import sizing_resolver

# Tier-3 (the sorted-run merge): in PASS2 of the two-pass
# aggregator, route the K==1 / 1-D-key direct-unique shape (Q18's inner
# ``GROUP BY l_orderkey``) through a sorted-run MERGE instead of
# ``torch.unique(cat(running, tick))``. The merge's only sort is a
# tick-local ``torch.unique`` over the new slice, so the radix-sort
# working set never covers the cumulative distinct set (whose contiguous
# CUB scratch ask grows ~340 MB per tick toward ~5 GB at SF500 and dies
# against a fragmented arena). False restores the exact original path.
SORTED_RUN_MERGE = True

# Compact K==1 integer domains carrying only partial states can bypass both
# cumulative unique and sorted-run shifting: pass-2 ticks scatter directly
# into one fixed raw slab. False rolls back to the sorted-run/generic paths.
DENSE_AGGREGATION_SLAB = True

# A flush of pass 2's stage slab merges it in compute views of at most
# _AGG_MAX_COMPUTE_VIEW_ROWS rows; the slab itself stays resident.  A view's
# merge scratch that does not fit the arena misses, and the allocator lends
# from the cache region (``memory/cuda_memory_allocator.cc``).
_AGG_MAX_COMPUTE_VIEW_ROWS = 1 << 26

# Partitioned aggregation.  A keyed aggregation whose pass-2 run is
# predicted not to fit the partition budget (``--partition_budget_mb``,
# ``operators/partitions.agg_partitions``) takes B hash buckets before pass 1:
# pass 1 writes each output's partials bucket by bucket (one record-table
# stream per bucket, like the join's sort pipe), and pass 2 packs whole
# measured buckets into rounds within the budget and aggregates, emits and
# resets one round at a time, so the resident run holds one round's
# partials.  At B = 1, or when the whole run fits, pass 2 is one round.
# AGGREGATION_PARTITIONS: 0 chooses B from the budget; > 0 forces B, a power
# of two, dense shapes included.
AGGREGATION_PARTITIONS = 0

# A result's first op (``get_aggregation_op``) as the torch scatter_reduce_
# flavour its partials merge by; any other op keeps its name.
MERGE_OPS = {'avg': 'mean', 'count': 'sum', 'max': 'amax', 'min': 'amin'}

_INT_DTYPES = (torch.int64, torch.int32, torch.int16, torch.int8)


def _narrow_key_values(dtype, width: int):
  """How many values a key of int8 or int16 ``dtype`` and ``width``
  elements can take, and what gives it; None for any other dtype."""
  if dtype == torch.int8:
    return 256 ** width, f"int8x{width}"
  if dtype == torch.int16:
    return 65536 ** width, f"int16x{width}"
  return None


# mix64's constants as int64 (two's complement), and the low 31 bits: an
# arithmetic shift right by 33 masked to them is the logical shift.
_MIX64_M1 = 0xff51afd7ed558ccd - (1 << 64)
_MIX64_M2 = 0xc4ceb9fe1a85ec53 - (1 << 64)
_LOW31 = (1 << 31) - 1


def _mix64(h: torch.Tensor) -> torch.Tensor:
  """mix64, the Murmur3 64-bit finalizer the join hashes its keys with
  (``operators/kernels/kernel_utils.cuh``), over an int64 tensor in torch's
  own elementwise ops: int64 multiplication wraps mod 2^64, and
  ``(k >> 33) & _LOW31`` is the logical shift torch lacks on int64."""
  k = h ^ ((h >> 33) & _LOW31)
  k = k * _MIX64_M1
  k = k ^ ((k >> 33) & _LOW31)
  k = k * _MIX64_M2
  return k ^ ((k >> 33) & _LOW31)


def _bucket_ids(h: torch.Tensor, B: int) -> torch.Tensor:
  """The bucket of each grouping hash ``h`` (int64) among B, a power of
  two, as int32: the top log2(B) bits of its mix64, read unsigned, the
  join's partition function (``operators/kernels/partition_sort.cu``).  A
  group's hash is the same in every tick, so are its bucket and the
  bucket's rows in pass 2."""
  if B == 1:
    return torch.zeros(h.shape[0], dtype=torch.int32, device=h.device)
  shift = 64 - (B.bit_length() - 1)
  return ((_mix64(h) >> shift) & (B - 1)).to(torch.int32)


def _partition_outputs(outs, bucket, counts, counts_t):
  """One pass-1 tick's outputs ``outs`` as ``PartitionedTensor``s in the
  sort pipe's padded layout (``batched_compress.aligned_partition_layout``):
  bucket b's ``counts[b]`` rows from a whole ``lcm(row bytes, 512)`` unit,
  where a compressed record's input must start.  ``bucket`` is each row's
  bucket (``_bucket_ids``), ``counts_t`` the counts on the GPU, so the
  layout costs no host read.  Rows keep their order (the hash's) within a
  bucket; each output takes one ``index_copy_`` to the destinations of its
  row width."""
  n = bucket.shape[0]
  order = torch.sort(bucket, stable=True).indices
  pos = torch.empty_like(order)
  pos[order] = torch.arange(n, dtype=order.dtype, device=order.device)
  align = constants.COMPRESSION_ALIGNMENT
  layouts = {}
  partitioned = []
  for t in outs:
    row_b = t.element_size()
    for extent in t.shape[1:]:
      row_b *= int(extent)
    if row_b not in layouts:
      unit_b = row_b * align // gcd(row_b, align)
      total = aligned_partition_layout(counts, row_b, unit_b)[2]
      # A row moves by the padding of the buckets before its own.
      unit = unit_b // row_b
      pad = (counts_t + unit - 1) // unit * unit - counts_t
      shift = torch.cumsum(pad, 0) - pad
      layouts[row_b] = (pos + shift.index_select(0, bucket), total)
    dest, total = layouts[row_b]
    out = torch.empty((total,) + tuple(t.shape[1:]),
                      dtype=t.dtype, device=t.device)
    out.index_copy_(0, dest, t)
    partitioned.append(PartitionedTensor(out, counts))
  return partitioned


def _agg_run_row_b(key_route: bool, keys, values) -> int:
  """Pass 2's run bytes per row, predicted before pass 1 from the emitted
  keys, ``(dtype, width)`` each, and the values, ``(dtype, state width)``
  each (a partial state holds its fields, ``_state_width``; any other value,
  width 0, one element): the key route's run holds the key and the values;
  the hash route's the 8-byte hash, the emitted keys and the values, which
  bounds the packed route's (the hash and the values)."""
  key_b = sum(dtype_size(dtype) * max(1, width) for dtype, width in keys)
  value_b = sum(dtype_size(dtype) * max(1, width) for dtype, width in values)
  if key_route:
    return key_b + value_b
  return 8 + key_b + value_b


def pass2_rounds(bucket_rows, row_b: int, budget_b: int,
                 cap_rows=None):
  """Pass 2's rounds of whole buckets: each round's end bucket
  (exclusive).  One round, ``[B]``, when the whole run fits ``budget_b`` at
  ``row_b`` bytes a row (its rows, or ``cap_rows``, the key range a
  key-route run is capped by, if fewer); else the tick cutter's greedy
  packing of adjacent whole buckets up to the budget (``tick_cutter.pack``,
  as the join packs bucket pairs into core ticks): a bucket over the budget
  alone is a round of its own."""
  points = np.zeros(len(bucket_rows) + 1, dtype=np.int64)
  np.cumsum(np.asarray(bucket_rows, dtype=np.int64), out=points[1:])
  total = int(points[-1])
  held = total if cap_rows is None else min(total, int(cap_rows))
  if held * row_b <= budget_b:
    return [len(bucket_rows)]
  return tick_cutter.pack(points * row_b, budget_b).tolist()


def _round_steps(lo: int, rows: int, ends, k: int):
  """How pass 2 consumes one feed: ``rows`` rows from row ``lo`` of its
  row space, against the rounds ending at ``ends`` (cumulative rows), the
  first one not yet closed ``k``.  Returns the steps in order,
  ``('take', a, b)`` the feed's rows [a, b) into the open round and
  ``('close', j)`` round j ending there (every round closes once, as soon
  as its rows are in; an empty one where it stands), and the first round
  not yet closed after the feed."""
  steps = []
  offset = 0
  while True:
    cur = lo + offset
    while k < len(ends) and ends[k] == cur:
      steps.append(("close", k))
      k += 1
    if offset >= rows:
      return steps, k
    assert k < len(ends), (
        f"pass 2 feed rows {cur}..{lo + rows} past the last round's end "
        f"{ends[-1] if ends else 0}")
    take = min(rows - offset, ends[k] - cur)
    steps.append(("take", offset, offset + take))
    offset += take


def _find_agg_call(tree):
  """Walk an aggregation expression tree and return the ``[fn, arg]``
  subtree where ``fn`` ∈ {sum, count, min, max, avg}, or ``None`` if no
  such node is present."""
  tree = normalize_expr_tree(tree)
  if isinstance(tree, list) and len(tree) == 2:
    fn = tree[0]
    if isinstance(fn, str) and fn in ("sum", "avg", "count", "min", "max"):
      return tree
  if isinstance(tree, list):
    for e in tree:
      r = _find_agg_call(e)
      if r is not None:
        return r
  return None


# A result whose per-chunk values cannot be merged by one op is carried as a
# partial state.  AVG cannot be merged from per-chunk means: groups
# contribute different row counts to different chunks.  Nor can an
# expression over two or more aggregate calls: a ratio of two sums (Q14's
# promo_revenue, Q8's mkt_share) merged by SUM is the sum of the per-chunk
# ratios.  Such a result keeps its calls' sufficient statistics in one
# existing logical intermediate slot as a 2-D tensor, one field per SUM or
# COUNT call and two per AVG call (its sum and count), in the expression's
# evaluation order (``[sum_0, count_0, sum_1, ...]``).  Every field merges by
# SUM; the expression is evaluated once, over the merged fields
# (``_finalize_state``).  Packing the fields into the value slot preserves
# the dataflow/applier topology (one inter_val tag per SQL result) while
# carrying enough information for the exact final value.  A result with one
# aggregate call and no AVG still merges by its call's op, which is exact for
# the call alone and for a constant multiple of it (Q11's ``sum * 0.0001``,
# Q17's ``sum / 7.0``).
_STATE_ID_BASE = 900_000_000

_AGG_FNS = ("sum", "count", "min", "max", "avg")
# The fields a partial state holds for each of its aggregate calls.
_STATE_FIELDS = {"sum": ("sum",), "count": ("count",),
                 "avg": ("sum", "count")}


def _collect_agg_calls(tree):
  """Every aggregate call of an expression, ``(fn, argument)``, in
  expression-evaluation order."""
  tree = normalize_expr_tree(tree)
  if isinstance(tree, list):
    if (len(tree) == 2 and isinstance(tree[0], str)
        and tree[0] in _AGG_FNS):
      return [(tree[0], normalize_expr_tree(tree[1]))]
    out = []
    for child in tree:
      out.extend(_collect_agg_calls(child))
    return out
  return []


def _state_calls(expr):
  """The aggregate calls a result's partial state holds, ``[]`` when the
  result merges by its op: a state for an expression with an AVG call or
  with two or more aggregate calls."""
  calls = _collect_agg_calls(expr)
  if not (any(fn == "avg" for fn, _ in calls) or len(calls) >= 2):
    return []
  unsupported = sorted({fn for fn, _ in calls if fn not in _STATE_FIELDS})
  if unsupported:
    raise NotImplementedError(
        f"an aggregate result over {', '.join(unsupported)} and other "
        f"aggregate calls has no partial state (its fields merge by SUM): "
        f"{expr}")
  return calls


def _state_width(calls) -> int:
  """The fields of a partial state over ``calls`` (0 for none)."""
  return sum(len(_STATE_FIELDS[fn]) for fn, _ in calls)


def _replace_agg_calls(tree, replacement_ids):
  """Replace aggregate calls with synthetic column references.

  The rewritten expression is evaluated only after pass 2 has merged every
  state field. Existing ``#`` wrappers and outer arithmetic remain intact,
  so Q17's ``0.2 * AVG(l_quantity)`` applies the scale exactly once to the
  global average rather than once per partial mean, and Q14's ratio divides
  the global sums.
  """
  cursor = 0

  def _walk(node):
    nonlocal cursor
    node = normalize_expr_tree(node)
    if isinstance(node, list):
      if (len(node) == 2 and isinstance(node[0], str)
          and node[0] in _AGG_FNS):
        if cursor >= len(replacement_ids):
          raise RuntimeError(
              "aggregate call replacement list is shorter than expression")
        rid = replacement_ids[cursor]
        cursor += 1
        return f"_agg_state_{cursor - 1}#{rid}"
      return [_walk(child) for child in node]
    return node

  rewritten = _walk(tree)
  if cursor != len(replacement_ids):
    raise RuntimeError(
        f"aggregate call replacement list has {len(replacement_ids)} entries "
        f"but expression contains {cursor} aggregate calls")
  return rewritten


def _make_partial_state(device_name, columns, calls, cnt_rows,
                        cnt_groups, inverted_index, target_dtype):
  """Build a partial state's fields (``_STATE_FIELDS`` per call) for one
  input chunk."""
  fields = []
  for fn, arg in calls:
    for field in _STATE_FIELDS[fn]:
      fields.append(evaluate_aggregate(
          device_name, columns, [field, arg],
          cnt_rows, cnt_groups, inverted_index).to(target_dtype))
  if not fields:
    raise RuntimeError("partial state requested for an expression without "
                       "aggregate calls")
  return torch.stack(fields, dim=1)


def _reduce_partial_values(combined_vals, agg_ops, state_flags,
                           inverse_index, cnt_merged):
  """Reduce pass-2 value columns into the newly merged gid space.

  Ordinary aggregate values retain their historical reduction. Partial
  states always reduce every field by SUM; the expression is evaluated only
  after the final tick, over the global fields (``_finalize_state``).
  """
  out = []
  for value, op, is_state in zip(
      combined_vals, agg_ops, state_flags):
    if is_state:
      if value.dim() != 2:
        raise RuntimeError(
            f"a partial state must be [rows, fields], got {tuple(value.shape)}")
      result = torch.zeros(
          (cnt_merged, value.shape[1]),
          dtype=value.dtype, device=value.device)
      expanded_index = inverse_index.unsqueeze(1).expand(
          -1, value.shape[1])
      result.scatter_reduce_(
          0, expanded_index, value, reduce="sum", include_self=False)
    else:
      result = torch.zeros(
          cnt_merged, dtype=value.dtype, device=value.device)
      # include_self=False so the destination's zero sentinel doesn't poison
      # amax/amin/mean — the first scattered value overwrites, then later
      # values combine via the aggregate operation.
      result.scatter_reduce_(
          0, inverse_index, value, reduce=op, include_self=False)
    out.append(result)
  return out


def _merge_dynamic_batch(running_state, tick_hash, tick_keys, tick_vals,
                         have_running, agg_ops, state_flags):
  """The dynamic accumulator's merge of one pass-2 batch into
  ``running_state`` (``keys``, ``vals``, ``hash``).  Each column's
  concatenation drops the running state behind it as soon as it exists.
  Its allocations are plain arena ones: with the granule cache attached, one
  that misses borrows the cache region's lowest blocks inside the allocator
  (``memory/README.md``, "Lending")."""
  K, N = len(tick_keys), len(tick_vals)
  use_direct_unique = K == 1 and tick_keys[0].dim() == 1
  # Concatenate each column once and drop the running state behind it.
  comb_keys = []
  for i in range(K):
    comb_keys.append(
        torch.cat([running_state["keys"][i], tick_keys[i]])
        if have_running else tick_keys[i])
    running_state["keys"][i] = None
  comb_vals = []
  for i in range(N):
    comb_vals.append(
        torch.cat([running_state["vals"][i], tick_vals[i]])
        if have_running else tick_vals[i])
    running_state["vals"][i] = None
  comb_hash = None
  if not use_direct_unique:
    comb_hash = (torch.cat([running_state["hash"], tick_hash])
                 if have_running else tick_hash)
    running_state["hash"] = None

  if use_direct_unique:
    uniq, new_inv = torch.unique(comb_keys[0], return_inverse=True)
    comb_keys[0] = None
  else:
    uniq, new_inv = torch.unique(comb_hash, return_inverse=True)
    comb_hash = None
  cnt_merged = uniq.numel()

  if use_direct_unique:
    new_keys = [uniq]
  else:
    new_keys = []
    for key in comb_keys:
      if key.dim() == 1:
        merged = torch.zeros(
            cnt_merged, dtype=key.dtype, device=key.device)
        merged.scatter_(0, new_inv, key)
      else:
        width = key.shape[1]
        merged = torch.zeros(
            (cnt_merged, width),
            dtype=key.dtype, device=key.device)
        merged.scatter_(
            0, new_inv.unsqueeze(1).expand(-1, width), key)
      new_keys.append(merged)

  new_vals = _reduce_partial_values(
      comb_vals, agg_ops, state_flags,
      new_inv, cnt_merged)
  running_state["keys"] = new_keys
  running_state["vals"] = new_vals
  running_state["hash"] = None if use_direct_unique else uniq


def _finalize_state(expr, state, calls, target_dtype):
  """Turn a merged partial state into the SQL expression's final value:
  each aggregate call replaced by its merged fields (an AVG's sum over its
  count), the expression evaluated once."""
  if state.numel() == 0:
    return torch.empty(0, dtype=target_dtype, device=state.device)
  expected_width = _state_width(calls)
  if state.dim() != 2 or state.shape[1] != expected_width:
    raise RuntimeError(
        f"a partial state over {len(calls)} aggregate calls must have width "
        f"{expected_width}, got {tuple(state.shape)}")

  replacement_ids = [_STATE_ID_BASE + i for i in range(len(calls))]
  replacements = {}
  field = 0
  for rid, (fn, _) in zip(replacement_ids, calls):
    if fn == "avg":
      replacements[rid] = state[:, field] / state[:, field + 1]
    else:
      replacements[rid] = state[:, field]
    field += len(_STATE_FIELDS[fn])
  rewritten = _replace_agg_calls(expr, replacement_ids)
  result = evaluate_aggregate(
      state.device.type, replacements, rewritten,
      state.shape[0], state.shape[0], None)
  return result.to(target_dtype)


# ----------------------------------------------------------------------------
# Streaming group-by helpers
#
# The streaming aggregator partitions rows by a hash over the concatenation of
# its key columns.  Each key is hashed independently in its native dtype so
# heterogeneous schemas (int8 strings, int64 keys, float64 decimals) never get
# promoted to a common wide type.  To avoid the int8→int64 widening copy that
# would blow up wide string columns by 8×, byte-level view-packing reinterprets
# each row's raw bytes as int64 words: zero-copy when the row width is already
# 8-byte-aligned, a small zero-padded copy (≤7 bytes/row) otherwise.
# ----------------------------------------------------------------------------


def _packed_width(col: torch.Tensor) -> int:
  """Number of int64 columns after view-packing a (N, W) tensor's bytes."""
  return (col.shape[1] * col.element_size() + 7) // 8


def _pack_as_int64(t: torch.Tensor) -> torch.Tensor:
  """View-pack a 2D tensor's row bytes into int64 columns for hashing.

  Handles any element type uniformly:
    - 8-byte types (int64, float64): zero-copy ``view(int64)``.
    - Sub-8-byte types (int8, int16, int32, float32, …): reinterpret each
      row's contiguous bytes as int64 words.  When the row byte-width is a
      multiple of 8, this is still zero-copy; otherwise a small zero-padded
      buffer is allocated (cost = N × ⌈row_bytes/8⌉ × 8, which is ≤ the
      original tensor size and typically much smaller than an element-wise
      ``.to(int64)`` widening).

  Returns
  -------
  torch.Tensor – shape (N, ⌈W·elem_size / 8⌉), dtype int64.
  """
  N, W = t.shape
  elem_size = t.element_size()
  row_bytes = W * elem_size
  target_bytes = ((row_bytes + 7) // 8) * 8

  raw = t.contiguous().view(torch.uint8).reshape(N, row_bytes)

  if row_bytes == target_bytes:
    return raw.view(torch.int64)

  buf = torch.zeros(N, target_bytes, dtype=torch.uint8, device=t.device)
  buf[:, :row_bytes] = raw
  return buf.view(torch.int64)


# Fixed, VALIDATED base seed for the grouping-hash coefficients.  The linear
# hash sum_word(packed_word * coeff) collides distinct STRUCTURED keys (TPC-H
# s_name, the Q16 (p_brand,p_type,p_size) composite, …) for a minority (~1/40)
# of *random* coeff draws — the Q16/Q21 wrong-rows flake.  A seed validated
# collision-free over every grouping-key domain the suite reaches
# (tests/operators/test_hash_seed_injective.py) removes the flake WITHOUT the
# torch.unique(dim=0) lexicographic-sort cost of lossless grouping.
#
# Injectivity over a key's full domain is a global property of the coeff vs the
# domain's byte-difference lattice, so it holds for every filtered subset, SF,
# and plan (a subset ⊆ the domain).  The coeffs MUST be a pure function of this
# constant + key shape — never the global RNG stream — so the coeff that runs
# is byte-identical to the coeff that was validated; see _make_key_coeffs.
_COEFF_BASE_SEED = int(os.environ.get('TQP_HASH_SEED', '2029')) & 0x7FFF_FFFF_FFFF_FFFF


def _make_key_coeffs(key_tensors: List[torch.Tensor], device: str) -> List[torch.Tensor]:
  """Deterministic int64 hash coefficients, one vector per packed key column.

  A LOCAL generator reseeded to the fixed base each call (rather than the
  process-global ``torch.randint`` stream) makes the coefficients a pure
  function of ``_COEFF_BASE_SEED`` and the key shapes — independent of how much
  randomness other operators / earlier queries consumed.  That is what makes a
  "validated seed" meaningful: the coeff produced here is identical on every
  run and every query order, so it matches exactly what the injectivity test
  checked.  Coefficient length matches the *packed* int64 width (not the
  original element width) so the same coefficients apply to both the running
  global state and a freshly arrived chunk for compatible hashing.
  """
  g = torch.Generator(device=device)
  g.manual_seed(_COEFF_BASE_SEED)
  return [
    torch.randint(2, 10_000_000, (_packed_width(col),), dtype=torch.int64,
                  device=device, generator=g)
    for col in key_tensors
  ]


def _hash_key_columns(key_tensors: List[torch.Tensor],
                      coeffs: List[torch.Tensor]) -> torch.Tensor:
  """Combine per-row hashes of each key column into a single (N,) int64 vector.

  Each column is byte-packed into int64 words via ``_pack_as_int64`` before
  multiplying by its coefficient vector; no element-wise widening or cross-key
  ``torch.cat`` ever occurs.
  """
  hash_tensor = None
  for col, coeff in zip(key_tensors, coeffs):
    packed = _pack_as_int64(col)
    contribution = (packed * coeff).sum(dim=1)
    hash_tensor = contribution if hash_tensor is None else hash_tensor + contribution
  return hash_tensor


def _scatter_unique_per_key(key_tensors: List[torch.Tensor],
                            inverse_index: torch.Tensor,
                            n_unique: int) -> List[torch.Tensor]:
  """Build one representative tensor per key in its native dtype.

  Each output has shape (n_unique, width_k); the scatter picks an arbitrary
  surviving row for each unique group (same semantics as the prior monolithic
  scatter, but per-column so nothing gets promoted to float64).
  """
  out: List[torch.Tensor] = []
  for col in key_tensors:
    width = col.shape[1]
    rep = torch.zeros((n_unique, width), dtype=col.dtype, device=col.device)
    rep.scatter_(0, inverse_index.unsqueeze(1).expand(-1, width), col)
    out.append(rep)
  return out


def _materialize_chunk_keys(columns: Dict[int, torch.Tensor],
                            key_ids_d: List[int]) -> List[torch.Tensor]:
  """Return the chunk's key columns as a list of 2D views (no concat)."""
  out: List[torch.Tensor] = []
  for key_id in key_ids_d:
    col = columns[key_id]
    if col.dim() < 2:
      col = col.unsqueeze(1)
    out.append(col)
  return out


def _pack_keys_composite(keys: List[torch.Tensor],
                         offsets: List[int]) -> torch.Tensor:
  """Pack K non-negative 1-D integer key columns into ONE int64 composite:
  ``composite = sum_i keys[i].to(int64) << offsets[i]``.  Lossless when each
  key < 2**bits[i] and the total bit-width <= 62 (the caller guarantees this
  via the TIER3_PACK gate: per-key global max sets the widths, min>=0 sets
  non-negativity, total<=62 keeps the value positive in a signed int64).
  Collapses a K>=2 grouped aggregate into the K==1 1-D shape the bounded
  Tier-3 sorted-run merge serves, so the cumulative torch.unique is avoided."""
  comp = torch.zeros(keys[0].shape[0], dtype=torch.int64, device=keys[0].device)
  for k, off in zip(keys, offsets):
    comp = comp | (k.to(torch.int64) << off)
  return comp


def _unpack_keys_composite(comp: torch.Tensor, offsets: List[int],
                           bits: List[int], dtypes) -> List[torch.Tensor]:
  """Inverse of :func:`_pack_keys_composite`: recover the K original key
  columns (each cast back to its declared dtype)."""
  out: List[torch.Tensor] = []
  for off, b, dt in zip(offsets, bits, dtypes):
    out.append(((comp >> off) & ((1 << b) - 1)).to(dt))
  return out


def tqp_hash_aggregate_two_pass(tensor_group, args, cache_region, cpu_mem_pool,
                                 name, applier, page_cache):
  """Two-pipeline grouped aggregation with fixed pass-2 state.

  Pass 1 (VortexPipeline) reads input columns and, per pipeline tick,
  emits a tuple
      [chunk_unique_hash, chunk_key_rep_0, ..., chunk_agg_val_0, ...]
  where every output is at chunk-unique cardinality. The pipeline
  writes these as one record table per output column across ticks.

  Pass 2 wraps those record tables as CCRs. Its pipeline chunk remains only the
  transfer/decompression feed quantum. Mergeable shapes accumulate in one
  fixed-capacity representation selected from:

    * direct-addressed dense state for compact partial-state domains;
    * an in-place scalar sorted run;
    * an in-place lossless packed-composite sorted run; or
    * a sorted hash run carrying the full emitted-key payload.

  Pass 2 merges one round (adjacent whole buckets of pass 1's partials, every
  row at B = 1) at a time and emits and resets the state at each round's end.
  A fixed stage slab of at most one compute view combines feeds before
  compute, and tick-local unique/search work is consumed in bounded views, so
  neither the cumulative group count nor the staging creates cumulative radix
  scratch. Non-mergeable or unadmitted shapes retain the historical dynamic
  accumulator.

  Restricted to ``args['keys']`` is non-empty. No-key (global) aggs
  go through the legacy ``tqp_hash_aggregate`` path.
  """
  assert args['keys'], "two-pass agg requires non-empty 'keys'"
  perf_logger().start(f"{name}")
  device_name = 'cuda'
  transfer_set, transfer_types, agg_mask_map = agg_transfer_set(
      args, tensor_group)

  # The keys and result expressions (``KeyedAggregate``): ``key_ids`` is
  # the subset of GROUP BY keys that ALSO appear in the SELECT (the operator
  # emits only these), ``key_ids_d`` the FULL GROUP BY (what the per-row
  # group hash is over).  These differ in queries like Q13.agg10 where the
  # GROUP BY key (c_custkey) is not emitted.
  spec = KeyedAggregate(args, tensor_group, transfer_set, transfer_types)
  key_ids, key_ids_new = spec.key_ids, spec.key_ids_new
  key_types, key_types_name = spec.key_types, spec.key_types_name
  agg_ids, agg_types = spec.agg_ids, spec.agg_types
  state_widths = spec.state_width
  key_ids_d = spec.key_ids_d
  agg_ops_norm = spec.ops
  state_flags = spec.state
  merge_ops = spec.merge_ops
  merge_ops_ok = spec.merge_ops_ok

  # ── Pass 1: per-chunk reduction (``KeyedAggregate.pass1``) ──────────
  # Returns:
  #   [chunk_unique_hash (1-D, int64)]
  # + [chunk_key_rep_i for each kid in key_ids] (1-D if width==1 else 2-D)
  # + [chunk_agg_val_i for each agg_id in agg_ids]
  #     (1-D normally; [groups, fields] for a partial state)

  # Per-emitted-key global max/min (0-dim GPU tensors), accumulated across
  # pass1 chunks and consumed after pass1 to size the Tier-3 composite-key
  # pack.  Sized by the emitted-key count (key_ids).
  key_max_ref = [None] * len(key_ids)
  key_min_ref = [None] * len(key_ids)

  # The two graph nodes of a keyed aggregate: pass 1 reads the input
  # columns' current instances (plus the mask) and produces the per-chunk
  # intermediates; pass 2's inputs ARE those intermediates.
  pass1_node, pass2_node = args['nodes']['pass1'], args['nodes']['pass2']
  pass1_pipe_name = pass1_node.name
  # Pass 1's chunk, resolved once: B's tick prediction and the pipe read it.
  pass1_chunk_b = sizing_resolver.resolve("agg", pass1_pipe_name).chunk_b

  # ── Partition count B, fixed before pass 1 ──────────────────────────
  # From the partition budget (``operators/partitions.agg_partitions``):
  #   P  the input rows, times the selectivity the filter recorded on the
  #      mask pass 1 reads (as the join reads its sides');
  #   T  pass 1's ticks at its chunk, over every column it reads;
  #   G  a static bound on the groups: the product of each GROUP BY key's
  #      dtype's values (int8 and int16, per element of its width: a
  #      string's bytes); none if a key is wider;
  #   w  pass 2's run bytes per row on the route the shape allows;
  # and, on the key route, a bound on the key's range (its largest minus its
  # smallest value plus one), at which pass 2 caps the run: the key's
  # dtype's values (int8, int16); none for a wider key.
  # G and the range only choose B: pass 2 sizes every buffer from the rows
  # pass 1 wrote and the key range it observed.
  # A dense shape takes 1 by choice: an admitted dense state spans the key
  # domain whatever the buckets, so bucketing pass 1 would only cost.  A
  # declined one falls back to the key route unpartitioned, a run of the
  # whole observed key range, and a key whose range is over its partial
  # rows never gets the dense state and runs the key route on all of them;
  # either run can be over the budget.
  p_pre = tensor_group[transfer_set[0]].col_length()
  p_rows = p_pre
  agg_masks = set(agg_mask_map.values())
  if len(agg_masks) == 1:
    sel = tensor_group[next(iter(agg_masks))].selectivity
    if sel is not None:
      p_rows = int(round(p_pre * sel))
  in_row_b = sum(
      dtype_size(tensor_group[c].get_type_torch())
      * max(1, tensor_group[c].col_width()) for c in transfer_set)
  p_ticks = max(1, -(-p_pre * in_row_b // max(1, pass1_chunk_b)))
  one_scalar_key = bool(
      len(key_ids) == 1 and key_ids == key_ids_d
      and tensor_group[key_ids[0]].col_width() == 1)
  key_route = bool(SORTED_RUN_MERGE and merge_ops_ok and one_scalar_key)
  dense_shape = bool(
      DENSE_AGGREGATION_SLAB and merge_ops_ok and all(state_flags)
      and len(agg_ids) > 0 and one_scalar_key
      and key_types[0] in _INT_DTYPES
      and key_types_name[0] in ("int", "date", "uint32"))
  key_range = (None, "?")
  if key_route and key_types[0] in _INT_DTYPES:
    key_range = _narrow_key_values(key_types[0], 1) or key_range
  partition_budget_b = constants.get_partition_budget_b()
  p_route = "dense" if dense_shape else "key" if key_route else "hash"
  agg_part = agg_partition_count(spec, p_rows, p_ticks, p_route, key_range)
  log_agg_partition_count(name, agg_part)
  agg_B, pred_b = agg_part.B, agg_part.pred_b

  def _reduce_pass1_view(columns):
    outs, buckets = spec.pass1(device_name, columns, agg_B)
    for col in list(columns.keys()):
      del columns[col]

    # Track per-emitted-key global max/min for the Tier-3 composite-key pack
    # layout.  Cheap: K small reductions per chunk; the .item()
    # host sync is deferred until after pass1 completes.
    for _i, _kr in enumerate(outs[1:1 + len(key_ids)]):
      if _kr.numel() == 0:
        continue
      _cmx = _kr.max()
      _cmn = _kr.min()
      key_max_ref[_i] = _cmx if key_max_ref[_i] is None else torch.maximum(key_max_ref[_i], _cmx)
      key_min_ref[_i] = _cmn if key_min_ref[_i] is None else torch.minimum(key_min_ref[_i], _cmn)
    return outs, buckets

  def pass1_op(columns):
    # Unpartitioned (B = 1) or an empty tick: the outputs as they are (a
    # 0-row output is never written).
    outs, buckets = _reduce_pass1_view(columns)
    if buckets is None:
      return outs
    bucket, counts, counts_t = buckets
    return _partition_outputs(outs, bucket, counts, counts_t)

  pass1_types = [torch.int64] + key_types + agg_types

  pass1_layout = (OutputLayout.partitioned(len(pass1_types), agg_B)
                  if agg_B > 1 else None)
  pass1_pipe = VortexPipeline(
      node=pass1_node,
      inputs=pass1_node.input_pairs(),
      layout=pass1_layout,
      output_columns_type=(
          [t for t in pass1_types for _ in range(agg_B)]
          if agg_B > 1 else pass1_types),
      operator=pass1_op,
      mask_map=agg_mask_map,
      cpu_mem_pool=cpu_mem_pool,
      chunk_size=pass1_chunk_b,
      sizing_key=("agg", pass1_pipe_name),
      # Pass-1's chunk budget has NO out-of-pipeline consumer, so the
      # sectioned in-pipe walk may re-latch it mid-run (per-pipe override of
      # the agg class default): the feed reserves and the ticks_remaining
      # countdown all key off PASS-2's chunk; pass-1's op is
      # chunk-geometry-pure (hash coefficients built once in the closure,
      # partial states additive, per-emitted-key min/max
      # accumulative, B and each group's bucket fixed before the pipe, and
      # pass-2's merge accepts arbitrary per-chunk partials).  B's tick
      # prediction read the chunk once, above; it only predicts.
      sectionable=True,
      applier=applier,
      page_cache=page_cache,
  )

  pass1_pipe.do_exchange()
  pass1_pipe.get_result()
  # Each pass-1 output as a column (a record table of regions), typed
  # below.
  bucket_rows = None
  if agg_B == 1:
    pass1_vars = [pass1_pipe.output_variable(i)
                  for i in range(len(pass1_types))]
  else:
    # A partitioned table's row space is its B streams in order, so pass 2
    # reads the partials bucket by bucket.
    pass1_tables = pass1_pipe.result_tables()
    pass1_vars = []
    for i, dtype in enumerate(pass1_types):
      table = pass1_tables.get(i)
      pass1_vars.append(
          TableVariable(table, torch_to_type(table.stored_dtype))
          if table is not None else Variable(tensor_type=torch_to_type(dtype)))
      rows_i = ([0] * agg_B if table is None
                else [int(r) for r in table.stream_rows()])
      if bucket_rows is None:
        bucket_rows = rows_i
      assert rows_i == bucket_rows, (
          f"{name}: pass-1 bucket rows diverged across outputs: "
          f"{rows_i} vs {bucket_rows}")
  # Every pass-1 output is host-backed after get_result()'s sync. Drop
  # the pipeline object before sizing/allocating the resident pass-2 run;
  # otherwise its completed tick stages retain GPU source views and can make a
  # 16.8 GB Q18 run look inadmissible despite those bytes being dead.
  del pass1_pipe
  torch.cuda.synchronize()
  # pass1_vars layout:
  #   pass1_vars[0]                    — chunk_unique_hash
  #   pass1_vars[1 .. K]               — key_rep, one per key
  #   pass1_vars[K+1 .. K+1+N_aggs-1]  — agg_val, one per agg

  # Build CCRs that pass 2 will consume. The "id" space is synthetic
  # (just positional indices). K is the number of EMITTED keys
  # (key_ids), not GROUPING keys (key_ids_d) — pass 1 only carried
  # forward the keys that appear in the SELECT.
  K = len(key_ids)
  N = len(agg_ids)

  # ── Tier-3 composite-key pack layout ─────────────────────────────────
  # Decide whether the K>=2 grouped aggregate can be losslessly packed into
  # ONE int64 (composite = sum_i key_i << offset_i) so it flows through the
  # bounded sorted-run merge instead of the cumulative torch.unique(comb_hash)
  # that OOMs Q20.  Bit widths come from the per-key global max tracked in
  # pass1; the pack is admitted ONLY when every emitted key is a non-negative
  # integer column, emit==group (key_ids==key_ids_d), the ops are mergeable,
  # and the total width fits 62 bits (sign-safe).  Otherwise TIER3_PACK stays
  # False and the query keeps its existing (K==1 Tier-3 or cumulative) path.
  _key_max_h = [int(t.item()) if t is not None else 0 for t in key_max_ref]
  _key_min_h = [int(t.item()) if t is not None else 0 for t in key_min_ref]
  key_pack_bits = [max(1, m.bit_length()) for m in _key_max_h]
  key_pack_offsets, _off = [], 0
  for _b in key_pack_bits:
    key_pack_offsets.append(_off)
    _off += _b
  TIER3_PACK = bool(
      SORTED_RUN_MERGE and merge_ops_ok and K >= 2
      and not any(state_flags)
      and key_ids == key_ids_d
      and all(t in _INT_DTYPES for t in key_types)
      # Integer-backed strings are 2-D byte matrices (Q7/Q9/Q16).  Their
      # element dtype is packable but each ROW is not one scalar key; treating
      # widths 4/10/25 as scalar operands broadcasts incompatible matrices
      # and, even at equal widths, would not be a lossless row encoding.
      and all(tensor_group[key_id].col_width() == 1
              for key_id in key_ids)
      and all(n >= 0 for n in _key_min_h)
      and _off <= 62)
  if TIER3_PACK:
    message_logger().critical(
        "[%s] Tier-3 composite pack ON: K=%d key_bits=%s total=%d/62",
        name, K, key_pack_bits, _off)

  # tensor_type is a TQP type name, not a torch dtype.
  pass1_vars[0].tensor_type = 'int'
  for i in range(K):
    pass1_vars[1 + i].tensor_type = key_types_name[i]
  for i in range(N):
    pass1_vars[1 + K + i].tensor_type = torch_to_type(agg_types[i])
  hash_ccr = CombinedColumnRep([pass1_vars[0]])
  key_ccrs = [CombinedColumnRep([pass1_vars[1 + i]]) for i in range(K)]
  val_ccrs = [CombinedColumnRep([pass1_vars[1 + K + i]]) for i in range(N)]

  # The pass-1 instances receive their payloads under the type names the
  # kernel needs (a key column's own type name, the agg result's class).
  assert len(pass1_node.outputs) == 1 + K + N, (len(pass1_node.outputs), K, N)
  pass1_node.outputs[0].ccr = hash_ccr
  for i in range(K):
    pass1_node.outputs[1 + i].ccr = key_ccrs[i]
  for i in range(N):
    pass1_node.outputs[1 + K + i].ccr = val_ccrs[i]
  # Pass 2's kernel-private keys 0 .. K+N pair with pass 1's outputs
  # positionally (the builder linked the two nodes' instances).
  pass2_inputs = [(i, inst) for i, inst in enumerate(pass2_node.inputs)]
  assert pass2_node.inputs == pass1_node.outputs, "pass 2 must read pass 1's instances"

  # PASS2 remains independently resolvable by the normal aggregation sizing
  # controller. Its chunk is a feed quantum only; persistent state below is
  # owned by a fixed aggregation slab/buffer and never grows per tick.
  pass2_name = pass2_node.name
  pass2_chunk_size_b = sizing_resolver.resolve(
      "agg", pass2_name).chunk_b

  # For a compact scalar integer domain carrying only partial states,
  # direct-address every partial row into one preallocated byte slab. Q17's
  # l_partkey domain is dense (1..140M at SF700), so this removes cumulative
  # torch.unique and repeated sorted-run shifts entirely. Sparse/non-state
  # domains retain the existing fixed sorted-run or generic route.
  def _physical_spec(ccr, fallback_dtype):
    rep = ccr.reps[0]
    # A pass-1 output that wrote no row is an empty Variable.
    dtype = (rep.record_table().stored_dtype if rep.table_backed
             else fallback_dtype)
    width = ccr.col_width()
    return dtype, ((width,) if width > 1 else ())

  value_specs = []
  for dtype, state_width, ccr in zip(
      agg_types, state_widths, val_ccrs):
    physical_dtype, trailing = _physical_spec(ccr, dtype)
    if state_width:
      if trailing != (state_width,):
        raise RuntimeError(
            f"a pass-1 partial state must have width {state_width}, "
            f"got {trailing or 1}")
    value_specs.append((physical_dtype, trailing))

  key_specs = [
      _physical_spec(ccr, dtype)
      for dtype, ccr in zip(key_types, key_ccrs)]

  key_domain_capacity = None
  if (K == 1 and key_types[0] in _INT_DTYPES
      and key_max_ref[0] is not None and key_min_ref[0] is not None):
    key_domain_capacity = _key_max_h[0] - _key_min_h[0] + 1

  dense_slab = None
  dense_slab_shape_ok = bool(
      DENSE_AGGREGATION_SLAB and merge_ops_ok and all(state_flags)
      and N > 0 and K == 1
      and len(key_ids_d) == 1 and key_ids == key_ids_d
      and key_types[0] in _INT_DTYPES
      and key_types_name[0] in ("int", "date", "uint32")
      and key_ccrs[0].col_width() == 1
      and key_domain_capacity is not None
      and key_domain_capacity <= hash_ccr.col_length())
  if dense_slab_shape_ok:
    slab_b = DenseAggregationSlab.required_bytes(
        key_domain_capacity, value_specs)
    # A slab the arena cannot serve, lending included, leaves the shape to
    # the sorted-run routes below.
    try:
      dense_slab = DenseAggregationSlab(
          _key_min_h[0], _key_max_h[0], key_types[0],
          value_specs, merge_ops, device_name)
    except RuntimeError:
      dense_slab = None
    if dense_slab is not None:
      message_logger().critical(
          "[%s] fixed aggregation slab ON: keys=[%d,%d] capacity=%d "
          "slab=%.3f GiB pass2_feed=%.3f GiB",
          name, _key_min_h[0], _key_max_h[0], key_domain_capacity,
          slab_b / 1024 ** 3, pass2_chunk_size_b / 1024 ** 3)
    else:
      message_logger().critical(
          "[%s] fixed aggregation slab declined: slab=%.3f GiB",
          name, slab_b / 1024 ** 3)

  # ── Pass 2: fixed resident run + stage slab ─────────────────────────
  # Every mergeable shape gets a fixed-capacity resident representation:
  #   dense  — direct-addressed key domain (Q17);
  #   key    — one scalar key sorted run (Q18);
  #   packed — lossless composite int64 sorted run (Q20);
  #   hash   — sorted hash plus the full emitted-key payload (Q10/wide keys).
  # The hash route makes the same collision-free hash assumption as the
  # historical cumulative-unique path; it changes storage, not equality.
  # Non-mergeable shapes, and a run the arena cannot serve, retain the
  # historical dynamic path.
  scalar_key_route_ok = bool(
      SORTED_RUN_MERGE and merge_ops_ok and K == 1
      and len(key_ids_d) == 1 and key_ids == key_ids_d
      and key_ccrs[0].col_width() == 1)

  fixed_route = None
  run_specs = []
  run_merge_ops = []
  if dense_slab is not None:
    fixed_route = "dense"
  elif scalar_key_route_ok:
    fixed_route = "key"
    run_specs = [key_specs[0]] + value_specs
    run_merge_ops = list(merge_ops)
  elif TIER3_PACK:
    fixed_route = "packed"
    run_specs = [(torch.int64, ())] + value_specs
    run_merge_ops = list(merge_ops)
  elif SORTED_RUN_MERGE and merge_ops_ok and (K + N) > 0:
    fixed_route = "hash"
    run_specs = [(torch.int64, ())] + key_specs + value_specs
    # Equal hashes carry equal emitted keys under the existing hash
    # invariant. amax selects that identical payload while aggregate states
    # use their physical associative merge operation.
    run_merge_ops = ["amax"] * K + list(merge_ops)

  # Pass 2 packs pass 1's whole measured buckets into rounds of adjacent
  # buckets within the partition budget, at the run's bytes per row (the
  # hash route's for the dynamic accumulator), the key route's run capped
  # by the observed key range, and holds one round at a time
  # (``pass2_op``).  At B = 1, or
  # when the whole run fits, the bucket-major row space is one round.  Only
  # measured rows size the run and the rounds.  The dense state is the
  # whole key domain whatever the round, so it is always one round.
  total_rows = hash_ccr.col_length()
  key_cap_rows = (
      key_domain_capacity
      if (key_domain_capacity is not None and K == 1
          and key_ids == key_ids_d
          and key_ccrs[0].col_width() == 1)
      else None)
  round_row_b = FixedColumnSlab.row_bytes(
      run_specs or [(torch.int64, ())] + key_specs + value_specs)
  round_ends = [total_rows]
  if agg_B > 1 and dense_slab is None:
    # Each round's end row in the bucket-major row space.
    bucket_points = np.concatenate(([0], np.cumsum(bucket_rows)))
    round_ends = [int(bucket_points[j]) for j in pass2_rounds(
        bucket_rows, round_row_b, partition_budget_b, key_cap_rows)]
  round_rows = np.diff(np.concatenate(([0], round_ends)))
  merge_capacity = int(round_rows.max())
  if key_cap_rows is not None:
    merge_capacity = min(merge_capacity, key_cap_rows)
  fixed_run_slab = None
  fixed_run_b = 0
  if fixed_route not in (None, "dense") and merge_capacity > 0:
    fixed_run_b = FixedColumnSlab.required_bytes(
        merge_capacity, run_specs)
    try:
      fixed_run_slab = FixedColumnSlab(
          merge_capacity, run_specs, device_name)
    except RuntimeError:
      fixed_run_slab = None
    if fixed_run_slab is None:
      message_logger().critical(
          "[%s] fixed aggregation run declined: route=%s capacity=%d "
          "run=%.3f GiB", name, fixed_route, merge_capacity,
          fixed_run_b / 1024 ** 3)
      fixed_route = None
    else:
      message_logger().critical(
          "[%s] fixed aggregation run ON: route=%s capacity=%d run=%.3f GiB",
          name, fixed_route, merge_capacity, fixed_run_b / 1024 ** 3)
  elif fixed_route not in (None, "dense"):
    fixed_route = None

  running_state = {
      "keys": [None] * K,
      "vals": [None] * N,
      "hash": None,
  }
  running_state_present = [False]
  ticks_remaining = [0]  # populated after pipe construction

  _finalize_agg_values = spec.finalize

  def _consume_pass2_batch(tick_hash, tick_keys, tick_vals):
    """Merge one aggregation-owned batch; never emits pipeline output."""
    if fixed_route == "dense":
      if tick_keys and tick_keys[0].numel() > 0:
        dense_slab.add(tick_keys[0], tick_vals)
      return

    rows = 0 if tick_hash is None else tick_hash.numel()
    if fixed_route in ("key", "packed"):
      rows = tick_keys[0].shape[0] if tick_keys else 0
    if rows == 0:
      return

    have_running = (
        fixed_run_slab.length > 0
        if fixed_run_slab is not None else running_state_present[0])
    if fixed_route == "key":
      batch_key, batch_payload = reduce_tick_slice(
          tick_keys[0], tick_vals, merge_ops)
      # Physical pass-1 records normally match the resident dtype.  Keep this
      # boundary explicit so a future narrower pass-1 representation cannot
      # silently feed mixed key dtypes to searchsorted.
      if batch_key.dtype != fixed_run_slab.columns[0].dtype:
        batch_key = batch_key.to(fixed_run_slab.columns[0].dtype)
    elif fixed_route == "packed":
      composite = _pack_keys_composite(tick_keys, key_pack_offsets)
      batch_key, batch_payload = reduce_tick_slice(
          composite, tick_vals, merge_ops)
      del composite
    elif fixed_route == "hash":
      batch_key, batch_payload = reduce_tick_slice(
          tick_hash, list(tick_keys) + list(tick_vals), run_merge_ops)
    else:
      # Historical dynamic accumulator fallback.
      _merge_dynamic_batch(
          running_state, tick_hash, tick_keys, tick_vals, have_running,
          agg_ops_norm, state_flags)
      running_state_present[0] = True
      return
    fixed_run_slab.length = merge_sorted_unique_runs_inplace(
        fixed_run_slab.columns[0], fixed_run_slab.columns[1:],
        fixed_run_slab.length if have_running else 0,
        batch_key, batch_payload, run_merge_ops)

  def _empty_from_spec(spec):
    dtype, trailing = spec
    return torch.empty(
        (0,) + tuple(trailing), dtype=dtype, device=device_name)

  def _finish_pass2():
    """Materialize/finalize the already-consumed resident state."""
    if fixed_route == "dense":
      out_key, out_vals = dense_slab.finish()
      return [out_key] + _finalize_agg_values(out_vals)

    if fixed_route is not None:
      length = fixed_run_slab.length
      run_key = fixed_run_slab.columns[0][:length]
      payload = [
          column[:length] for column in fixed_run_slab.columns[1:]]
      if fixed_route == "key":
        out_keys = [run_key]
        out_vals = payload
      elif fixed_route == "packed":
        out_keys = _unpack_keys_composite(
            run_key, key_pack_offsets, key_pack_bits, key_types)
        out_vals = payload
      else:
        out_keys = payload[:K]
        out_vals = payload[K:]
      return out_keys + _finalize_agg_values(out_vals)

    if not running_state_present[0]:
      empty_keys = [_empty_from_spec(spec) for spec in key_specs]
      empty_vals = [_empty_from_spec(spec) for spec in value_specs]
      return empty_keys + _finalize_agg_values(empty_vals)
    out_keys = running_state["keys"]
    out_vals = running_state["vals"]
    running_state["keys"] = [None] * K
    running_state["vals"] = [None] * N
    running_state["hash"] = None
    running_state_present[0] = False
    return out_keys + _finalize_agg_values(out_vals)

  # The stage slab stores only columns the selected route reads. Hash is
  # omitted for direct/one-key/packed routes, so the slab never copies a
  # dead intermediate.
  stage_specs = []
  if fixed_route in ("dense", "key"):
    stage_specs = key_specs[:1] + value_specs
  elif fixed_route == "packed":
    stage_specs = key_specs + value_specs
  elif fixed_route == "hash":
    stage_specs = [(torch.int64, ())] + key_specs + value_specs

  def _to_stage_columns(tick_hash, tick_keys, tick_vals):
    if fixed_route in ("dense", "key"):
      return [tick_keys[0]] + list(tick_vals)
    if fixed_route == "packed":
      return list(tick_keys) + list(tick_vals)
    return [tick_hash] + list(tick_keys) + list(tick_vals)

  def _from_stage_columns(columns):
    if fixed_route in ("dense", "key"):
      return None, [columns[0]], columns[1:]
    if fixed_route == "packed":
      return None, columns[:K], columns[K:]
    return columns[0], columns[1:1 + K], columns[1 + K:]

  # ── Pass 2: one round at a time ──────────────────────────────────────
  # The pass-1 tables' row space is bucket-major, so each feed is split at
  # the round ends it crosses (``_round_steps``).  A round's rows are staged
  # into a slab of at most one compute view (or merged directly when it
  # does not fit), and at its end the state is flushed, emitted and reset;
  # the last round frees the slab before it emits.  Output that the next
  # round's merges would overwrite (a view of the fixed run) is cloned
  # first.
  part_state = {"row": 0, "round": 0, "slab": None}
  if fixed_route is not None and stage_specs:
    slab_rows = min(_AGG_MAX_COMPUTE_VIEW_ROWS, int(round_rows.max()))
    try:
      part_state["slab"] = FixedColumnSlab(
          slab_rows, stage_specs, device_name)
    except RuntimeError:
      part_state["slab"] = None
  if agg_B > 1:
    # The largest round's run: its rows, or the key range a key-route run
    # is capped by, if fewer.
    largest_round_b = merge_capacity * round_row_b
    stage_slab = ("%d rows" % part_state["slab"].capacity
                  if part_state["slab"] is not None else "none")
    message_logger().critical(
        "[%s] partitioned pass 2: B=%d rounds=%d route=%s run_row=%d B "
        "largest round=%.1fM rows %.1f MB%s capacity=%d run=%.3f GiB "
        "stage_slab=%s%s", name, agg_B, len(round_ends),
        fixed_route or "dynamic", round_row_b, merge_capacity / 1e6,
        largest_round_b / (1 << 20),
        "" if pred_b is None
        else " (predicted %.1f MB)" % (pred_b / (1 << 20)),
        merge_capacity, fixed_run_b / 1024 ** 3, stage_slab,
        " OVER" if largest_round_b > partition_budget_b else "")

  def _flush_stage_slab():
    slab = part_state["slab"]
    rows = slab.length
    columns = slab.tensors()
    tick_hash, tick_keys, tick_vals = _from_stage_columns(columns)
    lo = 0
    while lo < rows:
      hi = min(lo + _AGG_MAX_COMPUTE_VIEW_ROWS, rows)
      hash_slice = (
          None if tick_hash is None else tick_hash[lo:hi])
      _consume_pass2_batch(
          hash_slice,
          [key[lo:hi] for key in tick_keys],
          [value[lo:hi] for value in tick_vals])
      lo = hi
    del tick_hash, tick_keys, tick_vals, columns
    slab.reset()

  def _stage_round_rows(tick_hash, tick_keys, tick_vals):
    slab = part_state["slab"]
    if slab is None:
      _consume_pass2_batch(tick_hash, tick_keys, tick_vals)
      return
    columns = _to_stage_columns(tick_hash, tick_keys, tick_vals)
    rows = columns[0].shape[0]
    offset = 0
    while offset < rows:
      take = min(slab.remaining, rows - offset)
      slab.append([column[offset:offset + take] for column in columns])
      offset += take
      if slab.remaining == 0:
        _flush_stage_slab()

  def _end_round(final):
    """Flush, emit and reset the round's state.  ``final``: no row is
    merged after it, so its output may stay a view of the fixed run, and
    the stage slab is freed before the output is materialized."""
    slab = part_state["slab"]
    if slab is not None and slab.length > 0:
      _flush_stage_slab()
    if final and slab is not None:
      # The free is stream-ordered in the custom allocator: drain it so the
      # finish (a dense state's spans its whole key domain) can reuse the
      # slab's blocks.
      part_state["slab"] = None
      del slab
      torch.cuda.current_stream().synchronize()
    out = _finish_pass2()
    if fixed_run_slab is not None:
      if not final:
        out = [t.clone() for t in out]
      fixed_run_slab.length = 0
    return out

  def pass2_op(input_dict):
    ticks_remaining[0] -= 1
    is_last_tick = (ticks_remaining[0] == 0)
    tick_hash = input_dict[0]
    tick_keys = [input_dict[1 + i] for i in range(K)]
    tick_vals = [input_dict[1 + K + i] for i in range(N)]
    rows = tick_hash.shape[0]
    steps, part_state["round"] = _round_steps(
        part_state["row"], rows, round_ends, part_state["round"])
    outs = []
    for step in steps:
      if step[0] == "close":
        # An empty round emits nothing.
        out = _end_round(final=(round_ends[step[1]] == total_rows))
        if out and out[0].shape[0] > 0:
          outs.append(out)
        continue
      sl = slice(step[1], step[2])
      _stage_round_rows(
          tick_hash[sl],
          [key[sl] for key in tick_keys],
          [value[sl] for value in tick_vals])
    part_state["row"] += rows
    if is_last_tick:
      assert (part_state["row"] == total_rows
              and part_state["round"] == len(round_ends)), (
          f"{name}: pass 2 read {part_state['row']} of {total_rows} rows "
          f"and closed {part_state['round']} of {len(round_ends)} rounds")
    if not outs:
      return []
    if len(outs) == 1:
      return outs[0]
    return [torch.cat([out[i] for out in outs]) for i in range(K + N)]

  pass2_types = key_types + agg_types

  # Pass 1 output lives as host regions (the region producer's record
  # tables, or the granule cache's slots). Pass 2 re-reads + decompresses
  # them through a VortexPipeline. Its independently
  # resolved ``agg`` chunk budget controls only the pass-2 feed quantum;
  # the accumulated state remains in the fixed workspace above.
  # Output order is key_ids_new + agg_ids, the node's output order; a
  # pass-2 output that feeds a UVA gather is host-resident on its instance.
  assert [inst.local for inst in pass2_node.outputs] == key_ids_new + agg_ids, (
      [inst.local for inst in pass2_node.outputs], key_ids_new, agg_ids)

  pass2_pipe = VortexPipeline(
      node=pass2_node,
      inputs=pass2_inputs,
      output_columns_type=pass2_types,
      operator=pass2_op,
      cpu_mem_pool=cpu_mem_pool,
      chunk_size=pass2_chunk_size_b,
      sizing_key=("agg", pass2_name),
      # NEVER sectionable: pass-2's construction-time chunk is read into
      # the ``ticks_remaining[0] = stages - 2`` final-chunk countdown below,
      # which a mid-run re-split breaks (Q17 SF1 empty result).  It stays on
      # the cross-run axis.
      sectionable=False,
      applier=applier,
      page_cache=page_cache,
  )
  # Count of compute ticks the pipeline will run (stages - prologue
  # - epilogue). Set before do_exchange so the first operator call
  # sees the right value.
  ticks_remaining[0] = pass2_pipe.stages - 2
  pass2_pipe.do_exchange()
  pass2_pipe.get_result()
  # A key keeps its column's own type name (a date key is not the float its
  # dtype suggests).
  pass2_vars = (
      [pass2_pipe.output_variable(i, key_types_name[i]) for i in range(K)]
      + [pass2_pipe.output_variable(K + i, torch_to_type(agg_types[i]))
         for i in range(N)])
  pass2_pipe_output_name = pass2_pipe.name
  # ``get_result`` has synchronized all output D2H and every output is
  # host-backed. Drop the aggregation-owned GPU workspace now, rather than
  # relying on collection of the pipeline/closure graph later in the query.
  # Q18 immediately starts another large aggregation; retaining the first
  # 16.8 GB resident run until cyclic GC leaves enough total free bytes but
  # fragments the largest run below the next pass-1 CUB request.
  running_state["keys"] = [None] * K
  running_state["vals"] = [None] * N
  running_state["hash"] = None
  part_state["slab"] = None
  dense_slab = None
  fixed_run_slab = None
  del pass2_pipe
  torch.cuda.synchronize()

  # Wire results back: the pass-2 instances' payloads under the type names
  # above; the cross-operator map points at the instances.
  for i, kid_new in enumerate(key_ids_new):
    inst = pass2_node.outputs[i]
    inst.ccr = CombinedColumnRep([pass2_vars[i]])
    tensor_group[kid_new] = inst.ccr

  for i, aid in enumerate(agg_ids):
    inst = pass2_node.outputs[K + i]
    inst.ccr = CombinedColumnRep([pass2_vars[K + i]])
    tensor_group[aid] = inst.ccr

  output_ids = set(args.get('output ids', key_ids_new + agg_ids))
  for input_col in args['input ids']:
    if input_col in tensor_group and input_col not in output_ids:
      tensor_group[input_col].free_underlying_mem(
          cpu_pool=cpu_mem_pool, gpu_pool=cache_region)
      del tensor_group[input_col]

  perf_logger().stop(f"{name}")
  return None


class AggregateResults:
  """An aggregation's result expressions, from their normalized plan trees
  (the key expressions left out): per result, its id (``agg_ids``), dtype
  (``agg_types``), the op its partials merge by (``ops``, the
  scatter-reduce flavour of the expression's first op), the expression
  (``exprs``) and the aggregate calls its partial state holds
  (``state_calls``, ``[]`` for none: ``_state_calls``); a state result's
  partial is its fields, ``state_width`` of them (``state``).
  ``evaluate`` gives one chunk's partials and ``finalize`` the results, so
  every engine mode evaluates and finalizes an aggregation the same way."""

  def __init__(self, trees, transfer_set, transfer_types):
    self.agg_ids: List[int] = []
    self.agg_types: List[torch.dtype] = []
    self.plan_ops: List[str] = []
    self.exprs = []
    self.state_calls = []
    for tree in trees:
      expr = tree[0]
      calls = _state_calls(expr)
      self.agg_ids.append(get_id(right_most_expr(tree)))
      # Parser output aliases use ``[expr, "#", output_id]``. The output
      # id is not an input column, so generic lookup historically falls
      # back to int64 even though AVG's SQL result is float64. Keep both
      # packed state fields and the finalized result in the true AVG
      # output dtype.
      self.agg_types.append(
          constants.float_dtype if any(fn == "avg" for fn, _ in calls)
          else get_result_type(tree, transfer_set, transfer_types))
      self.plan_ops.append(get_aggregation_op(tree))
      self.exprs.append(expr)
      self.state_calls.append(calls)
    # The torch scatter_reduce_ flavours, so a merge dispatches directly.
    self.ops = [MERGE_OPS.get(op, op) for op in self.plan_ops]
    self.state_width = [_state_width(calls) for calls in self.state_calls]
    self.state = [width > 0 for width in self.state_width]

  def evaluate(self, device_name, columns, cnt_rows, cnt_groups, inv):
    """One chunk's partial per result, in ``agg_ids`` order: its value per
    group over ``columns`` (the chunk's ``cnt_rows`` rows, masked rows
    already dropped), each row in group ``inv[row]`` of ``cnt_groups``; a
    state result as its partial state."""
    values = []
    for i, expr in enumerate(self.exprs):
      if self.state[i]:
        values.append(_make_partial_state(
            device_name, columns, self.state_calls[i],
            cnt_rows, cnt_groups, inv, self.agg_types[i]))
      else:
        values.append(evaluate_aggregate(
            device_name, columns, expr, cnt_rows, cnt_groups, inv))
    return values

  def finalize(self, values) -> List[torch.Tensor]:
    """Merged values in ``agg_ids`` order, each partial state evaluated."""
    finalized = []
    for i, value in enumerate(values):
      if self.state[i]:
        value = _finalize_state(
            self.exprs[i], value, self.state_calls[i], self.agg_types[i])
      finalized.append(value)
    return finalized


class KeyedAggregate(AggregateResults):
  """A keyed aggregation's keys and result expressions, from its plan args
  and the columns in ``tensor_group`` (their types and widths), and pass
  1's per-chunk work over them, so every engine mode groups, evaluates and
  buckets a chunk the same way.

  ``key_ids`` are the GROUP BY keys the SELECT emits (``key_ids_new`` their
  output ids), ``key_ids_d`` the whole GROUP BY, which the group hash is
  over.  The results are ``AggregateResults``'; a partial state merges by
  SUM (``merge_ops``)."""

  def __init__(self, args, tensor_group, transfer_set, transfer_types):
    self.key_ids: List[int] = []
    self.key_ids_new: List[int] = []
    self.key_types: List[torch.dtype] = []
    self.key_types_name: List[str] = []
    result_trees = []
    for tree in args['results']:
      tree = normalize_expr_tree(tree)
      if is_group_key_expr(tree):
        self.key_ids.append(get_id(tree if isinstance(tree, str) else tree[0]))
        self.key_ids_new.append(
            get_id(tree if isinstance(tree, str) else tree[-1]))
        self.key_types.append(tensor_group[self.key_ids[-1]].get_type_torch())
        self.key_types_name.append(tensor_group[self.key_ids[-1]].tensor_type())
      else:
        result_trees.append(tree)
    super().__init__(result_trees, transfer_set, transfer_types)
    self.key_widths = [tensor_group[k].col_width() for k in self.key_ids]
    self.key_ids_d = [get_id(k) for k in args['keys']]
    self.key_d_specs = [
        (tensor_group[k].get_type_torch(), tensor_group[k].col_width())
        for k in self.key_ids_d]

    # The physical reductions over intermediate state. A partial state is a
    # row-shaped value of SUM and COUNT fields, and every field merges by
    # SUM.
    self.merge_ops = [
        "sum" if is_state else op
        for op, is_state in zip(self.ops, self.state)]
    self.merge_ops_ok = all(op in MERGEABLE_OPS for op in self.merge_ops)
    # The group hash's coefficients, made from the first chunk's key
    # shapes and kept: a key must hash to the same value in every chunk,
    # or pass 2's merge would find no group twice.
    self._coeffs = None

  def group_bound(self):
    """A static bound on the groups, the product of each GROUP BY key's
    values when every key is int8 or int16 (per element of its width: a
    string's bytes), else None; and per key, ``(bound or None, what gives
    it)``."""
    key_bounds = []
    for dtype, width in self.key_d_specs:
      narrow = _narrow_key_values(dtype, max(1, width))
      key_bounds.append(narrow if narrow is not None else (None, "?"))
    groups = None
    if all(bound is not None for bound, _ in key_bounds):
      groups = 1
      for bound, _ in key_bounds:
        groups *= bound
    return groups, key_bounds

  def run_row_b(self, key_route: bool) -> int:
    """Pass 2's run bytes per row on the key route or the hash route
    (``_agg_run_row_b``)."""
    return _agg_run_row_b(
        key_route,
        list(zip(self.key_types, self.key_widths)),
        list(zip(self.agg_types, self.state_width)))

  def pass1(self, device_name, columns, B: int):
    """One chunk's partials: ``columns`` (by id, the chunk's rows, masked
    rows already dropped) reduced to one row per group.  Returns
    ``[hash] + keys + values`` and the buckets.

    ``hash`` is each group's grouping hash (int64, ascending, unique);
    ``keys`` each emitted key's value in the group, in its own dtype (a
    width-1 key 1-D, a wider one 2-D); ``values`` each result's partial
    over the group's rows, a state result as its partial state.  The
    buckets are None at B = 1 or with no group, else ``(bucket, counts,
    counts_t)``: each group's bucket among B (``_bucket_ids``, int32) and
    the groups per bucket, on the host and on the GPU."""
    cnt_rows = next(iter(columns.values())).size(0)

    chunk_key_tensors = _materialize_chunk_keys(columns, self.key_ids_d)
    # Group by a linear hash sum_word(packed_word * coeff) with
    # DETERMINISTIC, validated coefficients (see _COEFF_BASE_SEED /
    # _make_key_coeffs): the seed is proven collision-free over every
    # grouping-key domain the suite reaches, so this is correct while keeping
    # the fast 1-D unique (no dim=0 sort cost).
    if self._coeffs is None:
      self._coeffs = _make_key_coeffs(chunk_key_tensors, device_name)
    hash_tensor = _hash_key_columns(chunk_key_tensors, self._coeffs)

    chunk_unique_hash, inv = torch.unique(hash_tensor, return_inverse=True)
    del hash_tensor

    cnt_chunk_groups = chunk_unique_hash.numel()
    # Partitioned: each group's bucket, and the buckets' rows read right
    # after torch.unique's own synchronization, with nothing else queued
    # (the chunk's one extra host read).
    buckets = None
    if B > 1 and cnt_chunk_groups > 0:
      bucket = _bucket_ids(chunk_unique_hash, B)
      counts_t = torch.bincount(bucket, minlength=B)
      buckets = (bucket, counts_t.tolist(), counts_t)

    # Only scatter the EMITTED keys (key_ids). Grouping keys not in the
    # SELECT (Q13.agg10) don't need to be carried through.
    emitted_chunk_key_tensors = _materialize_chunk_keys(columns, self.key_ids)
    key_reps = _scatter_unique_per_key(
        emitted_chunk_key_tensors, inv, cnt_chunk_groups)
    del chunk_key_tensors, emitted_chunk_key_tensors

    agg_vals = self.evaluate(device_name, columns, cnt_rows,
                             cnt_chunk_groups, inv)

    # Squeeze (N, 1) -> (N,) for narrow keys (the aggregate's output
    # convention).
    squeezed_key_reps = [
        t.squeeze(1) if (t.dim() == 2 and t.shape[1] == 1) else t
        for t in key_reps]
    return [chunk_unique_hash] + squeezed_key_reps + agg_vals, buckets


class GlobalAggregate(AggregateResults):
  """A global (no-key) aggregation's result expressions, from its plan
  args.  ``partials`` evaluates one chunk, ``merge`` folds a chunk's
  partials into the running results and ``finalize`` gives the results
  (lists in ``agg_ids`` order), so every engine mode evaluates and merges a
  global aggregation the same way."""

  def __init__(self, args, transfer_set, transfer_types):
    trees = [normalize_expr_tree(tree) for tree in args['results']]
    assert not any(is_group_key_expr(tree) for tree in trees), \
        "no-key aggregate cannot have a key expression in results"
    super().__init__(trees, transfer_set, transfer_types)
    assert self.agg_ids == args['out agg ids']
    assert self.plan_ops == args['agg ops']

  def partials(self, device_name, columns) -> List[torch.Tensor]:
    """One chunk's partial per result: its value over the chunk's rows, a
    state result as its partial state."""
    cnt_rows = next(iter(columns.values())).size(0)
    # Every row scatters into group 0; the per-chunk reduction is done
    # by ``evaluate_aggregate`` over this inverted_index.
    inverted_index = torch.zeros(cnt_rows, dtype=torch.int64, device=device_name)
    return self.evaluate(device_name, columns, cnt_rows, 1, inverted_index)

  def merge(self, results: List[torch.Tensor], partials) -> None:
    """Fold a chunk's ``partials`` into ``results``, in place: a partial
    state by addition, every other result by its expression's first op over
    the two values (exact for one aggregate call alone or scaled)."""
    for i in range(len(self.agg_ids)):
      if self.state[i]:
        # Every field is an additive sufficient statistic. Evaluate the
        # expression only after the final chunk, never per chunk.
        results[i] = results[i] + partials[i]
        continue
      combined = torch.cat([results[i], partials[i]])
      op = self.ops[i]
      if op == 'sum':
        results[i] = combined.sum().reshape(1)
      elif op == 'amax':
        results[i] = combined.max().reshape(1)
      elif op == 'amin':
        results[i] = combined.min().reshape(1)
      else:
        raise RuntimeError(f"Unsupported op: {op}")


def agg_transfer_set(args, tensor_group):
  """An aggregation's transfer set, its types and its mask map:
  ``args['input ids']``, extended in place by the mask of each input in the
  plan's ``mask map`` (so the operator drops the masks with its inputs);
  the types of the input ids before the masks; and ``{input id: mask id}``
  for the masked inputs."""
  transfer_set = args['input ids']
  transfer_types = [tensor_group[i].get_type_torch() for i in transfer_set]
  agg_mask_map = {}
  for k, v in (args.get('mask map') or {}).items():
    if k in transfer_set:
      agg_mask_map[k] = v
      if v not in transfer_set:
        transfer_set.append(v)
  return transfer_set, transfer_types, agg_mask_map


class AggPartitionCount(NamedTuple):
  """A keyed aggregation's B and what it was chosen from
  (``agg_partition_count``)."""
  B: int
  pred_b: Optional[float]   # the predicted largest round, None if not predicted
  reason: str
  rows: int                 # P
  units: int                # T: ticks or morsels
  groups: Optional[int]     # G, None if unbounded
  key_bounds: list          # per GROUP BY key, (bound or None, what gives it)
  row_b: int                # w
  route: str                # 'dense', 'key' or 'hash'
  key_range: tuple          # the key route's (range bound or None, source)


def agg_partition_count(spec, p_rows, units, route, key_range=(None, "?")):
  """A keyed aggregation's B from the partition budget
  (``operators/partitions.agg_partitions``): P ``p_rows`` (the input rows
  times the mask's selectivity), T ``units`` (pass 1's ticks or morsels),
  G ``spec``'s static bound on the groups and w its run bytes per row on
  pass 2's ``route`` (a dense shape's run, when its state is declined, is
  the key route's while that route is on), with the key route's run capped
  by ``key_range``.  ``AGGREGATION_PARTITIONS`` > 0 forces B; a dense shape
  takes 1."""
  groups, key_bounds = spec.group_bound()
  row_b = spec.run_row_b(
      route == "key" or (route == "dense" and SORTED_RUN_MERGE))
  pred_b = None
  if AGGREGATION_PARTITIONS > 0:
    B, reason = AGGREGATION_PARTITIONS, "forced"
    assert B & (B - 1) == 0, (
        f"AGGREGATION_PARTITIONS={B} is not a power of two")
  elif route == "dense":
    B, reason = 1, "dense"
  else:
    B, pred_b, reason = agg_partitions(
        p_rows, units, groups, row_b, key_range[0],
        constants.get_partition_budget_b())
  return AggPartitionCount(B, pred_b, reason, p_rows, units, groups,
                           key_bounds, row_b, route, key_range)


def log_agg_partition_count(name, part: AggPartitionCount, unit="ticks"):
  """The ``[agg_b]`` line: B, what it was chosen from (T in ``unit``) and
  the predicted largest round against the budget."""
  budget_b = constants.get_partition_budget_b()
  message_logger().info(
      "[agg_b] Q%s %s: B=%d (%s)%s (%d rows, %d %s, <= %s groups [%s], "
      "%d B/row, route %s%s) against %.1f MB%s",
      constants.get_query_id(), name, part.B, part.reason,
      "" if part.pred_b is None else " for %.1f MB" % (part.pred_b / (1 << 20)),
      part.rows, part.units, unit,
      "?" if part.groups is None else part.groups,
      " x ".join(source for _, source in part.key_bounds), part.row_b,
      part.route,
      ", key range <= %s [%s]" % (
          "?" if part.key_range[0] is None else part.key_range[0],
          part.key_range[1])
      if part.route == "key" else "",
      budget_b / (1 << 20),
      " OVER" if part.pred_b is not None and part.pred_b > budget_b else "")


def tqp_hash_aggregate(tensor_group, args, cache_region, cpu_mem_pool, name="aggregation", applier=None, page_cache=None):
  # Keyed aggregations go through the two-pipeline chunked end-merge
  # path (see ``tqp_hash_aggregate_two_pass``). The body below handles the
  # no-key (global) case where every row collapses into a single group and
  # the cross-chunk reduction is a flat ``cat`` + op (``GlobalAggregate``).
  if args.get('keys'):
    return tqp_hash_aggregate_two_pass(
        tensor_group, args, cache_region, cpu_mem_pool,
        name, applier, page_cache)

  perf_logger().start(f"{name}")
  device_name = 'cuda'
  transfer_set, transfer_types, agg_mask_map = agg_transfer_set(
      args, tensor_group)

  spec = GlobalAggregate(args, transfer_set, transfer_types)
  agg_ids, agg_types = spec.agg_ids, spec.agg_types

  # The running results, in agg_ids order (None before the first chunk).
  global_aggregation_results = None
  global_stage_countdown = None

  def agg_wrapper(columns):
    nonlocal global_aggregation_results, global_stage_countdown
    global_stage_countdown -= 1

    partials = spec.partials(device_name, columns)

    for col in list(columns.keys()):
      del columns[col]

    if global_aggregation_results is not None:
      spec.merge(global_aggregation_results, partials)
    else:
      global_aggregation_results = partials

    if global_stage_countdown == 0:
      return spec.finalize(global_aggregation_results)
    return []

  node = args['nodes']['pipe']
  agg_pipe_name = node.name
  assert [inst.local for inst in node.outputs] == agg_ids
  agg_pipeline = VortexPipeline(
    node=node,
    inputs=node.input_pairs(),
    output_columns_type=agg_types,
    operator=agg_wrapper,
    mask_map=agg_mask_map,
    cpu_mem_pool=cpu_mem_pool,
    chunk_size=sizing_resolver.resolve("agg", agg_pipe_name).chunk_b,
    sizing_key=("agg", agg_pipe_name),
    # NEVER sectionable: ``global_stage_countdown = stages - 2`` below is
    # latched once from the construction geometry; a mid-run re-split
    # desynchronizes the finalization countdown.  Cross-run axis only.
    sectionable=False,
    applier=applier,
    page_cache=page_cache,
  )

  global_stage_countdown = agg_pipeline.stages - 2
  agg_pipeline.do_exchange()
  agg_pipeline.get_result()

  for i, aid in enumerate(agg_ids):
    inst = node.outputs[i]
    inst.ccr = CombinedColumnRep([agg_pipeline.output_variable(
        i, torch_to_type(agg_types[i]))])
    tensor_group[aid] = inst.ccr

  output_ids = set(args.get('output ids', agg_ids))
  for input_col in args['input ids']:
    if input_col in tensor_group and input_col not in output_ids:
      tensor_group[input_col].free_underlying_mem(cpu_pool=cpu_mem_pool, gpu_pool=cache_region)
      del tensor_group[input_col]

  perf_logger().stop(f"{name}")
  return None
