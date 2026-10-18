"""Sorted-run merge primitives for the two-pass aggregator's PASS2.

Tier-3 (this sorted-run merge path): for the K==1 / 1-D-key
direct-unique shape (Q18's inner ``GROUP BY l_orderkey``), PASS2 used to
re-run ``torch.unique(cat(running, tick))`` every tick, so the radix-sort
working set covered the CUMULATIVE distinct set (growing toward ~750M keys
/ a ~5 GB contiguous CUB scratch ask at SF500). The two functions here
replace that with:

  1. ``reduce_tick_slice`` — one ``torch.unique`` over THIS TICK's slice
     only (O(tick) radix scratch, never cumulative), collapsing duplicate
     keys within the slice. Needed because a PASS2 tick spans many PASS1
     chunk outputs: the slice is neither globally unique nor sorted, no
     matter how PASS1 orders its per-chunk output.
  2. ``merge_sorted_unique_runs_inplace`` — merges the (unique, ascending)
     running run with the (unique, ascending) tick run in its fixed slab
     using ``searchsorted`` rank arithmetic + ``scatter_reduce_``. Linear
     O(R + T) scratch; no sort at all over the cumulative set.

Only torch is imported here so the math is unit-testable on CPU without
pulling in the TQPlib CUDA extension (see
``tests/aggregate_merge/test_sorted_run_merge.py``).
"""

from typing import List, Sequence, Tuple

import torch

# Aggregation flavours (after the agg_ops_norm normalisation in
# operators/aggregate.py: count→'sum', max→'amax', min→'amin') for which
# merging two partial results per matched key as ``op(running, tick)`` is
# exact: 'sum' is associative+commutative, 'amax'/'amin' are
# associative+commutative+idempotent. A logical AVG qualifies only after
# operators/aggregate.py has decomposed it to a row-shaped ``[sum, count]``
# state whose physical merge op is ``sum``.
MERGEABLE_OPS = ('sum', 'amax', 'amin')

# Slice size for running-run placement in merge_sorted_unique_runs_inplace:
# bounds the per-slice index temporaries to _PLACE_CHUNK x 8 B (512 MB at
# the default) regardless of run length.  Module-level so tests can force
# tiny values to exercise the multi-slice path on small inputs.
_PLACE_CHUNK = 1 << 26   # 64M rows


def _row_index(index: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
  """Expand a row index across ``value``'s trailing state dimensions."""
  if value.dim() == 1:
    return index
  return index.view(
      (index.numel(),) + (1,) * (value.dim() - 1)).expand(
          (index.numel(),) + tuple(value.shape[1:]))


def reduce_tick_slice(keys: torch.Tensor,
                      vals: Sequence[torch.Tensor],
                      ops: Sequence[str]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
  """Collapse duplicate keys within one tick slice.

  Parameters
  ----------
  keys : (T,) tensor, arbitrary order, may contain duplicates.
  vals : list of ``(T, ...)`` tensors row-aligned with ``keys``.
  ops  : per-val reduce flavour, each in ``MERGEABLE_OPS``.

  Returns
  -------
  (unique ascending keys, list of reduced vals aligned with them).

  This is the ONLY sort in the Tier-3 path and it covers just the tick
  slice. ``include_self=False`` so the fresh-zero destination never
  poisons amax/amin: the first scattered value overwrites, subsequent
  ones combine via ``op``.
  """
  u_keys, inv = torch.unique(keys, return_inverse=True)
  out_vals: List[torch.Tensor] = []
  for v, op in zip(vals, ops):
    res = torch.zeros(
        (u_keys.numel(),) + tuple(v.shape[1:]),
        dtype=v.dtype, device=v.device)
    res.scatter_reduce_(
        0, _row_index(inv, v), v, reduce=op, include_self=False)
    out_vals.append(res)
  return u_keys, out_vals


def merge_sorted_unique_runs_inplace(key_buf: torch.Tensor,
                                     val_bufs: Sequence[torch.Tensor],
                                     run_len: int,
                                     tick_keys: torch.Tensor,
                                     tick_vals: Sequence[torch.Tensor],
                                     ops: Sequence[str],
                                     place_chunk_rows: int = None) -> int:
  """Merge a sorted-unique tick run into fixed resident buffers in place.

  The running run occupies ``key_buf[:run_len]`` / ``val_bufs[i][:run_len]``
  of buffers pre-allocated ONCE at a guaranteed capacity (the pass2 total
  input rows — a merge of unique sets can never exceed the sum of its
  inputs).  The merged run is materialised into the SAME buffers at
  ``[:M]`` and the new length M is returned.  This eliminates BOTH failure
  modes measured at SF500-Q18 across five iterations of the two-buffer
  design: per-tick grow-free-realloc churn (9-region arena fragmentation)
  and dual-set residency (two full key+val sets resident, ~23 GB at 750M
  groups) — peak drops to ONE set + chunk-sized temporaries.

  In-place safety: every running key's destination rank is >= its source
  rank (``dst[j] = j + #new_below >= j``), so the shift is performed
  back-to-front in ``_PLACE_CHUNK`` slices — the memmove discipline for
  overlapping ranges.  A slice's writes land either inside itself (its
  content is snapshotted via ``.clone()`` first — index_put with aliased
  source/destination is otherwise undefined) or at higher positions,
  whose original content belongs to already-processed (higher) slices.
  Destinations never fall below the slice start.
  """
  place_chunk_rows = (
      _PLACE_CHUNK if place_chunk_rows is None
      else max(1, int(place_chunk_rows)))
  T = tick_keys.numel()
  if T == 0:
    return run_len
  if run_len == 0:
    key_buf[:T] = tick_keys
    for vb, t_v in zip(val_bufs, tick_vals):
      vb[:T] = t_v
    return T

  R = run_len
  device = key_buf.device
  run_keys = key_buf[:R]

  # Tick-sized rank computation — all reads of the running run happen
  # BEFORE any in-place mutation below.
  pos = torch.searchsorted(run_keys, tick_keys)
  in_range = pos < R
  is_match = in_range & (run_keys[pos.clamp(max=R - 1)] == tick_keys)
  new_mask = ~is_match
  new_keys = tick_keys[new_mask]
  num_new = new_keys.numel()
  M = R + num_new
  assert M <= key_buf.shape[0], (
      f"merge capacity exhausted: M={M} > buf={key_buf.shape[0]} — the "
      f"buffers must be sized to the pass2 total input rows")
  j_match = pos[is_match]
  dst_match = j_match + torch.searchsorted(new_keys, tick_keys[is_match])
  dst_new = (pos[new_mask]
             + torch.arange(num_new, device=device))

  if num_new > 0:
    # Back-to-front chunked shift of the running run into its merged ranks.
    for a in reversed(range(0, R, place_chunk_rows)):
      b = min(a + place_chunk_rows, R)
      src_k = key_buf[a:b].clone()
      dst_c = torch.searchsorted(new_keys, src_k)
      dst_c += torch.arange(a, b, device=device)
      key_buf[dst_c] = src_k
      src_k = None
      for vb in val_bufs:
        src_v = vb[a:b].clone()
        vb[dst_c] = src_v
        src_v = None
      dst_c = None

    # Insert the new keys/vals at their (now vacant) ranks.
    key_buf[dst_new] = new_keys
    for vb, t_v in zip(val_bufs, tick_vals):
      vb[dst_new] = t_v[new_mask]

  # Combine matched tick rows into their running slots (tick-sized).
  if dst_match.numel() > 0:
    for vb, t_v, op in zip(val_bufs, tick_vals, ops):
      matched_vals = t_v[is_match]
      vb[:M].scatter_reduce_(
          0, _row_index(dst_match, matched_vals), matched_vals,
          reduce=op, include_self=True)
  return M
