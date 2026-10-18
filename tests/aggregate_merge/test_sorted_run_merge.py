"""CPU-only unit tests for the PASS2 fixed-slab sorted-run merge.

Exercises ``operators.sorted_run_merge`` — the math behind the
``SORTED_RUN_MERGE`` route in ``operators/aggregate.py`` ``pass2_op`` —
against a reference implementation that reproduces the ORIGINAL path's
semantics: one ``torch.unique`` over the full concatenation +
``scatter_reduce_(include_self=False)``.

Everything runs on plain CPU tensors. No CUDA import, no TQPlib, no GPU
touch — ``operators.sorted_run_merge`` deliberately imports only torch.
Run with ``CUDA_VISIBLE_DEVICES=""`` to keep tests/conftest.py from
bootstrapping the custom GPU allocator on a machine that has a GPU.
"""

import random

import pytest
import torch

from operators.sorted_run_merge import (
    MERGEABLE_OPS,
    merge_sorted_unique_runs_inplace,
    reduce_tick_slice,
)

OPS3 = ["sum", "amax", "amin"]


# ──────────────────────────────────────────────────────────────────────
# Reference: the original PASS2 semantics over the raw concatenation.
# ──────────────────────────────────────────────────────────────────────

def reference_reduce(all_keys, all_vals, ops):
    u_keys, inv = torch.unique(all_keys, return_inverse=True)
    out = []
    for v, op in zip(all_vals, ops):
        res = torch.zeros(
            (u_keys.numel(),) + tuple(v.shape[1:]), dtype=v.dtype)
        index = inv
        if v.dim() > 1:
            index = inv.view(
                (inv.numel(),) + (1,) * (v.dim() - 1)).expand_as(v)
        res.scatter_reduce_(0, index, v, reduce=op, include_self=False)
        out.append(res)
    return u_keys, out


# ──────────────────────────────────────────────────────────────────────
# Driver: feed raw tick slices through the production per-tick path
# (tick-local reduce, then sorted-run merge), exactly as pass2_op does.
# ──────────────────────────────────────────────────────────────────────

def merge_once(run_keys, run_vals, tick_keys, tick_vals, ops,
               place_chunk_rows=None):
    """Exercise one merge through the production fixed-slab interface."""
    capacity = run_keys.numel() + tick_keys.numel()
    key_buf = torch.empty(
        capacity, dtype=run_keys.dtype, device=run_keys.device)
    key_buf[:run_keys.numel()] = run_keys
    val_bufs = [
        torch.empty(
            (capacity,) + tuple(v.shape[1:]),
            dtype=v.dtype,
            device=v.device)
        for v in run_vals
    ]
    for dst, src in zip(val_bufs, run_vals):
        dst[:run_keys.numel()] = src
    merged_len = merge_sorted_unique_runs_inplace(
        key_buf,
        val_bufs,
        run_keys.numel(),
        tick_keys,
        tick_vals,
        ops,
        place_chunk_rows=place_chunk_rows)
    return (key_buf[:merged_len].clone(),
            [v[:merged_len].clone() for v in val_bufs])


def run_streaming(tick_keys_slices, tick_vals_slices, ops):
    nonempty = [i for i, keys in enumerate(tick_keys_slices)
                if keys.numel() > 0]
    if not nonempty:
        return None, None

    first = nonempty[0]
    capacity = sum(keys.numel() for keys in tick_keys_slices)
    key_template = tick_keys_slices[first]
    val_templates = tick_vals_slices[first]
    key_buf = torch.empty(
        capacity, dtype=key_template.dtype, device=key_template.device)
    val_bufs = [
        torch.empty(
            (capacity,) + tuple(v.shape[1:]),
            dtype=v.dtype,
            device=v.device)
        for v in val_templates
    ]

    run_len = 0
    for keys, vals in zip(tick_keys_slices, tick_vals_slices):
        if keys.numel() == 0:
            # pass2_op short-circuits empty ticks before the route; the
            # helper is still expected to no-op on them (tested below).
            continue
        tick_keys, tick_vals = reduce_tick_slice(keys, vals, ops)
        run_len = merge_sorted_unique_runs_inplace(
            key_buf, val_bufs, run_len, tick_keys, tick_vals, ops)
    return (key_buf[:run_len].clone(),
            [v[:run_len].clone() for v in val_bufs])


def assert_matches_reference(tick_keys_slices, tick_vals_slices, ops,
                             exact=True):
    got_keys, got_vals = run_streaming(tick_keys_slices, tick_vals_slices, ops)

    nonempty = [i for i, k in enumerate(tick_keys_slices) if k.numel() > 0]
    assert nonempty, "test case must contain at least one non-empty tick"
    all_keys = torch.cat([tick_keys_slices[i] for i in nonempty])
    all_vals = [torch.cat([tick_vals_slices[i][j] for i in nonempty])
                for j in range(len(ops))]
    ref_keys, ref_vals = reference_reduce(all_keys, all_vals, ops)

    # Output invariants: ascending and unique (a sorted-unique run).
    assert got_keys.dim() == 1
    if got_keys.numel() > 1:
        assert bool((got_keys[1:] > got_keys[:-1]).all()), \
            "merged keys must be strictly ascending"

    assert torch.equal(got_keys, ref_keys), \
        f"keys mismatch:\n got {got_keys}\n ref {ref_keys}"
    for got_v, ref_v, op in zip(got_vals, ref_vals, ops):
        assert got_v.dtype == ref_v.dtype
        if exact:
            assert torch.equal(got_v, ref_v), \
                f"vals mismatch for op={op}:\n got {got_v}\n ref {ref_v}"
        else:
            torch.testing.assert_close(got_v, ref_v)


def make_vals(keys, ops, dtype=torch.int64, generator=None, low=-100, high=100):
    """One value column per op, element-aligned with ``keys``."""
    n = keys.numel()
    out = []
    for _ in ops:
        v = torch.randint(low, high, (n,), generator=generator)
        out.append(v.to(dtype))
    return out


def k(*xs):
    return torch.tensor(xs, dtype=torch.int64)


# ──────────────────────────────────────────────────────────────────────
# Hand-computed micro cases (paranoia about the index arithmetic).
# ──────────────────────────────────────────────────────────────────────

def test_hand_computed_merge_all_three_ops():
    # running:  keys [1, 5, 9], vals aligned
    # tick raw: keys [5, 5, 7] (duplicate 5 inside the tick slice)
    run_keys = k(1, 5, 9)
    run_vals = [
        torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64),  # sum
        torch.tensor([10, 20, 30], dtype=torch.int64),          # amax
        torch.tensor([10, 20, 30], dtype=torch.int64),          # amin
    ]
    raw_tick = k(5, 5, 7)
    raw_vals = [
        torch.tensor([1.0, 2.0, 5.0], dtype=torch.float64),
        torch.tensor([1, 2, 5], dtype=torch.int64),
        torch.tensor([1, 2, 5], dtype=torch.int64),
    ]
    t_keys, t_vals = reduce_tick_slice(raw_tick, raw_vals, OPS3)
    assert torch.equal(t_keys, k(5, 7))
    assert torch.equal(t_vals[0], torch.tensor([3.0, 5.0], dtype=torch.float64))
    assert torch.equal(t_vals[1], torch.tensor([2, 5]))  # amax(1, 2)
    assert torch.equal(t_vals[2], torch.tensor([1, 5]))  # amin(1, 2)

    m_keys, m_vals = merge_once(
        run_keys, run_vals, t_keys, t_vals, OPS3)
    assert torch.equal(m_keys, k(1, 5, 7, 9))
    # matched key 5: op(running, tick); others verbatim.
    assert torch.equal(m_vals[0],
                       torch.tensor([10.0, 23.0, 5.0, 30.0],
                                    dtype=torch.float64))
    assert torch.equal(m_vals[1], torch.tensor([10, 20, 5, 30]))  # max(20,2)
    assert torch.equal(m_vals[2], torch.tensor([10, 1, 5, 30]))   # min(20,1)


def test_hand_computed_destination_ranks():
    # Adversarial insertion pattern: new keys at both ends, several new
    # keys between one pair of running keys, matches at first and last
    # running key. Expected merged set worked out by hand:
    #   run  [10, 20, 30]
    #   tick [1, 2, 10, 15, 16, 17, 30, 40, 41]
    #   merged [1, 2, 10, 15, 16, 17, 20, 30, 40, 41]
    run_keys = k(10, 20, 30)
    run_vals = [torch.tensor([100, 200, 300])]
    t_keys = k(1, 2, 10, 15, 16, 17, 30, 40, 41)
    t_vals = [torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])]
    m_keys, m_vals = merge_once(
        run_keys, run_vals, t_keys, t_vals, ["sum"])
    assert torch.equal(m_keys, k(1, 2, 10, 15, 16, 17, 20, 30, 40, 41))
    assert torch.equal(
        m_vals[0], torch.tensor([1, 2, 103, 4, 5, 6, 200, 307, 8, 9]))


def test_matched_only_at_boundaries():
    run_keys = k(-5, 0, 5)
    run_vals = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)]
    # tick matches exactly the first and last running keys, nothing new
    t_keys = k(-5, 5)
    t_vals = [torch.tensor([10.0, 30.0], dtype=torch.float64)]
    m_keys, m_vals = merge_once(
        run_keys, run_vals, t_keys, t_vals, ["sum"])
    assert torch.equal(m_keys, run_keys)
    assert torch.equal(m_vals[0],
                       torch.tensor([11.0, 2.0, 33.0], dtype=torch.float64))


# ──────────────────────────────────────────────────────────────────────
# Required edge cases, each against the reference.
# ──────────────────────────────────────────────────────────────────────

def test_disjoint_runs():
    g = torch.Generator().manual_seed(1)
    ticks = [k(3, 1, 2, 5, 4), k(10, 12, 11), k(-20, -19, -18, -17)]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_fully_overlapping_runs():
    g = torch.Generator().manual_seed(2)
    ticks = [k(7, 3, 5, 1), k(1, 3, 5, 7), k(5, 1, 7, 3)]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_interleaved_runs():
    g = torch.Generator().manual_seed(3)
    ticks = [k(1, 3, 5, 7, 9), k(2, 4, 6, 8, 10), k(3, 4, 9, 10, 11)]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_duplicates_within_tick_slice():
    g = torch.Generator().manual_seed(4)
    ticks = [k(5, 5, 5, 2, 2, 9), k(9, 9, 1, 5, 1, 1)]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_empty_tick_mid_stream():
    g = torch.Generator().manual_seed(5)
    empty = torch.empty(0, dtype=torch.int64)
    ticks = [k(4, 2), empty, k(2, 6)]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_empty_tick_run_into_merge_is_noop():
    # Direct helper-level checks for the empty early-outs.
    run_keys = k(1, 2, 3)
    run_vals = [torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)]
    empty_k = torch.empty(0, dtype=torch.int64)
    empty_v = [torch.empty(0, dtype=torch.float64)]

    m_keys, m_vals = merge_once(
        run_keys, run_vals, empty_k, empty_v, ["sum"])
    assert torch.equal(m_keys, run_keys)
    assert torch.equal(m_vals[0], run_vals[0])

    m_keys, m_vals = merge_once(
        empty_k, empty_v, run_keys, run_vals, ["sum"])
    assert torch.equal(m_keys, run_keys)
    assert torch.equal(m_vals[0], run_vals[0])

    t_keys, t_vals = reduce_tick_slice(empty_k, empty_v, ["sum"])
    assert t_keys.numel() == 0 and t_vals[0].numel() == 0


def test_empty_running_first_tick():
    # First tick: running state absent; the route takes the tick run as-is.
    g = torch.Generator().manual_seed(6)
    ticks = [k(9, -1, 9, 4)]
    vals = [make_vals(ticks[0], OPS3, generator=g)]
    assert_matches_reference(ticks, vals, OPS3)


def test_single_element_runs():
    g = torch.Generator().manual_seed(7)
    # same single key in both ticks (match), then a different one (new)
    ticks = [k(42), k(42), k(41)]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_negative_keys():
    g = torch.Generator().manual_seed(8)
    ticks = [k(-3, -1, -2, 0), k(-2, 1, -3, -100), k(-100, 0, 100)]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_amax_all_negative_values_zero_sentinel():
    # If include_self leaked the fresh-zero destination into amax, the
    # result would be 0 instead of the true (negative) max.
    ticks = [k(1, 2, 1), k(2, 3)]
    vals = [
        [torch.tensor([-5, -7, -3], dtype=torch.int64)],
        [torch.tensor([-9, -11], dtype=torch.int64)],
    ]
    got_keys, got_vals = run_streaming(ticks, vals, ["amax"])
    assert torch.equal(got_keys, k(1, 2, 3))
    assert torch.equal(got_vals[0], torch.tensor([-3, -7, -11]))


def test_amin_all_positive_values_zero_sentinel():
    ticks = [k(1, 2, 1), k(2, 3)]
    vals = [
        [torch.tensor([5, 7, 3], dtype=torch.int64)],
        [torch.tensor([9, 11], dtype=torch.int64)],
    ]
    got_keys, got_vals = run_streaming(ticks, vals, ["amin"])
    assert torch.equal(got_keys, k(1, 2, 3))
    assert torch.equal(got_vals[0], torch.tensor([3, 7, 11]))


def test_float64_sum_q18_shape():
    # Q18's actual shape: one float64 'sum' val (l_quantity is
    # integer-valued in TPC-H, so exact comparison is well-defined).
    g = torch.Generator().manual_seed(9)
    ticks = [torch.randint(0, 50, (200,), generator=g)
             for _ in range(4)]
    vals = [make_vals(t, ["sum"], dtype=torch.float64, generator=g,
                      low=1, high=51)
            for t in ticks]
    assert_matches_reference(ticks, vals, ["sum"])


def test_packed_sum_count_row_state():
    # Q17's physical AVG state is one logical value column with two fields.
    # Both tick-local duplicate reduction and cross-tick merge operate on
    # rows, preserving [sum, count] alignment.
    ticks = [
        k(3, 1, 3, 2),
        k(2, 1, 4, 3, 4),
        k(1, 4),
    ]
    vals = [[torch.tensor(rows, dtype=torch.float64)] for rows in (
        [[3, 1], [10, 2], [7, 1], [5, 1]],
        [[11, 2], [2, 1], [8, 1], [13, 2], [9, 3]],
        [[20, 4], [6, 1]],
    )]
    assert_matches_reference(ticks, vals, ["sum"])


def test_random_float_values_close():
    # True random floats: summation ORDER differs between the streaming
    # merge and the reference cat-reduce, so compare with tolerances.
    g = torch.Generator().manual_seed(10)
    ticks = [torch.randint(-20, 20, (150,), generator=g) for _ in range(5)]
    vals = []
    for t in ticks:
        vs = []
        for _ in OPS3:
            vs.append(torch.randn(t.numel(), dtype=torch.float64,
                                  generator=g))
        vals.append(vs)
    assert_matches_reference(ticks, vals, OPS3, exact=False)


def test_fuzz_int64_exact():
    # Randomised streams: ragged tick sizes (incl. empties), negative and
    # colliding keys, all three ops at once, exact integer comparison.
    for seed in range(20):
        g = torch.Generator().manual_seed(100 + seed)
        rng = random.Random(seed)
        n_ticks = rng.randint(1, 6)
        ticks, vals = [], []
        for _ in range(n_ticks):
            n = rng.choice([0, 1, 2, rng.randint(3, 200)])
            keys = torch.randint(-50, 50, (n,), generator=g)
            ticks.append(keys)
            vals.append(make_vals(keys, OPS3, generator=g))
        if all(t.numel() == 0 for t in ticks):
            ticks.append(k(0))
            vals.append(make_vals(ticks[-1], OPS3, generator=g))
        assert_matches_reference(ticks, vals, OPS3)


def test_fuzz_wide_key_range():
    # Keys spanning the int64 range sign boundary and large magnitudes
    # (l_orderkey at SF500 exceeds 2^31 — stay int64-clean).
    g = torch.Generator().manual_seed(11)
    big = 3_000_000_000
    ticks = [
        torch.tensor([big, -big, 0, big - 1, 7], dtype=torch.int64),
        torch.tensor([big, big + 5, -big, -1], dtype=torch.int64),
    ]
    vals = [make_vals(t, OPS3, generator=g) for t in ticks]
    assert_matches_reference(ticks, vals, OPS3)


def test_mergeable_ops_exclude_mean():
    # A logical mean must never be sent directly to scatter_reduce_. AVG is
    # admitted only after aggregate.py maps its [sum, count] state to SUM.
    assert set(MERGEABLE_OPS) == {"sum", "amax", "amin"}
    assert "mean" not in MERGEABLE_OPS


def test_aggregate_flag_exists():
    # The kill switch must exist at module level in operators.aggregate.
    # Importing operators.aggregate pulls CUDA-adjacent deps (TQPlib),
    # which may be absent on a CPU-only box — parse the source instead.
    import ast
    import os
    src_path = os.path.join(os.path.dirname(__file__), "..", "..",
                            "operators", "aggregate.py")
    tree = ast.parse(open(src_path).read())
    flags = [n for n in tree.body if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == "SORTED_RUN_MERGE"
                     for t in n.targets)]
    assert flags, "SORTED_RUN_MERGE module-level flag missing"
    assert isinstance(flags[0].value, ast.Constant)
    assert flags[0].value.value in (True, False)


def test_chunked_placement_matches_reference(monkeypatch):
    """Force tiny _PLACE_CHUNK so the running-run placement spans many
    slices; results must stay bit-identical to the reference path."""
    import operators.sorted_run_merge as srm
    monkeypatch.setattr(srm, "_PLACE_CHUNK", 3)
    g = torch.Generator().manual_seed(7)
    ops = ["sum", "amin"]
    ticks = [torch.randint(-30, 30, (n,), generator=g) for n in (29, 17, 31)]
    valsets = [make_vals(t, ops, generator=g) for t in ticks]
    assert_matches_reference(ticks, valsets, ops)


def test_inplace_merge_matches_reference(monkeypatch):
    """The in-place single-buffer variant (back-to-front chunked shift)
    must reproduce the reference reduce bit-exactly, across many ticks,
    forced multi-slice shifts, duplicates, and matched-key combines."""
    import operators.sorted_run_merge as srm
    monkeypatch.setattr(srm, "_PLACE_CHUNK", 4)
    g = torch.Generator().manual_seed(21)
    ops = ["sum", "amax", "amin"]
    ticks = [torch.randint(-40, 40, (n,), generator=g)
             for n in (33, 19, 27, 1, 22)]
    valsets = [make_vals(t, ops, generator=g) for t in ticks]

    cap = sum(t.numel() for t in ticks)        # guaranteed upper bound
    key_buf = torch.empty(cap, dtype=torch.int64)
    val_bufs = [torch.empty(cap, dtype=v.dtype) for v in valsets[0]]
    m = 0
    for t, vs in zip(ticks, valsets):
        tk, tv = reduce_tick_slice(t, vs, ops)
        m = merge_sorted_unique_runs_inplace(key_buf, val_bufs, m, tk, tv, ops)

    all_keys = torch.cat(ticks)
    all_vals = [torch.cat([valsets[i][j] for i in range(len(ticks))])
                for j in range(len(ops))]
    ref_keys, ref_vals = reference_reduce(all_keys, all_vals, ops)
    assert torch.equal(key_buf[:m], ref_keys)
    for vb, rv in zip(val_bufs, ref_vals):
        assert torch.equal(vb[:m], rv)


def test_inplace_merge_supports_row_shaped_state(monkeypatch):
    import operators.sorted_run_merge as srm
    monkeypatch.setattr(srm, "_PLACE_CHUNK", 2)

    ticks = [k(5, 1, 5), k(2, 5, 1), k(9, 2)]
    valsets = [[torch.tensor(rows, dtype=torch.float64)] for rows in (
        [[5, 1], [1, 1], [7, 2]],
        [[2, 1], [11, 3], [3, 2]],
        [[9, 1], [4, 2]],
    )]
    cap = sum(t.numel() for t in ticks)
    key_buf = torch.empty(cap, dtype=torch.int64)
    val_bufs = [torch.empty((cap, 2), dtype=torch.float64)]

    m = 0
    for keys, values in zip(ticks, valsets):
        tick_keys, tick_vals = reduce_tick_slice(keys, values, ["sum"])
        m = merge_sorted_unique_runs_inplace(
            key_buf, val_bufs, m, tick_keys, tick_vals, ["sum"])

    ref_keys, ref_vals = reference_reduce(
        torch.cat(ticks), [torch.cat([v[0] for v in valsets])], ["sum"])
    assert torch.equal(key_buf[:m], ref_keys)
    assert torch.equal(val_bufs[0][:m], ref_vals[0])
