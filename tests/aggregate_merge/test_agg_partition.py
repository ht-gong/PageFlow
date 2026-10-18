"""The pieces of a partitioned aggregation (operators/aggregate.py): each
group's bucket (``_bucket_ids``, by ``_mix64``), pass 1's
outputs in the padded bucket layout (``_partition_outputs``), pass 2's
rounds of whole buckets (``pass2_rounds``) and how a feed is cut at their
ends (``_round_steps``), and the run bytes per row B is chosen from
(``_agg_run_row_b``).

The rounds, the steps and the row bytes are CPU-only, the mix64 on the CPU
and the GPU; the buckets and the outputs run on the GPU.
"""
from __future__ import annotations

from math import gcd

import numpy as np
import pytest
import torch

from IO.codec import region_producer
from IO.codec.batched_compress import aligned_partition_layout
from operators.aggregate import (_agg_run_row_b, _bucket_ids, _mix64,
                                 _partition_outputs, _round_steps,
                                 pass2_rounds)
from operators.aggregation_slab import FixedColumnSlab
from tests.mix64 import mix64, mix64_np

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required")


# ── Buckets ──────────────────────────────────────────────────────────────

@requires_cuda
@pytest.mark.parametrize("B", [1, 2, 256])
def test_a_groups_bucket_is_the_joins_partition_function(B):
    """The top log2(B) bits of the hash's mix64, read unsigned, as the
    join's kernel takes them (the reference of tests/operators/
    test_partition_sort.py), from any int64 hash, the extremes included."""
    i64 = np.iinfo(np.int64)
    h = np.random.default_rng(B).integers(i64.min, i64.max, 100_000,
                                          dtype=np.int64)
    h[:4] = [0, -1, i64.min, i64.max]
    h = torch.from_numpy(h)
    got = _bucket_ids(h.cuda(), B)
    assert got.dtype == torch.int32 and got.shape == h.shape
    want = np.zeros(len(h), dtype=np.int64)
    if B > 1:
        mixed = mix64_np(h.numpy())
        want = (mixed >> np.uint64(64 - (B.bit_length() - 1))).astype(
            np.int64)
    assert np.array_equal(got.cpu().numpy(), want)


@pytest.mark.parametrize("device", [
    "cpu", pytest.param("cuda", marks=requires_cuda)])
def test_mix64_is_the_finalizer(device):
    """``_mix64``, in torch's elementwise ops, is the Murmur64 finalizer bit
    for bit (the reference's uint64 math), the extremes and both signs
    included."""
    i64 = np.iinfo(np.int64)
    h = np.random.default_rng(7).integers(i64.min, i64.max, 100_000,
                                          dtype=np.int64)
    h[:6] = [0, 1, -1, i64.min, i64.max, 1 << 33]
    h = torch.from_numpy(h)
    got = _mix64(h.to(device))
    assert got.dtype == torch.int64 and got.device.type == device
    assert torch.equal(got.cpu(), mix64(h))


@requires_cuda
def test_buckets_spread_a_small_keys_linear_hash():
    """A one-key linear grouping hash is the key times a coefficient: its
    top bits barely move.  Its mix64 spreads every 2^20 keys over all 256
    buckets within a quarter of the mean."""
    h = torch.arange(1, (1 << 20) + 1, dtype=torch.int64,
                     device="cuda") * 9_999_991
    counts = torch.bincount(_bucket_ids(h, 256), minlength=256)
    assert int(counts.min()) > 0.75 * 4096
    assert int(counts.max()) < 1.25 * 4096


# ── Pass 1's partitioned outputs ─────────────────────────────────────────

def _layout(counts, row_b):
    unit_b = row_b * 512 // gcd(row_b, 512)
    return aligned_partition_layout(counts, row_b, unit_b)


@requires_cuda
@pytest.mark.parametrize("B,n", [(2, 1), (8, 63), (64, 1_000),
                                 (256, 70_001)])
def test_outputs_are_laid_out_padded_in_bucket_order(B, n):
    """Each output holds bucket b's rows from its padded start, in their
    order within the tick, whatever its row width (a 1-byte code, an int64
    hash or key, a 16-byte AVG state, a 25-byte string), and the region
    producer reads every bucket from where it lies."""
    gen = torch.Generator().manual_seed(B * 7919 + n)
    # Some buckets stay empty.
    used = torch.randperm(B, generator=gen)[:max(1, B * 3 // 4)]
    bucket = used[torch.randint(0, len(used), (n,), generator=gen)]
    bucket = bucket.to(torch.int32).cuda()
    outs = [
        torch.randint(-2 ** 62, 2 ** 62, (n,), dtype=torch.int64,
                      generator=gen),
        torch.randint(-128, 127, (n,), dtype=torch.int8, generator=gen),
        torch.rand((n, 2), dtype=torch.float64, generator=gen),
        torch.randint(-128, 127, (n, 25), dtype=torch.int8, generator=gen),
    ]
    outs = [t.cuda() for t in outs]
    counts_t = torch.bincount(bucket, minlength=B)
    counts = counts_t.tolist()
    got = _partition_outputs(outs, bucket, counts, counts_t)
    assert len(got) == len(outs)
    for t, pt in zip(outs, got):
        assert pt.partition_row_counts == counts
        assert pt.tensor.dtype == t.dtype
        assert pt.tensor.shape[1:] == t.shape[1:]
        row_b = t.element_size() * (t.shape[1] if t.dim() > 1 else 1)
        _, starts, total = _layout(counts, row_b)
        assert pt.tensor.shape[0] == total
        for b, (c, s) in enumerate(zip(counts, starts)):
            assert torch.equal(pt.tensor[s // row_b:s // row_b + c],
                               t[bucket == b]), f"bucket {b}"
        flat = pt.tensor if pt.tensor.dim() > 1 else pt.tensor.unsqueeze(1)
        assert region_producer._bucket_starts(
            flat, np.asarray(counts, dtype=np.int64)).tolist() == starts


# ── Pass 2's rounds ──────────────────────────────────────────────────────

def test_a_run_that_fits_is_one_round():
    assert pass2_rounds([10, 20, 30], 16, 60 * 16) == [3]
    assert pass2_rounds([0, 0], 16, 0) == [2]
    # The key route's run holds the key range when that is fewer rows.
    assert pass2_rounds([10, 20, 30], 16, 40 * 16) == [2, 3]
    assert pass2_rounds([10, 20, 30], 16, 40 * 16, cap_rows=40) == [3]
    assert pass2_rounds([10, 20, 30], 16, 40 * 16, cap_rows=41) == [2, 3]


def test_rounds_pack_adjacent_buckets_up_to_the_budget():
    # 100-row buckets at 1 B a row against 250 B: two to a round.
    assert pass2_rounds([100] * 5, 1, 250) == [2, 4, 5]
    # A bucket over the budget is a round of its own.
    assert pass2_rounds([100, 1000, 100], 1, 500) == [1, 2, 3]
    # Empty buckets ride along: the leading ones with the first round, the
    # trailing ones with the last.
    assert pass2_rounds([0, 0, 300, 300, 0, 0], 1, 400) == [3, 6]


# ── A feed against the round ends ────────────────────────────────────────

def test_feeds_are_cut_at_round_ends():
    ends = [10, 20]
    assert _round_steps(0, 5, ends, 0) == ([("take", 0, 5)], 0)
    # A round that ends at the feed's end closes in it.
    assert _round_steps(5, 5, ends, 0) == (
        [("take", 0, 5), ("close", 0)], 1)
    assert _round_steps(10, 10, ends, 1) == (
        [("take", 0, 10), ("close", 1)], 2)
    # Several rounds close within one feed.
    assert _round_steps(0, 10, [2, 4, 6, 10], 0) == (
        [("take", 0, 2), ("close", 0), ("take", 2, 4), ("close", 1),
         ("take", 4, 6), ("close", 2), ("take", 6, 10), ("close", 3)], 4)
    # A round spanning feeds: neither closes.
    assert _round_steps(3, 4, [100], 0) == ([("take", 0, 4)], 0)
    # Empty rounds: leading ones close before the first take, the rest
    # where they stand.
    assert _round_steps(0, 5, [0, 0, 5, 5, 5], 0) == (
        [("close", 0), ("close", 1), ("take", 0, 5), ("close", 2),
         ("close", 3), ("close", 4)], 5)
    # An empty feed takes nothing.
    assert _round_steps(7, 0, [10], 0) == ([], 0)
    assert _round_steps(0, 0, [0, 3], 0) == ([("close", 0)], 1)


def test_a_feed_past_the_last_round_fails():
    with pytest.raises(AssertionError, match="past the last round"):
        _round_steps(8, 4, [10], 0)


def test_every_round_gets_its_rows_then_closes():
    """Random rounds (some empty) and feeds (some empty): every row goes to
    the round it lies in, in order, and every round closes once, in order,
    after its last row and before the next round's first."""
    for seed in range(20):
        _check_rounds_and_feeds(np.random.default_rng(seed))


def _check_rounds_and_feeds(rng):
    sizes = rng.integers(0, 50, int(rng.integers(1, 12)))
    sizes[rng.random(len(sizes)) < 0.3] = 0
    ends = np.cumsum(sizes).tolist()
    total = ends[-1]
    cuts = np.sort(rng.integers(0, total + 1, int(rng.integers(0, 8))))
    bounds = [0] + cuts.tolist() + [total]
    got = [[] for _ in ends]
    closed = []
    k = 0
    for lo, hi in zip(bounds, bounds[1:]):
        steps, k = _round_steps(lo, hi - lo, ends, k)
        for step in steps:
            if step[0] == "close":
                closed.append(step[1])
                continue
            a, b = step[1], step[2]
            assert 0 <= a < b <= hi - lo
            # Into the first round not yet closed.
            got[len(closed)].extend(range(lo + a, lo + b))
    assert closed == list(range(len(ends))) and k == len(ends)
    for j, (a, e) in enumerate(zip([0] + ends[:-1], ends)):
        assert got[j] == list(range(a, e))


# ── Run bytes per row ────────────────────────────────────────────────────

def _bytes(specs):
    return FixedColumnSlab.row_bytes(specs)


@pytest.mark.parametrize("keys,values", [
    # Q1: two 1-byte codes; four sums, three AVGs' (sum, count) states and
    # a count.
    ([(torch.int8, 1), (torch.int8, 1)],
     [(torch.float64, 0)] * 4 + [(torch.float64, 2)] * 3
     + [(torch.int64, 0)]),
    # Q10: int64 key, 25-byte name, decimal, 15-byte phone, nation, address
    # and comment strings; a revenue sum.
    ([(torch.int64, 1), (torch.int8, 25), (torch.float64, 1),
      (torch.int8, 15), (torch.int8, 25), (torch.int8, 40),
      (torch.int8, 117)], [(torch.float64, 0)]),
    # An int32 key and two AVG calls in one partial state (4 fields).
    ([(torch.int32, 1)], [(torch.float64, 4)]),
    # Q8: an int64 year and a ratio of two sums (2 fields).
    ([(torch.int64, 1)], [(torch.float64, 2)]),
])
def test_the_predicted_run_row_is_the_runs(keys, values):
    """The prediction, from the emitted keys' ``(dtype, width)`` and the
    values' ``(dtype, state width)`` before pass 1, is the row of the run
    pass 2 builds from the same columns: a key ``width`` wide is a
    ``(width,)`` spec, a partial state ``(fields,)``.  The
    runs pass 2 actually builds are checked against the logged prediction
    in tests/operators/test_agg_partitioned_sf1.py."""
    key_specs = [(d, (w,) if w > 1 else ()) for d, w in keys]
    value_specs = [(d, (w,) if w else ()) for d, w in values]
    hash_b = _agg_run_row_b(False, keys, values)
    # The hash route's run: the hash, the emitted keys, the values.
    assert hash_b == _bytes([(torch.int64, ())] + key_specs + value_specs)
    # The packed route's run (the composite and the values) is no wider.
    assert hash_b >= _bytes([(torch.int64, ())] + value_specs)
    if len(keys) == 1 and keys[0][1] == 1:
        # The key route's run: the key and the values.
        assert _agg_run_row_b(True, keys, values) == _bytes(
            key_specs + value_specs)


def test_q1s_hash_row_and_q18s_key_row():
    # Q1: 8 + two codes + four sums, three AVG states and a count; Q18: an
    # int64 key and a sum.
    q1_values = ([(torch.float64, 0)] * 4 + [(torch.float64, 2)] * 3
                 + [(torch.int64, 0)])
    assert _agg_run_row_b(False, [(torch.int8, 1)] * 2, q1_values) == 98
    assert _agg_run_row_b(True, [(torch.int64, 1)],
                          [(torch.float64, 0)]) == 16
