"""The partition count B of a join and of a keyed aggregation from the
partition budget (``operators/partitions.py``)."""

from __future__ import annotations

import math

import pytest

import operators.partitions as partitions
from operators.partitions import (AGG_B_MAX, B_MIN, JOIN_B_MAX,
                                  agg_partitions, max_load, partition_count)

MiB = 1 << 20


def _join_b(sides, budget_b):
    """The join's B (``hashjoin.tqp_join``'s call)."""
    return partition_count(sides, budget_b, JOIN_B_MAX)[0]


def _largest(sides, b):
    """The rule's predicted largest bucket pair at ``b``, in bytes."""
    total = 0.0
    for side in sides:
        rows, width = side[0], side[1]
        keys = min(side[2], rows) if len(side) > 2 else rows
        if rows and keys:
            total += rows / keys * max_load(keys, b) * width
    return total


def test_max_load():
    # Heavily loaded: Raab and Steger's mean plus sqrt(2 mean ln n).
    assert max_load(4096 * 128, 128) == pytest.approx(
        4096 + math.sqrt(2 * 4096 * math.log(128)))
    # Few balls: the binomial tail at about 90%.  25 nation keys put up to
    # four in the fullest of 32 bins and three of 64 or 128; 5 region keys
    # three of 4 bins (TPC-H's own region keys do: 0, 2 and 3 share one).
    assert max_load(25, 32) == 4
    assert max_load(25, 64) == max_load(25, 128) == 3
    assert max_load(5, 4) == 3
    assert max_load(1, 2) == max_load(1, 128) == 1
    assert max_load(0, 8) == 0
    # One bin holds every ball.
    assert max_load(7, 1) == 7 and max_load(1e6, 1) == 1e6
    # The two regimes agree where they meet.
    assert max_load(4095, 128) == pytest.approx(max_load(4096, 128), rel=0.05)


def test_small_join_takes_the_floor():
    # One bucket: the whole join fits the budget.
    assert B_MIN == 1
    assert _join_b([(1000, 16), (10, 16)], 512 * MiB) == B_MIN
    assert _join_b([(0, 16), (0, 24)], 512 * MiB) == B_MIN
    # Just over the budget: two.
    rows = 512 * MiB // 32 + 1
    assert _join_b([(rows, 16), (rows, 16)], 512 * MiB) == 2


def test_one_power_of_two_per_budget_multiple():
    # 100M rows x 16 B a side, 3.2 GB in all: 6x a 512 MiB budget, so 8.
    sides = [(100_000_000, 16), (100_000_000, 16)]
    assert _join_b(sides, 512 * MiB) == 8
    # Halving the budget doubles B; doubling it halves B.
    assert _join_b(sides, 256 * MiB) == 16
    assert _join_b(sides, 1024 * MiB) == 4
    # Every row its own key is the default.
    assert _join_b([(100_000_000, 16, 100_000_000)] * 2,
                   512 * MiB) == 8


def test_mean_on_the_budget_takes_the_next_power():
    # A mean bucket pair of exactly the budget leaves no room for the
    # hash's excess over the mean, so the next power of two is taken.
    rows = 4 * (512 * MiB // 16)
    assert _join_b([(rows, 16)], 512 * MiB) == 8
    assert _join_b([(rows - rows // 100, 16)], 512 * MiB) == 4


def test_cap_when_nothing_fits():
    # 7.5B rows x 16 B = 112 GiB against 512 MiB needs 224 buckets: 256.
    assert _join_b([(6_000_000_000, 16), (1_500_000_000, 16)],
                   512 * MiB) == 256
    # 30B rows, 447 GiB, need 894: the cap.
    assert JOIN_B_MAX == 512
    assert _join_b([(24_000_000_000, 16), (6_000_000_000, 16)],
                   512 * MiB) == JOIN_B_MAX


def test_few_keys_fill_whole_buckets():
    # A join on nation keys: 25 rows on the unique side, the other side's
    # rows spread over the same 25 keys.
    nation = (25, 16)
    # 100M rows, 64 MB a key: up to seven keys share the fullest of 8
    # buckets (448 MB), eleven of 4 (704 MB, over 512 MiB).
    assert _join_b([nation, (100_000_000, 16, 25)], 512 * MiB) == 8
    # 800M rows, 512 MB a key: two keys may share a bucket even at 256 and
    # 512, so nothing fits; 256 is the smallest B at that least pair.
    assert _join_b([nation, (800_000_000, 16, 25)], 512 * MiB) == 256
    # Region keys, 16M rows (256 MB) each: two a bucket fit, at 16.
    assert _join_b([(1, 16, 1), (80_000_000, 16, 5)], 512 * MiB) == 16
    # 30M rows (480 MB) each: only one a bucket fits, at 128.
    assert _join_b([(5, 16), (150_000_000, 16, 5)], 512 * MiB) == 128


@pytest.mark.parametrize("rows,budget_mb,keys", [
    (0, 512, None),               # nothing to hold
    (1, 1, None),                 # one row
    (3_000_000, 1, None),         # many buckets of a small join
    (600_000_000, 512, None),     # an SF100 join
    (600_000_000, 64, 25),        # few keys: buckets stop shrinking
    (600_000_000, 2, 10_000),     # nothing fits: the least pair
])
def test_smallest_fitting_power_of_two(rows, budget_mb, keys):
    sides = [(rows, 24), (rows // 4, 16) if keys is None
             else (rows // 4, 16, keys)]
    budget = budget_mb * MiB
    b = _join_b(sides, budget)
    assert B_MIN <= b <= JOIN_B_MAX and b & (b - 1) == 0
    powers = [1 << k for k in range(0, 10)]
    fits = [p for p in powers if _largest(sides, p) <= budget]
    if fits:
        assert b == fits[0]
    else:
        least = min(_largest(sides, p) for p in powers)
        assert b == min(p for p in powers if _largest(sides, p) == least)


@pytest.mark.parametrize("sides,budget_mb,b_max", [
    ([(100_000_000, 16), (100_000_000, 16)], 512, JOIN_B_MAX),   # fits at 8
    ([(100_000_000, 16), (100_000_000, 16)], 64, 8),             # cap binds
    ([(24_000_000_000, 16), (6_000_000_000, 16)], 512, 64),      # over at cap
    ([(25, 16), (800_000_000, 16, 25)], 512, JOIN_B_MAX),        # few keys
    ([(0, 16), (0, 24)], 1, 1),                                  # empty
    ([(24_000_000_000, 16), (6_000_000_000, 16)], 1, 256),       # tiny budget
])
def test_partition_count_returns_its_prediction_up_to_b_max(sides, budget_mb,
                                                            b_max):
    # The B returned is the rule's over [1, b_max], and the bytes returned
    # its predicted largest unit at that B.
    budget = budget_mb * MiB
    b, pred = partition_count(sides, budget, b_max)
    powers = [1 << k for k in range(b_max.bit_length())]
    assert b in powers
    assert pred == _largest(sides, b)
    fits = [p for p in powers if _largest(sides, p) <= budget]
    if fits:
        assert b == fits[0] and pred <= budget
    else:
        least = min(_largest(sides, p) for p in powers)
        assert pred == least > budget
        assert b == min(p for p in powers if _largest(sides, p) == least)


def test_partition_count_stops_at_b_max():
    # 7.5B rows, 112 GiB: 256 under the join's cap, the cap itself under a
    # lower one, with its pair over the budget.
    sides = [(6_000_000_000, 16), (1_500_000_000, 16)]
    b, pred = partition_count(sides, 512 * MiB, JOIN_B_MAX)
    assert b == 256 and pred <= 512 * MiB
    b, pred = partition_count(sides, 512 * MiB, 64)
    assert b == 64 and pred > 512 * MiB


def test_agg_run_that_fits_takes_one():
    # 1M partial rows x 16 B: 16 MB, under the budget.
    assert agg_partitions(1_000_000, 4, None, 16, None,
                          512 * MiB) == (1, 16e6, "fits")
    # Nothing to hold.
    assert agg_partitions(0, 1, None, 98, None, 512 * MiB) == (1, 0.0, "fits")
    # Just over the budget: two, as for a join.
    rows = 512 * MiB // 16
    assert agg_partitions(rows, 1, None, 16, None,
                          512 * MiB)[::2] == (1, "fits")
    assert agg_partitions(rows + 1, 1, None, 16, None,
                          512 * MiB)[::2] == (2, "budget")


def test_agg_ticks_times_groups_bound():
    # 5.92B rows over at most 6 groups, 98 B a run row, some 800 pass-1
    # ticks.  A tick emits at most one partial row per group, so pass 2
    # holds 4,800 rows, under 1 MB.
    p, t = 5_916_000_000, 800
    assert agg_partitions(p, t, 6, 98, None, 512 * MiB) == (
        1, 4_800 * 98.0, "fits")
    # With no bound on the groups every row is its own: 580 GB, over the
    # budget even at the cap.
    b, pred, why = agg_partitions(p, t, None, 98, None, 512 * MiB)
    assert (b, why) == (AGG_B_MAX, "at cap") and pred > 512 * MiB


def test_agg_key_route_caps_the_run_at_the_key_range():
    # 227M rows over at most 10M groups in a key range of 10M, 16 B a run
    # row, some 540 ticks.  The partial rows are 3.6 GB, but the key route's
    # run holds at most the key range: 10M rows, 160 MB.
    p, t, g = 227_000_000, 540, 10_000_000
    assert agg_partitions(p, t, g, 16, g, 512 * MiB) == (1, 160e6, "fits")
    # Without the cap: eight buckets of about 455 MB.
    b, pred, why = agg_partitions(p, t, g, 16, None, 512 * MiB)
    assert (b, why) == (8, "budget") and 450e6 < pred <= 512 * MiB


def test_agg_key_range_is_not_the_groups():
    # 120M rows over at most 30M groups, 16 B a run row, 8 ticks.  30M rows
    # (480 MB) would fit, but pass 2 caps the run at the key range, here
    # four times the groups: 1.9 GB, four buckets.
    p, t, g = 120_000_000, 8, 30_000_000
    assert agg_partitions(p, t, g, 16, g, 512 * MiB)[::2] == (1, "fits")
    b, pred, why = agg_partitions(p, t, g, 16, 4 * g, 512 * MiB)
    assert (b, why) == (4, "budget") and 480e6 < pred <= 512 * MiB


def test_agg_at_the_cap():
    # 6.0B rows over at most 1.5B groups in a key range of 6.0B, 16 B a run
    # row, 358 ticks.  256 buckets of about 376 MB fit 512 MiB, not 64 MiB.
    shape = (6_001_215_000, 358, 1_500_000_000, 16, 6_000_000_000)
    b, pred, why = agg_partitions(*shape, 512 * MiB)
    assert (b, why) == (256, "budget") and 370e6 < pred < 380e6
    b, pred, why = agg_partitions(*shape, 64 * MiB)
    assert (b, why) == (AGG_B_MAX, "at cap") and pred == pytest.approx(
        375.6e6, rel=1e-3)


def test_agg_cap_is_read_at_call_time(monkeypatch):
    # A lower AGG_B_MAX holds the shape above at the new cap, over the
    # budget.
    monkeypatch.setattr(partitions, "AGG_B_MAX", 64)
    b, pred, why = agg_partitions(6_001_215_000, 358, 1_500_000_000, 16,
                                  6_000_000_000, 512 * MiB)
    assert (b, why) == (64, "at cap") and pred > 512 * MiB


@pytest.mark.parametrize("rows,ticks,groups,cap_rows,row_b", [
    (0, 1, None, None, 16),                          # nothing to hold
    (1, 1, None, None, 98),                          # one row
    (150_000, 1, 6, None, 98),                       # few groups
    (6_000_000, 23, 10_000_000, 25, 16),             # a tiny key range
    (600_000_000, 36, None, None, 254),              # wide rows, no bound
    (600_000_000, 36, 25 * 2557, None, 49),          # ticks x groups binds
    (6_001_215_000, 358, 1_500_000_000, 6_000_000_000, 16),  # Q18, SF1000
    (6_001_215_000, 358, None, 10_000_000, 16),      # the cap at B = 1 only
])
def test_agg_b_properties(rows, ticks, groups, cap_rows, row_b):
    last = None
    for budget_mb in (1, 2, 64, 512, 1024, 64 * 1024):
        budget = budget_mb * MiB
        b, pred, why = agg_partitions(rows, ticks, groups, row_b, cap_rows,
                                      budget)
        # A power of two up to the cap, never rising with the budget.
        assert B_MIN <= b <= AGG_B_MAX and b & (b - 1) == 0
        assert last is None or b <= last
        last = b
        assert (why == "at cap") == (pred > budget)
        g = rows if groups is None else min(groups, rows)
        r1 = min(rows, ticks * g)
        if why == "fits":
            held = r1 if cap_rows is None else min(r1, cap_rows)
            assert b == 1 and pred == held * row_b
        else:
            # The search over the partial rows, which never takes B = 1
            # within the budget once the B = 1 test has failed.
            assert (b, pred) == partition_count([(r1, row_b, g)], budget,
                                                AGG_B_MAX)
            assert why == "at cap" or b > 1


# Each keyed aggregation of the 22 queries and its B at SF1, SF100 and
# SF1000 (256 MiB chunks, the 512 MiB default budget) and in the e2e ladder
# (SF1, 4 MiB chunks, a 2 MiB budget).  The facts are SF1's, as
# operators/aggregate.py passes them to the rule (main.py --SF 1) and logs
# them in its [agg_b] line, which tests/operators/test_agg_partitioned_sf1.py
# checks against these cells at the SF1 and e2e geometries: P, the rows past
# the mask's selectivity; the bytes pass 1 reads, its pre-mask rows times its
# input row bytes; per GROUP BY key its dtype's bound (a string, int8
# [rows, width], 256 values a byte), None for a wider key; w and the route.
# Rows scale with the scale factor; the ticks are those bytes over the
# chunk.  No key-route key is narrow, so no run is capped by a key range.
# Q17 (dense) takes 1 without the rule.
GEOMETRIES = {"SF1": (1, 256, 512), "SF100": (100, 256, 512),
              "SF1000": (1000, 256, 512), "e2e": (1, 4, 2)}


def _str(width):
    """The values a ``width``-byte string key can take."""
    return 256 ** width


AGG_CELLS = [
    # name, query, its aggregations, P, pass-1 bytes, key bounds, w, route,
    # B per geometry
    ("Q1", 1, ["aggregation 6"], 5_916_591, 6_001_215 * 35,
     [_str(1), _str(1)], 98, "hash", (1, 1, 16, 256)),
    ("Q2", 2, ["aggregation 30"], 158_960, 158_960 * 16, [None], 16, "key",
     (1, 1, 8, 2)),
    ("Q3", 3, ["aggregation 17"], 30_519, 30_519 * 36, [None] * 3, 36,
     "hash", (1, 1, 4, 1)),
    ("Q4", 4, ["aggregation 12"], 52_523, 52_523 * 15, [_str(15)], 31,
     "hash", (1, 1, 4, 1)),
    ("Q5", 5, ["aggregation 32"], 7_243, 7_243 * 41, [_str(25)], 41, "hash",
     (1, 1, 1, 1)),
    ("Q7", 7, ["aggregation 30"], 5_924, 5_924 * 66,
     [_str(25), _str(25), None], 74, "hash", (1, 1, 1, 1)),
    # mkt_share, a ratio of two sums, is a two-field partial state: the
    # dense shape, as Q17's AVG.
    ("Q8", 8, ["aggregation 42"], 2_603, 2_603 * 41, [None], 24, "dense",
     (1, 1, 1, 1)),
    ("Q9", 9, ["aggregation 31"], 319_404, 319_404 * 41, [_str(25), None],
     49, "hash", (1, 4, 32, 8)),
    ("Q10", 10, ["aggregation 22"], 114_705, 114_705 * 254,
     [None, _str(25), None, _str(15), _str(25), _str(40), _str(117)], 254,
     "hash", (1, 8, 64, 16)),
    ("Q11", 11, ["aggregation 16"], 31_680, 31_680 * 24, [None], 16, "key",
     (1, 1, 1, 1)),
    ("Q12", 12, ["aggregation 11"], 30_988, 30_988 * 25, [_str(10)], 34,
     "hash", (1, 1, 2, 1)),
    # Its key is not emitted: the hash route.
    ("Q13a", 13, ["aggregation 10"], 1_533_923, 1_533_923 * 16, [None], 16,
     "hash", (1, 8, 64, 16)),
    ("Q13b", 13, ["aggregation 13"], 150_000, 150_000 * 8, [None], 16, "key",
     (1, 1, 8, 2)),
    # Q15's revenue view twice: without its key emitted (the hash route),
    # then with it (the key route).
    ("Q15 agg 21", 15, ["aggregation 21"], 225_954, 6_001_215 * 25, [None],
     16, "hash", (1, 1, 8, 2)),
    ("Q15 agg 8", 15, ["aggregation 8"], 225_954, 6_001_215 * 25, [None],
     16, "key", (1, 1, 8, 2)),
    ("Q16a", 16, ["aggregation 15"], 118_274, 118_274 * 51,
     [_str(10), _str(25), None, None], 59, "hash", (1, 2, 16, 4)),
    ("Q16b", 16, ["aggregation 18"], 118_250, 118_250 * 51,
     [_str(10), _str(25), None], 59, "hash", (1, 2, 16, 4)),
    ("Q17", 17, ["aggregation 13"], 6_001_215, 6_001_215 * 16, [None], 24,
     "dense", (1, 1, 1, 1)),
    ("Q18 aggs 9, 21", 18, ["aggregation 9", "aggregation 21"], 6_001_215,
     6_001_215 * 16, [None], 16, "key", (1, 32, 256, 64)),
    ("Q18 agg 31", 18, ["aggregation 31"], 399, 399 * 61,
     [_str(25), None, None, None, None], 69, "hash", (1, 1, 1, 1)),
    ("Q20", 20, ["aggregation 21"], 9_741, 9_741 * 24, [None, None], 32,
     "hash", (1, 1, 1, 1)),
    ("Q21", 21, ["aggregation 31"], 4_141, 4_141 * 25, [_str(25)], 41,
     "hash", (1, 1, 1, 1)),
    # substring(c_phone, 1, 2): two int8 bytes.
    ("Q22", 22, ["aggregation 9"], 6_384, 6_384 * 10, [_str(2)], 26, "hash",
     (1, 1, 1, 1)),
]


def agg_cell_facts(cell, geometry):
    """What ``cell``'s aggregations pass the rule at ``geometry``: P, T, G
    (None: unknown) and w, and the key route's range bound, None: no key-route
    key is narrow."""
    _, _, _, p, pass1_b, keys, w, _, _ = cell
    sf, chunk_mb, _ = GEOMETRIES[geometry]
    groups = 1
    for bound in keys:
        groups = None if groups is None or bound is None else groups * bound
    ticks = max(1, -(-pass1_b * sf // (chunk_mb * MiB)))
    return p * sf, ticks, groups, w, None


def test_agg_b_per_aggregation_and_scale():
    for cell in AGG_CELLS:
        for column, geometry in enumerate(GEOMETRIES):
            want = cell[-1][column]
            if cell[7] == "dense":
                assert want == 1, cell[0]
                continue
            b, _, _ = agg_partitions(*agg_cell_facts(cell, geometry),
                                     GEOMETRIES[geometry][2] * MiB)
            assert b == want, (cell[0], geometry, b)

