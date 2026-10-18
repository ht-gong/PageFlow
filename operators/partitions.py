"""The hash-partition count B of a join or a keyed aggregation, from the
partition budget (``--partition_budget_mb``).

B is a power of two (at 1 every row is in bucket 0).  An operator holds one
*unit* whole: a join a bucket pair, a keyed aggregation a pass-2 round of
adjacent buckets.  Each takes the smallest B whose largest unit is predicted
to fit the budget, then packs whole measured buckets.  The prediction spreads
an input's rows over an estimate of its distinct keys, all of a key's rows in
one bucket, with the fullest bucket's keys at about 90% (``max_load``).  Skew
among the keys (one key much heavier than the rest) is not predicted.

Stdlib only: the rule is tested without a GPU
(``tests/operators/test_partitions.py``).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

B_MIN = 1
# The partition kernel's limit (operators/kernels/partition_sort.cu, BMAX);
# operators/partition_sort.py checks B against it.
JOIN_B_MAX = 512
# An aggregation's pass 1 writes B streams per output.  Read as
# partitions.AGG_B_MAX at call time.
AGG_B_MAX = 256


def max_load(m: float, n: int) -> float:
    """The fullest bin's load, at about 90%, of ``m`` equal balls thrown into
    ``n`` bins.  Heavily loaded (m/n >= 32): Raab and Steger's m/n +
    sqrt(2 (m/n) ln n).  Else the largest k that a tenth of a bin is expected
    to reach, from the binomial tail: n P(X >= k) >= 0.1, X ~ Binomial(m,
    1/n); by the union bound the fullest bin exceeds it with probability
    under 0.1.  The two agree where they meet (m/n = 32)."""
    if m <= 0:
        return 0.0
    if n <= 1:
        return float(m)
    mean = m / n
    if mean >= 32:
        return mean + math.sqrt(2.0 * mean * math.log(n))
    m = int(math.ceil(m))
    p = 1.0 / n
    pmf, below = (1.0 - p) ** m, 0.0         # P(X = k - 1), P(X <= k - 2)
    for k in range(1, m + 1):
        below += pmf                         # P(X <= k - 1)
        if n * (1.0 - below) < 0.1:          # under a tenth of a bin reaches k
            return float(k - 1)
        pmf *= (m - k + 1) / k * p / (1.0 - p)
    return float(m)


def partition_count(parts: Sequence[tuple], budget_b: int,
                    b_max: int) -> tuple[int, float]:
    """The smallest power of two B in [B_MIN, ``b_max``] whose largest unit
    is predicted to fit ``budget_b`` bytes, and that unit's predicted bytes.
    When none does, the smallest B with the least predicted unit, and that
    unit's bytes: ``b_max`` for many keys, less for few keys, whose buckets
    stop shrinking once every bucket holds at most a key or two.

    ``parts`` holds one ``(rows, row_bytes[, keys])`` per input a unit holds
    a bucket of (a join's two sides, an aggregation's partial rows): its
    rows, its bytes per row and an estimate of its distinct keys (default:
    every row distinct).  All of a key's rows land in one bucket, so a part's
    largest bucket is predicted as rows/keys rows per key times the fullest
    bucket's number of keys (``max_load``), and a unit as the sum of its
    parts' largest buckets: with keys = rows at SF100 and above, rows/B plus
    far below 1%.
    """
    assert b_max >= B_MIN and b_max & (b_max - 1) == 0, b_max
    best_b, best = B_MIN, math.inf
    for b in (1 << k for k in range(B_MIN.bit_length() - 1,
                                     b_max.bit_length())):
        largest = 0.0
        for part in parts:
            rows, row_bytes = part[0], part[1]
            keys = min(part[2], rows) if len(part) > 2 else rows
            if rows > 0 and keys > 0:
                largest += rows / keys * max_load(keys, b) * row_bytes
        if largest <= budget_b:
            return b, largest
        if largest < best:
            best_b, best = b, largest
    return best_b, best


def agg_partitions(rows: int, ticks: int, groups: Optional[int], row_b: int,
                   cap_rows: Optional[int],
                   budget_b: int) -> tuple[int, float, str]:
    """A keyed aggregation's B, its largest pass-2 round's predicted bytes
    and the reason: ``fits`` (B = 1, the whole run fits the budget),
    ``budget`` (the smallest B whose largest bucket fits) or ``at cap`` (no
    B up to AGG_B_MAX fits; the least predicted round's B).

    ``rows`` is P, the aggregation's input rows times its mask's recorded
    selectivity; ``ticks`` T, pass 1's predicted ticks; ``groups`` G, a
    static upper bound on its groups (None: unknown, every row its own
    group); ``row_b`` w, pass 2's run bytes per row; ``cap_rows`` a bound on
    the key route's run besides its partial rows: the range of its one key's
    values (largest minus smallest plus one), which pass 2 caps that run at
    (None: the hash route, or a key with no bound on its range).  A tick
    emits at most one partial row per group, so pass 2 holds R1 = min(P, T x
    G) partial rows over at most G keys, and B = 1 is tested against
    min(R1, ``cap_rows``) rows.  The range is not G: a key's values can be
    sparse in it.  For B > 1 the cap cannot bind before a bucket's rows
    fit: ``cap_rows`` rows are over the budget, so a bucket that fits holds
    fewer.  The search takes R1 alone and agrees with that test.
    """
    g = rows if groups is None else min(groups, rows)
    r1 = min(rows, ticks * g)
    held = r1 if cap_rows is None else min(r1, cap_rows)
    if held * row_b <= budget_b:
        return 1, float(held * row_b), "fits"
    b, pred = partition_count([(r1, row_b, g)], budget_b, AGG_B_MAX)
    return b, pred, "budget" if pred <= budget_b else "at cap"
