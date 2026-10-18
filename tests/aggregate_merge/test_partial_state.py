"""Partial states of aggregate results (``operators/aggregate.py``).

A result whose per-chunk values cannot be merged by one op is carried as a
partial state: one SUM or COUNT field per call, two per AVG call, every
field merged by SUM and the expression evaluated once over the merged
fields.  Before it, a ratio of two sums (Q14's ``promo_revenue``, Q8's
``mkt_share``) was evaluated per chunk and its chunks' ratios added up.

* Which results are states (``_state_calls``), on the plans' own trees:
  an expression with an AVG call or two or more aggregate calls; one call
  alone, or scaled by a constant, merges by its op; a MIN or MAX among
  several calls raises.
* ``_agg_run_row_b`` counts a state's fields.
* A global aggregation over several chunks (``GlobalAggregate``, which both
  engine modes run) gives the one-chunk answer: a ratio of sums, a ratio
  over an AVG and a COUNT, and a bare SUM next to them.
* The same partials merged per group (``_reduce_partial_values``, pass 2's
  dynamic accumulator) and finalized give each group's ratio.

The SF1 runs of Q8 and Q14 over several aggregation ticks are in
``tests/operators/test_agg_partial_state_sf1.py``.
"""
from __future__ import annotations

import pytest
import torch

from operators.aggregate import (GlobalAggregate, _agg_run_row_b,
                                 _reduce_partial_values, _state_calls,
                                 _state_width)

# The plans' result expressions (``queries/pickles``), the ``AS`` alias cut
# as ``AggregateResults`` cuts it.
Q14 = [['100.0', '*', [['sum', ['CASE', 'WHEN', ['StartsWith', ['p_type#13', 'PROMO']], 'THEN', ['l_extendedprice#97', '*', ['1.0', '-', 'l_discount#98']], 'ELSE', '0.0', 'END']], '#', '123']], '/', [['sum', ['l_extendedprice#97', '*', ['1.0', '-', 'l_discount#98']]], '#', '124']]
Q8 = [[['sum', ['CASE', 'WHEN', ['nation#124', '=', 'BRAZIL'], 'THEN', 'volume#123', 'ELSE', '0.0', 'END']], '#', '131'], '/', [['sum', 'volume#123'], '#', '130']]
Q17_SUB = ['0.2', '*', [['avg', ['l_quantity#130']], '#', '124']]
Q17_MAIN = [[['sum', ['l_extendedprice#97']], '#', '142'], '/', '7.0']
Q1_SUM = [['sum', ['l_quantity#96']], '#', '131L']
Q1_AVG = [['avg', ['l_quantity#96']], '#', '133']
Q1_COUNT = [['count', ['1']], '#', '130L']
Q15_MAX = [['max', ['total_revenue#127']], '#', '128']


def test_ratios_of_sums_are_states():
    assert [fn for fn, _ in _state_calls(Q14)] == ["sum", "sum"]
    assert [fn for fn, _ in _state_calls(Q8)] == ["sum", "sum"]
    # The calls' arguments, in evaluation order.
    assert _state_calls(Q8)[1] == ("sum", "volume#123")
    assert _state_width(_state_calls(Q14)) == 2


def test_an_avg_is_a_state_of_two_fields():
    assert _state_calls(Q17_SUB) == [("avg", "l_quantity#130")]
    assert _state_width(_state_calls(Q17_SUB)) == 2
    assert _state_width(_state_calls(Q1_AVG)) == 2


def test_one_call_merges_by_its_op():
    for expr in (Q1_SUM, Q1_COUNT, Q15_MAX, Q17_MAIN):
        assert _state_calls(expr) == []


def test_min_or_max_among_several_calls_raises():
    spread = [[['max', 'x#1'], '#', '10'], '-', [['min', 'x#1'], '#', '11']]
    with pytest.raises(NotImplementedError, match="max, min"):
        _state_calls(spread)
    with pytest.raises(NotImplementedError, match="max"):
        _state_calls([[['avg', 'x#1'], '#', '10'], '-',
                      [['max', 'x#1'], '#', '11']])


def test_the_run_row_counts_a_states_fields():
    # The hash route: the hash, an int32 key, a three-field float64 state
    # and a plain int64 value.
    assert _agg_run_row_b(
        False, [(torch.int32, 1)],
        [(torch.float64, 3), (torch.int64, 0)]) == 8 + 4 + 3 * 8 + 8


RATIO = [[[['sum', ['a#1', '*', 'b#2']], '#', '10'], '/',
          [['sum', 'b#2'], '#', '11']], 'AS', 'ratio#20']
MEAN = [[[['avg', 'a#1'], '#', '12'], '*',
         [['count', '1'], '#', '13']], 'AS', 'avg_times_count#21']
TOTAL = [[['sum', 'b#2'], '#', '14'], 'AS', 'total#22']


@pytest.fixture
def chunks(cuda_allocator):
    if not cuda_allocator:
        pytest.skip("needs the GPU")
    gen = torch.Generator().manual_seed(7)
    a = torch.rand(10_000, generator=gen, dtype=torch.float64)
    b = torch.rand(10_000, generator=gen, dtype=torch.float64)
    return a.cuda(), b.cuda()


def _global(results):
    args = {"results": results,
            "out agg ids": [20, 21, 22][:len(results)],
            "agg ops": ["sum", "avg", "sum"][:len(results)]}
    return GlobalAggregate(args, [1, 2], [torch.float64, torch.float64])


@pytest.mark.parametrize("n_chunks", [1, 2, 7])
def test_a_global_ratio_over_chunks_is_the_whole_ratio(chunks, n_chunks):
    a, b = chunks
    spec = _global([RATIO, MEAN, TOTAL])
    assert spec.state == [True, True, False]
    results = None
    for a_c, b_c in zip(a.chunk(n_chunks), b.chunk(n_chunks)):
        partials = spec.partials("cuda", {1: a_c, 2: b_c})
        if results is None:
            results = partials
        else:
            spec.merge(results, partials)
    ratio, mean_times_count, total = spec.finalize(results)
    torch.testing.assert_close(ratio, ((a * b).sum() / b.sum()).reshape(1))
    torch.testing.assert_close(
        mean_times_count, (a.mean() * a.numel()).reshape(1))
    torch.testing.assert_close(total, b.sum().reshape(1))


def test_a_keyed_ratio_merged_per_group_is_each_groups_ratio(chunks):
    a, b = chunks
    groups = torch.arange(a.numel(), device="cuda") % 5
    spec = _global([RATIO])
    # Pass 1 per chunk (each chunk's groups 0..4), pass 2 one merge into
    # the five groups, then the finalize.
    parts, gids = [], []
    for a_c, b_c, g_c in zip(a.chunk(3), b.chunk(3), groups.chunk(3)):
        uniq, inv = torch.unique(g_c, return_inverse=True)
        parts.append(spec.evaluate("cuda", {1: a_c, 2: b_c}, a_c.numel(),
                                   uniq.numel(), inv)[0])
        gids.append(uniq)
    merged = _reduce_partial_values(
        [torch.cat(parts)], ["sum"], spec.state, torch.cat(gids), 5)
    (ratio,) = spec.finalize(merged)
    want = torch.stack([(a[groups == g] * b[groups == g]).sum()
                        / b[groups == g].sum() for g in range(5)])
    torch.testing.assert_close(ratio, want)
