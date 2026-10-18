"""The join core's CUDA kernels (``operators/join_core.py``) and their
dispatch in ``operators/hashjoin.join_kernel_new``.

Every result is compared with a brute-force reference: the dense
build x probe comparison of keys (and residual).  Semi and anti results are
probe rows in ascending order, so they are compared exactly; inner and outer
pairs as multisets, since the sort/search core orders a probe row's matches
by an unstable sort.
"""
from __future__ import annotations

import pytest
import torch

import constants
import operators.join_core as join_core
from operators.hashjoin import _residual_summary, join_kernel_new
from tests.mix64 import bucket_of

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

_OPS = {'<>': torch.ne, '<': torch.lt, '<=': torch.le, '>': torch.gt,
        '>=': torch.ge}
_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1


def _cuda(values):
    return torch.tensor(values, dtype=torch.int64, device="cuda")


def _rand(n, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi, (n,), generator=g, dtype=torch.int64).cuda()


@pytest.fixture
def calls(monkeypatch):
    """The join_core entry points each call took, by name."""
    seen = []
    for name in ("expand", "semi_anti", "unique_hash_pairs"):
        real = getattr(join_core, name)

        def counted(*a, _name=name, _real=real, **k):
            seen.append(_name)
            return _real(*a, **k)
        monkeypatch.setattr(join_core, name, counted)
    return seen


# ------------------------------------------------------------------ expand --

def _expand_reference(sorted_ids, start, length):
    rows = torch.arange(length.numel(), device="cuda")
    cr = torch.repeat_interleave(rows, length)
    first = torch.repeat_interleave(start, length)
    run_start = torch.repeat_interleave(torch.cumsum(length, 0) - length, length)
    within = torch.arange(cr.numel(), device="cuda") - run_start
    return sorted_ids[first + within], cr


@pytest.mark.parametrize("lengths", [
    [], [0], [0, 0, 0], [1], [3], [1, 0, 2, 0, 0, 3, 0], [0, 0, 5],
    [5, 0, 0], [2, 2, 2, 2],
])
def test_expand_matches_reference_with_empty_runs(lengths):
    length = _cuda(lengths)
    n_build = max(8, sum(lengths) + 3)
    sorted_ids = torch.randperm(n_build, device="cuda")
    # Each run inside the build; runs may overlap (equal probe keys).
    start = _cuda([(3 * i) % (n_build - n) if n_build > n else 0
                   for i, n in enumerate(lengths)])
    cs, cr = join_core.expand(sorted_ids, start, length)
    ref_cs, ref_cr = _expand_reference(sorted_ids, start, length)
    assert torch.equal(cs, ref_cs) and torch.equal(cr, ref_cr)


def test_expand_random_runs():
    n_probe, n_build = 60_000, 200_000
    length = _rand(n_probe, 0, 7, 1)
    length[_rand(n_probe, 0, 10, 2) < 4] = 0
    start = torch.minimum(_rand(n_probe, 0, n_build, 3), n_build - length)
    sorted_ids = torch.randperm(n_build, device="cuda")
    cs, cr = join_core.expand(sorted_ids, start, length)
    ref_cs, ref_cr = _expand_reference(sorted_ids, start, length)
    assert torch.equal(cs, ref_cs) and torch.equal(cr, ref_cr)


# ------------------------------------------------------------- semi / anti --

def _semi_anti_reference(bk, pk, anti, bv=None, pv=None, op=None):
    hit = bk[:, None] == pk[None, :]
    if op is not None:
        hit &= _OPS[op](bv[:, None], pv[None, :])
    keep = hit.any(dim=0) if bk.numel() else torch.zeros(
        pk.numel(), dtype=torch.bool, device="cuda")
    if anti:
        keep = ~keep
    return torch.nonzero(keep).view(-1)


@pytest.mark.parametrize("anti", [False, True])
@pytest.mark.parametrize("op", [None, '<>', '<', '<=', '>', '>='])
@pytest.mark.parametrize("n_build,n_probe", [
    (0, 5), (5, 0), (1, 1), (7, 300), (2000, 1500), (40_000, 3000)])
def test_semi_anti_matches_reference(anti, op, n_build, n_probe):
    seed = n_build * 7 + n_probe
    # Duplicate-heavy build keys; probe keys half inside their range.
    bk = _rand(n_build, -50, max(n_build // 3, 1), seed)
    pk = _rand(n_probe, -60, 2 * max(n_build // 3, 1), seed + 1)
    # Few distinct values, so every operator meets ties and equal groups.
    bv = _rand(n_build, 0, 4, seed + 2)
    pv = _rand(n_probe, 0, 4, seed + 3)
    summary = (bv, pv, op) if op is not None else None
    got = join_core.semi_anti(bk, pk, anti, summary)
    assert torch.equal(got, _semi_anti_reference(bk, pk, anti, bv, pv, op))


@pytest.mark.parametrize("anti", [False, True])
def test_summary_all_true_and_all_false(anti):
    bk = _cuda([1, 1, 1, 2, 2, 3])
    pk = _cuda([1, 2, 3, 4, 1])
    same = _cuda([9] * 6)
    # Every build row of a key holds the probe row's value: <> never holds.
    got = join_core.semi_anti(bk, pk, anti, (same, _cuda([9] * 5), '<>'))
    assert got.tolist() == ([0, 1, 2, 3, 4] if anti else [])
    # Every build value differs from every probe value: <> holds for every
    # probe row whose key is in the build.
    got = join_core.semi_anti(bk, pk, anti, (same, _cuda([5] * 5), '<>'))
    assert got.tolist() == ([3] if anti else [0, 1, 2, 4])


def test_semi_anti_extreme_keys_and_null_sentinel():
    keys = [_I64_MIN + 1, -1, 0, constants.null, _I64_MAX, 42]
    bk = _cuda(keys[:4])
    pk = _cuda(keys + [_I64_MIN])
    assert join_core.semi_anti(bk, pk, False).tolist() == [0, 1, 2, 3]
    assert join_core.semi_anti(bk, pk, True).tolist() == [4, 5, 6]


def test_semi_anti_refuses_the_empty_slot_key():
    assert join_core.semi_anti(_cuda([3, _I64_MIN]), _cuda([3]), False) is None


def _one_bucket_keys(n, B=128, bucket=5, seed=0):
    """n distinct keys that all hash into one radix bucket of B: one tick's
    keys share their top mix64 bits."""
    cand = torch.randperm(n * B * 2, generator=torch.Generator().manual_seed(seed))
    cand = cand[bucket_of(cand, B) == bucket][:n]
    assert cand.numel() == n
    return cand.cuda()


def test_tables_with_keys_of_one_radix_bucket():
    keys = _one_bucket_keys(20_000)
    probe = torch.cat((keys[::3], keys[1::7] + (1 << 40)))
    got = join_core.semi_anti(keys, probe, False)
    assert torch.equal(got, _semi_anti_reference(keys, probe, False))
    build_idx, probe_idx = join_core.unique_hash_pairs([keys], [probe])
    assert torch.equal(probe[probe_idx], keys[build_idx])
    assert probe_idx.tolist() == list(range(keys[::3].numel()))


# -------------------------------------------------------------- unique hash --

@pytest.mark.parametrize("two", [False, True])
def test_unique_hash_random_tuples(two):
    n_build, n_probe = 30_000, 50_000
    k0 = torch.randperm(n_build * 4, device="cuda")[:n_build]
    build = [k0 // 3, k0 % 3] if two else [k0]
    picks = _rand(n_probe, 0, n_build, 11)
    miss = _rand(n_probe, 0, 4, 12) == 0
    p0 = torch.where(miss, k0[picks] + n_build * 8, k0[picks])
    probe = [p0 // 3, p0 % 3] if two else [p0]
    build_idx, probe_idx = join_core.unique_hash_pairs(build, probe)
    assert torch.equal(probe_idx, torch.nonzero(~miss).view(-1))
    assert torch.equal(build_idx, picks[probe_idx])


# ----------------------------------------------------------- residual match --

def test_residual_summary_matches_one_build_probe_comparison():
    left = {1: _cuda([1]), 3: _cuda([7])}
    right = {2: _cuda([1]), 4: _cuda([8])}

    def op_of(cond):
        found = _residual_summary(cond, left, right)
        if found is None:
            return None
        build, probe, op = found
        assert build is left[3] and probe is right[4]
        return op

    # Q21's lowered form, wrapped as the plan wraps it.
    assert op_of(['NOT', ['l_suppkey#3', '=', 'l_suppkey#4']]) == '<>'
    assert op_of([['NOT', [['#3'], '=', ['#4']]]]) == '<>'
    assert op_of(['#3', '<>', '#4']) == '<>'
    assert op_of(['#3', '<', '#4']) == '<'
    assert op_of(['#4', '<', '#3']) == '>'      # probe first: flipped
    assert op_of(['#4', '>=', '#3']) == '<='
    assert op_of(['NOT', ['#3', '<', '#4']]) == '>='
    assert op_of(['NOT', ['NOT', ['#3', '<>', '#4']]]) == '<>'
    # Not answered: equality, a negated <>, a literal, a cast, two columns of
    # one side, a compound condition.
    assert op_of(['#3', '=', '#4']) is None
    assert op_of(['NOT', ['#3', '<>', '#4']]) is None
    assert op_of(['#3', '<', '5']) is None
    assert op_of(['#3 as double', '<', '#4']) is None
    assert op_of(['#1', '<', '#3']) is None
    assert op_of([['#3', '<', '#4'], 'AND', ['#3', '>', '#4']]) is None


def test_residual_summary_needs_int64_columns():
    left = {1: _cuda([1]), 3: _cuda([7]).to(torch.int32)}
    right = {2: _cuda([1]), 4: _cuda([8])}
    assert _residual_summary(['#3', '<>', '#4'], left, right) is None


def test_residual_summary_refuses_an_id_on_both_sides():
    # evaluate_join would read #3 from the left side for both operands.
    left = {1: _cuda([1]), 3: _cuda([7])}
    right = {2: _cuda([1]), 3: _cuda([8]), 4: _cuda([8])}
    assert _residual_summary(['#4', '<>', '#3'], left, right) is None


# ------------------------------------------------------------ join dispatch --

def _tick(keys, B):
    """A tick's partition-major layout: rows grouped by radix bucket (stable)
    and the per-bucket row counts."""
    bucket = bucket_of(keys, B)
    order = torch.argsort(bucket, stable=True)
    counts = torch.bincount(bucket, minlength=B).tolist()
    return order, counts


def _join(join_type, lk, rk, lv=None, rv=None, condition=None, B=None):
    """Run one core tick; returns (left gidx, right gidx) of the output."""
    lgi = torch.arange(lk.numel(), device="cuda") + 1000
    rgi = torch.arange(rk.numel(), device="cuda") + 5000
    bucket_counts = None
    if B is not None:
        lo, lc = _tick(lk, B)
        ro, rc = _tick(rk, B)
        lk, lgi, rk, rgi = lk[lo], lgi[lo], rk[ro], rgi[ro]
        if lv is not None:
            lv, rv = lv[lo], rv[ro]
        bucket_counts = (lc, rc)
    left, right = {1: lk}, {2: rk}
    if lv is not None:
        left[3], right[4] = lv, rv
    out = join_kernel_new(
        left, right, 1, 2, [condition] if condition else [], [], True,
        join_type, lgi, rgi, bucket_counts=bucket_counts,
        equality_keys=([1], [2]),
        unique_key_sets=([], []),
        hash_scratch_limit_bytes=1 << 30)
    return out[0], out[1], (lk, rk, lv, rv, lgi, rgi)


def _join_reference(join_type, op, data):
    lk, rk, lv, rv, lgi, rgi = data
    hit = lk[:, None] == rk[None, :]
    if op is not None:
        hit &= _OPS[op](lv[:, None], rv[None, :])
    if join_type in ('right-semi', 'right-anti'):
        any_hit = hit.any(dim=0) if lk.numel() else torch.zeros(
            rk.numel(), dtype=torch.bool, device="cuda")
        keep = any_hit if join_type == 'right-semi' else ~any_hit
        return [], rgi[keep].tolist()
    li, ri = torch.nonzero(hit, as_tuple=True)
    pairs = list(zip(lgi[li].tolist(), rgi[ri].tolist()))
    if join_type == 'right-outer':
        matched = hit.any(dim=0) if lk.numel() else torch.zeros(
            rk.numel(), dtype=torch.bool, device="cuda")
        # Every unmatched right row with a null left side, also when
        # nothing matched.
        pairs += [(constants.null, r) for r in rgi[~matched].tolist()]
    return sorted(pairs)


_CASES = [
    # (join type, residual op, B)
    ('inner', None, None), ('inner', None, 16), ('inner', '<>', 16),
    ('inner', '<', None),
    ('right-semi', None, None), ('right-semi', None, 16),
    ('right-semi', '<>', 16), ('right-semi', '>=', 8),
    ('right-anti', None, 16), ('right-anti', '<>', 16),
    ('right-anti', '<=', None),
    ('right-outer', None, 16), ('right-outer', None, None),
]


def _check(join_type, op, got_l, got_r, data):
    ref = _join_reference(join_type, op, data)
    if join_type in ('right-semi', 'right-anti'):
        assert got_l.numel() == 0
        assert got_r.tolist() == ref[1]
    else:
        assert sorted(zip(got_l.tolist(), got_r.tolist())) == ref


@pytest.mark.parametrize("join_type,op,B", _CASES)
@pytest.mark.parametrize("n_left,n_right", [(0, 9), (9, 0), (1, 1), (1500, 900)])
def test_join_kernel_matches_reference(join_type, op, B, n_left, n_right):
    seed = n_left + 3 * n_right
    lk = _rand(n_left, 0, max(n_left // 4, 2), seed)
    rk = _rand(n_right, 0, max(n_left // 2, 3), seed + 1)
    lv = _rand(n_left, 0, 3, seed + 2)
    rv = _rand(n_right, 0, 3, seed + 3)
    cond = ['#3', op, '#4'] if op is not None else None
    got_l, got_r, data = _join(join_type, lk, rk, lv, rv, cond, B)
    _check(join_type, op, got_l, got_r, data)


# Q21's two residuals as the plan lowers them.
_Q21_RESIDUAL = ['NOT', ['l_suppkey#3', '=', 'l_suppkey#4']]


@pytest.mark.parametrize("join_type", ['right-semi', 'right-anti'])
def test_semi_anti_take_the_hash_set(calls, join_type):
    lk, rk = _rand(500, 0, 100, 41), _rand(400, 0, 200, 42)
    lv, rv = _rand(500, 0, 3, 43), _rand(400, 0, 3, 44)
    got_l, got_r, data = _join(join_type, lk, rk, B=16)
    _check(join_type, None, got_l, got_r, data)
    got_l, got_r, data = _join(join_type, lk, rk, lv, rv, _Q21_RESIDUAL,
                               B=16)
    _check(join_type, '<>', got_l, got_r, data)
    assert calls == ['semi_anti', 'semi_anti']


def _equal_values_reference(join_type, data):
    lk, rk, lv, rv, _, rgi = data
    hit = ((lk[:, None] == rk[None, :])
           & (lv[:, None] == rv[None, :])).any(dim=0)
    return rgi[hit if join_type == 'right-semi' else ~hit].tolist()


@pytest.mark.parametrize("join_type", ['right-semi', 'right-anti'])
def test_unanswerable_residual_takes_the_pairs(calls, join_type):
    lk, rk = _rand(500, 0, 100, 51), _rand(400, 0, 200, 52)
    lv, rv = _rand(500, 0, 3, 53), _rand(400, 0, 3, 54)
    # Equality is not summarized (a second join key arrives this way).
    got_l, got_r, data = _join(join_type, lk, rk, lv, rv, ['#3', '=', '#4'],
                               B=16)
    assert calls == ['expand']
    assert got_l.numel() == 0
    assert got_r.tolist() == _equal_values_reference(join_type, data)


@pytest.mark.parametrize("join_type", ['right-semi', 'right-anti'])
@pytest.mark.parametrize("residual", [False, True])
def test_empty_slot_build_key_takes_the_pairs(calls, join_type, residual):
    lk = torch.cat((_rand(300, 0, 60, 71), _cuda([_I64_MIN, _I64_MIN])))
    rk = torch.cat((_rand(200, 0, 120, 72), _cuda([_I64_MIN])))
    lv, rv = _rand(302, 0, 3, 73), _rand(201, 0, 3, 74)
    cond = _Q21_RESIDUAL if residual else None
    got_l, got_r, data = _join(join_type, lk, rk, lv, rv, cond)
    assert calls == ['semi_anti', 'expand']
    _check(join_type, '<>' if residual else None, got_l, got_r, data)


def test_inner_join_without_proof_takes_the_pairs(calls):
    _join('inner', _rand(300, 0, 50, 61), _rand(200, 0, 80, 62), B=8)
    assert calls == ['expand']
