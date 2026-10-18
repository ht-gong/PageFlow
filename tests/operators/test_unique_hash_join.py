"""Correctness and dispatch tests for the unique-build join pair core."""

from __future__ import annotations

import pytest
import torch

import operators.hashjoin as hashjoin
import operators.join_core as join_core
from operators.hashjoin import join_kernel_new, set_join_core_mode
from operators.join_core import unique_hash_pairs
from tests.mix64 import mix64_np


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _cuda(values):
    return torch.tensor(values, dtype=torch.int64, device="cuda")


@pytest.fixture(autouse=True)
def _legacy_mode():
    """Each test starts in "legacy" mode; the process's mode is restored after
    it (the in-process smoke runs later in the same process)."""
    previous = hashjoin._join_core_mode
    set_join_core_mode("legacy")
    try:
        yield
    finally:
        set_join_core_mode(previous)


@pytest.fixture
def hash_calls(monkeypatch):
    """One ``(key count, build rows)`` per unique-hash core call: the join
    took the fast path once per entry, building from the side with that many
    rows."""
    calls = []
    real = join_core.unique_hash_pairs

    def counted(build_keys, probe_keys):
        calls.append((len(build_keys), build_keys[0].numel()))
        return real(build_keys, probe_keys)

    monkeypatch.setattr(join_core, "unique_hash_pairs", counted)
    return calls


@pytest.fixture
def fallbacks(monkeypatch):
    """The reason of every refusal of the unique-hash core."""
    reasons = []

    def record(reason):
        reasons.append(reason)
        return None

    monkeypatch.setattr(hashjoin, "_join_core_fallback", record)
    return reasons


@pytest.fixture
def residual_evals(monkeypatch):
    """Calls of the residual-condition evaluator after the fast path (each
    one is a dynamic compaction)."""
    calls = []
    real = hashjoin.evaluate_join

    def counted(*a, **k):
        calls.append(a[3])
        return real(*a, **k)

    monkeypatch.setattr(hashjoin, "evaluate_join", counted)
    return calls


def _first_slots(keys, cap):
    """join_core.cu's slot_of: multiply-shift of mix64's bits below its top
    9 onto [0, cap)."""
    return [((int(h) << 9) % (1 << 64)) * cap >> 64 for h in mix64_np(keys)]


def test_single_key_pairs_are_stable_and_collision_safe():
    # Find six distinct keys with the same initial slot in the table of six
    # build rows (12 slots) so exact key comparison, rather than fingerprint
    # equality, decides every probe.
    candidates = torch.arange(1, 2048, dtype=torch.int64)
    slots = _first_slots(candidates.numpy(), 12)
    bucket = next(slot for slot in range(12) if slots.count(slot) >= 6)
    colliders = [int(candidates[i]) for i, slot in enumerate(slots) if slot == bucket][:6]

    build = _cuda(colliders)
    probe_values = [colliders[5], -999, colliders[0], colliders[3], colliders[0]]
    probe = _cuda(probe_values)
    build_idx, probe_idx = unique_hash_pairs([build], [probe])

    assert probe_idx.cpu().tolist() == [0, 2, 3, 4]
    assert build_idx.cpu().tolist() == [5, 0, 3, 0]

    # Stable compaction must produce byte-identical index order repeatedly.
    build_idx2, probe_idx2 = unique_hash_pairs([build], [probe])
    assert torch.equal(build_idx, build_idx2)
    assert torch.equal(probe_idx, probe_idx2)


def test_two_column_tuple_comparison_and_repeated_probe_keys():
    build0 = _cuda([1, 1, 2, 2])
    build1 = _cuda([7, 8, 7, 8])
    probe0 = _cuda([2, 1, 2, 3, 1, 1])
    probe1 = _cuda([8, 8, 9, 7, 7, 7])

    build_idx, probe_idx = unique_hash_pairs(
        [build0, build1], [probe0, probe1]
    )
    assert probe_idx.cpu().tolist() == [0, 1, 4, 5]
    assert build_idx.cpu().tolist() == [3, 1, 0, 0]


@pytest.mark.parametrize(
    ("build_values", "probe_values", "expected_build", "expected_probe"),
    [
        ([], [1, 2], [], []),
        ([1, 2], [], [], []),
        ([1, 2], [3, 4], [], []),
        ([1, 2, 3], [3, 2, 1], [2, 1, 0], [0, 1, 2]),
    ],
)
def test_empty_no_match_and_all_match_edges(
    build_values, probe_values, expected_build, expected_probe
):
    build_idx, probe_idx = unique_hash_pairs(
        [_cuda(build_values)], [_cuda(probe_values)]
    )
    assert build_idx.cpu().tolist() == expected_build
    assert probe_idx.cpu().tolist() == expected_probe


def test_capacity_boundaries_match_a_reference_lookup():
    for n_build in (7, 8, 9, 15, 16, 17, 31, 32, 33):
        build = torch.arange(n_build, dtype=torch.int64, device="cuda") * 17 + 3
        probe = torch.cat((build.flip(0), _cuda([-1]), build[:3]))
        build_idx, probe_idx = unique_hash_pairs(
            [build], [probe]
        )
        expected_build = list(range(n_build - 1, -1, -1)) + [0, 1, 2]
        expected_probe = list(range(n_build)) + [n_build + 1, n_build + 2, n_build + 3]
        assert build_idx.cpu().tolist() == expected_build
        assert probe_idx.cpu().tolist() == expected_probe


def _run_join(left_key, right_key, *, left_unique, right_unique,
              left_second=None, right_second=None, scratch=1 << 30,
              conditions=None):
    left = {1: left_key}
    right = {2: right_key}
    left_eq, right_eq = [1], [2]
    if left_second is not None:
        left[3] = left_second
        right[4] = right_second
        left_eq.append(3)
        right_eq.append(4)

    left_gi = torch.arange(left_key.numel(), dtype=torch.int64, device="cuda")
    right_gi = torch.arange(right_key.numel(), dtype=torch.int64, device="cuda")
    return join_kernel_new(
        left, right, 1, 2, conditions or [], [], True, "inner", left_gi, right_gi,
        equality_keys=(left_eq, right_eq),
        unique_key_sets=(left_unique, right_unique),
        hash_scratch_limit_bytes=scratch,
    )


def test_dispatch_preserves_left_right_orientation_for_both_build_roles(
        hash_calls):
    set_join_core_mode("unique_hash")

    # Proven-left build: output is stable logical-right probe order.
    result = _run_join(
        _cuda([10, 20, 30]), _cuda([30, 10, 10, 99]),
        left_unique=[[1]], right_unique=[],
    )
    assert result[0].cpu().tolist() == [2, 0, 0]
    assert result[1].cpu().tolist() == [0, 1, 2]

    # Proven-right build flip: map back to logical (left, right) orientation.
    result = _run_join(
        _cuda([5, 5, 8, 99]), _cuda([5, 8]),
        left_unique=[], right_unique=[[2]],
    )
    assert result[0].cpu().tolist() == [0, 1, 2]
    assert result[1].cpu().tolist() == [0, 0, 1]

    # Built from the left (3 rows), then from the right (2 rows).
    assert hash_calls == [(1, 3), (1, 2)]


def test_composite_dispatch_requires_and_hashes_the_complete_unique_key(
        hash_calls, residual_evals):
    set_join_core_mode("unique_hash")
    result = _run_join(
        _cuda([1, 1, 2, 2]), _cuda([2, 1, 2, 1, 3]),
        left_unique=[[1, 3]], right_unique=[],
        left_second=_cuda([7, 8, 7, 8]),
        right_second=_cuda([8, 8, 9, 7, 7]),
        conditions=[["#3", "=", "#4"]],
    )
    assert result[0].cpu().tolist() == [3, 1, 0]
    assert result[1].cpu().tolist() == [0, 1, 3]
    assert [keys for keys, _ in hash_calls] == [2]
    # The exact second equality is already guaranteed by tuple probing, so it
    # must not be evaluated again as a residual (a second dynamic nonzero).
    assert residual_evals == []


def test_both_unique_sides_choose_the_smaller_build(hash_calls):
    set_join_core_mode("unique_hash")
    result = _run_join(
        _cuda([5, 8, 9, 10]), _cuda([5, 8]),
        left_unique=[[1]], right_unique=[[2]],
    )
    assert result[0].cpu().tolist() == [0, 1]
    assert result[1].cpu().tolist() == [0, 1]
    assert hash_calls == [(1, 2)]      # the right side, 2 rows


def test_oversized_build_probe_ratio_falls_back_for_both_orientations(
        hash_calls, fallbacks):
    set_join_core_mode("unique_hash")
    large = _cuda(list(range(9)))
    small = _cuda([4])

    left_result = _run_join(
        large, small, left_unique=[[1]], right_unique=[]
    )
    assert left_result[0].cpu().tolist() == [4]
    assert left_result[1].cpu().tolist() == [0]

    right_result = _run_join(
        small, large, left_unique=[], right_unique=[[2]]
    )
    assert right_result[0].cpu().tolist() == [0]
    assert right_result[1].cpu().tolist() == [4]

    assert hash_calls == []
    assert fallbacks == ["build_probe_ratio_guard"] * 2


def test_true_residual_condition_and_materialized_columns_are_preserved(
        hash_calls, residual_evals):
    set_join_core_mode("unique_hash")
    left = {1: _cuda([1, 2, 3]), 5: _cuda([10, 20, 30])}
    right = {2: _cuda([1, 2, 3]), 6: _cuda([15, 15, 35])}
    left_gi = _cuda([100, 101, 102])
    right_gi = _cuda([200, 201, 202])

    result = join_kernel_new(
        left, right, 1, 2, [["#5", ">", "#6"]], [5, 6], True,
        "inner", left_gi, right_gi,
        equality_keys=([1], [2]),
        unique_key_sets=([[1]], []),
        hash_scratch_limit_bytes=1 << 30,
    )

    assert [tensor.cpu().tolist() for tensor in result] == [
        [20], [15], [101], [201]
    ]
    # The hash core, then the true residual evaluated once.
    assert len(hash_calls) == 1
    assert residual_evals == [["#5", ">", "#6"]]


def test_unproven_or_over_budget_calls_fall_back_to_legacy_core(fallbacks):
    set_join_core_mode("unique_hash")
    left, right = _cuda([1, 2, 3]), _cuda([3, 1, 9])

    # No proof covers the selected key.
    result = _run_join(left, right, left_unique=[], right_unique=[])
    assert result[0].cpu().tolist() == [2, 0]
    assert result[1].cpu().tolist() == [0, 1]

    # A valid proof still falls back before allocation when the bound is tiny.
    result = _run_join(
        left, right, left_unique=[[1]], right_unique=[], scratch=1
    )
    assert result[0].cpu().tolist() == [2, 0]
    assert result[1].cpu().tolist() == [0, 1]

    assert fallbacks == ["selected_keys_not_proven_unique", "scratch_guard"]
