"""Aggregation pass 2's dynamic accumulator (operators/aggregate.py
``_merge_dynamic_batch``): one batch merged into the running state, by the
keys themselves for one 1-D key column, else by the row hash.  Its
allocations are plain arena ones (a miss lends the cache region inside the
allocator, ``tests/allocator/test_arena_lend.py``), so there is no retry.

CPU only.
"""
from __future__ import annotations

import torch

from operators import aggregate


def _state(keys, vals, hash_=None):
    return {"keys": keys, "vals": vals, "hash": hash_}


def test_one_key_column_merges_by_its_keys():
    state = _state([torch.tensor([1, 3], dtype=torch.int64)],
                   [torch.tensor([10, 30], dtype=torch.int64)])
    aggregate._merge_dynamic_batch(
        state, None, [torch.tensor([3, 4], dtype=torch.int64)],
        [torch.tensor([1, 2], dtype=torch.int64)], True, ["sum"], [False])
    assert state["keys"][0].tolist() == [1, 3, 4]
    assert state["vals"][0].tolist() == [10, 31, 2]
    assert state["hash"] is None


def test_two_key_columns_merge_by_the_row_hash():
    run_k = [torch.tensor([1, 2], dtype=torch.int64),
             torch.tensor([5, 6], dtype=torch.int64)]
    tick_k = [torch.tensor([2, 4], dtype=torch.int64),
              torch.tensor([6, 8], dtype=torch.int64)]
    state = _state([k.clone() for k in run_k],
                   [torch.tensor([1.0, 2.0], dtype=torch.float64)],
                   run_k[0] * 100 + run_k[1])
    aggregate._merge_dynamic_batch(
        state, tick_k[0] * 100 + tick_k[1], tick_k,
        [torch.tensor([3.0, 4.0], dtype=torch.float64)], True, ["sum"],
        [False])
    rows = sorted(zip(state["keys"][0].tolist(), state["keys"][1].tolist(),
                      state["vals"][0].tolist()))
    assert rows == [(1, 5, 1.0), (2, 6, 5.0), (4, 8, 4.0)]
    assert state["hash"].tolist() == sorted([105, 206, 408])


def test_a_first_batch_becomes_the_state():
    state = _state([None], [None])
    aggregate._merge_dynamic_batch(
        state, None, [torch.tensor([7, 7, 2], dtype=torch.int64)],
        [torch.tensor([1, 2, 3], dtype=torch.int64)], False, ["sum"], [False])
    assert state["keys"][0].tolist() == [2, 7]
    assert state["vals"][0].tolist() == [3, 3]
