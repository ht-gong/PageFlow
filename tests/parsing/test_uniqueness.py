"""CPU-only checks for uniqueness in the static plan pipeline."""

from __future__ import annotations

import pickle
from pathlib import Path

from parsing import prepare_query_plans


_ROOT = Path(__file__).resolve().parents[2]


def _all_plans():
    for qid in range(1, 23):
        with (_ROOT / "queries" / "pickles" / f"planobj-q{qid}.pkl").open("rb") as f:
            plans = pickle.load(f)
        prepare_query_plans(plans, qid)
        yield qid, plans


def _covers(keys, unique_sets):
    keys = set(keys)
    return any(set(unique_key) <= keys for unique_key in unique_sets)


def test_all_inner_joins_have_a_relationally_proven_unique_side():
    counts = {"left_only": 0, "right_only": 0, "both": 0, "neither": 0}
    total = 0
    for _qid, plans in _all_plans():
        for plan in plans:
            for op_id in plan.ops_idx:
                args = plan.ops_dict[op_id]
                if args["name"] != "Join" or args["join type"] != "Inner":
                    continue
                total += 1
                left = _covers(args["left keys"], args["left unique keys"])
                right = _covers(args["right keys"], args["right unique keys"])
                bucket = (
                    "both" if left and right
                    else "left_only" if left
                    else "right_only" if right
                    else "neither"
                )
                counts[bucket] += 1

    assert total == 53
    assert counts == {
        "left_only": 42,
        "right_only": 9,
        "both": 2,
        "neither": 0,
    }


def test_composite_only_build_keys_are_not_misproved_as_singletons():
    indexed = {}
    for qid, plans in _all_plans():
        for plan_index, plan in enumerate(plans):
            for op_id in plan.ops_idx:
                args = plan.ops_dict[op_id]
                if args["name"] == "Join":
                    indexed[(qid, plan_index, op_id)] = args

    q9 = indexed[(9, 0, 22)]
    assert [37, 38] in q9["left unique keys"]
    assert [37] not in q9["left unique keys"]
    assert [38] not in q9["left unique keys"]

    q20 = indexed[(20, 0, 23)]
    assert [37, 38] in q20["left unique keys"]
    assert [93, 94] in q20["right unique keys"]
    assert [37] not in q20["left unique keys"]
    assert [93] not in q20["right unique keys"]
