"""The menu-based compression controller.

CPU-only.  The decision-universe and pick-vector tests sweep the 22 plans'
in-memory graphs with synthetic rep presence (conftest); the benefit
ranking is checked on synthetic (raw_bytes, stored_bytes) menus.
"""
import pytest

from compression.format_policy import CompressionPolicy
from adaptive.compression_controller import CompressionController, RAW
from adaptive.decision_menu import (
    ColumnChoices,
    DecisionMenu,
    build_input_choices,
    validate_pick_vector,
)
from tests.adaptive.conftest import build_plan_input_menus
from tests.dataflow.plan_graphs import query_dataflow as _plan


# --------------------------------------------------------------------------- #
# Decision universe from menus
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qid", list(range(1, 23)))
def test_two_entry_ranges_exactly_where_both_reps_exist(qid):
    graph = _plan(qid)
    ap, entries = build_plan_input_menus(graph, CompressionPolicy.adaptive())
    assert entries
    for node, inst, cc in entries:
        # Two-entry iff both reps exist: scan-produced AND
        # the policy resolves a candidate.  Everything else — untyped,
        # RAW-policy classes, operator-output-fed — is a singleton, with no
        # exclusion rule anywhere on the adaptive side.
        expected_two = (inst.producer.kind == "scan"
                        and ap.candidate_for(inst) is not None)
        assert cc.is_decision == expected_two, (qid, node, inst)
        if expected_two:
            assert cc.choices == (RAW, ap.candidate_for(inst))


def test_deferred_payload_instances_yield_singletons():
    # The REAL deferral path, on a real plan: q10's deferred payload columns
    # are produced by their deferred gather, typed as their columns, so the
    # static resolution is the policy's codec and every consumer input
    # range is a singleton of it (the producer wrote one rep), with no
    # adaptive-side exclusion rule to get wrong.
    from dataflow.compression_applier import CompressionApplier

    def _range_for(inst, ap):
        return build_input_choices({ap.codec(inst): 400}, 1000).choices

    deferred = _plan(10)
    ap_def = CompressionApplier(deferred, policy=CompressionPolicy.adaptive())
    victims = [inst for inst in deferred.instances if inst.producer.kind == "deferred"]
    assert victims, "q10 defers no payload column — pick another plan"
    for inst in victims:
        assert ap_def.candidate_for(inst) == ap_def.codec(inst) != RAW
        assert _range_for(inst, ap_def) == (ap_def.codec(inst),)


# --------------------------------------------------------------------------- #
# Pick vectors for all k over all menus
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qid", list(range(1, 23)))
def test_pick_vectors_all_plans_all_k(qid):
    graph = _plan(qid)
    _, entries = build_plan_input_menus(graph, CompressionPolicy.adaptive())
    by_op = {}
    for node, _inst, cc in entries:
        by_op.setdefault(node.name, []).append(cc)
    for op, cols in by_op.items():
        menu = DecisionMenu(pipe_name=op, inputs=tuple(cols), outputs=())
        ctrl = CompressionController(menu)
        decision = set(ctrl.decision_indices())
        n = ctrl.decision_count()
        for k in range(n + 1):
            vec = ctrl.pick_vector(k)
            assert len(vec) == len(cols)
            # Both sides police the same contract: every emitted
            # vector passes the boundary validation.
            validate_pick_vector(menu.inputs, vec)
            compressed = 0
            for i, (cc, pick) in enumerate(zip(cols, vec)):
                assert pick in cc.choices, (qid, op, k, i)
                if i in decision:
                    compressed += pick != RAW
                else:
                    assert pick == cc.choices[0], (qid, op, k, i)
            assert compressed == k, (qid, op, k)


# --------------------------------------------------------------------------- #
# Benefit ranking: menu byte counts, not a ratio table
# --------------------------------------------------------------------------- #

def _menu(cols):
    return DecisionMenu(pipe_name="synth", inputs=tuple(cols), outputs=())


def _two(raw_b, stored_b, cand="ANS"):
    return build_input_choices({RAW: raw_b, cand: stored_b}, raw_b)


def test_ranking_orders_by_bytes_saved():
    #            raw  stored ->  benefit = raw - stored (≡ bytes·(1-1/ratio))
    # col 0      100      50 ->   50
    # col 1     1000     800 ->  200
    # col 2       80       8 ->   72
    # col 3      500     500 ->    0
    ctrl = CompressionController(_menu([
        _two(100, 50), _two(1000, 800), _two(80, 8), _two(500, 500)]))
    assert ctrl.ranked_indices() == [1, 2, 0, 3]
    assert ctrl.benefit(1) == pytest.approx(200.0)
    assert ctrl.benefit(3) == 0.0


def test_ranking_deterministic_stable_tiebreak():
    # Equal benefits -> positional order, stable across calls.
    ctrl = CompressionController(_menu([
        _two(100, 50), _two(200, 150), _two(300, 250)]))
    first = ctrl.ranked_indices()
    assert first == [0, 1, 2]
    assert ctrl.ranked_indices() == first


def test_missing_stored_bytes_degrade_to_zero_benefit():
    # A decision range whose candidate carries no stored byte count (built
    # directly, bypassing the from-reps builder) ranks zero, deterministic.
    no_bytes = ColumnChoices(choices=(RAW, "ANS"), raw_bytes=1000,
                             stored_bytes={RAW: 1000})
    ctrl = CompressionController(_menu([no_bytes, _two(100, 90)]))
    assert ctrl.benefit(0) == 0.0
    assert ctrl.ranked_indices() == [1, 0]


def test_nonshrinking_candidate_clamps_to_zero():
    ctrl = CompressionController(_menu([_two(100, 130), _two(100, 99)]))
    assert ctrl.benefit(0) == 0.0
    assert ctrl.ranked_indices() == [1, 0]


def test_singletons_never_rank():
    ctrl = CompressionController(_menu([
        build_input_choices({RAW: 100}, 100),
        build_input_choices({"BITCOMP": 40}, 100),
        _two(100, 50)]))
    assert ctrl.decision_indices() == [2]
    assert ctrl.ranked_indices() == [2]
    assert ctrl.candidate(0) is None and ctrl.candidate(1) is None
