"""Unit tests for the CompressionApplier.

Pure CPU: the applier is built from the in-memory dataflow graph of each
checked-in plan, no engine / CUDA.  Asserts the single chokepoint honours
the typed policy rule (typed instances resolve their ``(mapped_type,
is_sorted)`` class through the CompressionPolicy — adaptive per-class best,
or ``fixed(X)`` forcing one codec); untyped instances degrade to RAW.  A
gather's sources follow the policy like any instance.
"""
import pytest

from compression.format_policy import CompressionPolicy
from dataflow.compression_applier import CompressionApplier, RAW
from tests.dataflow.plan_graphs import build_applier, query_dataflow


def _scan_instances(graph):
    return sorted((i for i in graph.instances if i.producer.kind == "scan"),
                  key=lambda i: i.name)


def _gather_sources(graph):
    """The non-scan sources of the graph's gathers, by name."""
    return sorted({src.name: src for node in graph.nodes
                   if node.kind in ("rearrange", "compose", "deferred")
                   for src in node.inputs[:-1]
                   if src.producer.kind != "scan"}.values(),
                  key=lambda i: i.name)


# --------------------------------------------------------------------------- #
# Fixed policy: forces one codec on every typed instance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qid", [1, 3, 9])
@pytest.mark.parametrize("codec", ["ANS", "BITCOMP", "CASCADED"])
def test_fixed_policy_rule(qid, codec):
    ap = build_applier(qid, policy=CompressionPolicy.fixed(codec))
    for inst in ap.graph.instances:
        expected = RAW if inst.type_sorted is None else codec
        assert ap.codec(inst) == expected, inst


def test_fixed_raw_policy_resolves_everything_raw():
    # A RAW "fixed" policy is a no-op plan: every instance resolves RAW
    # (main.exec simply builds no applier in RAW mode).
    ap = build_applier(3, policy=CompressionPolicy.fixed(None))
    for inst in ap.graph.instances:
        assert ap.codec(inst) == RAW


def test_gather_sources_follow_the_policy():
    # q3's gathers read intermediates (a rearrange's output, an out_index
    # lineage); they take the policy's codec like any typed instance (a
    # gather streams a compressed source onto the GPU).
    ap = build_applier(3, policy=CompressionPolicy.fixed("ANS"))
    sources = _gather_sources(ap.graph)
    assert sources, "expected q3 to have non-scan gather sources"
    for inst in sources:
        assert inst.type_sorted is not None, inst
        assert ap.codec(inst) == "ANS", inst


# --------------------------------------------------------------------------- #
# Adaptive policy
# --------------------------------------------------------------------------- #

def test_adaptive_typed_instances_use_policy():
    ap = build_applier(9, policy=CompressionPolicy.adaptive())
    scans = _scan_instances(ap.graph)
    assert scans, "expected scan instances in q9"
    # Inject a typing for the first scan instance: (int, sorted) -> CASCADED.
    scans[0].type_sorted = ("int", True)
    assert ap.codec(scans[0]) == "CASCADED"
    # (string, unsorted) -> ANS
    scans[1].type_sorted = ("string", False)
    assert ap.codec(scans[1]) == "ANS"


def test_adaptive_untyped_instance_degrades_to_raw():
    ap = build_applier(9, policy=CompressionPolicy.adaptive())
    for inst in ap.graph.instances:
        inst.type_sorted = None
    for inst in ap.graph.instances:
        assert ap.codec(inst) == RAW


def test_adaptive_self_consistency_with_scan_formats():
    # The codec the applier picks for a typed instance must be one the scan
    # path's formats_for() would materialize for the SAME (type, sorted) —
    # i.e. the selector never asks for a rep the scan did not build.
    policy = CompressionPolicy.adaptive()
    ap = build_applier(9, policy=policy)
    cases = [("int", True), ("int", False), ("float", True),
             ("string", False), ("date", True)]
    for inst, (mt, srt) in zip(_scan_instances(ap.graph), cases):
        inst.type_sorted = (mt, srt)
        picked = ap.codec(inst)
        materialized = policy.formats_for(mt, srt)
        assert picked in materialized, (inst, mt, srt, picked, materialized)


# --------------------------------------------------------------------------- #
# Pins (--compression_search fixed)
# --------------------------------------------------------------------------- #

_TAG = "aggregation 17 two-pass aggregation pass1:inter_hash"   # q3, int -> BITCOMP


def test_pin_overrides_typed_policy():
    plain = build_applier(3, policy=CompressionPolicy.adaptive())
    assert plain.codec(plain.graph.instance_by_name(_TAG)) == "BITCOMP"
    ap = build_applier(3, policy=CompressionPolicy.adaptive(), pins={_TAG: "ANS"})
    inst = ap.graph.instance_by_name(_TAG)
    assert inst.pin == "ANS"
    assert ap.codec(inst) == "ANS"
    assert ap.candidate_for(inst) == "ANS"
    # A RAW pin resolves RAW and publishes no candidate.
    ap = build_applier(3, policy=CompressionPolicy.adaptive(), pins={_TAG: RAW})
    inst = ap.graph.instance_by_name(_TAG)
    assert ap.codec(inst) == RAW
    assert ap.candidate_for(inst) is None


def test_pin_applies_to_a_gather_source():
    source = _gather_sources(query_dataflow(3))[0].name
    ap = build_applier(3, policy=CompressionPolicy.adaptive(), pins={source: "CASCADED"})
    assert ap.codec(ap.graph.instance_by_name(source)) == "CASCADED"


def test_pin_leaves_other_instances_unchanged():
    pinned = "join 13 0 sort pipeline:sorted_col_92"
    plain = build_applier(3, policy=CompressionPolicy.adaptive())
    ap = build_applier(3, policy=CompressionPolicy.adaptive(), pins={pinned: "ANS"})
    reference = {inst.name: plain.codec(inst) for inst in plain.graph.instances}
    for inst in ap.graph.instances:
        if inst.name != pinned:
            assert ap.codec(inst) == reference[inst.name], inst


def test_unknown_pin_raises():
    with pytest.raises(KeyError):
        build_applier(3, policy=CompressionPolicy.adaptive(), pins={"nope:1": "ANS"})


def test_build_applier_signature():
    ap = build_applier(9, policy=CompressionPolicy.adaptive())
    assert isinstance(ap, CompressionApplier)
    ap = build_applier(9, policy=CompressionPolicy.fixed("ANS"))
    assert ap._policy.mode == "fixed"


def test_output_and_input_codecs_follow_the_nodes():
    ap = build_applier(3, policy=CompressionPolicy.adaptive())
    for node in ap.graph.nodes:
        assert ap.output_codecs(node) == [ap.codec(i) for i in node.outputs]
        assert ap.input_codecs(node) == [ap.codec(i) for i in node.inputs]
