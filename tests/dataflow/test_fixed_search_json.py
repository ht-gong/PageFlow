"""Unit tests for ``--compression_search fixed`` pin loading and validation.

Pure CPU, over the in-memory dataflow graphs.  Covers the file loader
(``load_fixed_search_json``) and every rejection ``validate_pins`` makes,
plus the checked-in ``tests/verification/SF1_fixed_search.json`` the e2e
rung replays.
"""
import json
import os

import pytest

from compression.format_policy import CompressionPolicy
from dataflow.compression_applier import (
    RAW,
    load_fixed_search_json,
    validate_pins,
)
from tests.dataflow.plan_graphs import build_applier, query_dataflow as _df

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CHECKED_IN = os.path.join(_ROOT, "tests", "verification", "SF1_fixed_search.json")


def _write(tmp_path, data):
    p = tmp_path / "pins.json"
    p.write_text(json.dumps(data))
    return str(p)


def _scan_candidate(policy):
    return lambda inst: policy.compressed_format(*inst.type_sorted)


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #

def test_load_returns_query_entry_and_uppercases(tmp_path):
    p = _write(tmp_path, {"3": {"scan 5:71": "raw", "x:y": "ans"}, "9": {}})
    assert load_fixed_search_json(p, 3) == {"scan 5:71": "RAW", "x:y": "ANS"}
    assert load_fixed_search_json(p, 9) == {}


def test_load_missing_query_is_empty(tmp_path):
    p = _write(tmp_path, {"3": {"scan 5:71": "RAW"}})
    assert load_fixed_search_json(p, 7) == {}


@pytest.mark.parametrize("data", [
    [],                                   # top level not an object
    {"3": ["scan 5:71", "RAW"]},         # query entry not an object
    {"3": {"scan 5:71": 1}},             # codec not a string
])
def test_load_rejects_bad_shapes(tmp_path, data):
    with pytest.raises(ValueError):
        load_fixed_search_json(_write(tmp_path, data), 3)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def test_validate_accepts_safe_pins():
    pins = {"scan 5:71": "RAW",
            "aggregation 17 two-pass aggregation pass1:inter_hash": "ANS"}
    assert validate_pins(_df(3), pins) == []


def test_validate_rejects_unknown_codec():
    out = validate_pins(_df(3), {"scan 5:71": "ZSTD"})   # LZ4 is a real codec now
    assert len(out) == 1 and "scan 5:71" in out[0] and "unknown codec" in out[0]


def test_validate_rejects_unknown_tag():
    out = validate_pins(_df(3), {"scan 99:1": "ANS"})
    assert len(out) == 1 and "not in the query's dataflow" in out[0]


def _gather_source(graph):
    """A non-scan source of one of the graph's gathers."""
    return sorted(src.name for node in graph.nodes
                  if node.kind in ("rearrange", "compose", "deferred")
                  for src in node.inputs[:-1]
                  if src.producer.kind != "scan")[0]


def test_validate_accepts_a_gather_source_pinned_compressed():
    # A gather streams a compressed source onto the GPU, so a gather source
    # takes any codec.
    graph = _df(3)
    tag = _gather_source(graph)
    assert validate_pins(graph, {tag: "ANS"}) == []
    assert validate_pins(graph, {tag: "RAW"}) == []


def test_validate_accepts_a_lineage_seed_out_index():
    # A seed join's out_index is the lineage the next join's compose
    # gathers: it takes any codec.
    found = None
    for qid in range(1, 23):
        graph = _df(qid)
        for node in graph.nodes:
            if node.kind == "compose":
                found = (graph, node.inputs[0].name)
                break
        if found:
            break
    assert found, "expected a lineage compose in some query"
    graph, tag = found
    assert validate_pins(graph, {tag: "BITCOMP"}) == []


def test_validate_accepts_a_deferred_gather_output():
    # A deferred column is produced by its deferred gather, typed as the
    # column; its joins have no rearrange output for it at all.
    graph = _df(22)
    out = validate_pins(graph, {"join 5 rearrange right:54": "ANS"})
    assert len(out) == 1 and "not in the query's dataflow" in out[0]
    tag = "deferred gather 6 g0:54"       # c_phone, gathered before aggregation 6
    assert graph.instance_by_name(tag).type_sorted == ("string", False)
    assert validate_pins(graph, {tag: "ANS"}) == []


def test_validate_accepts_a_compose_output():
    # A lineage compose's output is typed as an int.
    for qid in range(1, 23):
        graph = _df(qid)
        node = next((n for n in graph.nodes if n.kind == "compose"), None)
        if node is not None:
            break
    assert node is not None, "expected a lineage compose in some query"
    assert node.outputs[0].type_sorted == ("int", False)
    assert validate_pins(graph, {node.outputs[0].name: "BITCOMP"}) == []


def test_validate_rejects_an_untyped_instance_pinned_compressed():
    graph = _df(3)
    inst = graph.instance_by_name(
        "aggregation 17 two-pass aggregation pass1:inter_hash")
    inst.type_sorted = None
    out = validate_pins(graph, {inst.name: "ANS"})
    assert len(out) == 1 and "must be RAW (untyped)" in out[0]
    assert validate_pins(graph, {inst.name: "RAW"}) == []


def test_validate_scan_pin_limited_to_materialized_rep():
    graph = _df(3)
    cand = _scan_candidate(CompressionPolicy.adaptive())
    # scan 5:71 is (date, unsorted) -> BITCOMP under the default verdict.
    assert validate_pins(graph, {"scan 5:71": "BITCOMP"}, scan_candidate=cand) == []
    assert validate_pins(graph, {"scan 5:71": "RAW"}, scan_candidate=cand) == []
    out = validate_pins(graph, {"scan 5:71": "ANS"}, scan_candidate=cand)
    assert len(out) == 1 and "materializes only RAW and BITCOMP" in out[0]
    # Intermediates are not subject to the scan check.
    assert validate_pins(
        graph, {"aggregation 17 two-pass aggregation pass1:inter_hash": "ANS"},
        scan_candidate=cand) == []


def test_validate_reports_every_violation():
    graph = _df(3)
    cand = _scan_candidate(CompressionPolicy.adaptive())
    out = validate_pins(graph, {"scan 5:71": "ANS", "nope:1": "ANS",
                                _gather_source(graph): "ZIP"},
                        scan_candidate=cand)
    assert len(out) == 3


# --------------------------------------------------------------------------- #
# The checked-in file the e2e rung replays
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qid", [3, 9])
def test_checked_in_file_is_valid_and_overrides(qid):
    policy = CompressionPolicy.adaptive()
    pins = load_fixed_search_json(_CHECKED_IN, qid)
    assert pins, f"q{qid}: checked-in file pins nothing"
    assert validate_pins(_df(qid), pins, scan_candidate=_scan_candidate(policy)) == []
    ap = build_applier(qid, policy=policy, pins=pins)
    for tag, codec in pins.items():
        assert ap.codec(ap.graph.instance_by_name(tag)) == codec, tag
    # At least one pin differs from what the policy would have chosen, so
    # the file exercises the override rather than restating the default.
    plain = build_applier(qid, policy=policy)
    assert any(plain.codec(plain.graph.instance_by_name(t)) != c for t, c in pins.items())
