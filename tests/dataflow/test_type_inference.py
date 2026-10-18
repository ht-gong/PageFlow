"""Unit tests for gen-time static type inference.

Pure CPU: exercises the expression typer in dataflow_gen, the typing of the
in-memory dataflow graphs' instances, and the adaptive-applier resolution of
typed intermediates.  No engine / CUDA.
"""
import pytest

from compression.format_policy import CompressionPolicy
from dataflow.compression_applier import RAW
from dataflow.dataflow_gen import (
    _SCHEMA_CLASS,
    _expr_class,
    _as_bindings,
)
from tests.dataflow.plan_graphs import build_applier, query_dataflow as _df, typed_map


def _types(qid, **kw):
    return {name: list(ts) for name, ts in typed_map(_df(qid, **kw)).items()}


# --------------------------------------------------------------------------- #
# The gen-side schema-class mirror stays in sync with the scan path's map
# --------------------------------------------------------------------------- #

def test_schema_class_mirrors_type_mapping():
    from IO.data_scanner import TYPE_MAPPING
    assert _SCHEMA_CLASS == TYPE_MAPPING


# --------------------------------------------------------------------------- #
# Expression typer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("expr,classes,want", [
    ("l_extendedprice#933", {933: "float"}, "float"),
    ("sum_qty#305L", {305: "int"}, "int"),          # long-marker suffix
    ("1.0", {}, "float"),
    ("10", {}, "int"),
    ("UNKNOWN_TOKEN", {}, None),
    # arithmetic lub: float dominates int
    (["l_extendedprice#933", "*", ["1.0", "-", "l_discount#934"]],
     {933: "float", 934: "float"}, "float"),
    # division is always float
    ([["sum", "volume#1"], "/", ["sum", "volume#1"]], {1: "float"}, "float"),
    # aggregates
    (["count", "1"], {}, "int"),
    (["avg", "l_quantity#242"], {242: "int"}, "float"),
    (["sum", "l_quantity#242"], {242: "int"}, "int"),
    (["min", "o_orderdate#866"], {866: "date"}, "date"),
    # known scalar functions
    (["year", ["l_shipdate#2184"]], {2184: "date"}, "int"),
    (["substring", ["c_phone#1156", "1", "2"]], {1156: "string"}, "string"),
    # cast target wins
    (["cast", ["l_quantity#2820 as double"]], {2820: "int"}, "float"),
    # CASE: only THEN/ELSE branches count — the string condition must not
    # pollute the lub
    (["CASE", "WHEN", ["nation#5", "=", "BRAZIL"], "THEN", "volume#6",
      "ELSE", "0.0", "END"], {5: "string", 6: "float"}, "float"),
])
def test_expr_class(expr, classes, want):
    assert _expr_class(expr, classes) == want


def test_as_bindings_flat_and_nested():
    out = {}
    _as_bindings([[["sum", ["x#1"]], "AS", "revenue#995"]], out)
    assert 995 in out
    out = {}
    # nested alias form (subquery aggregates): target is [expr, '#', '5805L']
    _as_bindings([[[["sum", ["l_quantity#5810"]], "#", "5804L"], "AS",
                   [["sum", ["l_quantity#5740"]], "#", "5805L"]]], out)
    assert 5805 in out


# --------------------------------------------------------------------------- #
# The typing: coverage + ground-truth classes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("qid", list(range(1, 23)))
def test_all_intermediates_typed(qid):
    """Every intermediate is typed, the lineage gathers' outputs included
    (a compose's lineage an int, a deferred gather's columns as their
    columns)."""
    untyped = [inst for inst in _df(qid).instances
               if inst.producer.kind != "scan" and inst.type_sorted is None]
    assert not untyped, untyped


@pytest.mark.parametrize("qid,substr,want_cls", [
    (1, "pass2:122", "int"),        # sum(l_quantity) — int operand stays int
    (3, "pass2:122", "float"),      # revenue = sum(float expr)
    (3, "inter_hash", "int"),
    (7, "project pipeline:124", "int"),     # year(l_shipdate)
    (22, "project pipeline:122", "string"),  # substring(c_phone)
    (15, "pass2:123", "int"),      # supplier_no — key rename of l_suppkey
    (2, "sorted_col_124", "float"),  # min(ps_supplycost) through a join sort
    (13, "global_index", "int"),
])
def test_known_intermediate_classes(qid, substr, want_cls):
    hits = [(t, v) for t, v in _types(qid).items() if substr in t]
    assert hits, f"q{qid}: no tag matching {substr!r}"
    for tag, (cls, srt) in ((t, tuple(v)) for t, v in hits):
        assert cls == want_cls, f"{tag}: {cls} != {want_cls}"
        assert srt is False or tag.startswith("scan")


def test_scan_tags_sorted_from_sorted_cols():
    """Scan instance typing == the runtime scan's rule (no drift): sorted
    exactly where ``sorted_cols`` names the column, unsorted without it, as
    ``main.py`` builds the graph."""
    assert tuple(_types(3)["scan 9:92"]) == ("int", False)  # L_ORDERKEY (PK)
    types = _types(3, sorted_cols=frozenset({"L_ORDERKEY"}))
    assert tuple(types["scan 9:92"]) == ("int", True)
    assert tuple(types["scan 5:71"]) == ("date", False)    # O_ORDERDATE
    assert [tag for tag, (_, srt) in types.items() if srt] == ["scan 9:92"]


# --------------------------------------------------------------------------- #
# Adaptive applier resolution
# --------------------------------------------------------------------------- #

def test_instances_carry_tuples():
    graph = _df(3)
    assert graph.instance_by_name(
        "aggregation 17 two-pass aggregation pass2:122").type_sorted == ("float", False)


def test_deferred_payload_is_gathered_typed():
    """A deferred payload column has no rearrange output at its joins; its
    deferred gather's output is typed as the column (c_phone, a string)."""
    ap = build_applier(22, policy=CompressionPolicy.adaptive())
    assert ap.graph.instance_by_name("join 5 rearrange right:54") is None
    inst = ap.graph.instance_by_name("deferred gather 6 g0:54")
    assert inst is not None and inst.type_sorted == ("string", False)
    assert ap.codec(inst) == "ANS"


def test_adaptive_applier_codecs_typed_intermediates():
    ap = build_applier(3, policy=CompressionPolicy.adaptive())
    by_name = ap.graph.instance_by_name
    # typed intermediates resolve per class...
    assert ap.codec(by_name("aggregation 17 two-pass aggregation pass1:inter_hash")) == "BITCOMP"
    assert ap.codec(by_name("aggregation 17 two-pass aggregation pass2:122")) == "ANS"
    # ...gather sources included: an int out_index lineage
    for node in ap.graph.nodes:
        if node.kind in ("rearrange", "compose", "deferred"):
            for inst in node.inputs[:-1]:
                if inst.producer.kind != "scan":
                    assert ap.codec(inst) != RAW, inst
