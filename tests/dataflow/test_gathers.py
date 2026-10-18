"""Gathers are nodes (B3 of docs/materialization_unification_plan.md).

CPU-only, over the 22 prepared plans, deferral on and off: the builder's
three invariants hold (each gather is the last consumer of its sources; no
node reads a deferred column's pre-chain instance after its chain starts;
every group's lineage is produced before it is read and no two groups share
one), every deferred column is gathered exactly once, every group's lineage
is seeded exactly once before its first read, and the roles set the flags.
"""
import pytest

from tests.dataflow.plan_graphs import prepared_plans, query_dataflow
from dataflow.dataflow_gen import build_query_dataflow, load_schema

GATHER = ("rearrange", "compose", "deferred")


def _lineage_read(node):
    """The lineage instance a compose (its source) or a deferred gather (its
    index) reads."""
    return node.inputs[0] if node.kind == "compose" else node.inputs[-1]


@pytest.mark.parametrize("qid", list(range(1, 23)))
def test_builder_invariants_hold(qid):
    graph = query_dataflow(qid)
    for node in graph.nodes:
        if node.kind not in GATHER:
            continue
        *sources, index = node.inputs
        assert len(sources) == len(node.outputs) >= 1
        for src in sources:
            assert src.consumers[-1] is node
        assert index.producer.position < node.position


@pytest.mark.parametrize("qid", list(range(1, 23)))
def test_every_deferred_column_is_gathered_exactly_once(qid):
    plans = prepared_plans(qid)
    graph = build_query_dataflow(plans, load_schema(), qid)
    for plan, plan_graph in zip(plans, graph.plans):
        deferred = sorted(spec["col"] for args in plan.ops_dict.values()
                          for spec in args.get("gather_deferred") or [])
        produced = sorted(inst.local for n in plan_graph.nodes if n.kind == "deferred"
                          for inst in n.outputs)
        assert produced == deferred
        assert len(set(produced)) == len(produced)
        # No rearrange node produces a deferred column.
        for n in plan_graph.nodes:
            if n.kind == "rearrange":
                assert not {inst.local for inst in n.outputs} & set(deferred), n


@pytest.mark.parametrize("qid", list(range(1, 23)))
def test_each_group_lineage_is_seeded_once_before_its_first_read(qid):
    for plan in query_dataflow(qid).plans:
        reads = {}
        for n in plan.nodes:
            if n.kind in ("compose", "deferred"):
                reads.setdefault(int(n.name.rsplit("g", 1)[1]), []).append(n)
        seeds = []
        for g, nodes in reads.items():
            first = _lineage_read(nodes[0])
            # The seed: a core's out_index, produced before the first read.
            assert first.producer.kind == "core" and str(first.local).startswith("out_index")
            assert first.producer.position < nodes[0].position
            seeds.append(first)
            current = first
            for n in nodes:
                assert _lineage_read(n) is current, (g, n)
                if n.kind == "compose":
                    current = n.outputs[0]
        assert len({id(s) for s in seeds}) == len(seeds), "two groups share a lineage"


def test_q22_c_phone_is_gathered_at_its_consumers_not_at_the_join():
    graph = query_dataflow(22)
    for n in graph.nodes:
        if n.kind == "rearrange":
            assert 54 not in [inst.local for inst in n.outputs], n
    g6 = graph.node_by_name("deferred gather 6 g0")
    g9 = graph.node_by_name("deferred gather 9 g0")
    assert g6 is not None and [inst.local for inst in g6.outputs] == [54]
    assert g9 is not None and [inst.local for inst in g9.outputs] == [55]
    # A split group: both gathers read the same lineage, freed after the last.
    assert _lineage_read(g6) is _lineage_read(g9)
    assert _lineage_read(g6).consumers[-1] is g9


@pytest.mark.parametrize("qid", list(range(1, 23)))
def test_roles_set_the_flags(qid):
    graph = query_dataflow(qid)
    sources, indices = set(), set()
    for n in graph.nodes:
        if n.kind in GATHER:
            sources.update(id(i) for i in n.inputs[:-1])
            indices.add(id(n.inputs[-1]))
    for inst in graph.instances:
        is_source, is_index = id(inst) in sources, id(inst) in indices
        expect_host = is_source or is_index or str(inst.local).startswith("out_index")
        assert inst.host_resident == expect_host, inst
        if inst.producer.kind in ("compose", "deferred"):
            assert inst.type_sorted is not None, inst
