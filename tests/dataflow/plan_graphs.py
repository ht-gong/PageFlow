"""Build a query's dataflow graph from its checked-in pickle.

The graph is never checked in: the runtime builds it from the prepared plans
on every occurrence.  Tests build it the same way, here.  CPU-only.
"""
from __future__ import annotations

import os
import pickle

from dataflow.compression_applier import CompressionApplier
from dataflow.dataflow_gen import DataflowGraph, build_query_dataflow, load_schema
from parsing import prepare_query_plans

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def prepared_plans(qid):
    with open(os.path.join(_ROOT, "queries", "pickles", f"planobj-q{qid}.pkl"), "rb") as f:
        plans = pickle.load(f)
    return prepare_query_plans(plans, qid)


def query_dataflow(qid, *, sorted_cols=frozenset()) -> DataflowGraph:
    """The graph ``main.py`` builds for ``qid`` (no sorted scan column, as
    the scan marks none).  A fresh graph on every call: instances are
    mutable (payloads, pins)."""
    return build_query_dataflow(
        prepared_plans(qid), load_schema(_ROOT), qid, sorted_cols=sorted_cols)


def build_applier(qid, *, policy, pins=None) -> CompressionApplier:
    return CompressionApplier(query_dataflow(qid), policy=policy, pins=pins)


def typed_map(graph) -> dict:
    """``{instance name: (class, is_sorted)}`` over the typed instances."""
    return {inst.name: inst.type_sorted for inst in graph.instances
            if inst.type_sorted is not None}
