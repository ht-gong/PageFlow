#!/usr/bin/env python3
"""Build the static dataflow graph of a TPC-H query from its prepared plans.

``build_query_dataflow(query_plans, schema, qid)`` walks each prepared
``QueryPlanRep`` (``parsing.prepare_query_plans`` has run) and returns a
:class:`DataflowGraph`: one :class:`PlanGraph` per plan (a subquery plan
first), each a list of :class:`PipeNode` objects in execution order, whose
inputs and outputs are :class:`ColumnInstance` objects.  The runtime holds
these objects: every operator finds the nodes it will run in
``args['nodes']``, a pipeline is constructed from its node, and a codec, a
cache position, a consumer count and a residency flag are read off an
instance.  Names (``node.name``, ``inst.name``) are labels for perf
reports, sizing keys, the fixed-search pins JSON and logs; nothing is
looked up by name at runtime.

The graph is built in memory by ``main.py`` on every query occurrence, from
the very plans that occurrence executes, so it cannot drift from the
runtime's deferral decision; nothing is checked in.  Building all 22 queries
takes about 17 ms.  ``DataflowGraph.to_dict()`` is the JSON shape the
checked-in graph used to have, for the debug dump and the tests.

Late materialization model
--------------------------
We apply a SIMPLE form of late materialization for Filter: a filter only
reads the columns that appear in its predicate (``materialize_in``) and
emits a mask; the pass-through data columns are NOT considered consumed or
re-produced by the filter.  Downstream operators (aggregates, joins, sorts)
pick up the instances the filter passes through plus its mask.  If the filter
ends up with no materialized inputs (e.g. the predicate was elided), it
contributes nothing to the dataflow.

Every gather is a node whose inputs carry a role: its *sources* (every
input but the last: instances read over UVA or streamed onto the GPU, then
freed) and its *index* (the last input, read through ``col_to_gpu``).  A
join's per-side ``rearrange`` gathers the side's payload by the core's
``out_index``.  Payload deferral (``materialize_deferred``): a deferred column
is not gathered by its joins' rearranges; it rides a row lineage across the
chain — the seed join's ``out_index`` instance, then one ``compose`` node per
later join (source: the group's current lineage; index: that join's
``out_index``) — and one ``deferred`` node per group gathers the group's
columns just before the op that first reads them (``deferred gather <op>
g<group>``; source: the columns' current instances; index: the lineage).
``host_resident`` follows from the roles: every source and index, plus
every ``out_index``.  A compose's lineage is typed as an int, a deferred
gather's outputs as their columns.  The lifetime rule the runtime
applies is one: an intermediate is freed after its last consumer node has
run (the gather frees its sources itself).

Debug dump (local inspection only, never checked in):
    python -m dataflow.dataflow_gen --q 1 3 6 21 --out /tmp/dataflow
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from parsing.expressions import column_ref

# ---------------------------------------------------------------------------
# Canonical naming — the labels are the runtime pipe names
# (filter.py: "{op} filter pipe", project.py: "{op} project pipeline",
#  aggregate.py: "{op} aggregation pipe" for the global no-key path AND
#    "{op} two-pass aggregation pass1" / "{op} two-pass aggregation pass2"
#    for the keyed chunked end-merge path,
#  hashjoin.py: "{op} {side_int} sort pipeline" / "{op} core pipeline" /
#    "{op} rearrange {left|right}").
# ---------------------------------------------------------------------------

_PIPELINE_OPS = frozenset({'Scan', 'Filter', 'Project', 'Aggregate', 'Join'})

# Post-pipeline operators: they run OUTSIDE the page-cache pipeline DAG
# (materialising their inputs via ``col_to_gpu`` rather than a cached
# VortexPipeline), so they never advance the cache's op position.  They are
# recorded as *consumers* of the columns they read — without that, a column
# whose only in-DAG consumer is e.g. a HAVING Filter is marked dead and
# dead-dropped after that Filter, leaving a dangling ``page.tensor`` that the
# Sort's ``col_to_gpu`` then reads as GPU-resident garbage (the exact Q11
# dataflow lost-row bug).  Their node has
# position None so they never shift the counter, which only pipeline nodes
# advance.
_TERMINAL_OPS = frozenset({'Sort', 'take ordered', 'AdaptiveSparkPlan'})

# Canonical name helpers matching the runtime pipe names
def _name_scan(op_id: int) -> str:        return f"scan {op_id}"
def _name_filter(op_id: int) -> str:      return f"filter {op_id} filter pipe"
def _name_project(op_id: int) -> str:     return f"project {op_id} project pipeline"
# Aggregate has two runtime shapes:
#   - global (no-key) aggregate: one VortexPipeline named "{op} aggregation pipe"
#   - keyed two-pass aggregate:  two VortexPipelines, pass1 emits per-chunk
#     dedup intermediates that pass2 re-H2Ds + merges across ticks.  The graph
#     records both as separate nodes so the page_cache can plan around the
#     pass1 -> pass2 intermediate column instances.
def _name_aggregate(op_id: int) -> str:        return f"aggregation {op_id} aggregation pipe"
def _name_aggregate_pass1(op_id: int) -> str:  return f"aggregation {op_id} two-pass aggregation pass1"
def _name_aggregate_pass2(op_id: int) -> str:  return f"aggregation {op_id} two-pass aggregation pass2"

# Join sub-pipelines — side is an integer 0 (LEFT) / 1 (RIGHT) for sort pipes,
# and "left"/"right" for rearrange (matches hashjoin.py).
def _name_join_sort(op_id: int, side_int: int) -> str:
    return f"join {op_id} {side_int} sort pipeline"
def _name_join_core(op_id: int) -> str:
    return f"join {op_id} core pipeline"
def _name_join_rearrange(op_id: int, side_label: str) -> str:
    return f"join {op_id} rearrange {side_label}"

# Intermediate labels (no ':' so "producer:label" instance names parse):
#   sort outputs:      "sorted_col_{orig_col_id}" and "global_index"
#   core out_index:    "out_index_left" / "out_index_right"
#   pass 1 outputs:    "inter_hash", "inter_key_{i}", "inter_val_{i}"
def _sort_data_label(c: int) -> str: return f"sorted_col_{c}"
_GIDX_LABEL = "global_index"
_OUT_IDX = {"left": "out_index_left", "right": "out_index_right"}
_INTER_HASH = "inter_hash"


# ---------------------------------------------------------------------------
# Static type inference
#
# Every produced tag gets a mapped-type class ``(class, is_sorted)`` inferred
# ONCE at gen time and published in the JSON as ``col_types``, so the runtime
# adaptive applier can codec-pick intermediates instead of degrading them to
# RAW.  Classes are the policy classes ("int"/"float"/"date"/"string"), NOT
# tensor dtypes — width differences (int32 vs int64 accumulators) don't matter.
# ``is_sorted`` is True only for the scan columns in ``sorted_cols``, which
# main.py leaves empty (scan_to_ram marks no column sorted);
# intermediates are stamped unsorted (the sorted-leading-key upgrade is a
# separate, deferred step).  Anything the inference can't type is
# simply absent from ``col_types`` and keeps degrading to RAW at runtime.
# ---------------------------------------------------------------------------

# Mirrors IO.data_scanner.TYPE_MAPPING (schema type -> policy class); kept
# local so standalone gen never imports the torch-heavy scan module.  A unit
# test (tests/dataflow/test_type_inference.py) asserts the two stay in sync.
_SCHEMA_CLASS = {
    "identifier": "int",
    "integer": "int",
    "decimal": "float",
    "date": "date",
    "variable text": "string",
    "fixed text": "string",
}

# Least-upper-bound priority for mixed-operand expressions: float dominates
# (any arithmetic with a float operand is float), then date (min/max of
# dates), then string, then int.
_CLASS_RANK = {"int": 0, "string": 1, "date": 2, "float": 3}

# Result class of known scalar functions (expression heads).
_FN_CLASS = {"year": "int", "month": "int", "day": "int", "substring": "string"}

# Aggregates with a fixed result class regardless of operand; sum/min/max
# take the operand's class (via lub over the aggregated expression).
_AGG_FIXED_CLASS = {"count": "int", "avg": "float", "mean": "float"}

def _cast_target_class(token: str):
    """``"l_quantity#2820 as double"`` -> "float" (None if unrecognized)."""
    low = token.lower()
    if " as " not in low:
        return None
    target = low.rsplit(" as ", 1)[1].strip().rstrip(">)")
    for prefix, cls in (("double", "float"), ("float", "float"),
                        ("decimal", "float"), ("bigint", "int"),
                        ("int", "int"), ("long", "int"),
                        ("date", "date"), ("string", "string"),
                        ("varchar", "string")):
        if target.startswith(prefix):
            return cls
    return None


def _lub(classes):
    known = [c for c in classes if c]
    return max(known, key=_CLASS_RANK.__getitem__) if known else None


def _expr_class(node, col_class: Dict) -> "str | None":
    """Class of a parsed plan expression (nested lists of str tokens).

    Column refs resolve through *col_class*; literals type themselves
    (``1.0`` float, ``10`` int); known function/aggregate heads override;
    any division makes the result float; everything else is the lub of its
    typed children.  Unknown -> None (the tag stays untyped -> RAW).
    """
    if isinstance(node, str):
        cid = column_ref(node)
        if cid is not None:
            return col_class.get(cid)
        tok = node.strip()
        if tok.replace(".", "", 1).replace("-", "", 1).isdigit():
            return "float" if "." in tok else "int"
        return None
    if not isinstance(node, (list, tuple)) or not node:
        return None

    if any(tok == "/" for tok in node if isinstance(tok, str)):
        return "float"

    head = node[0].lower() if isinstance(node[0], str) else None
    if head in _AGG_FIXED_CLASS:
        return _AGG_FIXED_CLASS[head]
    if head in _FN_CLASS:
        return _FN_CLASS[head]
    if head == "cast":
        # ['cast', ['l_quantity#2820 as double']] — target type wins, else
        # fall through to the operand's class.
        for tok in node[1] if isinstance(node[1], (list, tuple)) else [node[1]]:
            if isinstance(tok, str):
                cls = _cast_target_class(tok)
                if cls:
                    return cls
        return _expr_class(node[1], col_class)
    if head in ("sum", "min", "max"):
        return _lub(_expr_class(child, col_class) for child in node[1:])

    # CASE WHEN cond THEN v [ELSE w] END — only the THEN/ELSE branches carry
    # the result class; condition refs (e.g. a string equality) must not
    # pollute the lub.
    if any(tok in ("THEN", "ELSE") for tok in node if isinstance(tok, str)):
        branches = [node[i + 1] for i, tok in enumerate(node[:-1])
                    if isinstance(tok, str) and tok in ("THEN", "ELSE")]
        return _lub(_expr_class(b, col_class) for b in branches)

    return _lub(_expr_class(child, col_class) for child in node)


def _id_token(token) -> "int | None":
    """A bare attribute-id token: ``"305"``/``"305L"`` -> 305, else None."""
    if isinstance(token, str) and token.rstrip("L").isdigit() and token[0].isdigit():
        return int(token.rstrip("L"))
    return None


def _as_target_id(node) -> "int | None":
    """The bound id of an ``AS`` alias target.

    Flat form: ``'total_revenue#4796'``; nested form (subquery aggregates):
    ``[['sum', [...]], '#', '5805L']`` — the id is the trailing bare token.
    """
    if isinstance(node, str):
        return column_ref(node)
    if (isinstance(node, (list, tuple)) and len(node) == 3
            and node[1] == "#"):
        return _id_token(node[2])
    return None


def _as_bindings(node, out: Dict[int, object]) -> None:
    """Collect ``[expr, 'AS', <target>]`` bindings from a results tree.

    Covers plain agg outputs (``'revenue#995'`` targets), key renames
    (``['l_suppkey#4729', 'AS', 'supplier_no#4795']``) and the nested alias
    form subquery aggregates use (see :func:`_as_target_id`).
    """
    if not isinstance(node, (list, tuple)):
        return
    if len(node) == 3 and node[1] == "AS":
        tid = _as_target_id(node[2])
        if tid is not None:
            out[tid] = node[0]
            return
    for child in node:
        _as_bindings(child, out)


def _agg_out_classes(args: dict, col_class: Dict) -> Dict[int, str]:
    """Class per id bound in the aggregate's ``results`` tree.

    Covers the ``out agg ids`` AND any key renames bound there (e.g. q15's
    ``supplier_no``).  Falls back per out-agg-id to the op-only rule
    (count -> int, avg -> float) when the tree yields nothing; sum/min/max
    without a typeable operand stay untyped.
    """
    bindings: Dict[int, object] = {}
    _as_bindings(args.get("results") or [], bindings)

    out: Dict[int, str] = {}
    for bid, expr in bindings.items():
        cls = _expr_class(expr, col_class)
        if cls:
            out[bid] = cls
    agg_ids = list(args.get("out agg ids") or [])
    agg_ops = list(args.get("agg ops") or [])
    for i, aid in enumerate(agg_ids):
        if aid not in out and i < len(agg_ops):
            cls = _AGG_FIXED_CLASS.get(agg_ops[i].lower())
            if cls:
                out[aid] = cls
    return out


# ---------------------------------------------------------------------------
# The object graph
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class ColumnInstance:
    """One produced column; a col id has one instance per generation.

    ``local`` is the col id (an int, the key in ``tensor_group``) or a label
    for a within-operator intermediate (``"sorted_col_92"``,
    ``"global_index"``, ``"out_index_left"``, ``"inter_hash"``, ...).
    ``consumers`` lists the nodes that read it, in execution order.
    """
    producer: "PipeNode"
    local: object
    consumers: List["PipeNode"] = field(default_factory=list)
    # (class, is_sorted) from the static typer; None = untyped (resolves RAW).
    type_sorted: Optional[Tuple[str, bool]] = None
    # Never admitted to the granule cache (spills to pinned host).
    host_resident: bool = False
    # --compression_search fixed, resolved once at load.
    pin: Optional[str] = None
    # The runtime payload, set by the producer.
    ccr: object = None

    @property
    def name(self) -> str:
        """``"<producer name>:<local>"`` — display only."""
        return f"{self.producer.name}:{self.local}"

    @property
    def col_id(self):
        return self.local

    @property
    def next_use_op(self) -> int:
        """Position of the first consumer, or the plan's terminal rank when
        there is none or it is a terminal (post-pipeline) op."""
        if self.consumers and self.consumers[0].position is not None:
            return self.consumers[0].position
        return self.producer.plan.terminal_rank

    @property
    def expected_reads(self) -> int:
        """Consumer reads to expect (1 for an implicit terminal consumer)."""
        return len(self.consumers) if self.consumers else 1

    def __repr__(self):
        return f"ColumnInstance({self.name!r})"


@dataclass(eq=False)
class PipeNode:
    """One runtime pipeline (or a terminal op), in its plan's execution order.

    ``kind``: "scan", "filter", "project", "agg", "pass1", "pass2", "sort",
    "core", "rearrange", "compose", "deferred" or "terminal".  ``position`` is the node's index in
    the execution order of its plan's pipelines (what the cache
    counts); None for scans and terminal ops.  ``inputs[i]`` is read under
    the consumer-local key ``input_keys[i]`` (the two differ across a
    Project rename).
    """
    op_id: int
    kind: str
    name: str
    plan: "PlanGraph"
    position: Optional[int] = None
    inputs: List[ColumnInstance] = field(default_factory=list)
    input_keys: List[object] = field(default_factory=list)
    outputs: List[ColumnInstance] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.kind == "terminal"

    def input_pairs(self):
        """``[(key, instance)]`` — what a VortexPipeline is built from."""
        return list(zip(self.input_keys, self.inputs))

    def __repr__(self):
        return f"PipeNode({self.name!r}, position={self.position})"


class PlanGraph:
    """The nodes of one plan, in execution order."""

    def __init__(self, index: int):
        self.index = index
        self.nodes: List[PipeNode] = []

    @property
    def pipeline_nodes(self) -> List[PipeNode]:
        return [n for n in self.nodes if n.position is not None]

    @property
    def terminal_rank(self) -> int:
        """The rank after every pipeline: where a column with no in-DAG
        consumer, or a terminal consumer, is placed."""
        return len(self.pipeline_nodes)

    @property
    def instances(self) -> List[ColumnInstance]:
        return [inst for n in self.nodes for inst in n.outputs]

    def assign_positions(self) -> None:
        """Number the pipeline nodes: not scans, not terminals, and not a
        node with neither inputs nor outputs (it never runs a pipeline)."""
        pos = 0
        for n in self.nodes:
            if n.kind in ("scan", "terminal") or not (n.inputs or n.outputs):
                n.position = None
            else:
                n.position = pos
                pos += 1


class DataflowGraph:
    """One query's graph: every plan's nodes and instances."""

    def __init__(self, query_id):
        self.query_id = query_id
        self.plans: List[PlanGraph] = []

    @property
    def nodes(self) -> List[PipeNode]:
        return [n for p in self.plans for n in p.nodes]

    @property
    def instances(self) -> List[ColumnInstance]:
        return [inst for p in self.plans for inst in p.instances]

    def instance_by_name(self, name: str) -> Optional[ColumnInstance]:
        """Resolve a hand-written tag (the fixed-search pins JSON)."""
        for inst in self.instances:
            if inst.name == name:
                return inst
        return None

    def node_by_name(self, name: str) -> Optional[PipeNode]:
        for n in self.nodes:
            if n.name == name:
                return n
        return None

    # -- the JSON shape (debug dump, tests) --------------------------------

    def to_dict(self) -> Dict:
        plans_out = []
        for plan in self.plans:
            op_views = [{
                'canonical_name': n.name,
                'op_id': n.op_id,
                'input_cols': [{'tag': inst.name, 'col_id': key}
                               for key, inst in zip(n.input_keys, n.inputs)],
                'output_cols': [inst.name for inst in n.outputs],
                'terminal': n.terminal,
            } for n in plan.nodes]
            insts = plan.instances

            def _sort_key(inst):
                cid = inst.local
                return (0, cid, inst.name) if isinstance(cid, int) else (1, 0, inst.name)

            column_view = [{
                'col_id': inst.local,
                'producer': inst.producer.name,
                'consumers': [c.name for c in inst.consumers],
            } for inst in sorted(insts, key=_sort_key)]
            plans_out.append({
                'plan_index': plan.index,
                'operator_view': op_views,
                'column_view': column_view,
                'col_types': {inst.name: [inst.type_sorted[0], inst.type_sorted[1]]
                              for inst in sorted(insts, key=lambda i: i.name)
                              if inst.type_sorted is not None},
            })
        return {'query_id': self.query_id, 'plans': plans_out}


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

class _Builder:
    """Walks a QueryPlanRep and emits the plan's PipeNodes and instances."""

    def __init__(self, plan_graph: PlanGraph, schema: Dict = None, seed_class: Dict = None,
                 sorted_cols=frozenset()):
        self.plan = plan_graph
        # col_id -> current instance
        self._active: Dict[int, ColumnInstance] = {}
        # deferred group -> its current lineage instance (the seed join's
        # out_index, then each compose's output); per plan, like the groups.
        self._lineage: Dict[int, ColumnInstance] = {}
        # (deferred node, the seed core node of its group), for the
        # invariant check after positions are assigned.
        self._deferred_roots: List[Tuple[PipeNode, PipeNode]] = []
        self._lineage_root: Dict[int, PipeNode] = {}
        # deferred group -> the out_index instance that seeded it.
        self._lineage_seed: Dict[int, ColumnInstance] = {}
        # Static type inference (see the section above): col_id -> policy
        # class, maintained alongside _active so renames/aliases propagate
        # identically.  ``seed_class`` carries classes across plan boundaries
        # (attribute ids are global per query, so a subquery plan's outputs
        # type the consumer plan's references — see the two-pass loop in
        # build_query_dataflow).
        self._schema = schema or {}
        self._sorted = sorted_cols
        self._class: Dict[object, str] = dict(seed_class or {})

    # -- graph helpers -----------------------------------------------------

    def _node(self, op_id: int, kind: str, name: str) -> PipeNode:
        node = PipeNode(op_id=op_id, kind=kind, name=name, plan=self.plan)
        self.plan.nodes.append(node)
        return node

    def _consume(self, node: PipeNode, col_id) -> ColumnInstance:
        """Read the current instance of *col_id* under that key."""
        inst = self._active[col_id]
        self._read(node, inst, col_id)
        return inst

    @staticmethod
    def _read(node: PipeNode, inst: ColumnInstance, key) -> None:
        node.inputs.append(inst)
        node.input_keys.append(key)
        inst.consumers.append(node)

    def _produce(self, node: PipeNode, col_id) -> ColumnInstance:
        """A new generation of *col_id*, produced by *node*."""
        inst = ColumnInstance(producer=node, local=col_id)
        node.outputs.append(inst)
        self._active[col_id] = inst
        return inst

    @staticmethod
    def _produce_label(node: PipeNode, label: str) -> ColumnInstance:
        """A within-operator intermediate (never in ``tensor_group``)."""
        inst = ColumnInstance(producer=node, local=label)
        node.outputs.append(inst)
        return inst

    def _note_type(self, inst: ColumnInstance, col_id, is_sorted: bool = False) -> None:
        """Type *inst* from the col_id class map (no-op if untyped)."""
        cls = self._class.get(col_id)
        if cls:
            inst.type_sorted = (cls, is_sorted)

    # -- operator handlers -------------------------------------------------

    def _handle_scan(self, u: int, args: dict) -> None:
        node = self._node(u, "scan", _name_scan(u))
        # Type scan outputs with the SAME rule scan_to_ram uses (mapped class
        # from the schema; sorted iff in sorted_cols, empty at run time).
        table = self._schema.get(args.get('table name', '').upper(), {})
        names = args.get('output names') or []
        for i, c in enumerate(args['output ids']):
            inst = self._produce(node, c)
            name = names[i] if i < len(names) else None
            spec = table.get(name.upper()) if name else None
            cls = _SCHEMA_CLASS.get(spec.split(',')[0]) if spec else None
            if cls:
                self._class[c] = cls
                inst.type_sorted = (cls, name.upper() in self._sorted)
        args['nodes'] = {'scan': node}

    def _handle_filter(self, u: int, args: dict) -> None:
        # Simple late materialization: a filter only consumes the columns
        # that its predicate actually reads and produces its mask.  If it
        # materializes nothing, it's a no-op in the dataflow and gets no
        # node.  Pass-through data columns keep their upstream instance.
        mat_in = list(args.get('materialize_in', []))
        if not mat_in:
            return
        node = self._node(u, "filter", _name_filter(u))
        for c in mat_in:
            self._consume(node, c)
        if 'out mask id' in args:
            mid = args['out mask id']
            inst = self._produce(node, mid)
            # Masks are torch.bool at runtime (filter.py output_columns_type);
            # "int" is the policy class for narrow integral data.
            self._class[mid] = "int"
            self._note_type(inst, mid)
        args['nodes'] = {'pipe': node}

    def _handle_project(self, u: int, args: dict) -> None:
        # Match operators/project.py: the project pipeline is created only
        # when there are expressions to evaluate.  Pure renames run no
        # pipeline and get no node.
        rename_map = args.get('rename map', {})
        if not args.get('used expr'):
            # Still propagate rename aliases so downstream consumers resolve
            # to the upstream producer's instance.
            for orig, new in rename_map.items():
                if orig in self._active:
                    self._active[new] = self._active[orig]
                if orig in self._class:
                    self._class[new] = self._class[orig]
            return

        node = self._node(u, "project", _name_project(u))
        # Inputs: materialized_in only.  Renames are zero-cost aliases
        # handled by tensor_group.alias() — they don't go through the
        # pipeline, so they're not consumed at this stage.
        for c in args.get('materialize_in', []):
            self._consume(node, c)
        # Outputs: ONLY materialize_now — these are the columns the runtime
        # pipeline actually emits.  Type computed outputs from their defining
        # expressions (``[expr, 'AS', 'name#id']`` entries in ``used expr``).
        expr_by_id: Dict[int, object] = {}
        _as_bindings(args.get('used expr') or [], expr_by_id)
        for c in args.get('materialize_now', []):
            inst = self._produce(node, c)
            if c in expr_by_id:
                cls = _expr_class(expr_by_id[c], self._class)
                if cls:
                    self._class[c] = cls
            self._note_type(inst, c)
        for orig, new in rename_map.items():
            if orig in self._active:
                self._active[new] = self._active[orig]
            if orig in self._class:
                self._class[new] = self._class[orig]
        args['nodes'] = {'pipe': node}

    def _consumed_by_aggregate(self, args: dict) -> List:
        consumed = list(args.get('input ids', []))
        if 'mask map' in args:
            for mid in set(args['mask map'].values()):
                if mid not in consumed:
                    consumed.append(mid)
        return consumed

    def _handle_aggregate(self, u: int, args: dict) -> None:
        # Runtime gate (operators/aggregate.py::tqp_hash_aggregate): truthy
        # ``args['keys']`` -> keyed chunked two-pass path; else global no-key
        # path. The parsing package stores either a non-empty list or None.
        if args.get('keys'):
            self._handle_aggregate_chunked(u, args)
        else:
            self._handle_aggregate_global(u, args)

    def _handle_aggregate_global(self, u: int, args: dict) -> None:
        """Global (no-key) aggregate — single VortexPipeline.

        Every row collapses into a single group; the operator emits agg
        values only, no key columns.
        """
        node = self._node(u, "agg", _name_aggregate(u))
        for c in self._consumed_by_aggregate(args):
            self._consume(node, c)
        # No-key path emits only agg values (``out key ids`` is empty).
        self._class.update(_agg_out_classes(args, self._class))
        for c in args.get('out agg ids', []):
            self._note_type(self._produce(node, c), c)
        args['nodes'] = {'pipe': node}

    def _handle_aggregate_chunked(self, u: int, args: dict) -> None:
        """Keyed two-pass aggregate — two VortexPipelines.

        Pass 1 reads upstream columns, emits per-chunk dedup outputs as
        compressed pages: ``[chunk_unique_hash, *key_reps, *agg_vals]``
        with K = ``len(out key ids)`` and N = ``len(out agg ids)``.

        Pass 2 re-H2Ds those pages as CCRs and merges across ticks; the
        last tick emits the cross-chunk-merged key + agg tensors with the
        original col_ids.  Both passes are real cache/transfer boundaries,
        so they are separate nodes with the pass1 -> pass2 intermediate
        instances visible to the cache.
        """
        emitted_keys = list(args.get('out key ids', []))
        agg_ids = list(args.get('out agg ids', []))
        K, N = len(emitted_keys), len(agg_ids)

        # ── Pass 1 ────────────────────────────────────────────────────
        pass1 = self._node(u, "pass1", _name_aggregate_pass1(u))
        for c in self._consumed_by_aggregate(args):
            self._consume(pass1, c)
        # Positions at the runtime side:
        #   0:          inter_hash   (chunk_unique_hash, int64 1-D)
        #   1..K:       inter_key_i  (squeezed key reps)
        #   K+1..K+N:   inter_val_i  (per-agg reduced values)
        # Types: inter_hash is int64; inter_key_i carries the key column's
        # class; inter_val_i carries the (final) agg result's class — the
        # pass-1 partial has the same policy class as the merged result.
        agg_cls = _agg_out_classes(args, self._class)
        self._class.update(agg_cls)  # incl. key renames (supplier_no etc.)
        self._produce_label(pass1, _INTER_HASH).type_sorted = ("int", False)
        for i, kid in enumerate(emitted_keys):
            self._note_type(self._produce_label(pass1, f"inter_key_{i}"), kid)
        for i, aid in enumerate(agg_ids):
            inst = self._produce_label(pass1, f"inter_val_{i}")
            if aid in agg_cls:
                inst.type_sorted = (agg_cls[aid], False)

        # ── Pass 2 ────────────────────────────────────────────────────
        pass2 = self._node(u, "pass2", _name_aggregate_pass2(u))
        for inst in pass1.outputs:
            self._read(pass2, inst, inst.local)
        self._class.update(agg_cls)
        for c in emitted_keys + agg_ids:
            self._note_type(self._produce(pass2, c), c)
        args['nodes'] = {'pass1': pass1, 'pass2': pass2}

    def _handle_join(self, u: int, args: dict) -> None:
        left_group = set(args.get('left group', []))
        right_group = set(args.get('right group', []))
        join_in = set(args.get('join in', []))
        # The join's full output: gathered now plus deferred (keys are never deferred).
        mat_out = list(args.get('materialize_now', [])) + list(args.get('materialize_deferred') or [])
        mask_map = dict(args.get('mask map', {}))

        left_transfer = sorted(join_in & left_group)
        right_transfer = sorted(join_in & right_group)

        # Mask assignment: left gets priority
        left_mask_ids: set = set()
        for k, v in mask_map.items():
            if k in left_transfer:
                left_mask_ids.add(v)
        right_mask_ids: set = set()
        for k, v in mask_map.items():
            if k in right_transfer:
                right_mask_ids.add(v)
        right_mask_ids -= left_mask_ids

        core_outputs = [k for k in mat_out if k in join_in]
        rearrange_cols = [k for k in mat_out if k not in join_in]
        rearrange_required = len(rearrange_cols) > 0

        # -- Sort pipelines (side 0 = LEFT, 1 = RIGHT) --
        sort_nodes = []
        for side, transfer, mask_ids in ((0, left_transfer, left_mask_ids),
                                         (1, right_transfer, right_mask_ids)):
            node = self._node(u, "sort", _name_join_sort(u, side))
            for c in transfer + sorted(mask_ids):
                self._consume(node, c)
            for c in transfer:
                self._note_type(self._produce_label(node, _sort_data_label(c)), c)
            self._produce_label(node, _GIDX_LABEL).type_sorted = ("int", False)  # int64
            sort_nodes.append(node)

        # -- Core join pipeline --
        # Consumes ALL sort intermediates (both sides' data cols + global
        # indices).  Masks are NOT forwarded past the sort pipelines.
        core = self._node(u, "core", _name_join_core(u))
        for node in sort_nodes:
            for inst in node.outputs:
                self._read(core, inst, inst.local)
        for c in core_outputs:
            self._note_type(self._produce(core, c), c)
        out_index = {}
        if rearrange_required:
            for side_label in ("left", "right"):
                inst = self._produce_label(core, _OUT_IDX[side_label])
                inst.type_sorted = ("int", False)  # out_index: int64
                # The out_index exists only to drive a UVA gather (this
                # join's own rearrange and/or a lineage compose), so it is
                # host-resident whether or not a gather reads it (D-3).
                inst.host_resident = True
                out_index[side_label] = inst

        # -- Gathers, per side, in the order the runtime performs them --
        # A rearrange node when the side has sources: its materialize_now
        # minus the keys.  Then one
        # compose node per deferred group composed on this side: the group's
        # current lineage is its source, the side's out_index its index.  A
        # seed join (the group's first) emits no node: its out_index becomes
        # the lineage and gains the next gather as a consumer.  A side whose
        # payload is all deferred gets no rearrange node.
        now = list(args.get('materialize_now', []))
        deferred_cols = list(args.get('materialize_deferred') or [])
        group_of = args.get('deferred_groups', {})
        gathers = {"left": [], "right": []}
        for side_label, group in (("left", left_group), ("right", right_group)):
            sources = [k for k in now if k in group and k not in join_in]
            if sources:
                node = self._node(u, "rearrange", _name_join_rearrange(u, side_label))
                for c in sources:
                    self._consume(node, c)
                self._read(node, out_index[side_label], out_index[side_label].local)
                for c in sources:
                    self._note_type(self._produce(node, c), c)
                gathers[side_label].append(node)
            seen: List[int] = []
            for c in deferred_cols:
                if c not in group or group_of[c] in seen:
                    continue
                g = group_of[c]
                seen.append(g)
                if g not in self._lineage:
                    self._lineage[g] = self._lineage_seed[g] = out_index[side_label]
                    self._lineage_root[g] = core
                    continue
                node = self._node(u, "compose", f"join {u} lineage {side_label} g{g}")
                self._read(node, self._lineage[g], "lineage")
                self._read(node, out_index[side_label], out_index[side_label].local)
                self._lineage[g] = self._produce_label(node, "lineage")
                self._lineage[g].type_sorted = ("int", False)  # int64
                gathers[side_label].append(node)

        args['nodes'] = {'sort': sort_nodes, 'core': core, 'gathers': gathers}

    def _handle_post_pipeline(self, u: int, args: dict) -> None:
        """Record a Sort / take-ordered / output op as a consumer of the
        columns it reads after the pipeline DAG.

        These ops materialise their inputs via ``col_to_gpu`` (a bare,
        uncached ``DirectIn``).  If such a column's only in-DAG consumer has
        already run, the granule cache will have marked it dead and released
        its slots — ``col_to_gpu`` would then copy garbage from reused
        slots.  Recording
        the read here keeps the column alive (``remaining_reads`` never hits
        zero, so it is never note_read'd to phase-0 DEAD); under memory
        pressure it is evicted via the safe spill+rebind path instead.
        """
        name = args.get('name')
        if name == 'AdaptiveSparkPlan':
            read_ids = list(args.get('output ids') or [])
        else:  # Sort / take ordered, and the mask of a filter right below
            read_ids = list(args.get('input ids') or []) + \
                list(args.get('key ids') or []) + \
                sorted(set((args.get('mask map') or {}).values()))
        # De-dup, preserve order; only columns with a live producer instance
        # can be consumed (renamed/scan-only ids without an _active entry are
        # skipped — they were never cache-admitted intermediates).
        seen: set = set()
        cols = [c for c in read_ids
                if c in self._active and not (c in seen or seen.add(c))]
        if not cols:
            return
        node = self._node(u, "terminal", f"{name} {u}")
        for c in cols:
            self._consume(node, c)
        args['nodes'] = {'terminal': node}

    # -- top-level dispatch ------------------------------------------------

    def _deferred_gathers(self, u: int, args: dict) -> List[PipeNode]:
        """One ``deferred`` node per group gathered just before op *u*: its
        sources are the group's columns' current (pre-chain) instances, its
        index the group's current lineage; its outputs, typed as their
        columns, are what the op's nodes then read."""
        groups: Dict[int, List] = {}
        for spec in args.get('gather_deferred') or []:
            groups.setdefault(spec['group'], []).append(spec['col'])
        nodes = []
        for g, cols in groups.items():
            node = self._node(u, "deferred", f"deferred gather {u} g{g}")
            for c in cols:
                self._consume(node, c)
            assert g in self._lineage, (
                f"deferred gather {u} g{g}: the group's lineage is read before it is produced")
            self._read(node, self._lineage[g], "lineage")
            for c in cols:
                self._note_type(self._produce(node, c), c)
            self._deferred_roots.append((node, self._lineage_root[g]))
            nodes.append(node)
        return nodes

    def walk(self, plan) -> None:
        for u in plan.ops_idx:
            args = plan.ops_dict[u]
            name = args['name']
            deferred_nodes = self._deferred_gathers(u, args)
            if name in _PIPELINE_OPS:
                handler = {
                    'Scan': self._handle_scan,
                    'Filter': self._handle_filter,
                    'Project': self._handle_project,
                    'Aggregate': self._handle_aggregate,
                    'Join': self._handle_join,
                }[name]
                handler(u, args)
            elif name in _TERMINAL_OPS:
                self._handle_post_pipeline(u, args)
            if deferred_nodes:
                args.setdefault('nodes', {})['deferred'] = deferred_nodes

    def check_invariants(self) -> None:
        """The three gather invariants, after positions are assigned."""
        for node in self.plan.nodes:
            if node.kind not in _GATHER_KINDS:
                continue
            *sources, index = node.inputs
            # 1. Each gather is the last consumer of its sources: it frees them.
            for src in sources:
                assert src.consumers[-1] is node, (
                    f"{node.name} frees its source {src.name}, which "
                    f"{[c.name for c in src.consumers if c is not node]} also read")
            # 3. A lineage is produced before it is read.
            assert index.producer.position is None or index.producer.position < node.position, (
                f"{node.name} reads {index.name} before it is produced")
        # 2. No node between a chain's seed join and the deferred gather reads
        #    a deferred column's pre-chain instance.
        for node, root in self._deferred_roots:
            for src in node.inputs[:-1]:
                for other in src.consumers:
                    if other is node:
                        continue
                    assert other.position is not None and other.position < root.position, (
                        f"{other.name} reads {src.name}, a deferred column's pre-chain "
                        f"instance, after its chain starts at {root.name}")
        # 3. No two groups share a lineage instance (two groups seeded at
        #    the same join sit on different sides, so their seeds differ).
        seeds = list(self._lineage_seed.values())
        assert len({id(i) for i in seeds}) == len(seeds), (
            f"two deferred groups share a lineage: {[i.name for i in seeds]}")


_GATHER_KINDS = ("rearrange", "compose", "deferred")


def _mark_gather_roles(plan: PlanGraph) -> None:
    """Keep every gather's sources (every input but the last) and its index
    (the last input) host-resident, never admitted to the GPU cache: a
    gather reads them outside the cache's ticks (over UVA, through a tick
    pipe of its own, or ``col_to_gpu``)."""
    for node in plan.nodes:
        if node.kind not in _GATHER_KINDS:
            continue
        *sources, index = node.inputs
        for src in sources:
            src.host_resident = True
        index.host_resident = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_plan_graph(plan, index: int, schema: Dict = None, seed_class: Dict = None,
                     sorted_cols=frozenset()) -> Tuple[PlanGraph, Dict]:
    graph = PlanGraph(index)
    builder = _Builder(graph, schema=schema, seed_class=seed_class, sorted_cols=sorted_cols)
    builder.walk(plan)
    graph.assign_positions()
    _mark_gather_roles(graph)
    builder.check_invariants()
    return graph, builder._class


def build_query_dataflow(query_plans, schema: Dict, qid=None,
                         sorted_cols=frozenset()) -> DataflowGraph:
    """The static dataflow graph of one query.

    ``query_plans`` are the prepared plans the runtime executes (a subquery
    plan first).  Positions restart per plan: a pipeline's position is its
    index in the execution order of its own plan.  The nodes are attached to
    the operators they will run as ``args['nodes']``: ``scan``; ``pipe`` for a
    filter, project or global aggregate; ``pass1``/``pass2`` for a keyed
    aggregate; ``sort`` (both sides), ``core`` and ``gathers`` (per side, the
    rearrange then the composes) for a join; ``terminal`` for a Sort, take
    ordered or output; and ``deferred`` (the deferred gathers run before the
    op) on whichever op first reads a deferred group.

    ``sorted_cols`` names the scan columns that are sorted; every other scan
    column is unsorted.  ``main.py`` passes none, as the scan
    (``IO.data_scanner.scan_to_ram``) marks no column sorted.

    Two passes: attribute ids are global per query, but a subquery's outputs
    (e.g. q15's total_revenue) may be consumed by a plan walked earlier in the
    list.  Pass 1 accumulates every plan's class map; pass 2 re-walks with the
    union seeded, so cross-plan references resolve regardless of plan order.
    One subquery level -> two passes reach the fixpoint.
    """
    shared_class: Dict = {}
    for index, plan in enumerate(query_plans):
        _, cls = build_plan_graph(plan, index, schema=schema, seed_class=shared_class)
        shared_class.update(cls)

    graph = DataflowGraph(qid)
    for index, plan in enumerate(query_plans):
        plan_graph, _ = build_plan_graph(plan, index, schema=schema, seed_class=shared_class,
                                         sorted_cols=sorted_cols)
        graph.plans.append(plan_graph)
    return graph


def load_schema(project_root: str = None) -> Dict:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    with open(os.path.join(project_root, 'queries', 'schema.json'), encoding='utf-8') as f:
        return json.load(f)


def dump_all(qids: List[int], output_dir: str) -> None:
    """Debug dump: build each query's graph from its pickle and write
    ``<output_dir>/q<N>.json``.  Local inspection only; never checked in."""
    from parsing import prepare_query_plans

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    pickle_dir = os.path.join(project_root, 'queries', 'pickles')
    os.makedirs(output_dir, exist_ok=True)
    schema = load_schema(project_root)

    for qid in qids:
        pkl_path = os.path.join(pickle_dir, f'planobj-q{qid}.pkl')
        if not os.path.isfile(pkl_path):
            print(f"  SKIP Q{qid}: {pkl_path} not found")
            continue
        with open(pkl_path, 'rb') as f:
            query_plans = pickle.load(f)
        prepare_query_plans(query_plans, qid)
        result = build_query_dataflow(query_plans, schema, qid).to_dict()
        out_path = os.path.join(output_dir, f'q{qid}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Ensure project root is on sys.path for pickle deserialization
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    parser = argparse.ArgumentParser(
        description='Dump the in-memory dataflow graph of query plans as JSON (debug only)')
    parser.add_argument('--q', type=int, nargs='+', default=list(range(1, 23)),
                        help='Query IDs to process')
    parser.add_argument('--out', required=True, help='Output directory')
    args = parser.parse_args()

    dump_all(args.q, args.out)
