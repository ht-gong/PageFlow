"""Plan-driven codec resolution: the typed policy, per column instance.

Every operator writes each output in the codec its instance resolves to,
and every consumer reads it in that codec.  This module reads the query's
dataflow graph (``dataflow.dataflow_gen``): a typed instance takes the
policy's codec for its ``(type, sorted)`` class, an untyped one is RAW.  A
gather's sources need no RAW rule: a gather reads a source over UVA from a
RAW rep (a scan always keeps one) or streams the source's own records onto
the GPU (``IO/pipeline/rearrange_pipeline.py``).

Where the rule lives
--------------------
This is a static, plan-derived fact — it reads only the graph, holds no
runtime state, and produces exactly one deterministic decision per query — so
it does NOT belong in ``adaptive/`` (whose siblings make runtime-informed
decisions).  It lives here in ``dataflow/`` beside ``dataflow_gen.py``.

Why this is consistent
----------------------
Compression is intrinsic to a column instance: the producer picks one codec and
every consumer reads that same codec.  Producer and consumer both hold the
same :class:`ColumnInstance` object, and :meth:`CompressionApplier.codec`
answers for it either way (``VortexPipeline`` runs strict rep lookup when an
applier is present).

Codec choice itself is the typed per-``(type, sorted)`` policy rule
(``compression/format_policy.py`` + the instance's ``type_sorted``).  Under
``--compression_search fixed`` a JSON of per-tag codecs (the *pins*, see
:func:`load_fixed_search_json` / :func:`validate_pins`) overrides the policy
pick instance by instance.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from compression.format_policy import CompressionPolicy

RAW = "RAW"


# ``TQP_RAW_INTERMEDIATES`` (on-word, read at import): every intermediate
# RAW, the scan's reps still following the policy: an experiment switch, for
# any strategy (the free-list baseline's intermediates otherwise follow the
# policy like any other's).
RAW_INTERMEDIATES: bool = (os.environ.get("TQP_RAW_INTERMEDIATES") or ""
                           ).strip().lower() in ("1", "true", "yes", "on")


def _raw_intermediate(inst) -> bool:
    """Under ``TQP_RAW_INTERMEDIATES`` every intermediate is RAW; the scan's
    reps follow the policy."""
    return RAW_INTERMEDIATES and inst.producer.kind != "scan"


class CompressionApplier:
    """Plan-driven codec resolution: the static rule chain, one chokepoint.

    Built from a query's :class:`dataflow.dataflow_gen.DataflowGraph`.  Every
    producer-side and consumer-side question is :meth:`codec` of an instance:

    1. **RAW intermediates** — under ``TQP_RAW_INTERMEDIATES`` every
       non-scan instance is RAW.
    2. **Pin** — an instance pinned by ``--compression_search fixed``
       resolves to its pinned codec, validated at load by
       :func:`validate_pins` and written onto the instance here.
    3. **Typed policy rule** — a typed instance resolves its ``(mapped_type,
       is_sorted)`` class through the :class:`CompressionPolicy` (adaptive =
       profiled per-class best; ``CompressionPolicy.fixed(X)`` forces one
       codec).  The builder types scan outputs from the schema and
       intermediates by static inference; an untyped instance is RAW.

    This is the *static default* resolver behind the decision menu: it
    fills every column with no adaptive pick and carries no derived caches —
    adaptive decisions never enter this class; they ride the per-pipe menu
    (``adaptive/decision_menu.py``) and the per-tick observer return.
    :meth:`candidate_for` is the menu construction's policy input.
    """

    def __init__(self, graph, *, policy: CompressionPolicy,
                 pins: Optional[Dict[str, str]] = None):
        self._graph = graph
        # Single source of truth for codec choice (shared with the scan path).
        self._policy = policy
        # --compression_search fixed: resolve each hand-written tag to its
        # instance once (the caller validated the pins with validate_pins).
        self._pins: Dict[str, str] = dict(pins or {})
        for tag, codec in self._pins.items():
            inst = graph.instance_by_name(tag)
            if inst is None:
                raise KeyError(f"pin {tag!r}: no such column instance in the dataflow")
            inst.pin = codec

    @property
    def graph(self):
        return self._graph

    def codec(self, inst) -> str:
        """Resolve an instance to a codec; the single chokepoint.

        (1) ``TQP_RAW_INTERMEDIATES``'s RAW; (2) a pin from the fixed-search JSON;
        (3) typed instances resolve through the policy (degrading to the
        per-type default); (4) untyped instances are RAW.  Static — no
        adaptive state enters this resolution.
        """
        if _raw_intermediate(inst):                    # 1. TQP_RAW_INTERMEDIATES
            return RAW
        if inst.pin is not None:                       # 2. fixed-search pin
            return inst.pin
        if inst.type_sorted is not None:               # 3. typed policy pick
            algo = self._policy.compressed_format(inst.type_sorted[0], inst.type_sorted[1])
            if algo is not None:
                return algo
            return self._policy.default_for(inst.type_sorted[0]) or RAW
        return RAW                                     # 4. untyped -> RAW

    def candidate_for(self, inst) -> Optional[str]:
        """The instance's single non-RAW candidate:
        ``policy.compressed_format(type, is_sorted)``, or ``None`` when the
        instance is RAW under ``TQP_RAW_INTERMEDIATES``, untyped, or its class
        resolves RAW-only.  The policy input to decision-menu construction: a
        ``None`` here is exactly a singleton output range."""
        if _raw_intermediate(inst):
            return None
        if inst.pin is not None:
            return None if inst.pin == RAW else inst.pin
        if inst.type_sorted is None:
            return None
        return self._policy.compressed_format(inst.type_sorted[0], inst.type_sorted[1])

    def output_codecs(self, node) -> List[str]:
        """One codec per output instance of *node*, in output order."""
        return [self.codec(inst) for inst in node.outputs]

    def input_codecs(self, node) -> List[str]:
        """One codec per input instance of *node*, in input order."""
        return [self.codec(inst) for inst in node.inputs]


# ---------------------------------------------------------------------------
# --compression_search fixed: per-tag codec pins from a JSON file
# ---------------------------------------------------------------------------

def load_fixed_search_json(path: str, qid) -> Dict[str, str]:
    """Read the pins for query ``qid`` from a fixed-search JSON file.

    The file maps query id (as a string) to ``{tag: codec}``.  A query
    absent from the file has no pins.  Shape errors raise ``ValueError``
    naming the offending entry.
    """
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be an object of query id -> pins")
    pins = data.get(str(qid), {})
    if not isinstance(pins, dict):
        raise ValueError(f"{path}: entry for query {qid} must be an object of tag -> codec")
    for tag, codec in pins.items():
        if not isinstance(tag, str) or not isinstance(codec, str):
            raise ValueError(f"{path}: query {qid}: pin {tag!r}: {codec!r} is not a string pair")
    return {tag: codec.upper() for tag, codec in pins.items()}


def validate_pins(graph, pins: Dict[str, str], *,
                  scan_candidate=None) -> List[str]:
    """Check a pin map against a query's graph; return the violations.

    Each returned string names the tag and the codec.  An empty list means
    the pins are safe to apply.  Checks:

    * the codec is a registry name (``RAW`` or one of the seven codecs in
      ``compression_algorithms.algo_names()``);
    * the tag names an instance of the graph;
    * an untyped instance is not pinned to a non-RAW codec (no class, so no
      codec the policy would give it);
    * a scan instance is pinned only to RAW or to ``scan_candidate(inst)``
      (the one compressed rep ``scan_to_ram`` materializes for it), when the
      caller supplies that callable.
    """
    from compression.compression_algorithms import algo_names
    known = set(algo_names())
    out: List[str] = []
    for tag, codec in pins.items():
        if codec not in known:
            out.append(f"{tag!r} -> {codec!r}: unknown codec (known: {sorted(known)})")
            continue
        inst = graph.instance_by_name(tag)
        if inst is None:
            out.append(f"{tag!r} -> {codec}: tag is not in the query's dataflow")
            continue
        if codec != RAW and inst.type_sorted is None:
            out.append(f"{tag!r} -> {codec}: tag must be RAW (untyped)")
            continue
        if codec != RAW and scan_candidate is not None \
                and inst.producer.kind == "scan":
            cand = scan_candidate(inst)
            if codec != cand:
                out.append(f"{tag!r} -> {codec}: the scan path materializes "
                           f"only RAW and {cand} for this column")
    return out

