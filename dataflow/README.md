# `dataflow/` — the column dataflow graph and plan-driven codec resolution

This directory turns a query's prepared plans into a graph of the runtime
pipelines and the column instances that flow between them, and then uses that
graph to decide, per instance, which compressed representation its producer
must emit. Both steps are plan-only: they read the prepared plan, hold no
runtime state, and yield one deterministic answer per query. That is why they
live here rather than in `adaptive/`, whose modules make runtime-informed
decisions.

The two pieces fit together like this:

1. **`dataflow_gen.py`** builds the graph **in memory**. `main.py` calls
   `build_query_dataflow(query_plans, schema, qid)` right after
   `parsing.prepare_query_plans`, on every query occurrence (warmup included),
   so the graph always describes the plans that occurrence executes, with that
   occurrence's payload-deferral decision. Nothing is checked in. The result
   is a `DataflowGraph`: one `PlanGraph` per plan (a subquery plan first),
   each a list of `PipeNode` objects in execution order, whose inputs and
   outputs are `ColumnInstance` objects. **The runtime holds these objects.**
   The builder attaches to every operator the nodes it will run
   (`args['nodes']`: `scan`; `pipe` for a filter, project or global
   aggregate; `pass1`/`pass2` for a keyed aggregate; `sort[0]`, `sort[1]`,
   `core` and `gathers['left'|'right']` for a join; `deferred` on whichever
   operator first reads a deferred payload group; `terminal` for a Sort,
   `take ordered` or output). An operator builds its `VortexPipeline` from its
   node and `(key, instance)` input pairs, and writes each output instance's
   payload (`inst.ccr`). **Every gather is a node whose inputs carry a
   role**: its *sources* (every input but the last, read over UVA or
   streamed onto the GPU, and freed by the gather) and its *index* (the last
   input, read through `col_to_gpu`). A join's per-side `rearrange` gathers
   the side's payload (`materialize_now` minus the keys) by the core's
   `out_index`; a deferred column instead rides a row lineage across its
   join chain — the seed join's `out_index`, then one `compose` node per
   later join (`join <op> lineage <side> g<group>`) — and one `deferred`
   node per group (`deferred gather <op> g<group>`) gathers the group's
   columns just before the op that first reads them. `rearrange_pipeline`
   runs any of the three off its node. The graph therefore holds exactly the
   pipelines that run, in the order they run, which the granule cache's
   position counter relies on (`NativeGranuleCache.advance(node)` asserts it). One
   lifetime rule frees every intermediate after its last consumer: the
   gather frees its sources, and its index when it is the index's last
   consumer (a seed's `out_index` survives to the compose or deferred gather
   that reads it; a group split across two ops keeps its lineage until the
   second). A codec, a cache position (`next_use_op`), a
   consumer count (`expected_reads`) and a residency flag (`host_resident`)
   are read off the instance. Names (`node.name`, e.g. `join 7 core
   pipeline`; `inst.name`, `"<producer>:<local>"`) are labels for perf
   reports, sizing keys, the fixed-search pins JSON and logs; nothing is
   looked up by name at runtime. Positions restart per plan. Building all 22
   queries takes about 17 ms. `graph.to_dict()` is the JSON shape the
   checked-in graph used to have, for the debug dump and the tests.
2. **`compression_applier.py`** resolves codecs off the instances.
   `host_resident` follows from the gather roles: every source and index
   plus every `out_index`. `CompressionApplier.codec(inst)` returns RAW for
   an intermediate under `TQP_RAW_INTERMEDIATES`, the pin for a
   `--compression_search fixed` instance, the typed `CompressionPolicy` pick
   (`compression/format_policy.py`) for a typed instance, and RAW for an
   untyped one. A compose's lineage is typed as an int and a deferred
   gather's columns as their columns. A gather's sources follow the policy
   like any instance: a gather streams a compressed source onto the GPU
   (`IO/pipeline/rearrange_pipeline.py`). Producer and consumer hold the
   same instance, so they agree on its codec.

The granule cache's Dataflow policy counts pipelines with
`NativeGranuleCache.advance(node)` and ranks each granule by its producing
instance's `next_use_op`. The applier is built in `main.py` and consulted by
`IO/pipeline/vortex_pipeline.py` for every input and output instance. Tests build the
graph the same way the runtime does, through `tests/dataflow/plan_graphs.py`:
`test_gathers.py` (the builder's gather invariants, one gather per deferred
column, one seed per lineage, the roles),
`test_applier_modes.py` and `test_type_inference.py` (the applier), and
`--compression_search fixed` (a JSON of per-tag codecs applied through the
applier instead of a search) is covered by `test_fixed_search_json.py` with
`tests/verification/SF1_fixed_search.json` as the checked-in example. Design
background is in `adaptive/README.md` (the decision menu the applier feeds)
and `docs/materialization_unification_plan.md`.

To inspect a graph, dump it (local only; never check the output in):

```
python3 -m dataflow.dataflow_gen --q 1 3 6 21 --out /tmp/dataflow
```

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Empty marker that makes `dataflow` an importable Python package. |
| `dataflow_gen.py` | The object graph and its builder. `ColumnInstance` (producer, `local` col id or label, consumers in execution order, `type_sorted`, `host_resident`, `pin`, `ccr`), `PipeNode` (op id, kind, name, position, inputs with their consumer-local keys, outputs), `PlanGraph` and `DataflowGraph` (`instance_by_name`, `to_dict`). `build_query_dataflow(query_plans, schema, qid, sorted_cols)` walks each prepared plan twice (pass 1 collects every plan's class map so a subquery output consumed by an earlier-listed plan types the same). The `_Builder` records, per runtime pipeline, the instances consumed and produced. Scans produce base columns typed from the schema, sorted when `sorted_cols` names them (the scan path's own rule; `main.py` passes none, as the scan marks no column sorted); filters use simple late materialization (they consume only predicate columns and emit a mask); pure-rename projects propagate aliases without a node; keyed aggregates become two nodes (pass1 emits per-chunk `inter_hash`/`inter_key_i`/`inter_val_i` instances that pass2 reads); joins become left and right sort nodes, a core node with `sorted_col_*`, `global_index` and `out_index_*` instances, then per side, in run order, a rearrange node when the side has payload to gather now and one compose node per deferred group composed there; a deferred node per group is emitted before the op that first reads it. `_mark_gather_roles` sets `host_resident` from the roles and `check_invariants` asserts that each gather is the last consumer of its sources, that no node reads a deferred column's pre-chain instance after its chain starts, and that every lineage is produced before it is read and shared by no two groups. Sort, take-ordered and output operators are terminal nodes (position `None`) recorded as consumers so the granule cache keeps their inputs alive; a Sort's inputs include the mask of a filter right below it. A static type-inference pass maps plan expressions to the policy classes `int`/`float`/`date`/`string`; a compose's lineage is an `int`, a deferred gather's outputs take their columns' classes. `dump_all` and the `__main__` write the graph as JSON for local inspection. |
| `compression_applier.py` | Plan-driven codec resolution over instances. `CompressionApplier(graph, policy=..., pins=...)` writes the validated pins onto their instances and answers `codec(inst)`, `candidate_for(inst)` (the single non-RAW candidate the adaptive decision menu is built from), `output_codecs(node)` and `input_codecs(node)`, all through one chokepoint: `TQP_RAW_INTERMEDIATES`'s RAW first, then the pin, then the typed policy pick, then RAW for an untyped instance. Under `--compression_search fixed`, `load_fixed_search_json` reads per-tag codec pins and `validate_pins(graph, pins)` rejects an unknown codec or tag, a non-RAW pin on an untyped instance, and a scan pin naming a rep the scan path does not build. |
