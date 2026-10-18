# `parsing/` — Spark physical plan → typed query IR

This directory turns a TPC-H SQL query into the `QueryPlanRep` that the rest
of the engine executes. It has two halves that run at different times:

* **Plan generation** has two steps, both checked in. *Capture*
  (`ParsingLayer.explain`, driven by `regenerate.py`) starts a Spark session over
  empty Parquet tables and saves the *formatted* physical plan to
  `queries/plan-q<N>.txt`. *Lowering* (`ParsingLayer.lower`, pure Python) turns
  that text into a typed operator graph, pickled to
  `queries/pickles/planobj-q<N>.pkl`. The engine never starts Spark: it loads
  the pickle, or lowers the text when the pickle is missing.
* **Post-load analysis** (`pipeline.py`, `uniqueness.py`, `materialization.py`)
  runs on **every** load, fresh or cached. The pickles deliberately hold only
  the stable physical plan; query-specific fixes, unique-key proofs, and
  payload deferral are recomputed so a cached plan and a freshly parsed one
  take one code path.

The pieces fit together like this:

1. **`ParsingLayer.explain(query)`** runs `spark.sql(query).explain(mode=
   'formatted')`; **`ParsingLayer.lower(plan_text, qid)`** splits the text into the operator *tree* (indentation and
   `BuildLeft`/`BuildRight` markers) and the per-node *item* blocks
   (`Input`, `Output`, `Condition`, `Keys`, ...). `parse_item` converts each
   Spark operator into a dict under a canonical name (`Scan`, `Filter`,
   `Project`, `Aggregate`, `Sort`, `take ordered`, `Join`, `Exchange`,
   `AdaptiveSparkPlan`), and `parse_tree` fills `query_parent_map`. If the
   plan has a `===== Subqueries =====` section, the subquery becomes its own
   `QueryPlanRep`, listed **first** so it executes before the main plan. A
   handful of TPC-H lines that defeat the comma splitter (the
   `substring(c_phone, 1, 2)` country-code projection, Q20's
   `0.5 * sum(l_quantity)`) are matched by regex before the general path.
2. **Expression lowering.** Every `Condition`, `Output`, `Results`, and
   `Join condition` string goes through `expression_parser.parse`, a packrat
   PyParsing `infixNotation` grammar covering comparisons, arithmetic,
   `AND`/`OR`/`NOT`, `IN`, `LIKE`, `AS`, `CASE WHEN`, function calls, date
   literals, and Spark's `name#id` column references. Four TPC-H expressions
   that reference a `Subquery` are short-circuited to hand-written trees. The
   helpers in `expressions.py` then read those nested lists: column ids,
   rename maps, which inputs a projection really needs, and aggregate
   keys/ops.
3. **Plan rewrites**, in order: `clean_agg` collapses each of Spark's
   partial/final aggregate pairs onto its final aggregate, a group-only pair
   too (Q16's `count(DISTINCT ps_suppkey)` is planned as a group-only
   aggregate on `(p_brand, p_type, p_size, ps_suppkey)` below the count);
   `remove_exchange` splices out shuffle nodes; then the first two passes of
   `materialization.py` run: `seed_materialization` swaps join sides so the
   **build side is always physical left** (recording `left group`/`right
   group`/`join in`) and seeds every operator's `materialize_in`,
   `materialize_now` and empty `materialize_deferred` (a `Sort` or
   `take ordered` gets only `materialize_in`);
   `apply_simple_materialization` makes a `Filter` emit a mask column
   (ids from `MASK_COL_START = 10000`) and threads a `mask map` down to the
   first operator that must materialize: a join, an aggregate or a sort,
   which applies it (Q11's HAVING mask reaches its final `Sort`). This is
   what gets pickled.
   One per-query fix is part of the lowering too: Q11's subquery plan carries
   `subquery_scale = "1/SF"`, which the runtime applies through
   `subquery_value` when it captures the scalar. The text's SF1 fraction,
   0.0001, becomes 0.0001 / SF as qgen prints it, to 10 decimals
   (`q11_fraction`): the value the baselines' query stream carries.
4. **`prepare_query_plans(plans, qid)`** is the post-load entry point
   `main.py` calls after either branch. It runs `annotate_query_plans`, then
   deferred materialization.

Consumers: `main.py` (`ParsingLayer.lower` on a missing pickle, then
`prepare_query_plans` on every run), `dataflow/dataflow_gen.py` (builds the
in-memory dataflow graph from the prepared plans) and `profiling/sweep.py`
(reads the pickles for scan sets), and
`operators/{aggregate,output,operator_utils}.py` (the `get_id`/`get_name`
helpers and `column_ref`, the one rule that turns an expression token into a
column id and rejects the `Brand#NN` literals). Tests are in `tests/parsing/`,
all CPU-only: `test_lowering.py` checks that each pickle is the lowering of
its checked-in text, captured under the pinned pyspark; `test_uniqueness.py`
asserts the unique-side census over all 53 inner joins in the 22 cached
plans. `operators/hashjoin.py` is the consumer of the uniqueness annotations
and still gates on join type, key dtype, and scratch bounds before trusting
them.

## Regenerating a plan

The plan text and the pickles are checked in, so nothing here runs in a normal
`make q`; the dataflow graph is built in memory from the prepared plan on every
run. Regenerate after changing a query:

```bash
make plans Q="<N>"               # capture queries/plan-q<N>.txt, lower it to the pickle
```

After a change to the lowering code alone, `make plans ARGS=--lower-only`
re-lowers the checked-in text without starting Spark.

The column ids in a plan (`l_quantity#96`) are Spark expression ids, drawn from
one counter in the JVM: they depend on the pyspark version and on everything the
JVM ran before the query. `regenerate.py` fixes both. It refuses to capture
under any pyspark other than the one `requirements.txt` pins (`pyspark==…`), and
it captures each query in its own process, so its own JVM, which runs only the
table setup before the query. Regenerating any subset of queries, in any order,
therefore reproduces the same ids. Each text file's first line records the
pyspark version that captured it, and the Location lines name the scratch
directory as `./tmp` wherever the capture ran. Do not edit the text by hand;
`tests/parsing/test_lowering.py` fails when a pickle is not the lowering of
its text.

Moving the pin renumbers every plan. Rebuild the image, run `make plans` for
all queries, then update everything that names column ids:
`tests/verification/SF1_fixed_search.json`, `tests/parsing/test_uniqueness.py`
and `tests/dataflow/`.

`ParsingLayer` needs `JAVA_HOME`. It writes Spark scratch plus one empty Parquet
directory per table under its `scratch_dir` (default `./tmp/`; `regenerate.py`
uses a temporary directory). It reads column types from
`queries/schema.json` (the `PRIMARY_KEY` entry is skipped there and is **not**
the source of uniqueness proofs; see `uniqueness.py`). The generated plans
depend on the Spark version's optimizer output, so a Spark upgrade can change
node numbering; the lowering names no node id.

## Plan representation

`QueryPlanRep` holds these fields:

| Field | Contents |
|-------|----------|
| `ops_idx` | Spark node ids in execution order (leaves first). |
| `subquery_scale` | `"1/SF"` on Q11's subquery plan, `None` otherwise (a class-attribute default, so older pickles read `None`). |
| `ops_dict` | `node id -> args dict`. Every operator has `name` and `output ids`; most have `input ids`, `materialize_in`, `materialize_now`, `materialize_deferred`. |
| `query_parent_map` | `node id -> [input node ids]` (the operators it consumes). Joins have exactly two after `rewrite_ops`. |
| `query_child_map` | `node id -> consumer node id`. Single consumer per node. |
| `disk_read_cols` | `ScannedTuple(name, identifier, table)` for every base column any `Scan` produces. |

After `prepare_query_plans` the dicts also carry `left unique keys` /
`right unique keys` on joins, and a join's deferred
payload moved from `materialize_now` to `materialize_deferred`,
`deferred_groups` on joins and `gather_deferred` on the operator that
collects deferred payloads. An operator's full output is always
`materialize_now` + `materialize_deferred`; only joins defer. The dataflow builder
(`dataflow/dataflow_gen.py`) turns these into the gather nodes the runtime
executes and attaches them as `args['nodes']`.

## Files

| File | Purpose |
|------|---------|
| `__init__.py` | Public API: `ParsingLayer`, `QueryPlanRep`, `ScannedTuple`, `prepare_query_plans`, `annotate_query_plans`, `annotate_plan_unique_keys`, the three materialization passes plus `consumed_columns`, and the `get_id` / `get_name` / `right_most` / `column_ref` helpers. Importing it loads neither pyspark nor pyparsing. |
| `models.py` | The two dataclasses above. |
| `layer.py` | `ParsingLayer`: Spark session setup over empty Parquet tables built from `queries/schema.json` (pyspark is imported only here), `explain()` (the formatted plan text), and the static `lower()` with the nested `parse_tree` / `parse_item` / `clean_agg` / `remove_exchange` passes described above, the two lowering-time materialization passes, and the TPC-H line-level special cases. `lower()` imports the expression grammar on first use and allocates mask-column ids per query, across its plans. `generate()` is `lower(explain())`. |
| `regenerate.py` | `make plans`: captures `queries/plan-q<N>.txt` under the pinned pyspark, one fresh JVM per query, and lowers it to `queries/pickles/planobj-q<N>.pkl`. |
| `expression_parser.py` | PyParsing grammar for Spark expression strings, packrat-enabled with the recursion limit raised to 4096; loaded only by `lower()`, so those side effects stay out of every other process. `parse()` tries the whole string, then retries with the outer parentheses stripped. Prints every expression and its parse time to stdout, which is why plan generation is noisy. |
| `expressions.py` | Pure helpers over parsed expression trees: `get_id` / `get_name` split `name#id`; `column_ref` is the one rule that reads a token as a column reference (and rejects `Brand#NN`), used by `get_literals` here and by every resolver in `operators/operator_utils.py` and `dataflow/dataflow_gen.py`; `get_literals` collects referenced column ids (ignoring `isnotnull`); `get_project_output_ids`, `get_project_used_cols` (rename map, used inputs, computed outputs), `get_aggregation_key_and_values` (group keys, aggregate outputs, ops), and `is_simple_projection` (pure select/rename, eligible for late materialization). |
| `pipeline.py` | `prepare_query_plans`: the post-load dependency-ordered pipeline (uniqueness, optional deferral); `subquery_value`: applies a subquery plan's `subquery_scale` to the scalar it produced; `q11_fraction`: Q11's FRACTION at a scale factor, as qgen prints it. |
| `uniqueness.py` | Relational unique-key proof propagation. Carries its own `TPC_H_UNIQUE_KEYS` table (composite keys for `PARTSUPP` and `LINEITEM`), propagates key sets through Scan, Filter/Sort (plus `distinct ids`), Project (surviving ids or explicit renames only), Aggregate (group keys), and Join (inner: preserve a side when the other side's equality key is unique, then substitute joined-equal columns; semi/anti: physical right; left outer: right only if left is unique). Unknown operators fail closed. Resolves join sides by matching `left group`/`right group` against parent output ids, because `rewrite_ops` may have swapped list order. |
| `materialization.py` | The three materialization passes, top to bottom in execution order. `seed_materialization` (lowering): build-left join normalization and the `materialize_in` / `materialize_now` seed (with an empty `materialize_deferred`). `apply_simple_materialization` (lowering): filter masks and `mask map`. `compute_deferred_materialization` (every load): for each join, splits its output into columns that must be gathered now (join keys, sort/aggregate keys) and payloads that can ride lineage through a chain of joins to their first consumer after the chain, found with `consumed_columns` (the one table of which columns each operator reads, masks aside). Groups deferred columns by `(source scan, join chain)` so the runtime does one gather per group. |
