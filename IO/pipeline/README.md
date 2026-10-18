# `IO/pipeline/` — the tick loop and join materialization

This subpackage holds the two ways an operator runs over columns:

- `VortexPipeline` streams host-resident columns through the GPU in
  software-pipelined ticks;
- `rearrange_pipeline` materializes join outputs by gathering rows, over
  UVA or from the source streamed onto the GPU.

`common.py` holds the helpers they share. `pipeline/` sits at the top of the
`IO/` import graph: it imports from `pages/`, `telemetry/`, `codec/`, `cache/`
and `transfer/`, and reaches `adaptive/` only through the hooks in
`adaptive/adaptive_driver.py` and `adaptive/instage_optimization.py`. Nothing
else in `IO/` imports it.

Consumers:

- `VortexPipeline`: `operators/filter.py`, `operators/project.py`,
  `operators/aggregate.py` (pass 1, pass 2 and the single-pass pipe) and
  `operators/hashjoin.py` (the partitioned sort pipe and the join core).
- `rearrange_pipeline`: `operators/hashjoin.py` (join rearranges and lineage
  composes) and `main.py` (`_materialize_deferred`, the deferred gathers).
- `col_to_gpu`: `operators/sort.py`, `operators/output.py` and
  `rearrange_pipeline` itself.

`IO/README.md` ("One tick, end to end") walks through one steady-state tick.
This file covers what that walkthrough skips: how a pipeline is built, the
loop's order and invariants, the adaptive hooks, and the other two pipelines.

## Building a `VortexPipeline`

**Production form.** Every operator builds its pipeline from its dataflow node:

```python
VortexPipeline(node=node, inputs=node.input_pairs(), layout=..., output_columns_type=[...],
               operator=fn, applier=applier, page_cache=page_cache, cpu_mem_pool=...,
               chunk_size=sizing_resolver.resolve("<class>", name).chunk_b, sizing_key=("<class>", name))
```

- `inputs` is a list of `(key, ColumnInstance)` pairs. The key is the name the
  operator sees in its column dict. The instance supplies the payload (`ccr`,
  asserted non-`None`), the codec and the cache bookkeeping. Nothing is looked
  up by name, and the pipe's name is `node.name`.
- The **bare form** (`input_columns={key: CombinedColumnRep}`, `name=...`, no
  node) is only used by tests. It uses the fixed codec everywhere and
  placeholder cache bookkeeping, and it cannot take a layout or an applier.

**Codecs are resolved once, at construction.**

- With an applier, each input's codec is `applier.codec(inst)` and each output
  slot's is `applier.codec(node.outputs[i])`. Rep selection is strict: an input
  whose `CombinedColumnRep` lacks the resolved rep raises, naming the pipe and
  the column.
- Without an applier (`--compression RAW`, bare pipelines) every column uses
  `get_fixed_compression()`, and selection falls back to the first rep.
- `output_compression_algs=` overrides the output resolution, but no caller
  passes it.

**`OutputLayout`** maps physical output slots onto the node's logical output
instances.

- `flat(n)` gives one slot per instance.
- `partitioned(n, B)` gives *B* consecutive slots per instance: the hash-join
  sort pipe and a partitioned aggregation's pass 1 emit one slot per hash
  bucket. Each slot is `(instance, b)`.
- The pipeline derives every per-slot fact from the instance:
  - `next_use_op` and `expected_reads` for the granule cache;
  - `host_resident`, which keeps the instance out of the cache;
  - `b` as the stream (bucket) of its records, which the cache's Dataflow
    policy reads as the use-order hint (a lower partition is consumed sooner).
- The *B* slots of one instance form a **slot group**, keyed by its first
  (base) slot. The group must carry one codec: `_execute_post_comp` asserts it,
  and only the base slot publishes a write decision.

**Where results land.** `get_result()` calls `torch.cuda.synchronize()` and
returns nothing: the outputs are the region producer's tables. With a node and a flat layout it also sets each output
instance's `ccr`, once, as a `variable.TableVariable` over its frozen table,
or, for an instance that wrote no row, an empty `Variable` typed from the
declared output dtype. `result_tables()` gives those tables by instance (by
base slot for a bare pipeline). A partitioned layout is left to its operator:
`operators/hashjoin.py` wraps the sort's tables itself and
`operators/aggregate.py` its pass 1's (an output with no row as an empty
column); a table's row space is its *B* streams in order, so a reader of
the whole column reads it bucket by bucket. `OutputLayout.collapse` is used
only by tests.

**Regions.** Every pipe writes its outputs as regions
(`IO/codec/region_producer.py::RegionProducer`: one region per tick and
column, compressed at tick t, packed at t+1 (before the operator under the
tick barrier, after its launch with event-ordered ticks), copied home
by the home writer in the background, the last tick's packed by the drain),
so a pipeline needs `cpu_mem_pool` (asserted). An operator takes each output
with `output_variable(i, tensor_type)`: the published `TableVariable`, typed
under the operator's name for it, or an empty `Variable` when the instance
wrote no row.

**With the granule cache** (`IO/cache/native_cache.py`, passed as
`page_cache`) a pipe runs as without a cache, and its producer gets the cache and the output instances it
may hold (`cached_outputs`: every instance the graph does not keep in host
memory, with its consumer position and expected reads). The pipe calls the
cache's `advance(node)` at its start and `advance_tick()` at every tick, and
pins the granules each tick reads (`_read_granules`, the stages' cached byte
spans) before any admission of that tick. Under the barrier
(`_pin_granules`): ticks 0 and 1 before their launch, every later tick at
the previous tick's pre-plan. With event-ordered ticks: at the top of each
tick, before its bind, the granules of the chunk it issues and of the one it
computes (see "Event-ordered ticks"). At its end it drops the pins and counts
the last tick's reads (`settle_reads`) before the drain's pack admits.

**Construction also asserts** that every input rep is contiguous and pinned
(a record table of regions always is).

## Chunking and the tick count

- **Uniform geometry** (no `div`). All input columns must have the same row
  count (asserted). The tick cutter (`tick_cutter.py`) cuts them:
  - **In quanta.** Records hold a power-of-two number of rows, cut from the
    start of their row space or producer slice, so the input columns' record
    boundaries nest. The cutter takes the boundaries every representation of
    every input shares (`common_bounds`) and keeps those a whole number of
    the pipe's *quantum* past the last origin (`quantum_points`; the quantum
    is the most rows per record among the inputs and the 1-byte masks the
    pipe writes, so 2^20 rows for a pipe that reads or writes a mask; an
    origin is row 0 or the end of a slice's short last record). Each tick
    then takes the most quanta whose raw bytes, `rows × Σ(width × element
    size)`, fit the budget (`pack`): **the budget rounds down to whole
    quanta** (56 B a row with a mask: 224 MiB of 256), and a tick reads
    whole records of every column.
  - **By rows** (the fallback): when a quantum (the largest piece between
    two kept boundaries) costs more than the budget, or a representation is
    not a record table: `max(1, chunk_size // Σ(width × element size))` rows
    per chunk, the reader trimming the records at a tick's ends.
  - `chunk_size=None` means `constants.get_chunk_size_b()`, resolved at
    construction. Operators pass `sizing_resolver.resolve(...)` instead, so the
    sizing plane can set the budget.
  - `stages = num_chunks + 2`: the row boundaries are the chunks' ends
    followed by two zero-length chunks.
- **`div`** holds one list of cumulative row boundaries per input column, so
  columns of different lengths can be split tick by tick. Here
  `stages = len(div[0]) + 2`. The join core uses it: its left and right inputs
  differ in length, and each tick packs a group of whole radix partitions by
  the same greedy rule (`tick_cutter.pack`); a partition's boundaries are
  record boundaries of every column. A `div` pipe is not uniform, so it can
  never be re-cut.
- **Slices.** Each tick's slice of a column whose two ends are record
  boundaries of every representation reads those records whole
  (`TableVariable.whole_records`: the selection is the record range, with
  no row search, no trim after decoding and no row offset in a zero-copy
  read); any other slice is `get_sliced_rep` and trims. `tick_whole[t]`
  says which ticks read whole records; the debug log counts the others per
  pipe (`[ticks]`).

In-stage *t* carries chunk *t*, and the compute at tick *t* runs chunk *t−1*:

- Tick 0 is the transfer-only prologue.
- The last two ticks carry the empty chunks: tick `stages−2` transfers
  nothing new, and tick `stages−1` computes an empty chunk and drains the last
  real output.
- The producer does not pack at the last tick: the drain packs its
  predecessor's regions.

When every input of a tick is empty, `_execute_pre_comp_and_operator` returns
empty tensors without calling the operator. Operators that count invocations
in closure state rely on this: the join core's per-tick bucket schedule
skips all-empty ticks for exactly this reason.

Every tick's `TickStages` (its in-stages and pre-comp stages) is built at
construction. Each tick's `DirectIn`s get their compiled layout
(`DirectIn.install_compiled_tick_layout`).

### Per-column static decisions

`adaptive.instage_optimization.decide_column_type` runs once per input column.
It picks `zero_copy_mask` when the column has a RAW rep entirely in pinned
host memory and its mask keeps fewer rows than `--mask_selectivity`;
otherwise it picks `direct` with the resolved codec. Rearranges only happen in
`rearrange_pipeline`, which always gathers over UVA.

**Masks.** `mask_map` is `{data key: mask key}`.

- Mask columns are left out of the operator's dict unless `pass_mask=True`.
- Masked data columns are compacted in `PreCompStage.apply_mask`. The
  compaction uses a popcount that `DirectIn` queues on `h2d_stream`, which
  `_materialize_mask_counts` reads with `.item()` on that same stream after
  the transfer sync. With event-ordered ticks the count is stored into the
  pipe's pinned count ring over UVA at the input issue (`tqpcompress.uva_store`,
  so it does not queue on the copy engine behind the copies home in flight)
  and read a tick later, behind that issue's event (`_ring_mask_counts`,
  `_read_mask_counts`).
- With `fold_mask=True` (with `pass_mask`; hashjoin's sort pipes) the operator
  applies the masks itself: a direct column skips the compaction and reaches
  it with the mask's rows, and the operator is called as
  `operator(dict, counts)`, `counts` each mask's kept rows by its key. A
  zero-copy column is still compacted by its read.

**The operator contract.** `operator(dict)` (or `operator(dict, counts)` under
`fold_mask`) returns a tensor, a sequence of tensors, or `PartitionedTensor`s;
a 0-dim tensor is wrapped in a list. Items are consumed in slot order, and a
`PartitionedTensor` fills *B* consecutive slots.

## `do_exchange`, in the order the code runs it

A pipe runs its ticks one of two ways, chosen per pipe:

- **Under the tick barrier** (`_barrier_ticks`): every tick ends with the
  host waiting for the compute and exchange streams. This is the order
  below, and a pipe takes it while its tick observer (a compression
  search's, or an explicit one) is live: its samples assume a quiet tick.
- **Ordered by events** (`_event_ticks`, see "Event-ordered ticks"): every
  other pipe, with or without the granule cache, and the rest of a pipe
  whose observer has settled. A pipe with no observer runs events from its
  first tick; `--compression_search fixed` installs no observer, and the
  tick sizing session measures on events.

**The hand-over.** After each steady barrier tick, once the observer has
settled (`_searches_settled`: its `settled()` is true — it emits no more
picks), `_barrier_ticks` returns the next tick and `_event_ticks` takes over
from it: tick *start* is rebuilt from the final pick state and every later
tick built from other picks is dropped, to be built when the loop reaches it
(`_hand_over_to_events`; the event loop reads telemetry two ticks late, past
the t+2 rebuild horizon, so the observer is dropped), chunk *start−1*'s mask
counts are read from the GPU before the first issue, and its granules are
pinned with chunk *start*'s. An explicit observer (`tick_observer=`, the
probes') and an observer without `settled` keep the barrier throughout.

If the ticks raise with a region producer's copies home in flight,
`do_exchange` first calls `RegionProducer.abort`, which waits for those
copies (the buffers they read and the homes they write would otherwise be
freed under them) and gives the column chunks back. With event-ordered
ticks it first waits for the compute and exchange streams and then empties
every tick's hold list (`_abort_event_ticks`).

One host thread drives everything. Each pipeline has a compute stream (see
`common.py`); `h2d_stream` and `d2d_stream` belong to the
`IO/transfer/data_exchange.py` singleton that every pipeline shares, and the
home writer's stream (`IO/codec/home_writer.py`) takes the copies home.

1. **Entry.**
   - With the cache, `advance(node)` moves the dataflow policy's position.
   - `perf_logger().start(name)` starts the timer, and
     `cmp.reset_window_peak()` starts the arena-peak bracket.
   - A tick sizing session is opened when one applies (see "The tick sizing
     session" below), and a fresh `ResourceMonitor` is built.
2. **Tick 0.** With the cache, `advance_tick()` and pin `ts[0]`'s granules;
   launch `ts[0]`'s in-stages, sync, collect transfer times, `_emit_tick()`.
3. **Tick 1.**
   - With the cache, `advance_tick()` and pin `ts[1]`'s granules.
   - Launch `ts[1]`.
   - Compute chunk 0 on the compute stream (decompress, operator, the
     producer's compress).
   - With the cache, pin `ts[2]`'s granules.
   - Sync both sides, collect, `_emit_tick()`.
4. **Ticks 2 … `stages−1`**, until the observer settles. Each tick:
   1. With the cache, `advance_tick()`; then materialize `ts[i]`'s mask
      counts, then the `TQP_PICK_DEBUG` assert.
   2. `exchange.launch(ts[i].in_stage)`. The region producer learns the
      ticks still producing output (`ticks_left`).
   3. On the compute stream: the region producer's pack of the previous
      tick's records (not at the last tick), then decompress, zero-copy,
      mask, run the operator, post-compute (the region producer's compress).
   4. With the cache, pin `ts[i+1]`'s granules (host state only).
   5. `compute_stream.synchronize()` and `exchange.sync()`.
   6. Collect (`_tick_collect`): the tick's lanes
      (`exchange.get_last_times()`) go to the `ResourceMonitor`; the home
      writer's copies that have ended join the d2h lane, and the pack's
      span joins `compress_out`. Then
      `_emit_tick()`.
   7. `ts[i−1].clean_intermediates()` on the compute stream.
5. **Exit.**
   - Clean the last tick.
   - With the cache, release the pins and count the last tick's reads
     (`replace_consume_pins()`, `settle_reads()`).
   - Drain the region producer: the last tick's pack and a wait for every
     copy home.
   - Close the tick sizing session.
   - Record `peak_gpu_bytes` (`TQP_EXPERIMENT_TELEMETRY`), and stop the
     timer.

The invariants this order protects:

- **Pins come before admission.** A tick's cached granules are pinned
  before any admission that could reclaim their slots: ticks 0 and 1 before
  their launch, every later tick during the previous one, before the next
  tick's pack admits. A reclaimed slot read by the next copy would feed
  nvCOMP garbage.
- **Under the barrier the loop is fully synchronized every tick.** Overlap
  happens only inside a tick, between the exchange streams and the compute
  stream.
- **The window peak brackets the whole pipe.** It covers the operator
  space's resident allocations and this pipe's transients, never the cache
  region (`memory/README.md`). With a tick sizing session it
  is re-baselined each tick through `_WindowPeakTracker`, and the running
  maximum keeps the whole-pipe figure.

## Event-ordered ticks

Tick *i* issues chunk *i*'s input, computes chunk *i−1* and packs chunk
*i−2*'s outputs. Two events per tick, each from a ring of three per pipe:
**H_i**, recorded on `h2d_stream` after the input issue (which has already
waited for `d2d_stream`), and **T_i**, the tick event, recorded on the
compute stream after chunk *i−1*'s compress and a `wait_event(H_i)`: what
the barrier of tick *i* waited for. Each tick *i*:

0. With the granule cache: `advance_tick()` (the reads tick *i−1* bound
   settle), then pin the granules of chunks *i* and *i−1*
   (`replace_consume_pins`): chunk *i*'s copies run beside this tick's pack
   and a lend before its launch would move them, and chunk *i−1*'s compute
   reads its granules in place. Chunk *i−2*'s reads need no pin: every slot
   write that could reuse their slots from here on is on the compute stream
   behind T_{i−2}, which waited on their copies, or behind a lend's device
   sync.
1. Bind chunk *i*'s input, host only (`NaiveExchange.bind`:
   `DirectIn.bind_input`, the source pointers into the compiled layout). A
   cached region's home pieces carry the events of the copies into their
   homes (`BoundTickPlan.home_waits`); the H2D lane waits on them before its
   copies (`DirectIn.launch_h2d`).
2. `_tick_wait(i)`: wait for T_{i−2} (normally fired already), then release
   `ts[i−3]`, whose chunk tick *i−2* computed: its hold list and
   `clean_intermediates`. At most three ticks' inputs are alive.
3. `h2d_stream` waits for the compute work queued so far (chunk *i−2*'s;
   see the hold list below), then issue chunk *i*'s input
   (`exchange.launch(…, bound, holds=ts[i].holds)`: allocate, H2D and D2D,
   finalize), store each mask count into the count ring over UVA
   (`_ring_mask_counts`), record H_i. Chunk *i*'s copies overlap chunk
   *i−1*'s compute, launched next.
4. `_read_mask_counts(i)`: chunk *i−1*'s counts from the ring, behind
   H_{i−1} (only a pipe that reads a mask).
5. Compute chunk *i−1* on the compute stream (decode, zero-copy kernels,
   masks, the operator).
6. The region producer's pack of chunk *i−2*, after the operator's launch
   so the operator runs while the host waits for the compressed sizes; on
   an empty tick too, never at the last tick (the drain packs, as under the
   barrier).
7. Compress chunk *i−1*'s outputs, the end event, `wait_event(H_i)`, record
   T_i.
8. Tick *i−2*'s telemetry (`_tick_collect`): its own closed copy log (the
   launch returns it), its compute events (`_ComputeEvents`, a ring of
   three), then `_emit_tick()`; the log is released.

Tick 0 runs steps 1, 3 and T_0; tick 1 has no step 2, 6 or 8. After the
last tick the pipe waits once for the compute and exchange streams, collects
and releases the ticks left, in order, and drains.

**The hold list** (`TickStages.holds`) keeps every buffer a tick's input
issue allocates on `h2d_stream` that the compute stream reads: the agg_buf
and the decoded outputs (`DirectIn._alloc_agg_buf_and_bind_views`), the mask
counts (`InStage.issue_holds`), and a zero-copy stage's decoded mask (for a
RAW mask, a view of its own agg_buf). The allocator orders a free only on
the allocating stream, so without it the next tick's allocation could take a
buffer the compute stream still reads (`IO/transfer/README.md`, stream
rules). A buffer allocated on the compute stream and used only there is
freed as under the barrier. The wait of step 3 (the issue behind the
compute work queued so far) is empirical: without it SF100 Q16's
aggregation 15 pass 1 faults with an illegal memory access, even with the
allocator's force-drain guarded (`memory/README.md`); the race is between
the input issue and chunk *i−2*'s compute, and its cause is not yet pinned
down. The rule for new code: a buffer an in-stage
allocates on the h2d or d2d stream and the compute stream reads goes on the
hold list. The barrier fills no hold list.

**Memory.** Three ticks' inputs are alive against two under the barrier,
chunk *i−2*'s worst-case slab lives through chunk *i−1*'s operator, and
frees the host makes ahead of the GPU stay pending in the allocator until
their stream passes them (they shrink the largest free run, not the
allocated bytes).

## Adaptive hooks

With both searches off, none of this code runs, and the pipeline builds and
executes exactly the static plan.

**Menu and observer.**

- A `DecisionMenu` is built only when an observer will exist: either
  `tick_observer=` was passed (no production caller does), or a factory is
  installed and `stages ≥ tick_observer_min_ticks()`.
- `_build_decision_menu` identifies each rep's codec from its first record
  (`leading_codec_name`), since a rep is single-codec.
- Only two-rep (decision) columns pay for a per-record byte walk. Doing the walk
  on every column measured +3–35% wall time on join-heavy queries.
- Output ranges come from the resolved output codecs. The non-base slots of a
  slot group publish locked singletons.
- Per-slot output bytes are accumulated for the observer only when some
  output is a decision.

**`_emit_tick`** runs after every tick's collect, the prologue included.

- It calls the observer with the monitor. A `PickPair` return splits into
  input picks, output picks and `seal`; a plain sequence is input picks.
- Input picks are checked with `validate_pick_vector` and written into the
  live codec list, for `direct` columns only. A pick on a zero-copy column is
  legal but has no effect.
- Output picks or a seal go to `_actuate_write_picks`.
- Then it always calls `_jit_rebuild(last_tick + 2)`.

**`_jit_rebuild(t)`** (the *t+2* rule).

- It compares `_effective_pick_state()` with the state tick *t* was built
  from. The pick state is each column's codec, with RAW normalized and `None`
  for zero-copy columns. Equal states build identical stages, so the rebuild
  is skipped.
- Otherwise it rebuilds only `ts[t].in_stage`, installs its compiled layout,
  and re-points `ts[t+1].pre_comp[j].in_stage_ref` at the new stages. Without
  that re-point, the compute would read the old stages, which never launched.
- Tick *t* is pre-planned during tick *t−1*, which starts after this call,
  so the rebuild is always in time.
- `TQP_PICK_DEBUG` records the expected state at this horizon and asserts it
  when tick *t* launches.

**`_actuate_write_picks`** (*t+1*, `read-and-write` only). It runs inside
`_emit_tick`, after tick *t*'s sync and before tick *t+1*'s post-compute.
It validates the output vector, spreads each group's base pick to the whole
group, and retargets the live output codecs, which the next compress reads.

Every tick writes regions of its own, so a seal with no codec change (a null
boundary) changes nothing. The first real flip sets
`_write_flipped`.

**The tick sizing session and `_resection`.** A pipe gets a
`TickJumpSession` (`adaptive/sizing_controller.py`) when all of these hold:

- it has a `sizing_key`, and no explicit observer;
- its geometry is uniform, and its row width is non-zero;
- the factory accepts it: a re-cuttable class (sort, filter, project) or
  `sectionable=True` (aggregation pass 1), not `sectionable=False`, at least
  `tick_samples + 3` data chunks, and no size fixed by an earlier
  occurrence.

On event-ordered ticks each chunk after the first (the first whose previous
tick also ran on events, after a hand-over) goes to the session when its
tick is collected (`_observe_chunk`): its built budget, its input bytes, its
tick time — the gap between its compute's end and the previous chunk's on
the GPU timeline (the `_ComputeEvents` ring, four deep for this), plus its
readback: its outputs' stored bytes times their later crossings (home and
each consumer read; `_charge`, at the region producer's ship and pack) at
the probed PCIe rate — the arena's peak while its tick ran (the tick's
`_WindowPeakTracker` bracket, whose running max keeps the whole-pipe peak),
and the input bytes not yet issued.  The epilogue's empty chunk is not fed.

A budget the session returns calls `_resection(i + 1, budget,
build_ahead=k + 3)` at the end of tick *i*:

- It is refused (`False`, then `recut_refused()`) when only the epilogue
  remains or no rows remain.
- Rows before `boundaries[i]` keep their chunks: tick *i* is issued and tick
  *i+1*'s pre-comps point at its in-stages. The cutter re-cuts the
  remaining rows at the new budget, with two new empty chunks.
- It re-slices the host inputs (mask columns share the boundaries) and drops
  the ticks from *i+1* on; they are built when the loop reaches them
  (`_built`), the first `build_ahead` now, so the chunks the session
  measures carry no build of their own. It resets the built-pick records
  and per-tick budgets, and drops stale debug expectations.
- It mutates the stage lists in place, so the loop's `ts` alias stays valid.

- **Small helper.** `_is_raw_algo`.

## Files

| File | Purpose |
|------|---------|
| `tick_cutter.py` | The tick cutter: `common_bounds`, `quantum_points`, `cut` (whole quanta up to the budget, or `cut_rows`), `pack` (the greedy packing, shared with the join core's buckets) and `slice_column` (each tick's slices, whole records where both ends are record boundaries). |
| `vortex_pipeline.py` | `OutputLayout` and `VortexPipeline`. Covers construction, codec resolution and slot facts; chunking (through the tick cutter) and the pre-built `TickStages`; static column decisions; `do_exchange` and the region producer's pack and drain (`result_tables`); the menu, `_emit_tick`, `_jit_rebuild`, `_actuate_write_picks`, the tick sizing session's samples and `_resection`; and `get_result` / `_publish_outputs`. |
| `rearrange_pipeline.py` | `rearrange_pipeline(node, cpu_mem_pool, type, *, applier, page_cache)`; `_read_plan` (UVA or streamed, by `uva_bytes` against the streamed rep's stored bytes and `_fits_on_gpu`), `_stream_in` and `_raw_home` (the tick pipes that load a source onto the GPU or decode it home RAW), and `_stream_gather`: the windowed gather, compress and copy-home loop (regions), with the SZ-B sizing hooks. `_ARGSORT_OVERHEAD_PER_ROW = 24`. |
| `common.py` | `_WindowPeakTracker`, `col_to_gpu` and `PartitionedTensor`. |

## Environment variables

Parsing rules are in `AGENTS.md` ("Environment variables").

| Variable | Read in | Effect here |
|---|---|---|
| `TQP_PICK_DEBUG` | `IO/telemetry/gates.py`, once at import; `vortex_pipeline.py` copies it when a pipe is built | assert each tick launches with the in-stages its *t+2* horizon recorded |

`TQP_EXPERIMENT_TELEMETRY` is read once, by
`IO/telemetry/gates.py`.

## Tests

- `tests/telemetry/test_movement.py` (GPU): the movement ledger on bare
  pipes and gathers.
- `tests/telemetry/test_gates.py` (GPU): with the telemetry switch off, a
  pipe's tick observer still sees each tick's lanes and compute, a sized
  pipe's tick sizing session gets each chunk's tick time and peak, and a
  gather session gets its `GatherIterSample`s.
- Standalone probes: `tests/adaptive/pick_flip_probe.py` and
  `write_flip_probe.py` exercise the *t+2* rebuild and the write boundary.
- `tests/IO/test_tick_events.py` (GPU): a sizing session re-cuts
  event-ordered ticks twice and the rows come out whole.
- `tests/smoke` and the `tests/e2e` ladder run every path end to end; the
  e2e `mid_stream` rung re-cuts wherever its sessions jump, and checks
  results, not that a re-cut happened.

```bash
TQP_GPU=<idle> docker compose run --rm -T dev python3 test.py tests/telemetry
```

## Known gaps

- `PipeStep` (`common.py:100`) has no users.
