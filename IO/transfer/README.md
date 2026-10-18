# `IO/transfer/` — the per-tick stages and the exchange that drives them

This subpackage moves one tick's input. An input stage brings a column's
chunk onto the GPU; a pre-compute stage turns it into the tensor the operator
reads. The operator's output goes to the pipe's region producer
(`IO/codec/region_producer.py`), not through here. `NaiveExchange` owns the
three copy streams and issues every in-stage's copies for a tick in one
`launch` call. The copy streams are kept for the
process (`utility/cuda_streams.dedicated`), so no pipe's compute stream is one
of them.

In the `IO/` import graph, `transfer/` imports `pages/` and `telemetry/`;
only `pipeline/` sits above it.
It also uses `adaptive/resource_monitor.py` (`TransferTimes`), `variable.py`,
`TQPlib.tqpcompress` and `TQPlib.zero_copy_reader`. Consumers:
`IO/pipeline/vortex_pipeline.py` builds and drives every stage;
`IO/pipeline/common.py::col_to_gpu` and `IO/pipeline/rearrange_pipeline.py`
use the exchange directly; `adaptive/instage_optimization.py` builds the zero-copy stages; `main.py`
installs the exchange and reads the counters.

## Where this layer sits in a tick

The tick loop is documented in `IO/README.md`. Under the tick barrier, at
tick *t* the pipeline calls `exchange.launch(ts[t].in_stage)`, runs compute,
then waits on `compute_stream.synchronize()` and `exchange.sync()`. With
event-ordered ticks (`IO/pipeline/README.md`) it binds (`exchange.bind`),
waits on the tick event two ticks old, launches with the tick's hold list,
and records an event on `h2d_stream` that the compute stream waits on. One
`TickStages` holds stages for two different chunks:

| Stage list of `ts[t]` | Works on | Stream |
|---|---|---|
| `in_stage` | chunk *t* | `h2d_stream`, `d2d_stream` |
| `pre_comp` (points at `ts[t−1].in_stage`) | chunk *t−1* | compute |

Under the barrier no CUDA event links the transfer streams to the compute
stream: the host sync at the end of every tick is the ordering. Chunk *t*
lands before tick *t* returns, and its compute runs in tick *t+1*. The one
cross-stream edge is the bind's wait on the events of the copies into a
cached region's home (below). With event-ordered ticks the compute stream
waits on the event after each tick's input issue instead.

**Stream rules** (the allocator as it is): `record_stream` does nothing; a
free is ordered only on the stream that allocated the block, and another
stream can reuse the block once the allocating stream passes the free; so a
buffer used on another stream stays referenced until an event recorded on
that stream after its last use has completed. This holds for force-drained
blocks too: the allocator keeps an unfired free event as a guard and makes
any later taker on another stream wait for it (`memory/README.md`). The
barrier satisfies the rule for every per-tick buffer; with event-ordered
ticks the tick's hold list (`TickStages.holds`) does. A block is ordered
only for the stream that allocates it, so an event-ordered launch makes
`d2d_stream` wait for `h2d_stream` before its D2D fills, and `h2d_stream`
waits for the compute stream before each input issue (empirical, see
`IO/pipeline/README.md`).

With the granule cache the pipeline pins a tick's cached granules before
the pack that could admit and reclaim their slots: under the barrier, tick
*t+1*'s while compute is still running (`VortexPipeline._pin_granules`);
with event-ordered ticks, at the top of each tick, the granules of the
chunk it issues and of the one it computes (`IO/pipeline/README.md`).

## The exchange (`data_exchange.py`)

`NaiveExchange` owns `h2d_stream` and `d2d_stream`, and the copy log of the
launch in flight (`IO/telemetry/copy_events.py`). It is a process
singleton: `set_exchange_to_naive()` builds a fresh one (`main.py` at
startup, and the tests), and `exchange()` returns it.

`launch(in_stages, bound=None, holds=None)` never syncs the host. In order:

1. Without `holds` (the barrier): open a copy log; every copy batch issued
   until `sync()` lands in it.
   With `holds` (event-ordered ticks): open a log of the launch's own, which
   the launch closes and returns; the exchange keeps no log, so a nested
   launch and sync (`col_to_gpu` inside an operator) leaves it alone.
2. `bound` is `bind(in_stages)`'s result (bound here when `None`): the
   in-stages grouped by exact class, in first-seen order, so one class's
   planner sees all its stages at once, each group with what its class
   binds before allocating (`bind_input`; `DirectIn`'s source pointers).
3. On `h2d_stream`, per class: `alloc_input(stages, bound, holds)`, then
   `launch_h2d(state, plan)`. `holds` collects the buffers allocated there
   that the compute stream reads.
4. With `holds`, `d2d_stream` waits for `h2d_stream` (stream rules above).
   Per plan: `state.record_stream(d2d_stream)` when the state is a tensor,
   then `launch_d2d(state, plan)` on `d2d_stream`.
5. `h2d_stream.wait_stream(d2d_stream)`, so later readers on `h2d_stream` see
   the D2D fills.
6. On `h2d_stream`: `build_decompress_metadata` per plan, then
   `phase_finalize(d2d_stream)` for every in-stage. With `holds`, each
   in-stage's `issue_holds()` (its mask count; a zero-copy stage's decoded
   mask) joins the list.

Each copy site records its batch's start and end events right around its
copies (`IO/telemetry/README.md`). `sync()` blocks the host on
both streams, closes and releases the log and stores its lanes as an
`(h2d, d2d)` ms tuple (`get_last_times()`), which the pipeline wraps as the
tick's `TransferTimes` with the home writer's d2h lane: per stream, the span from the first batch's start to the last
batch's end, 0 when the stream copied nothing. A lane holds no host work
before its first copy or after its last; host gaps between its batches stay
in. The `h2d` lane includes, between the zero-copy stages' copies, a
compressed mask's decompress. The pipe's d2h lane is its home writer's
copies (`IO/codec/home_writer.py`), on a stream `sync()` does not drain. A
`sync()` with no launch since the last one leaves the lanes unchanged.

## Input stages

`InStage` (`in_stage.py`) defines the class-level methods the exchange
calls (`bind_input` and `alloc_input`, the two halves of
`plan_input_data_movement`, then `launch_h2d`, `launch_d2d`,
`build_decompress_metadata`) and the per-stage `phase_finalize` and
`issue_holds`; all default to no-ops (`alloc_input` to the whole plan). After the launch the compute side reads `gpu_col` (a `Variable`)
and `mask_count`, and the pipeline reads `raw_numel` to skip empty ticks.

### `DirectIn`: compile once, bind every launch

A `DirectIn` reads a *selection* of its column's record table
(`IO/pages/record_table.py`): a `variable.TableVariable`, what the region
producer wrote or a scan. Its regions are its own, one per producer tick and
column, each holding every bucket of that tick, their homes carved from
pinned chunks, or one per scan batch. `TableVariable.record_selection` picks
the records the tick's rows overlap. A compressed record is copied and
decoded whole; a RAW record is copied only for the rows the tick keeps. The
bind reads each run straight from its region (`_bind_regions`: a pinned
home or scan batch by H2D, a GPU buffer by D2D). A column with no records
(an empty `Variable`) reads nothing.

A tick the tick cutter cut in whole quanta (`IO/pipeline/tick_cutter.py`)
starts and ends on record boundaries of every column, so its slice selects
whole records: its record range, with no row search
(`TableVariable.whole_records`). The selected records of one region form one *run*, one contiguous byte
range and one copy. The operator's rows are then the decoded rows, trimmed inside
the first and last compressed record only for a tick cut by rows (the
fallback): the stage view's
`row_range`, which `phase_finalize` puts on `gpu_col`, and `Variable.data()`
applies. A zero-copy mask stage reads its mask the same way, and a
whole-record source's segments count rows from the slice's first
(`RecordTable.record_segments`), so its kernel reads from row 0. A
compressed record's index stays on the host (the table), so no metadata
crosses with the payload. **Compilation**
(`install_compiled_tick_layout` → `tick_layout.compile_tick_layout`) runs when
the pipeline builds each `TickStages` and on every just-in-time rebuild of a
tick's in-stages, so it stays off the launch path. It makes two passes:

1. **Selections and runs** (`_plan_stage_read`, `_plan_table_stage`). A
   stage whose selection is a contiguous range of records over page-shaped
   regions (one codec per region, a region's records in byte order and its
   RAW records back to back, checked once per table in `_RegionIndex`: a
   scan's batches or a one-stream producer's regions) is planned region by
   region (`_plan_by_region`): Python per region, numpy over the records,
   from per-table prefix sums of rows and decoded bytes. Otherwise, when its
   regions hold several streams (a
   partition sort's), with numpy per run (`_plan_table_read`): compressed
   records are copied in runs of their regions' byte order
   (`RecordTable.runs`: a range of buckets copies one run per region) and
   decoded to their row-order place, RAW records copy their kept rows in
   row order, one run per stretch adjacent both in the selection and in
   their region's bytes. The two give the same read; the runs come back as
   arrays (`run_arr`). Each
   stage's compressed runs are laid out back to back, each 512-aligned, so
   every record keeps the alignment its codec needs; its RAW records follow, dense and in row order, so a RAW
   stage's bytes are its rows. A *mixed* stage, whose producer switched
   codecs mid-run (a write flip under `read-and-write`), has both; its RAW
   runs are copied into the decoded output at launch. A stage holds at most
   one compressed codec. Stages are grouped by manager identity
   (`codec_key = (codec_idx, elem_size, chunk_b)`) in first-seen order; RAW
   stages form one group placed last.
2. **Aggregate buffer, pack entries and templates.**

   ```
   agg_buf = [ codec group 0: runs(stage a | stage b …) ]
             [ codec group 1: … ]
             [ RAW group:     rows(stage x | stage y …) ]
   ```

   A compressed stage starts 512-aligned, a RAW stage 8-aligned, and every
   group ends 8-aligned. Each compressed group also gets a decompressed
   output buffer (its stages' decoded rows × row bytes, each stage's slot
   8-aligned), allocated at launch: it holds the selected records, not the
   regions. Each run gets one pack entry: its `agg_buf` destination offset and
   bytes, two immutable lists the H2D and D2D lanes share. Each lane has one
   mutable source-pointer list. A stage's `_TableRuns` gives its entries and,
   per run, its region, its offset into the region and the bytes its records
   decode to. Each compressed group gets a
   `CompiledManagerTemplate`, compiled from the table: per selected record,
   its compressed and output offsets relative to the two base buffers as
   frozen numpy arrays, reusable absolute arrays and CPU int64 tensors that
   serve as the pybind arguments, and the records' saved
   `DecompressionConfig` handles when every record carries one, as every
   record the engine writes does.

The layout's geometry is trusted between compile and launch: a record
table's regions and records never change. Device residency and absolute
pointers are bound at every launch, so a cached granule that moved between
its slot and its home since compile needs no recompile.

**Binding** (`bind_tick_sources`, on the launch path) picks a lane per run,
by its region, and overwrites both lanes' source slot with the region's base
plus the run's offset; a zero in a lane means "skip this run there":

| Region | Lane | Source |
|---|---|---|
| a `TableVariable`'s host region | H2D | the region's home or scan batch |
| a `TableVariable`'s GPU region | D2D | the region's buffer |
| a `TableVariable`'s cached region (granule cache) | both | piece by piece (`NativeGranuleCache.pieces`: one call per tick for the stages that read only cached regions, `_TickCached`, and one for each other stage's cached runs): granules in slots by D2D, the rest from the home by H2D, after the copies into those home bytes (the events `pieces` returns go to `BoundTickPlan.home_waits`, and `DirectIn.launch_h2d` makes the H2D lane wait on them: the bind runs on the legacy default stream, which orders none of the exchange's non-blocking streams) |

A run of a cached region is split into pieces and its own entry is skipped
in both lanes. Each home piece becomes a pack entry, appended to the
layout's entries (the plan's `pack_args` and `d2d_pack_args`); the slot
pieces, merged where one continues the previous in source and destination,
go to the plan's `slot_pieces` and are copied by one `batched_memcpy_abs`
launch on the D2D lane (`launch_slot_pieces`, after `batched_d2d_pack`).
The stages of a tick that read only cached regions are bound together:
`compile_tick_layout` concatenates their cached spans, agg_buf offsets,
settle spans and granule ids once (`_TickCached`, with the read plan,
`TQP_CACHE_PLAN`), and the bind makes one `pieces` and one `defer_read`
call for all of them; a stage that also reads other regions is bound on
its own. Once the bind holds, the cached byte spans are handed to the
cache (`defer_read`), which counts them at the next tick boundary or
pipeline start, after the copies have ended (`cached_read_ranges`, arrays
of region indices and byte bounds, computed once per column, or the
layout's; a record the tick ends inside is left out, so its stream's last
read is counted once). The same spans, whole, are what the
pipeline pins for the tick (`VortexPipeline._pin_granules`).

Under `TQP_EXPERIMENT_TELEMETRY` the bind records each run in the movement
ledger (`IO/telemetry/movement.py`) by its region's kind: one page, its
selected records' decoded bytes and its bytes on the wire. A stage's
runs are `scan_h2d` (a scanned table's, `RecordTable.base`) or
`restore_h2d` (a producer's) from host, `resident_d2d` from the GPU, one
page per run; a cached region's pieces are `cache_hit` (in slots) and
`restore_evicted_h2d` (from the home: granules the cache evicted or never
admitted), one page per piece, decoded bytes shared by the pieces in
proportion to their bytes.

A layout binds exactly once, and a second bind raises. A launched
layout stays acquired because the compute stream reads its template's CPU
pointer arrays later; `BoundTickPlan.release()` releases it where nothing
is left to launch: a bind with no bytes to copy, and the zero-copy mask
stage once its copies and decode are issued (and tests).

**Launch.** `plan_input_data_movement` is `bind_input` then `alloc_input`.
`bind_input` (host only) gives empty stages an empty `Variable` and binds;
`alloc_input` allocates `agg_buf` on `h2d_stream` plus one shared output per
compressed group (both on the tick's hold list when one is given), and
points each stage's buffers and group context at their slices. `launch_h2d` issues **one** `batched_h2d_pack` for the tick;
`launch_d2d` issues **one** `batched_d2d_pack`, skipped when no run uses the
D2D lane. Each takes `(agg_buf, sources, destination offsets, sizes)`,
validates every destination before copying anything, and issues one
`cudaMemcpyAsync` per run with a non-zero source, with an explicit copy
direction. The two write disjoint regions of `agg_buf`, so neither waits on the
other. The slot pieces' `batched_memcpy_abs` goes through
`pages/abs_copy.py`'s `launch`, which stages its address arrays through the
pinned buffers that module keeps across launches (shared with the region
producer's packs; at most eight, a buffer too small for a launch is
replaced, not kept beside it); the kernel reads them over UVA, adds the
agg_buf's address to the destination offsets, and the launch and the
buffer's event record are one native call. `DirectIn` does not override `build_decompress_metadata`.
`phase_finalize` builds
`gpu_col`: a RAW stage's `_contiguous_data` is a view of its data slice, and a
column another column uses as its mask (`_compute_mask_sum`) is summed into a
GPU scalar, which the pipeline reads as an int after the tick's sync.

**Decompress** runs on the compute stream in the next tick:
`IO/codec/batched_decompress.py::BatchedDecompress.decompress_all` calls each
compressed stage's `decompress`. The first stage of a codec group runs
`_launch_mgr_group`, which copies a mixed stage's RAW runs into the shared
output, rebases the template's pointers onto the `agg_buf` and output bases
with `np.add`, and issues **one** `mgr_decompress`. Later stages of the
group only narrow their slice. A group with a record lacking a saved handle
(a cold fail-safe) parses its records' headers at launch
(`mgr_ingest_configs`, which synchronizes) for that template only, so no
parsed `DecompressionConfig` is shared with another decompression. The one
exception to this schedule is a compressed mask column: its `phase_finalize`
needs the mask's values, so it decompresses the whole codec group on
`h2d_stream` during the launch.

### Zero-copy stages (`zero_copy_stage.py`)

`adaptive/instage_optimization.py::decide_column_type` picks `ZeroCopyMaskIn`
once per column at pipeline construction, for a RAW column held entirely in
pinned host memory (a `TableVariable` whose records are RAW and whose
regions are in host memory, `is_host_raw`) whose mask keeps less than
`--mask_selectivity` of the rows. Only the mask crosses PCIe. There is no batched plan across stages:
each stage reads its mask like a one-stage `DirectIn`, whose layout it
compiles at construction. `launch_h2d` binds it, copies the mask's selected
records (one copy per run) and decodes them on `h2d_stream`, so a stage
decodes only the mask records its rows overlap, and
`phase_finalize` sums the mask. In the next tick, `execute_zero_copy` sizes
its output by that count and calls `zero_copy_reader.zero_copy_mask`, which
reads the selected rows from the table's RAW host records over UVA into a
new GPU tensor, through the column's segments (`TableVariable.uva_source`:
one per stretch of the table's records adjacent in one region, with
explicit row starts); the stage's column is
that tensor as one RAW record of a GPU region (`gpu_region_column`, a
`TableVariable`, freed with its last reference). Under `TQP_EXPERIMENT_TELEMETRY` the stage records both moves in the
movement ledger: its copy of the mask at the bind, as a `DirectIn`'s
(`scan_h2d` / `restore_h2d` per host run, `resident_d2d` per
GPU-resident one), and its UVA read (`scan_uva` / `restore_uva`, the bytes of
the rows gathered).

## Compute-side stages

`PreCompStage` (`pre_comp_stage.py`) runs four phases, in the order the
pipeline's `_execute_pre_comp_and_operator` calls them: `decompress` or
`execute_zero_copy`; `extract_raw` (a cached zero-length tensor per dtype for
an empty input, instead of an allocation); `apply_mask` (`compact_masked` with
the mask's precomputed count, skipped when the pipe folds the mask); and the
result in `ready_tensor`.

The operator's output is handed to the pipe's region producer by
`VortexPipeline._execute_post_comp`; nothing in this subpackage touches it.

## `TickStages` lifetime (`tick_stages.py`)

The pipeline builds every `TickStages` up front: tick 0 has in-stages only,
and every later tick adds its pre-compute stages.
Each tick's `DirectIn`s get their layout when the tick is built; a
just-in-time rebuild replaces tick *t+2*'s `in_stage` with a freshly compiled
layout and re-points tick *t+3*'s pre-compute references.

`clean_intermediates` drops every in-stage's `gpu_col`, `agg_buf` views and
mask buffers, and the pre-compute references. It may run only after the
compute stream and `exchange.sync()` have finished; earlier, it
frees memory a kernel or copy engine is still reading. The pipeline calls it
for `ts[t−1]` at the end of tick *t*, on the compute stream.

## The ledger's read hook

The runs a tick reads are the movement ledger's: `bind_tick_sources` is its
read hook (under `TQP_EXPERIMENT_TELEMETRY`). It files a stage's runs as
above, and records them once the bind has succeeded.

## Files

| File | Purpose |
|------|---------|
| `data_exchange.py` | `NaiveExchange` (two streams, a copy log per launch, `launch`, `sync`, `get_last_times` for the tick's input lanes), the singleton `exchange()` and `set_exchange_to_naive()`. The one transfer backend. |
| `tick_stages.py` | `TickStages`, the two stage lists of one tick, and `clean_intermediates` with its two preconditions. |
| `in_stage.py` | The `InStage` interface and `DirectIn`: layout install, `plan_input_data_movement` = `bind_input` (empty stages, bind) + `alloc_input` (allocate, bind views, the hold list), the two single-call packers, `phase_finalize` and group-batched `decompress`. |
| `tick_layout.py` | The compiled input geometry: `compile_tick_layout`, `CompiledTickLayout`, `CompiledManagerTemplate`, `bind_tick_sources`, `BoundTickPlan`, `launch_slot_pieces` (through `pages/abs_copy.py`) and `_launch_mgr_group`; the planners `_plan_by_region`, `_plan_table_stage` and `_plan_table_read`, a stage's `_TableRuns` and `_bind_regions`. Reads `gates.VARCHK` (`TQP_VARCHK`), which asserts the record indexes while a template compiles. |
| `zero_copy_stage.py` | `ZeroCopyMaskIn`, `gpu_region_column` (a stage's output as a GPU region). |
| `pre_comp_stage.py` | `PreCompStage` and the per-dtype cached empty tensor. |


## Tests

`tests/IO/test_compiled_input_layout.py` (CPU) covers the compiled geometry:
no absolute pointers at compile, pointer storage reused on rebind, the lane
of a GPU and a host region with the ledger's kinds (CUDA), manager templates,
and `launch_d2d` skipping. `tests/IO/test_record_reads.py` covers the record
selections: on the CPU, the runs, trims and decoded bytes of RAW, compressed
and mixed stages and the header-parse fail-safe; with CUDA, RAW, BITCOMP,
ANS and mixed columns a region producer wrote, host and GPU resident, read
back exactly over record boundaries and random row ranges (one stage and a five-stage tick) with the
ledger's decoded and wire bytes, and the zero-copy mask stage's mask; on
the CPU, the table planner over bucket-major regions (simulated copies and
decode against the rows, one run per region for a range of buckets) and
its agreement with the region planner on page-shaped regions.
`tests/page_cache/test_granule_producer.py` checks the slot pieces' launch
and their merging. `tests/telemetry/test_copy_events.py` checks the copy
log's lanes. Run:
`TQP_GPU=<idle> docker compose run --rm -T dev python3 test.py tests/IO tests/page_cache`.

## Known gaps

- **Dead code.** `InStage.transferred_bytes` (`in_stage.py:51`) has no
  callers.
