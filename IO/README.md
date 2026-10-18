# `IO/` — the host→GPU streaming pipeline for TQP-Vortex

This directory is the core of the engine: everything between a column sitting in
pinned host memory and an operator running on a chunk of it on the GPU. It holds
the tick loop that streams every operator (`VortexPipeline`), the stages that
move, decompress and compress data inside each tick, the region producer that
writes every pipe's output, the memory the loop draws from (the arena's cache
region and the pinned host pool), the GPU granule cache that lets later ticks skip
transfers, the Python adapters
over the nvCOMP manager extension, and the zero-copy CUDA kernels that read
selective columns straight from host memory over UVA.

The pieces fit together like this (paths are relative to `IO/`):

1. **Scan.** `data_scanner.py` loads each needed column from the datagen
   parquet, decoded on the GPU by `parquet_reader.py`, into the pinned host
   pool (`pages/pinned_mem.py::PinnedMemory`), as RAW plus at most one
   compressed rep. Each rep is a record table of one region per scan batch
   (`variable.TableVariable`), written through a
   `codec/region_producer.py::RegionProducer`, one tick per batch. It hands
   them to operators as `pages/combined_column_rep.py::CombinedColumnRep`s.
2. **Tick loop.** `pipeline/vortex_pipeline.py::VortexPipeline` cuts the input
   columns into chunks of `--chunk_size_mb`, rounded down to whole quanta of the
   records the columns share (`pipeline/tick_cutter.py`, so a tick reads whole
   records), and runs one operator over them in
   software-pipelined **ticks**. Each tick is a
   `transfer/tick_stages.py::TickStages`: an input stage per input column
   (`transfer/in_stage.py` for H2D, `transfer/zero_copy_stage.py` for
   zero-copy), a `transfer/pre_comp_stage.py` (decompress, mask) and the
   operator. The output goes to the pipe's
   `codec/region_producer.py::RegionProducer` (one region per tick and
   column, copied home in the background).
   `transfer/data_exchange.py` owns the transfer streams and drives the
   in-stages of a tick.
3. **Pools and cache.** `cache/cache_region.py::CacheRegion` is the cache's
   GPU memory: the top blocks of the arena that `memory/` manages, behind a
   boundary. `cache/native_cache.py::NativeGranuleCache` keeps granules of the
   producer's regions resident in those blocks, one per block, so a consumer tick skips their
   H2D, with LRU, MRU or dataflow eviction (its bookkeeping in the C++ core
   `cache/granule_core.cc`).
   `pages/pinned_mem.py` is the host side: every region's home is there.
4. **Codec adapters.** `codec/region_producer.py` with `codec/home_writer.py`
   (write a pipe's, a gather's or the scan's output as one region per tick
   and column, copied to the host in the background),
   `codec/manager_compress.py` (the codec facts the producer asks nvCOMP
   for) and `codec/batched_decompress.py` (the per-tick decode fan-out) are
   the Python side of `TQPlib.tqpcompress` (see `compression/README.md`).
5. **Native module.** `zero_copy_reader.cu` builds into
   `TQPlib.zero_copy_reader`: the UVA gather kernels over segments of pinned
   host memory behind `--mask_selectivity` and the join's gathers.
6. **Telemetry.** `telemetry/copy_events.py` times every copy batch with one
   CUDA event pair around its copies; the searches read the batches as one
   span per transfer stream. `telemetry/movement.py` is the movement
   ledger: under `TQP_EXPERIMENT_TELEMETRY` the copy sites of the stages, the
   producer and the granule cache record every host↔GPU copy once, by kind.
   `telemetry/gates.py` holds that switch, and the debug validators' switches;
   with the experiment switch off no ledger hook runs.

Consumers: every streaming operator in `operators/` constructs a
`VortexPipeline` (`filter`, `project`, `sort`, `aggregate`, `hashjoin`) and
`hashjoin` also calls `rearrange_pipeline`; `main.py` builds the pools, the cache and the scan,
and installs the exchange. The adaptive planes in `adaptive/` are reached only
through hooks the pipeline queries (`adaptive_driver`, `decision_menu`,
`resource_monitor`, `sizing_resolver`); the arena (`memory/`) lends the
cache region's lowest blocks to an allocation that misses, through the
granule cache's core. Tests: `tests/IO/`
(compiled input layout, record reads and tables, region producer, scan
regions, kept streams),
`tests/page_cache/` (the granule cache, its region, its producer, its ranges,
the reference model and its slot pool), `tests/allocator/test_arena_lend.py`
(lending),
`tests/allocator/test_slot_pool_blocks.py` (the reference model's slot pool),
`tests/compression/test_nvcomp_manager.py` (drives `manager_compress` and
`batched_decompress`) and `tests/telemetry/`. Design notes:
`cache/README.md` (the granule cache, cleaning and why its defaults are what
they are), `adaptive/README.md` and `docs/adaptive_intermediate_sizing.md` (the
hooks), and `compression/README.md` (the codec path and
the nvCOMP manager rules).

## One tick, end to end

`VortexPipeline.do_exchange` keeps three chunks in flight. At tick *t*, with
`exchange = data_exchange.exchange()`:

1. **Transfer** (`exchange.launch(ts[t].in_stage)`): the in-stages of chunk
   *t* are grouped by class; `DirectIn` compiles one aggregate GPU buffer per
   tick and fills it with **one** `batched_h2d_pack` call on `h2d_stream`
   (host records, and cached granules' home pieces) and **one**
   `batched_d2d_pack` call on `d2d_stream` (GPU regions),
   plus one `batched_memcpy_abs` for the cached granules' slot pieces.
2. **Compute** (`_execute_pre_comp_and_operator`, on the pipeline's compute
   stream): a region producer first packs chunk *t−2*'s compressed output
   into its regions, whose copies home start right behind on the home
   writer's stream; then chunk *t−1* is decompressed (one `mgr_decompress` per codec group,
   later stages in the group just narrow into the shared output), zero-copy
   kernels run, masks are applied, then the operator is called.
3. **Post-compute** (`_execute_post_comp`): outputs are compressed into
   worst-case slabs by the region producer (a `PartitionedTensor`, from the
   join's sort pipe or a partitioned aggregation's pass 1, is one instance of
   B buckets; a RAW output ships home at once); with the cache,
   the next tick's granules are pinned.
4. **Sync and account**: the exchange and compute streams are synchronized;
   the tick's copy batches (`telemetry/copy_events.py`, one event pair around
   each site's copies) give the `TransferTimes` lanes, one span per stream,
   which go with `ComputeTimes` to the `ResourceMonitor`; chunk *t−1*'s
   intermediates are released
   (`TickStages.clean_intermediates`).

That is the tick under the barrier, which runs while a search of the pipe
is live (a compression-search observer, until it settles). Otherwise, with or without the granule cache, the ticks are
ordered by events
(`pipeline/README.md`, "Event-ordered ticks"): no host sync per tick; the
compute stream waits on an event after each tick's input issue, the host
waits on the tick event two ticks old before it allocates, the pack of chunk
*t−2* follows the operator's launch, the buffers the compute stream reads
stay on the tick's hold list until that wait, and the telemetry is read two
ticks late.

The whole `do_exchange` is bracketed with `TQPlib.tqpmemory`'s window peak
(`reset_window_peak` / `get_window_peak_bytes`), which the sizing search reads
and, under `TQP_EXPERIMENT_TELEMETRY`, the perf report records as
`peak_gpu_bytes`.

## Files

The subpackages follow the import graph: each one imports only from the
subpackages in its last column (lazy imports inside functions included), so
there are no cycles.

| Subpackage | Holds | Imports from `IO/` |
|---|---|---|
| `pages/` | record tables and regions, the column of representations, the pinned host pool and the pinned staging of absolute-address copies | `telemetry/` (the switches) |
| `telemetry/` | the copy-batch event pairs, the movement ledger, the residency peaks, and the telemetry and debug switches | nothing |
| `codec/` | the Python side of the nvCOMP manager extension | `pages/`, `telemetry/` |
| `cache/` | the GPU granule cache, its policies and the cache setting | `pages/`, `telemetry/` |
| `transfer/` | the per-tick input stages and the exchange that drives them | `pages/`, `telemetry/` |
| `pipeline/` | the tick loop and join materialization | all of the above |

`data_scanner.py` (the load-time scan, which uses `pages/` and `codec/`),
`parquet_reader.py` (the decode it reads through, which imports nothing from
`IO/`) and the native module stay at the package root. Every `__init__.py`
is empty, so importing a subpackage imports none of its modules.

### Import rules

These keep the split free of cycles and of stale bindings:

- **Import each name from the module that defines it.** There are no
  compatibility shims and no re-exports from a sibling: the cache's result
  types come from `cache/native_cache.py`, never through a module that
  imports them.
- **Never import a rebindable module global by name from another module.**
  `from m import x` copies the binding at import time, so a later
  `global x; x = ...` in `m` is invisible to the importer. These are the
  globals some function rebinds today:
  - every switch in `telemetry/gates` (tests set them), which every site
    reads through a module import (`gates.EXPERIMENT`, `gates.VARCHK`);
  - `pages/region._next_region_id`;
  - `codec/manager_compress._mgr_max_compressed_size`, bound on first use;
  - `cache/config._buffer_strategy`, which `main.py` reads through
    `config.buffer_strategy()`;
  - `transfer/data_exchange._exchange`.

  Dicts, lists and instances that are only mutated in place (such as
  `telemetry/movement._totals`) are safe to import by name.
- **Lazy imports inside function bodies stay lazy.** Several break an import
  cycle, and hoisting them would also change import timing.

Each subpackage has its own README that covers how it works: object
lifetimes and ownership, stream and sync rules, the contracts its callers
rely on, its tests and its known gaps. The tables below are the summary.

### `pipeline/` — the tick loop and join materialization

In depth: `pipeline/README.md`.

| File | Purpose |
|------|---------|
| `vortex_pipeline.py` | The execution driver. `VortexPipeline` is built from its dataflow node (`node=`, a `dataflow.dataflow_gen.PipeNode`) and `(key, ColumnInstance)` input pairs: the instances supply the payloads, the codecs (through the applier) and the cache bookkeeping (next-use position, expected reads, host residency), and `OutputLayout` maps the physical output slots onto the node's output instances (`flat`, or `partitioned(n, B)` for B slots per column: the join's sort pipe, a partitioned aggregation's pass 1); for a flat layout the pipeline sets each output instance's payload at `get_result` (a `variable.TableVariable` for an instance the region producer wrote; `result_tables` gives those tables). Every pipe writes its outputs through its region producer (it needs `cpu_mem_pool`; `page_cache` is the granule cache or None). A bare pipeline (tests) takes `input_columns` (`Dict[int, CombinedColumnRep]`) and `name` with the fixed codec everywhere; its tables are `result_tables()` by base slot. It splits the inputs into chunks (`div`, else `constants.get_chunk_size_b()` rounded down to whole quanta by the tick cutter, `tick_cutter.py`; each tick's slice reads whole records where its ends are record boundaries), pre-builds one `TickStages` per tick, choosing `DirectIn` or `ZeroCopyMaskIn` per column via `adaptive.instage_optimization.decide_column_type`, and runs the loop above in `do_exchange`. Compression-search hooks: `_build_decision_menu` publishes each column's legal codecs, input picks are applied at *t+2* by a just-in-time in-stage rebuild (`_jit_rebuild`), output picks at *t+1* (`_actuate_write_picks`, which retargets the output codecs). Sizing hooks: the tick sizing session (each chunk's tick time, readback and arena peak on event-ordered ticks), and mid-run re-chunking (`_resection`). Reads `TQP_PICK_DEBUG`. |
| `rearrange_pipeline.py` | `rearrange_pipeline`, the join materialization path: it streams over output-row windows sized from the chunk budget minus 24 B/row of argsort scratch (`_ARGSORT_OVERHEAD_PER_ROW`), gathering each window over UVA (`zero_copy_rearrange` over a RAW rep's segments, `uva_source`; a movement-ledger `scan_uva` / `restore_uva` read) or from the source streamed onto the GPU through a tick pipe (`index_select`; the pipe's reads are the ledger's), whichever moves fewer bytes (`_read_plan`), then compressing it into a region the home writer copies home (`codec/region_producer.py`; the window charges the previous window's worst-case slab per row; the cache never holds a gather's output); under `--sizing_search mid_stream` the window is re-resolved between iterations. |
| `common.py` | What the pipelines share. `_WindowPeakTracker` composes per-iteration arena-peak brackets into a pipe-level peak. `col_to_gpu` is a one-shot read-and-decompress helper (a one-stage `DirectIn`), also used by `operators/sort.py` and `operators/output.py`. `PartitionedTensor` is the marker `hashjoin`'s partitioned sort and a partitioned aggregation's pass 1 return: one fat tensor holding one output instance's B buckets in the padded layout of `aligned_partition_layout`, handed to the region producer as one instance of B streams. Small helper: `_is_raw_algo`. |

### `transfer/` — the per-tick stages and the exchange

In depth: `transfer/README.md`.

| File | Purpose |
|------|---------|
| `data_exchange.py` | The transfer backend. `NaiveExchange` owns `h2d_stream` and `d2d_stream` (kept for the process, `utility/cuda_streams.py`) plus the copy log of the launch in flight (`telemetry/copy_events.py`); `launch(in_stages)` groups stages by class and runs each class's batched `plan_input_data_movement` → `launch_h2d` → `launch_d2d` → `build_decompress_metadata` → `phase_finalize` sequence; `sync` drains both streams, closes the log and takes the per-tick `TransferTimes` from it (per stream, the span from the first copy batch's start to the last one's end). Outputs never pass through it: the home writer's copies are the pipe's d2h lane. Module singleton: `set_exchange_to_naive()` (installed once by `main.py` after the arena is up) and `exchange()`. |
| `tick_stages.py` | `TickStages`: the two stage lists of one tick (`in_stage`, `pre_comp`) and `clean_intermediates`, which drops every GPU reference they hold. Safe only after the compute stream and the exchange of that tick have completed. |
| `in_stage.py` | Input stages. `InStage` defines the no-op per-tick interface the exchange calls. `DirectIn` is the H2D/D2D path: it reads the records its column's selection names (`TableVariable.record_selection`: those the tick's rows overlap); `install_compiled_tick_layout` attaches one layout from `tick_layout.py` to a tick's stages when the pipeline builds them; at launch `plan_input_data_movement` binds the tick's sources into it and allocates one aggregate buffer (`_alloc_agg_buf_and_bind_views`), which `launch_h2d` / `launch_d2d` fill with **one** `batched_h2d_pack` and **one** `batched_d2d_pack`; each codec group is then decompressed with one `mgr_decompress`. |
| `tick_layout.py` | The compiled per-tick input geometry `DirectIn` runs on. `compile_tick_layout` builds a pointer-free `CompiledTickLayout` per tick from each stage's record selection (`TableVariable.record_selection`: the records the tick's rows overlap, planned region by region): stages grouped by manager codec key with a trailing RAW group, one aggregate buffer holding one run per region (its selected records, 512-aligned; a RAW record only the rows used), a `CompiledManagerTemplate` of relative pointer arrays compiled from the record table plus cached `DecompressionConfig` handles, decoding only the selected records, each stage's kept rows (`row_range`) among the decoded ones, and one pack entry per run (destination offset and bytes) that `batched_h2d_pack` / `batched_d2d_pack` share; `bind_tick_sources` writes the tick's actual sources (a GPU region → D2D, a host region → H2D, each plus the run's offset; a cached region's run piece by piece, its granules in slots copied by one `batched_memcpy_abs`, `launch_slot_pieces`, and the rest from its home by H2D), defers the cached reads to the cache, and is the movement ledger's read hook (`cache_hit`, `resident_d2d`, `scan_h2d` for a base table's records, `restore_h2d` for a producer's, per run: the selected records' decoded bytes and the run's bytes; `restore_evicted_h2d` for a cached region's home pieces); `_launch_mgr_group` issues the one `mgr_decompress` per codec group. A bind trusts the compiled geometry: a record table's regions and records never change. Reads `gates.VARCHK` (`TQP_VARCHK`). |
| `zero_copy_stage.py` | The zero-copy input stages. `ZeroCopyMaskIn` moves only the mask column, reading only the mask records its rows overlap through a one-stage `DirectIn` layout, and calls `zero_copy_reader.zero_copy_mask` over the source's host segments (`TableVariable.uva_source`: its table's RAW records). It emits its rows as one RAW record of a GPU region (`gpu_region_column`, a `TableVariable`). Under `TQP_EXPERIMENT_TELEMETRY` the mask's copies are recorded at its `DirectIn` bind (`restore_h2d` / `scan_h2d`, or `resident_d2d`), and the stage records its UVA read (`scan_uva` / `restore_uva`) in the movement ledger. |
| `pre_comp_stage.py` | `PreCompStage`: one per input column between H2D and the operator. `decompress` / `execute_zero_copy` (phase 1), `extract_raw` (returns a cached zero-length GPU tensor for empty partitions instead of allocating), `apply_mask` (`compact_masked` against the mask column named by `mask_map_target`, skipped when the pipe folds the mask, or the zero-copy result). |

### `pages/` — pages, record tables and memory pools

In depth: `pages/README.md`.

| File | Purpose |
|------|---------|
| `combined_column_rep.py` | `CombinedColumnRep`: the set of representations of one column (RAW and at most one compressed rep, as `Variable`s); `get_raw`, `get_compressed`, `get_sliced_rep`, `free_underlying_mem`. This is the unit operators pass around. |
| `region.py` | `Region`: a byte range of records in one allocation (pinned host or GPU tensor, or a cached region's pinned home), with its location and how its memory is released (pinned block, the cache, or nothing for an arena tensor); its record table frees it. |
| `record_table.py` | `RecordTable`: the record index of one representation as numpy columns (region, offset, stored and raw bytes, row start, rows, codec, handle), CSR by stream; `RecordTableBuilder` appends blocks and `freeze`s; `Selection` (by rows, with first/last-record trims); `runs` (one byte range per region of adjacent selected records, laid out in one staging buffer in (region, byte) order, each run keeping its source alignment; that is row order only while regions store their records in row order); `free` frees the regions. A `variable.TableVariable` holds one per representation and the reader selects records through it. |
| `pinned_mem.py` | The host side. `MemoryPool` is a variable-block first-fit allocator (0.5 MB blocks in `main.py`) with lazy coalescing (one merge per six frees, and one more before a miss is reported) and a cached largest-free-run; `PinnedMemory` backs it with pinned, device-mapped segments of `PINNED_SEGMENT_GB` (128 GiB) from `TQPlib.tqpmemory.pinned_host_alloc_tensor`, every page bound to GPU 0's NUMA node after `utility.numa.check_node_capacity` confirms the node can hold them; one mapped region above roughly 500 GiB fails in a live CUDA context. Allocations never straddle a segment, so `cudaHostGetDevicePointer` in `zero_copy_reader.cu` always returns a valid alias. `get_storage` (raises `PinnedPoolExhausted` when no free run fits), `try_get_storage` (None instead, for the region producer's look-ahead chunk), `free`, `free_all`, `free_space`. |
| `abs_copy.py` | Copies by absolute device address (`batched_memcpy_abs`): `launch`, one kernel on the current stream whose arrays it stages and which reads them over UVA, launched with the staging buffer's event record in one native call (`batched_memcpy_abs_staged`, with an optional destination base), and `staging`, the pinned buffers the copies' address arrays are staged through, kept across launches and handed out by event (at most eight). The region producer's packs stage through it for kernels of their own; its RAW slot copies and the reader's slot pieces (`transfer/tick_layout.py`, `launch_slot_pieces`) use `launch`. |

### `cache/` — the GPU caches

In depth: `cache/README.md`.

| File | Purpose |
|------|---------|
| `config.py` | The cache setting of the process: the `BufferStrategy` enum behind `--buffer_strategy` and `set_global_buffer_strategy` / `buffer_strategy()`, which `main.py` sets once and reads to build the cache (`cpu`: none). |
| `granule_core.cc`, `native_cache.py` | `NativeGranuleCache`, the cache `--buffer_strategy` `lru`/`mru`/`dataflow` builds, over a C++ core (`TQPlib.tqpcache`, host only): cache-on runs take the region producer, the table scan and the region readers, as runs without a cache do. `GranuleCore` holds the granule, region and segment tables, the LRU, MRU and Dataflow policies and the slot allocator, every call over numpy int64 arrays; the wrapper keeps every CUDA object. Cached regions cut into slot-sized granules: `register` a region with its home and records by stream, `admit` a prefix of each offered region's granules, `reclaim(nbytes)` (dead, then live in policy order, clean before dirty on an equal key; dirty ones copied home through a caller's `copy_home`, their slots released behind its event), `cleaning_targets` / `commit_cleaning` (only while the cleaning gate, armed by the live victims of recent admissions and reclaims, is open), copies into homes as batches with one event each, consume pins per granule, `pieces` (byte ranges of many regions in one call as slot and home pieces, which the reader binds by D2D and H2D), `note_read` (the dead rule per stream; the reader defers it past the tick barrier), `drop`. Run per occurrence unless `config.CORE` (`TQP_CACHE_CORE`) is `cpu`. Its slots are the arena blocks of the cache region, handed out from the top. `tests/page_cache/reference_cache.py` is the same cache in Python, the reference model of its tests. |
| `cache_region.py` | `CacheRegion`: the arena's cache region (`--slot_pool_gb` of top blocks behind a boundary); its geometry, `set_size` (moves the boundary between queries; a refused grow raises) and `view` (one uint8 tensor over its bytes). |

### `codec/` — compression adapters (Python side of `TQPlib.tqpcompress`)

In depth: `codec/README.md`.

| File | Purpose |
|------|---------|
| `batched_compress.py` | `aligned_partition_layout` computes the padded per-partition byte starts of a partitioned output (written by the join's partition kernel, `operators/kernels/partition_sort.cu`, and a partitioned aggregation's pass 1, `operators/aggregate.py`; read by the region producer, which refuses buckets back to back unless that is the same layout). Defines `_MAX_OUT_BUF_BYTES` (256 MiB), the hard cap on worst-case compressed output per batched manager call, which bounds both the staging slab and nvCOMP's retained scratch. |
| `manager_compress.py` | What the region producer asks nvCOMP about a codec, cached per process: `_codec_info` (codec id and manager chunk; a `chunk_b` overrides the codec map's, for the sweep) and `_worst_case_aligned` (a record's worst-case compressed bound from `mgr_max_compressed_size`). Also an `atexit` hook that calls `mgr_shutdown` before interpreter teardown. |
| `region_producer.py` | `RegionProducer`: every pipe's output path, the gathers' (`pipeline/rearrange_pipeline.py`), the scan's compressed reps' and the codec sweep's (`chunk_b`; `spills=False` keeps the scan's copies out of the ledger's spills, `pool_tag="scan"` its homes out of the pinned pool's intermediate residency). Each tick's output of a column becomes one region (its records back to back, bucket by bucket) in a pinned home carved from a per-column chunk: compressed at tick t into worst-case slabs (pieces of at most `_MAX_OUT_BUF_BYTES`), packed at t+1 before the operator by one `batched_memcpy_abs` per piece into a dense GPU buffer, then shipped home; a RAW output ships at once. With the granule cache a cacheable output's region is offered to the cache at the pack, its admitted prefix written into slots by the same kernel (a tick admitted whole by one kernel for the tick, `_fill_slots`) and only the rest shipped home, and the cache's cleaning targets are shipped after; a dense buffer that does not fit is replaced by copies from the slab straight to the homes. `drain` packs the last tick and freezes one record table per column (one stream per bucket), which the pipeline publishes as a `variable.TableVariable`. The record cut (`record_rows`, `record_cap_b`, `row_quantum`) and `slab_bytes_per_row` (a gather window's charge for the previous window's slab). |
| `home_writer.py` | `HomeWriter`: the only code that copies into homes, on one background stream behind the pack's event, holding each source buffer until its copy has ended, at most two ticks' shipments per pipe in flight; its finished shipments are the pipe's d2h lane (`take_lane`). |
| `batched_decompress.py` | `BatchedDecompress.decompress_all`: the per-tick fan-out that calls `decompress` on each `DirectIn` stage (the first member of a codec group launches, the rest narrow). |

### `telemetry/` — copy batches, the movement ledger and the residency peaks

In depth: `telemetry/README.md`.

| File | Purpose |
|------|---------|
| `copy_events.py` | One CUDA event pair per copy batch (the copies one site issues on one stream in one launch), filed by its lane (`h2d`, `d2d`). A `CopyLog` per launch (`open_log` / `close_log`; the exchange opens one per tick) gives the tick's input lanes (`lane_ms`: per stream, the first batch's start to the last one's end, read by the searches as `TransferTimes`). Engine inputs, recorded whatever the telemetry switch says. |
| `gates.py` | The telemetry and debug switches, each parsed once at import (on-word): `EXPERIMENT` from `TQP_EXPERIMENT_TELEMETRY`, which every hook site tests before computing its arguments; the validators `VARCHK`, `PICK_DEBUG` and `TIMED_GC_VALIDATE`. |
| `movement.py` | The movement ledger: `record(kind, pages, logical_b, wire_b)` for the fifteen kinds (`scan_h2d`, `scan_uva`, the four `spill_*`, `restore_h2d`, `restore_evicted_h2d`, `restore_uva`, `cache_hit`, `resident_d2d`, `cache_fill`, `avoided_dead`, `avoided_clean`, `result_d2h`) into the current phase (`set_phase`: `query`, or `output` while `main.py` runs the output step); `reset` and `report`, which `main.py` writes as the perf report's `movement` section. `table_read_kind`, `uva_kind` and `record_uva_read` serve the hook sites. |
| `residency.py` | The perf report's `residency` section: `begin` where a query's timer starts (rebaselines the pinned pool's and the granule cache's peaks, notes what each tier holds), `report` after its operators: the arena's peak (`get_peak_bytes`), the pinned pool's peaks per tag (`intermediate`, `scan`) and in total, the granule cache's used-slot peak. |

### Package root

| File | Purpose |
|------|---------|
| `data_scanner.py` | The scan. `scan_to_ram` reads each needed column from the datagen parquet under `config.json`'s `"tables path"`, maps the schema type through `TYPE_MAPPING`, marks no column sorted, and materializes the reps the active `CompressionPolicy.formats_for` names: RAW plus at most one compressed rep. Each rep is a record table with one region per batch of a whole number of `row_quantum` rows, at most 256 MiB or one quantum when that is larger (only rows over 256 B: the quantum is 2^20 rows under the record row cap `RECORD_ROWS_MAX`) (`_batch_rows`; a `variable.TableVariable` marked `base`), its records cut on rows from row 0 by the producer's rule (`region_producer.record_rows`), so a batch boundary is a record boundary of every column. Per table, one chunked read of the needed columns on a reader stream (`parquet_table` keeps each table's footers for the process), under `utility/rmm_arena.small_buffers` (cuDF's buffers of at most 1 MiB share arena blocks); per column, a `_ParquetColumn` cuts the decoded rows into the batches across chunk and part-file boundaries and makes each batch one tick of a region producer per occurrence of the column, on a producer stream, whose outputs are the occurrence's reps. The next chunk is read while the current chunk's batches compress and copy home, and a table's producers are drained before the next table is read. On return it releases the nvCOMP managers' scratch (`mgr_release_scratch`; they are keyed by the scan's producer stream, which nothing after the scan uses), so the arena holds what it held before the scan. |
| `parquet_reader.py` | The datagen parquet in the engine's encoding (the module docstring is the contract: the 54 columns, their dtypes and shapes, row order, input guards). `ParquetTable(root, sf, table)` reads the footers of `<root>/sf<sf>/parquet/<table>/*.parquet` (`files` in sorted name order with their row counts, `rows`, `columns`) and checks each column's type; `encoding(column)` is its `(dtype, width)`; `batches(columns)` yields `(first_row, {COLUMN: tensor})` per chunk of cuDF's chunked reader, in row order, decoded on the GPU on the reader's stream (torch's current stream at the first `next()`). A batch's memory goes back to the arena ordered on that stream only, so a consumer on another stream waits on it and keeps the batch's tensors alive until its work on them completes. Errors are `ParquetEncodingError`. |
| `zero_copy_reader.cu` | UVA gather kernels, built by `CMakeLists.txt` into `TQPlib.zero_copy_reader`. A read is a list of *segments*, runs of whole rows back to back in pinned host memory (a region's records): `seg_ptrs` (each segment's device alias) and `seg_rows` (each segment's first row in the column's row space, then the end), int64 GPU tensors, plus the rows `[row_start, row_stop)` read. `device_alias(host)` is `cudaHostGetDevicePointer` (`Region.uva_ptr` caches it per region). `zero_copy_mask(seg_ptrs, seg_rows, mask, result, row_start, row_stop)` reads the rows a boolean GPU mask selects into `result`; `zero_copy_rearrange(seg_ptrs, seg_rows, rearrange, row_start, row_stop, dtype, width)` gathers rows by an int64 GPU index counted from `row_start`, iterating in argsorted order so host reads are sequential while the scatter to the output is by the original position, and writes `CONST_NULL` (−64) for null indices (right-outer joins). The kernels trap on a row outside the segments; both dispatch on dtype (uint8 and int8 through int64, float, double). |
| `CMakeLists.txt` | `Python3_add_library(zero_copy_reader MODULE zero_copy_reader.cu WITH_SOABI)`, linked against `pybind11::module`, Torch and `torch_python`, with a `$ORIGIN` rpath; the CUDA language and architectures come from the top-level `cmake/CrossTorchGPU.cmake`. `setup.py` stages `build/IO/*.so` into `TQPlib/zero_copy_reader/`. |
| `__init__.py` | Empty marker that makes `IO` an importable Python package (each subpackage has its own, also empty). |

## Environment variables read in this directory

All are documented, with parsing rules, in `AGENTS.md` ("Environment
variables"); this is only where each one is read.

| Variable | Read in |
|---|---|
| `TQP_EXPERIMENT_TELEMETRY`, `TQP_VARCHK`, `TQP_PICK_DEBUG`, `TQP_TIMED_GC_VALIDATE` | `telemetry/gates.py` (once, at import); the sites read `gates.<NAME>` |
| `TQP_MGR_RECORD_MB` (and the module global `RECORD_ROWS_MAX`) | `codec/region_producer.py` (read by `data_scanner.py` too) |
| `KVIKIO_NTHREADS` | `parquet_reader.py` sets it to 32 at import unless set; libkvikio reads it |

`TQP_MGR_SCRATCH` is read in `compression/`, not here; the per-codec manager
chunk size is not an environment variable at all (it comes from
`profiling/results/codec_map.json` through `compression/format_policy.py`).

## Rebuilding the native module

Python edits take effect immediately under the `dev` service. After editing
`zero_copy_reader.cu`, rebuild inside `dev` and copy the module into `TQPlib/`
(see "Notes for agents editing the engine" in `AGENTS.md`):

```bash
docker compose run --rm -T dev bash -c '
  cmake --build build -j && cp build/IO/*.so TQPlib/zero_copy_reader/'
```
