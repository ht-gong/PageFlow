# `IO/pages/` — record tables, regions, column reps and the pinned pool

This subpackage holds the units every other part of `IO/` reads, caches and
frees: the record table and the region (a representation is read as records
of a table, and allocated and freed as regions), `CombinedColumnRep`, the
per-column wrapper operators pass around, and the pinned host pool (the
granule cache's GPU memory is the arena's cache region, `IO/cache/`). It sits at the
bottom of the `IO/` import graph and imports nothing from the other `IO/`
subpackages. Its dependencies are `constants`, `variable.py` (the repo-root
`Variable`, which itself imports `record_table.py` and `region.py`) and
`TQPlib.tqpmemory`. Everything else in `IO/` imports it: `IO/codec/` builds
regions and record tables, `IO/cache/` binds granules of regions to slots,
`IO/transfer/` moves them, and `IO/pipeline/` and `IO/data_scanner.py`
create and free them. `main.py` builds the pools.

## Column reps

A `Variable` (`variable.py`) with no records is an empty column, or the GPU
tensor a read produced: the transfer stages fill its `_contiguous_data`
(trimmed to its `row_range`) so `data()` does no work. A `CombinedColumnRep`
is the column itself: a list of reps, a `tag` and an optional dataflow
`identity`. The "RAW plus at most one compressed rep" rule is enforced by
the scan and the codec policy, not by this class. Metadata queries delegate
to `reps[0]`. `get_sliced_rep` slices every rep with the same rows and keeps
the tag but drops the identity. A `TableVariable` (`variable.py`) is a rep
held as a record table of regions, what the scan and the region producer
(`IO/codec/region_producer.py`) write, and every column with rows
(`gpu_region_column` makes one of a GPU tensor: the zero-copy stages' and
the final sort's outputs). It reads rows `[lo, hi)` of the table's row
space; its slices are `TableVariable`s over the same table that hold no
reference, and only the column's own one (the full range) frees the table.

## Record tables and regions

`record_table.py` and `region.py` describe a representation by its records.
The tick reader selects records from a `TableVariable`'s table.

**A region** (`Region`) is a byte range holding records back to back,
written once and never changed (a spillable region's bytes move once, whole:
below): one uint8 tensor (`storage`), in pinned host memory or on the GPU,
with `nbytes`, a `location` (`host`, `device`, or `cached` for a region the granule cache holds: its storage is its pinned
home, and its bytes sit partly in the cache's slots), a process-unique
`region_id` (counter `_next_region_id`), a `release` that says how to return
the memory (a pinned-pool block index, `None` for an arena tensor, or the `(cache, cached region)` pair of a cached region,
whose free drops it from the cache and waits for the copies still writing its
home; its home goes back with the table's chunks). `free` returns the
memory and is a no-op the second time; after it `storage` is `None` and
`data_ptr` raises. Its record table frees it.
`uva_ptr` is a host region's device alias (`cudaHostGetDevicePointer`),
for the zero-copy readers, computed once per region; `device_ptr` is the
address a kernel reads a region at (its own on the GPU, the alias in host
memory).

**Spillable regions** (`device_regions.py`, `--buffer_strategy freelist`)
are the one kind of region that moves. Under the free-list baseline the
region producer keeps each output on the GPU in an arena allocation of its
own (a RAW output's tensor, an exact copy when it is a view into a larger
allocation; a compressed output's packed records) with a `home`, its pinned
bytes, carved but not written, and registers it (a compressed one with its
logical bytes, for the ledger). The
arena's miss hook (`memory/README.md`) is `evict_one`: with the device
synchronized and the allocation stalled, the arena calls it until the
allocation fits, each call evicting the least recently used registered
region with no `pins` (registration and each read, a pin, make a region the
most recent; ties are arbitrary): copied into its home on one stream
(`spill_to_home`), the stream waited for, and made a host region
(`settle_on_host`), its GPU tensor dropped. When no unpinned region is
left it draws on the reserve (`--freelist_reserve_gb`, `_draw`): free
blocks held at the arena's top behind its region boundary from each
query's `begin_query` to its `release_all`, of which the boundary gives up
what the request lacks beyond the free run below it, and which refill
(`_refill`) from that run at each `register` and `unpin` while short. It
returns 0 when the reserve is empty too, and the allocation fails. A
region's records
are offsets into it, so its record table stays valid; the per-table address
caches (`tick_layout._region_bases`, `RecordTable.uva_segments`) key on the
eviction `epoch()`. A read that takes a region's address before an
allocation pins it until its copies or kernel are queued: a tick's bind
until the exchange's launch (`BoundTickPlan.pins`, `unpin`, `release`) and a
rearrange gather's source for the gather. `arena_stats` is the arena's
stats with the unpinned regions counted free (`stats_if_freed`), for
aggregation pass 2's admission; `begin_query`, `take_stats` and
`release_all` bracket a query in `main.py`.

**A record table** (`RecordTable`) is the record index of one whole
representation, as numpy `int64` columns with one row per record: `region`
(an index into `regions`), `offset` (bytes from the region's start,
512-aligned for compressed records), `nbytes` (stored bytes), `raw_bytes`,
`row_start` (within the stream), `rows`, `codec` (codec id, 0 = RAW) and
`handle` (the nvCOMP decompression config handle, 0 = none). Records are
ordered by stream, then by rows; `stream_ptr` holds the CSR offsets, one
stream per partition. The constants are `stored_dtype`, `width`,
`logical_type`, `row_bytes`, and `codec_keys`, the manager key of each codec
id present. A table's row space concatenates its streams in order, the order
of a partition-major column today, so a contiguous row range is a contiguous
record range.

- **Building.** `RecordTableBuilder.add_region` registers a region and
  `append` adds a block of records in it, or in one region per record
  (any streams, any order; the records must be whole rows, inside their
  region and, when compressed, 512-aligned). `freeze` sorts by stream (stable, so each stream keeps its
  append order), computes per-stream row starts and checks no two records
  of a region overlap. A region belongs to one table.
- **Selecting.** `select_rows(start, stop)` returns a `Selection`:
  ascending record ids plus `head_rows` and `rows`, the rows to drop from the
  start of the first record and the rows kept, so only the first and last
  record are trimmed (a tick cut in whole quanta selects whole records with
  no trim: `TableVariable.whole_records`, whose selection is its record
  range). A selection holds no references, so a column's tick slices are
  selections over its one table.
- **Runs.** `runs(records, align)` groups records into runs: a
  run is the given records of one region with no other record
  between them in the region's byte order, copied as one byte range. The
  result (`Runs`) lays the runs out back to back in one staging buffer, each
  after an `align` boundary (512 by default; 1 packs them densely) and at
  its source offset modulo `align`, so every record keeps its source
  alignment, and gives each record's place in it (`record_dst`). The records
  of one stream in one region form one run, and so do those of adjacent
  streams in a region that holds its streams in order.
- **Byte order is not row order.** Runs are laid out in (region, byte)
  order. A reader that uses the staging buffer as the decoded rows (RAW
  records copied straight to their output place, as the tick reader does)
  gets row order only because every region holds each stream's records in
  row order and its streams in stream order: true of the scan's regions
  and of the region producer's (one per tick, bucket by bucket); the table
  planner places
  compressed records by `record_dst` and copies RAW rows in row order.
- **Zero-copy reads.** `uva_segments()` gives a RAW table's segments for
  the UVA readers: one per stretch of records that follow each other both
  in the row space and in one region's bytes (host, or GPU under the
  free-list baseline), as each one's device address and first row (plus
  the end), built once per table and eviction epoch;
  `record_segments(first, stop)` gives the same for a range of records,
  rows counted from its first (a whole-record slice's zero-copy read, with
  no row offset). `base` marks
  a scanned table (the movement ledger files its reads as scans).
- **Freeing.** `free(cpu_pool, gpu_pool)` frees every region, then the
  pinned allocations the regions were carved from (`host_blocks`, set by
  the region producer, whose regions have no `release` of their own). It is
  idempotent.

## The pinned host pool

`MemoryPool` is a first-fit allocator over fixed-size blocks. `main.py` uses
0.5 MiB blocks, so every allocation rounds up to whole blocks, and a small
region still takes 0.5 MiB. The free list holds `(start_block, length)` runs.

- `get_storage(nbytes)` takes the first run that fits, splits it and returns
  `(ptr, view)`, where `ptr` is the start block index. When nothing fits it
  raises `PinnedPoolExhausted`, whose message says to raise
  `--pinned_pool_gb`. `try_get_storage(nbytes)` returns `None` instead, for
  allocations a caller can do without (the region producer's look-ahead
  chunk).
- `free(ptr)` appends the run to the free list. Every sixth free since the
  last merge calls `coalesce()`, which sorts the list and merges runs that
  touch within a segment. So up to five freed runs can sit unmerged beside
  their neighbours, and a request that misses merges them and looks once more
  before it reports no fit. `free_space()` (the largest contiguous free
  region, in bytes) merges them first too, then reads a cached largest-run
  size.
- `shrink(ptr, nbytes)` keeps the first `nbytes` of an allocation (at least
  one block) and returns the blocks after them as a free does; the region
  producer trims each column's pinned chunk with it.
- `free_all()` reseeds the free list without touching host memory, and
  restarts the count of frees.
- Under `TQP_EXPERIMENT_TELEMETRY` the pool counts the blocks each *tag*
  holds, an allocation's tag being `get_storage`'s `tag` (`"intermediate"`
  unless named: the scan's region producer names `"scan"`), and their peaks,
  per tag and in total: `usage()` in bytes, `reset_peaks()` where a query's
  timer starts; `free_all` keeps the peaks. The perf report's `residency`
  section reads them (`IO/telemetry/residency.py`).

`PinnedMemory` backs the pool with `PINNED_SEGMENT_GB` (128 GiB) segments, each
its own pinned, device-mapped region from
`TQPlib.tqpmemory.pinned_host_alloc_tensor`: an anonymous mapping bound to the
GPU's NUMA node with `mbind`, then pinned with `cudaHostRegister`. The node
defaults to GPU 0's, and the constructor stops if that node cannot hold the
pool (`utility/numa.py::check_node_capacity`). Segments exist because of a
device limit: one monolithic mapped allocation fails above roughly
500–590 GiB in a live CUDA context, even with far more host RAM free, because
the device-mapping step rolls back. The free list is seeded with one run per
segment, and coalescing never merges across a boundary. So no allocation
spans two segments, and `cudaHostGetDevicePointer`
(`zero_copy_reader.device_alias`) always returns a valid device alias for a
region. A pool up to 128 GiB is one
segment; the SF1000 default (686 GiB, `main._pinned_capacity_gb`) is six. Any
single request must fit inside one segment: a 256 MiB scan batch
(`IO/data_scanner.py`), a pinned chunk or region of the region producer
(a RAW tick output, which grows with `--chunk_size_mb`). Pinning happens once at startup and
can take minutes at large SF. Each segment is unregistered and unmapped when
its last view dies.

The host pool tracks no stream state. A copy into pinned
memory is ordered for readers by its writer (the home writer's events, the
exchange sync), not by the pool.

## Files

| File | Purpose |
|------|---------|
| `combined_column_rep.py` | `CombinedColumnRep`: a column as a list of `Variable` reps plus `tag` and `identity`. |
| `region.py` | `Region`: a byte range of records in one allocation, its location, release, stream references and device alias (`uva_ptr`), and a spillable region's home, pins and move (`spill_to_home`, `settle_on_host`); the `_next_region_id` counter. |
| `device_regions.py` | The free-list baseline's spillable GPU regions: registration, pins, the eviction epoch, `evict_one` (the arena's miss hook: the least recently used unpinned region, called until the allocation fits, then the reserve), the reserve (`--freelist_reserve_gb`: `_draw`, `_refill`, `reserve_b`), the spillable-as-free arena stats and the per-query counters. |
| `record_table.py` | `RecordTable` (numpy record columns per representation, CSR streams), `RecordTableBuilder` (append blocks, `freeze`), `Selection` (rows, with trims), `Runs`, the zero-copy segments and free. |
| `pinned_mem.py` | `MemoryPool` (the first-fit block allocator) and `PinnedMemory` (the 128 GiB mapped-segment backing). |
| `abs_copy.py` | Copies by absolute device address: `launch` (one `batched_memcpy_abs` on the current stream, reading its staged arrays over UVA, launched with the staging buffer's event record in one native call) and `staging`, the pinned buffers their address arrays are staged through, kept across launches and handed out by event. Used by the region producer's packs (their own kernels over a staged buffer) and RAW slot copies (`codec/region_producer.py`), and by the reader's slot pieces (`transfer/tick_layout.py`). |

## Tests

The pinned pool's free list is covered by
`tests/allocator/test_pinned_pool_free_list.py` on a `MemoryPool` that hands out
block ranges instead of pinned memory (first fit, at most one merge per six
frees over 10,000 frees, a miss that fits after merging, `free_all` restarting
the count, `free_space` across touching runs, `shrink`), and `PinnedMemory` by
`tests/allocator/test_pinned_pool.py` (NUMA binding, a full pool). The
residency counters of both pools are covered by
`tests/allocator/test_pool_peaks.py`.
`CombinedColumnRep` and `PinnedMemory` appear as fixtures throughout
`tests/IO/` (for example `test_compiled_input_layout.py`),
`tests/page_cache/` and
`tests/telemetry/`. `tests/IO/test_record_table.py` covers the record
tables and regions: on the CPU, the builder and freeze, row selections
against brute force, runs and freeing (`tests/record_tables.py` holds the
record-table reads the tests check against).
`tests/IO/test_scan_regions.py` covers the scan's tables (one region per
batch in the producer's chunks, RAW and compressed, a tick slice freeing
nothing; `tests/scan_table.py` builds them from tensors), the zero-copy readers
over segments (`zero_copy_rearrange` honouring `row_start`) and, on SF1,
every scanned column's table, each table's columns scanned together, read
back against the reader's decode, also with batches that span the reader's
chunks. No test covers the pool's
segmenting directly. Run with:

```bash
TQP_GPU=<idle> docker compose run --rm -T dev python3 test.py tests/allocator
TQP_GPU=<idle> docker compose run --rm -T dev python3 test.py tests/IO
```
