# `IO/codec/` — compress and decompress records through the nvCOMP manager

This subpackage is the Python side of `TQPlib.tqpcompress`. It turns GPU
tensors into compressed records in regions and decodes the records a tick
reads. The native module it drives (built from `compression/nvcomp_manager.cu`
and `compression/compression.cu`), the codec registry and the codec map are
documented in `compression/README.md`; this file covers what happens on the
Python side of that boundary. There is one way to compress: `RegionProducer`
writes its caller's output as one region per tick and column, which
`HomeWriter` copies into pinned memory.

In the `IO/` import graph, `codec/` imports only `IO/pages/` (regions and
record tables) and `IO/telemetry/` (the switches and the movement ledger).
Outside `IO/` it reads `constants` (`COMPRESSION_ALIGNMENT = 512`),
`compression/compression_algorithms.py` (codec ids, chunk sizes). Consumers: `IO/pipeline/vortex_pipeline.py` owns
each pipe's `RegionProducer` (with the granule cache when one is configured)
and the per-tick `BatchedDecompress`; `IO/pipeline/rearrange_pipeline.py`
writes its chunks through a `RegionProducer`; `IO/data_scanner.py` writes
every scan representation, RAW included, through one per scanned column
occurrence (a tick per scan batch of at most 256 MiB, its copies home not
counted as spills). The join sort pipe's partition kernel
(`operators/kernels/partition_sort.cu`) and a partitioned aggregation's pass 1
(`operators/aggregate.py`, `_partition_outputs`) write their outputs in the
padded layout of `aligned_partition_layout`; the producer refuses buckets back
to back unless that is the same layout (every bucket but the last whole units).
`profiling/sweep.py` times every codec and chunk through a `RegionProducer`
(with an explicit `chunk_b`) and reads the regions back through a
`DirectIn`, the engine's reader.

## What a compressed record is

A region holds one or more **records**. A compressed record is one
self-describing `NVCOMP_NATIVE` object: nvCOMP compresses it in its own
internal chunks (the codec's chunk size from the codec map) and keeps their
directory in the object's header. The rules that follow:

- **Records start 512-byte aligned** inside the region
  (`COMPRESSION_ALIGNMENT`), as nvCOMP requires. The gap after a record is
  padding.
- **A record holds whole rows**: a power-of-two number, the largest within
  `TQP_MGR_RECORD_MB` raw (default 4 MiB) and within `RECORD_ROWS_MAX` rows
  (2^20; `None`: no row cap), raised to the fewest that keep it a multiple
  of 512 bytes (for rows over 8 KiB), cut from the first row of its
  (tick, bucket) slice, so each record's source address stays aligned and the
  boundaries of a narrower column of the same slice are boundaries of every
  wider one (`record_rows`, `record_cap_b`; `row_quantum` is the rows a
  1-byte column's record holds, 2^20).
- **The record index** is the record table's (`IO/pages/record_table.py`),
  held on the host only: per record its region, offset, compressed size,
  raw size, rows, codec id and `DecompressionConfig` handle, plus one
  `codec_key = (codec_idx, elem_size, chunk_b)` per codec. The handles come
  back from the compress launch. They are process-lifetime entries in the
  native module's registry, one per record. Nothing of the index is stored
  next to the payload or copied to the GPU.

Because the table carries the handles, decompress is host arithmetic plus
one `mgr_decompress` call per codec group: the tick's compiled template
builds pointer arrays from the index and passes the cached handles, and
nvCOMP reads each record's compressed size from the record's own header. It
reads nothing back from the GPU and parses no headers on the host. A record
without a handle is parsed at its template's launch (`mgr_ingest_configs`,
which synchronizes; the native module counts these as
`header_parse_syncs`).

## Regions: `RegionProducer` and `HomeWriter`

`IO/codec/region_producer.py` is the output path of every pipe, with or
without the granule cache (below): the sort and filter pipes, the direct
pipes (join core, aggregation passes, project; with a graph node, whose
output instances publish the tables; a bare pipeline's tables are its
`result_tables`) and the gathers' chunks
(`IO/pipeline/rearrange_pipeline.py`, which packs chunk k after chunk k+1's
gather); also the scan's representations (`IO/data_scanner.py`, which
packs each batch at the next batch's tick, with its producer stream current;
`spills=False` keeps its copies home out of the movement ledger's spill
kinds, and `pool_tag="scan"` its homes out of the pinned pool's
intermediate residency) and the codec sweep
(`chunk_b` overrides the codec map's chunk for every codec). One tick's
output of one column becomes one **region**: its records back to back,
512-aligned, bucket by bucket, each bucket's records in row order, in the
pinned pool (its home). A pipe's regions of a column
form one record table (`IO/pages/record_table.py`) with one stream per
bucket, frozen when the pipe drains; the column is then a
`variable.TableVariable` and its readers select records from the table
(`IO/transfer/tick_layout.py`).

- **Tick t, the producer's stream** (`compress`; the pipe's compute stream,
  which the pipeline makes current around every call, or the caller's
  current stream when none is set). Each
  bucket's rows are cut into records from the bucket's first row
  (`record_rows`, above). The
  columns of one `(codec, dtype, width)` are compressed together in pieces of at most
  `_MAX_OUT_BUF_BYTES` (256 MiB) of worst-case output, cut by an upper bound
  of each record's worst case; each piece is one `mgr_compress_begin` /
  `mgr_compress_launch` into a worst-case slab of its own, so no allocation
  of the tick is larger than one piece, with the actual sizes stored into
  pinned memory over UVA by one block on the stream (`tqpcompress.uva_store`;
  a copy would queue on the copy engine behind the copies home already
  issued, and the stream with it). Codec info and the device buffer of actual
  sizes are kept per producer, and each instance's records are named as
  they are cut (`_col_records`). Nothing blocks, and every slab stays alive
  until the pack. A failure while the pieces are set up (an allocation that
  does not fit) finishes every ticket begun.
- **Tick t+1, before the operator** (`pack`). The sizes have landed (the
  tick barrier synchronized the compute stream). One cumulative sum over
  the aligned sizes gives every record's place in its column's region (the
  column's records back to back), and one block of records for its table.
  Piece by piece, one `batched_memcpy_abs` compacts the piece's records from
  its slab into a dense GPU buffer of its own, and the slab goes (a tick
  with no cached output is packed by per-piece array arithmetic). The
  kernel reads its address arrays straight from a pinned staging buffer
  (`IO/pages/abs_copy.py`) over UVA: an upload of them on the compute stream would
  queue behind the next tick's input prefetch on the copy engine, and the
  tick's compute with it. A region's
  home is carved from a pinned chunk of its column, sized for the column's
  expected remaining output (mean region × ticks left, at most 16 MiB; the
  pipe sets `ticks_left`, and a caller that does not gets 8): small regions
  share the pool's 0.5 MiB block rounding. When the pool has no run for
  such a chunk, the region alone is allocated. A chunk that cannot take the
  next region, and every chunk at the drain, gives its unused blocks back
  (`MemoryPool.shrink`); the table returns its chunks when it is freed
  (`RecordTable.host_blocks`).
- **Right behind the pack** (`HomeWriter.ship`). The home writer copies
  each region's part of each dense buffer into its home (one copy per
  region and piece) and drops the buffers once the copies have ended.
- **RAW columns** need no deferral: the operator's output tensor is the
  dense buffer (each bucket's rows cut into records the same way; the
  padding between buckets ships with them), shipped at once in `compress`.
- **End of pipe** (`drain`): the last tick's pack, a wait for every
  shipment in flight (with the granule cache, the slot fills with them), and
  the tables' freeze. `VortexPipeline.get_result`'s device sync
  comes after it.

The **home writer** (`home_writer.py`) is the only code that copies into
homes. It runs the copies on one background stream behind an event recorded
after the pack, which the tick barrier does not wait for, and holds every
buffer it copies from until the event after its copies has fired, then
drops it (`reap`): the arena orders a free only on the allocating stream
(the compute stream), so dropping it earlier could hand the bytes to new
compute work while the copy still reads them. A pipe keeps at most two
ticks' shipments in flight (a tick's RAW shipment and its packed region
count as one tick); a third tick's pack or RAW shipment first waits on the
oldest (`wait_for_room`). If the pipe's ticks fail,
`VortexPipeline.do_exchange` calls `abort`, which waits for every copy in
flight before the failure leaves the pipe and gives the chunks back. A
shipment's event pair is the pipe's d2h lane once it has ended
(`take_lane`), and the pack's span counts in the tick's `compress_out`, so
the compression searches still see the output path.

A write-pick change of the read-and-write search takes effect at the next
tick's compress: a region holds one codec, and a column's table may mix RAW
and compressed regions. No seal or backlog is needed.

**With the granule cache** (`cache`, `IO/cache/native_cache.py`; the
pipeline sets `cached_outputs`, every output instance the graph does not keep
in host memory, by base, with its consumer position and expected reads):

- At the pack (a RAW output: at its compress), each cached output's home is
  carved as above and its region, of location `cached`, is registered with
  the cache; the tick's regions are offered in one `admit`, which takes a
  prefix of each region's granules. A tick whose outputs are all cached and
  admitted whole is written into its slots by one kernel for the tick
  (`_fill_slots`), and nothing goes home; in any other (`_issue_copies`)
  one `batched_memcpy_abs` per piece (absolute addresses) writes the
  records' bytes in the admitted prefix into their slots and the rest into
  the piece's dense buffer, the piece's span with the slot prefixes taken
  out (every part stays 512-aligned); the home writer copies the dense
  parts home and the cache waits on that copy before reading the home
  (`note_home_copy`). These kernels (the slot fills, and a RAW output's
  slot copies) run on the home writer's stream, in the tick's shipment
  before its copies home (`HomeWriter.ship`'s `fill`): behind the copies
  home of the dirty victims whose slots they reuse (the admission's
  events), so the compute stream never waits on a victim, and the shipment
  holds the slabs they read.
- With cleaning on (`TQP_CACHE_CLEAN`, off by default), after the tick's
  admissions the cache's cleaning targets go home on the writer, behind the
  tick's slot fills on its stream (`commit_cleaning`); a reclaim's dirty
  victims go home through `home_writer.copy_home`. A region under
  `TQP_CACHE_FLOOR` granules (default 1) is not offered to the cache.
- Slabs and dense buffers are plain arena allocations: with the cache
  attached, one that misses borrows the cache region's lowest blocks inside
  the allocator (`memory/README.md`, "Lending"), which may evict and move
  granules of earlier ticks, never this tick's, whose slot addresses the
  copy plan already holds. A slab that still does not fit fails the query.
  A dense buffer that does not fit (with or without a cache) is skipped,
  and its refused records are copied from the slab straight to their homes,
  the slab held until those copies end.
- The drain synchronizes the producer's stream, so the last pack's slot
  fills have ended before a later pipe reads them; `abort` drops the
  regions it registered.

## Decompress

- **The per-tick path** does not organize anything itself.
  `BatchedDecompress.decompress_all` calls `decompress()` on every `DirectIn`
  stage that has data. The tick's compiled input layout
  (`IO/transfer/tick_layout.py`) has already bound every compressed stage to a
  codec-group context. The first stage of a group launches one
  `mgr_decompress` for the whole group (`_launch_mgr_group`), and the other
  stages narrow their views into the shared output.
- With `TQP_VARCHK` set, the template compile asserts each record index:
  positive original sizes that sum to the decoded size, records that do not
  overlap (`IO/transfer/tick_layout.py`).

## Streams and synchronization

| Where | Stream | Host blocks? |
|---|---|---|
| `RegionProducer.compress`, `pack` | the producer's stream (the compute stream in a pipe) | no; a third tick's pack waits for the oldest shipment in flight (the cap) |
| `HomeWriter.ship` | the writer's background stream, behind the pack's event | no |
| `RegionProducer.drain` | compute stream | yes, until every copy home has ended |

## Shutdown

An `atexit` hook in `manager_compress.py` calls `mgr_shutdown` so cached
managers release their scratch before the allocator is torn down.

## Environment variables read here

| Variable | File | Effect |
|---|---|---|
| `TQP_MGR_RECORD_MB` | `region_producer.py` | the raw-byte cap of a record, `RECORD_MB` (default 4); the row cap `RECORD_ROWS_MAX` (2^20) is a module global only |

`AGENTS.md` has the parsing rules. The per-codec nvCOMP chunk size is not an
environment variable; it comes from `profiling/results/codec_map.json` through
`compression/format_policy.py`.

## Files

| File | Purpose |
|------|---------|
| `batched_compress.py` | `aligned_partition_layout`, the padded per-partition row counts and byte starts of a partitioned output (written by the join's partition kernel, `operators/kernels/partition_sort.cu`, and a partitioned aggregation's pass 1, `operators/aggregate.py`'s `_partition_outputs`; read by the producer's `_bucket_starts`), and `_MAX_OUT_BUF_BYTES` (256 MiB), the cap on the worst-case output of one batched manager call. |
| `manager_compress.py` | What the producer asks nvCOMP about a codec, cached per process: `_codec_info` (codec id and manager chunk; a `chunk_b` overrides the codec map's) and `_worst_case_aligned` (a record's worst-case compressed bound); and the `atexit` shutdown hook. |
| `region_producer.py` | `RegionProducer` (`chunk_b`: the sweep's chunk; `spills`: whether its copies home are ledger spills; `pool_tag`: what its homes hold, for the pinned pool's residency counts): per pipe, `compress` (tick t), `pack` (t+1, one region per column; with the granule cache, the admission and the slot fills), `drain` and `tables`, `collect_lane` for the pipeline's d2h lane, `abort`; the record cut (`record_rows`, `record_cap_b`, `row_quantum`, `_cut_records`). The internal records are `_Column` (a column's regions, blocks and pinned chunk), `_Piece` and `_Tick`. The per-tick steps: `compress`, `_col_records` and `_launch` (tick t), `_pack` (t+1: per-piece array arithmetic without a cached output, `_fill_slots` for a tick admitted whole, `_issue_copies` otherwise). |
| `home_writer.py` | `HomeWriter`: the shipments of one pipe on the background `writer_stream`, the two-tick cap (`wait_for_room`), `reap`, `take_lane`, `drain` and `abort` (`ship` issues the copies by switching the current stream directly, without the stream context); `copy_home`, the granule cache's reclaim copies. |
| `batched_decompress.py` | `BatchedDecompress.decompress_all`, the per-tick fan-out to `DirectIn.decompress`. |

## Tests

- `tests/compression/test_nvcomp_manager.py` covers the manager path for all
  seven codecs, through the producer and a `DirectIn`: round trips (with an
  empty partition), incompressible data, the record layout, guard canaries,
  handle portability across streams and sync-freedom (hot decompress from
  the table's handles under `torch.cuda.set_sync_debug_mode("error")` with
  no header parse).
- `tests/IO/test_region_producer.py` covers the region producer and the home
  writer: one region per tick of cumulative aligned records, bucket-major,
  read back whole and by bucket range (RAW, BITCOMP, ANS); a write flip
  mixing codecs in one table; a codec group split into pieces, no
  allocation above one piece, and no ticket left after a failed one; the
  writer holding a source until its copy has ended, a fill run on the
  writer's stream before the copies, the buffers only it reads dropped once
  it has run, and the lane's span covering both, the
  two-tick cap, RAW and compressed shipments of one tick sharing it;
  `abs_copy.launch` offsetting destinations from a base and recording its
  staging event behind the kernel; regions freed right
  after the drain taking no late copy; homes carved from chunks trimmed at
  the drain; work on a producer stream other than the caller's; a
  failed pipe waiting for its copies home; the one-bucket record shortcut;
  and the same random ticks with and without the granule cache: the same
  tables and record bytes under a deterministic stand-in compressor, the
  same rows decoded under nvCOMP.
- `tests/page_cache/test_granule_producer.py` covers the producer with the
  granule cache: a record cut between its slots and its home read back
  exactly, earlier ticks evicted home, an uncached output kept a host region,
  the slab-to-home copies when a dense buffer does not fit, and whole-admitted
  ticks filled into their slots by one kernel.

All of these need a GPU. Run:

```bash
TQP_GPU=<idle> docker compose run --rm -T dev python3 test.py tests/IO tests/compression
```
