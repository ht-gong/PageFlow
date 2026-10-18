# `IO/cache/` — the granule cache and its policies

This directory holds the GPU cache `--buffer_strategy` `lru`, `mru` and
`dataflow` build: the **granule cache**. It keeps granules of the regions
one pipeline produced resident in GPU slots, so the pipeline that consumes
them reads them with a device-to-device copy instead of an H2D transfer; a
dirty granule is copied home when it is evicted (cleaning, which copies
granules home ahead of eviction, is off unless `TQP_CACHE_CLEAN` is set).

In the `IO/` import graph, `cache/` imports only `telemetry/` (the movement
ledger and the switches). Consumers:

- `main.py` sets the strategy (`config.set_global_buffer_strategy`), sizes
  the arena's cache region (`cache_region.py`), builds the cache over it,
  resets its position per plan and clears it between queries.
- `IO/codec/region_producer.py`, the producer side, registers and admits
  regions at pack time, fills their slots and, with cleaning on, ships
  cleaning copies home.
- `IO/transfer/tick_layout.py` (`bind_tick_sources`), the consumer side,
  reads a cached region's pieces from its slots and its home and counts the
  reads.
- `IO/pipeline/vortex_pipeline.py` pins the next tick's granules, advances
  ticks and positions. `IO/pipeline/rearrange_pipeline.py` advances the
  position too.
- `memory/` (the arena) lends the region's lowest blocks to an allocation
  that misses, through the core's lend hook, once `main.py` has attached
  the occurrence's cache (`attach`); the core gives them back.

## How it works

With `--buffer_strategy` `lru`, `mru` or `dataflow`, `main.py` builds the
cache over the arena's cache region: `NativeGranuleCache(CacheRegion(),
policy, copy_home=home_writer.copy_home, clean_percent=CLEAN_PERCENT)` (25),
one per policy (`config.POLICY` may change it per occurrence), its
bookkeeping in the C++ core `GranuleCore` (`TQPlib.tqpcache`). An occurrence
runs it unless `config.CORE` (`TQP_CACHE_CORE`) is `cpu`. It attaches the
occurrence's cache to the arena once the region is sized (`attach`, so
allocations that miss borrow from it), calls `reset_position()` at the
start of every plan, and `detach()` and `clear()` after every query.

**The region.** The cache's GPU memory is the arena's top blocks, behind a
boundary (`memory/README.md`): `--slot_pool_gb` of 1 MiB blocks, out of the
arena's free list, so no operator allocation lands in them. `CacheRegion`
(`cache_region.py`) holds its geometry (`base`, `block`, `num_blocks`,
`boundary`, read live from the arena), moves the boundary between queries
(`set_size` after each query's scan and at its end) and gives one
non-owning uint8 tensor over its bytes (`view`). A slot is an arena block:
slot `i` is block `i`, at `base + i * block`, so the core's slot ids span
the whole arena and only the region's blocks, `[boundary, num_blocks)`,
are ever free. The core hands out the region's highest free slots first
(the top run that holds a request, else the top runs downward), so
granules gather at the arena's top, while the arena's operator allocations
are first fit from its bottom: the blocks next to the boundary stay free
while the region is not full, and they are the ones lending gives first;
no admission or eviction depends on which slots a granule takes. Between
queries the boundary moves while the cache is empty: `--slot_pool_gb_list`
moves it per query, and an occurrence under `TQP_CACHE_CORE=cpu` moves it to
the arena's top (no region) and the next one with the cache back. `clear()`
then takes the region's current blocks as the free slots. Within a query
only lending moves it (below). A region registered before a `clear` stays harmless: its
`drop` and `home_events` do nothing (each region carries the cache's epoch). Cache-on runs take the paths of
a run without a cache: the table scan, the region producer for every pipe
and gather, and the region readers; a gather's outputs are host regions,
never cached. `tests/page_cache/reference_cache.py` is the same cache in
Python, the reference model `test_native_core.py` compares the core's
decisions against; it is not an engine path.

It caches regions, not pages. A cached region is cut into granules of one
slot (one arena block) each, and each granule has one row in
the core's tables: its slot, clean, live, pins and the policy keys
(consumer position, lowest live hint, admission tick and order). States:
dirty (in a slot only), clean (in a slot and its home), evicted (home only),
dead (no read to come; its slot is released once its reads settle, at the
next tick boundary or pipeline start, or by the next reclaim if it was
pinned or admitted in the current tick; the ledger counts it as
`avoided_dead`).

- **Admission.** `register` gives the cache a region's size, its reserved
  home and its records by stream (the hint of a stream defaults to its index,
  a partitioned output's bucket: a join's sort pipe's, or a partitioned
  aggregation's pass 1's). `admit(regions)` admits a prefix of each
  region's granules in byte order, stopping at the first one the policy
  refuses: a free slot always admits; otherwise the granule is compared with
  the victim reclaim would free for it. LRU and MRU admit while there is
  one; under Dataflow a dead
  victim loses, a live one only to a sooner consumer or, on a tie, a
  strictly lower hint). Granules admitted in the current tick are not
  victims until `advance_tick`.
- **Reclaim.** `reclaim(nbytes)` frees dead granules, then live ones in
  policy order (a clean one before a dirty one only on an equal key),
  skipping pinned ones. Dirty ones
  are copied home through the `copy_home` callback (the home writer), and
  their slots are released behind its event. A granule whose streams have
  all been read is dead under every policy.
- **Lending (a shrink) and its return (a grow).** Attached
  (`attach`: the core's `tqpcache.lend_hooks()` registered with the arena's
  `set_lend_hook` over the home writer's stream, and the arena's
  `lend_api()` given to the core with `set_arena`), the cache is what an
  allocation that misses borrows from, all in C++ (`memory/README.md`,
  "Lending"). The arena synchronizes the device (no copy still reads or
  writes a slot; every copy into a home has ended, so the core marks every
  batch fired, and pending slots are free) and asks the core's `lend(k)` for
  the region's lowest `k` blocks. The core plans a shrink with no floor
  (`plan_shrink`): the surplus residents evicted in victim order (dead,
  clean, dirty), and the granules left in those blocks moved to the highest
  free slots above them, in slot order so a run of them moves as one run.
  It issues the copies on the lend stream through the arena's
  `arena_lend_copy` (the dirty victims home first, then the relocations, so
  a relocation into a dirty victim's slot follows its copy home; copies
  that run on merged into one), then commits (`commit_shrink`: the moved
  granules' slots rebound, the evicted dropped, the region's lowest slot
  raised); the arena synchronizes the stream before it hands the blocks
  out. A pinned granule (a tick's reads in flight, or a read outside the
  ticks, `tick_layout.pin_cached_reads`) or one admitted in the current
  tick (the producer holds its slot addresses through the tick's pack) is
  neither evicted nor moved: a lend that would need one gives nothing (all
  or nothing), and so does one with a copy home the wrapper has not
  reported issued (`note_issued`), since the lend counts every batch ended.
  Its live victims count toward the cleaning gate, as a reclaim's. The lent
  blocks come back (`reclaim_lent`: the arena's `arena_region_return` moves
  the boundary down over the free blocks just below it, up to those still
  lent, and the core adds them as free slots) at every `advance` (a
  pipeline's start), and only there: an admission short of slots evicts as
  it would with nothing lent (returned blocks would take this tick's
  granules, which cannot move, just above the blocks the next lend needs).
  Nothing holds a slot address across a lend: the readers resolve theirs
  from the core per tick (`pieces`, under the tick's pins) or pin a read
  outside the ticks, the producer writes only this tick's slots per pack,
  the cleaning and the copies home issue theirs before anything allocates,
  and the compiled read plans hold region indices and byte spans.
  What a lend could give now, without moving anything, is the core's
  `lendable_slots(0)`: the blocks below the lowest one holding a pinned or
  same-tick granule, at most the region less those granules, with a slot
  waiting on a copy home counted free (a lend synchronizes first). The
  arena's stats carry it as `lendable_b`, which the sizing search's budget
  and a gather's streamed-or-UVA choice count. The arena holds the core's
  address while
  attached: `detach` (and the finalizer `attach` registers, should the
  cache go while attached) takes it back.
  `lend_stats` / `take_lend_stats` give the counters `main.py` prints per
  query (`[lend]`) and files in the movement ledger.
- **Order.** The policies are integer keys per granule; the cache puts
  dead granules above the key and takes the first *k* with one partition. Ties
  go to the older region and, inside a region, to the later granule, so a
  region keeps a resident prefix and an evicted suffix.
- **Cleaning** copies dirty granules home ahead of eviction, only with
  `config.CLEANING` (`TQP_CACHE_CLEAN`, off by default: at SF100 with a 2 GiB
  cache it wrote 22 GB home to spare 14.5 GB of eviction copies, and the pass
  was 1.2% faster without it): `cleaning_targets` names the dirty granules
  among the front `clean_percent` (the engine's is `CLEAN_PERCENT`, 25) of
  the live, unpinned residents in policy order, clean ones there counting as
  covered. Cleaning runs only while the cleaning gate is open (constants, no
  setting): every admission adds its live victims (granules reclaimed with a
  read still to come, clean or dirty, in granule bytes) to a window of the
  last `CLEAN_GATE_WINDOW` (64) admissions, and a reclaim outside admission
  (`reclaim`, a lend's evictions) adds its live victims to the latest entry;
  the gate opens once the window's sum reaches `CLEAN_GATE_ARM_B` (4 MiB) and
  closes once it is 0 (at SF100 this keeps live write-backs at 0.3% / 0.2% of
  the evicted live bytes under Dataflow / LRU, with 6.2 / 7.0 GiB of
  cleaning; 64 MiB over 32 admissions left 3.9% under Dataflow).
  `commit_cleaning` takes the copy's event, and the granules are clean once
  it fires.
- **Copies into homes and slot reuse.** Every copy into a home is a batch
  with one event, recorded per granule: the producer's
  copy of a region's refused part (`note_home_copy`), cleaning's and a dirty
  reclaim's. A read of home bytes waits on their granules' batches
  (`pieces` returns the events; `home_events` gives a region's, which its
  free waits on). A slot is reused only after every copy on it has ended:
  a granule's slot is released behind its batch, and an admission that takes
  such a slot returns the events for the slot fill to wait on.
- **Reads.** Every range call takes regions (a `NativeRegion` or an array
  of region indices) with arrays of byte bounds, so a tick's ranges over
  many regions are one call. `replace_consume_pins(regions, lo, hi)` pins
  the granules the next tick reads (or the ids `granules` gave, which the
  reader keeps per column); `pieces(regions, lo, hi)` gives the
  ranges as pieces in slot runs or in the home, with the events of the
  copies into those home bytes still in flight; `note_read(regions, lo,
  hi)` counts one read of every stream whose last record in a region ends
  inside its range, and a granule is dead once every stream in it has had
  all its reads (under LRU and MRU the core keeps a count of each granule's
  segments with reads to come; under Dataflow it also recomputes the
  granule's lowest live hint). Dead granules are released from a list of
  candidates (those that died, were unpinned or were admitted dead since
  the last release, and the dead ones a release kept because they were
  pinned or admitted in the current tick), not by a scan of the region;
  only a Dataflow `advance`, which kills every granule whose consumer has
  run, scans every slot. `drop` releases a freed region's slots.
- **Tables.** The core keeps granules, regions and segments (one per run of
  a stream's records in a region, in region and byte order) in arrays, so a
  range finds its segments and granules by binary search. Every call takes
  and returns numpy int64 arrays; copies into homes are *batches*, integers
  created in order, which the wrapper maps to the events of the home
  writer's one stream, so waiting on the newest live batch of a set waits
  on all of it.

How the engine drives it:

- **The producer** (`IO/codec/region_producer.py`), at pack time (tick t+1;
  a RAW output at once), carves each cacheable output's home, registers the
  region (a `Region` of location `cached`, `IO/pages/region.py`) and offers
  the tick's regions in one `admit`. One copy kernel over absolute addresses
  (`batched_memcpy_abs`) writes the admitted prefix from the slab into its
  slots, at the addresses the core gives (`slot_addrs`), and the refused
  rest into the dense buffer, which the home writer copies home
  (`note_home_copy`). The kernel (the slot fill) runs on the home writer's
  stream, behind the copies home of the dirty victims whose slots it
  reuses (the admission's events) and before the tick's copies home, so
  the compute stream never waits on a victim; the writer holds the slabs it
  reads (`IO/codec/home_writer.py`, `ship`'s `fill`). Then, with cleaning
  on, `cleaning_targets` goes to the home writer, behind the tick's slot
  fills on its stream (`commit_cleaning`). A region under
  `config.FLOOR_GRANULES` granules (default 1) is not offered and goes home
  as a plain region. An instance the graph keeps in host memory (a gather
  source) is a plain host region. Its slabs and dense buffers are plain arena
  allocations: one that misses borrows the region's lowest blocks inside the
  allocator, which moves none of this tick's granules. A dense buffer that
  still does not fit is skipped, and the refused records go from the slab
  straight to their homes, while a slab that does not fit fails the query.
- **The reader** (`IO/transfer/tick_layout.py`, `bind_tick_sources`) binds
  the runs of cached regions of a tick's stages that read only cached
  regions with one `pieces` call per tick, from their spans concatenated
  when the tick's layout is compiled (`_TickCached`), and a stage that also
  reads other regions with a `pieces` call of its own: home pieces by H2D
  after the copies into them (the H2D lane waits on their events,
  `BoundTickPlan.home_waits`), each a pack entry of its own; slot pieces,
  merged where they continue each other, by one `batched_memcpy_abs`
  launch per tick on the D2D lane (`launch_slot_pieces`, one native call
  whose kernel reads the address arrays over UVA,
  `IO/pages/abs_copy.py`). The read is counted with `defer_read` once the
  bind holds, and `note_read` runs at the next `advance_tick` or `advance`
  (`settle_reads`). A granule the count leaves dead may then be released
  while its copy still runs: the slot fills that could reuse its slot run
  on the home writer's stream behind the pack's event, which the compute
  stream records after the tick event that waited on that copy (or behind
  the tick barrier), and a lend synchronises the device first. A record the
  tick ends inside is left out, so its stream's last read is counted once.
- **The pipeline** (`IO/pipeline/vortex_pipeline.py`) calls `advance(node)`
  at its start (which takes lent blocks back) and `advance_tick()` at every
  tick and once more after its drain (so the last pack's granules may move
  for a lend between pipes), pins the granules each tick reads
  (`replace_consume_pins`) before any admission or allocation of that tick
  (under the barrier tick 0 and 1 before their launch, later ticks at the
  previous tick's pre-plan; with event-ordered ticks, at the top of each
  tick, the chunk it issues and the one it computes), and drops the pins at
  its end. `rearrange_pipeline` advances the position. A read outside the
  ticks (`TableVariable.col_to_gpu`, `IO/pipeline/common.col_to_gpu`) pins
  its granules from its bind until its copies are issued
  (`tick_layout.pin_cached_reads`).
- **A freed column** frees its regions (`RecordTable.free`); a cached region's
  free calls `drop` and waits for the copies still writing its home.
- **The movement ledger** (experiment telemetry, `IO/telemetry/movement.py`)
  counts granules packed into
  slots as `cache_fill`, a refused suffix copied home as `spill_rejected`,
  slot pieces read as `cache_hit` and home pieces as `restore_evicted_h2d`, cleaning
  copies as `spill_mirror`, and reclaim's and a lend's victims as
  `spill_evict` (a dirty granule copied home), `avoided_clean` and
  `avoided_dead` (released with no copy; a lend's are filed at the query's
  end, `take_lend_stats`), a page per granule: its bytes as `wire_b` and, as `logical_b`, the
  decoded bytes of the records in it, each record's spread evenly over its
  stored bytes (`ledger_bytes`, from the decoded sizes `register` gets
  under experiment telemetry).
- **The residency report** (experiment telemetry, `IO/telemetry/residency.py`)
  reads the slots in use (`used_bytes`) and the most in use at once since
  `reset_peak` (`peak_used_bytes`), which the core counts as it hands slots
  out and takes them back.

## Files

| File | Purpose |
|------|---------|
| `config.py` | `BufferStrategy`, the enum behind `--buffer_strategy`; `set_global_buffer_strategy`, which `main.py` calls once, and `buffer_strategy()`, which it reads back to build the cache (none for `cpu`) and pick the policy. The per-occurrence switches `CORE` (`TQP_CACHE_CORE`: `native`, the default, or `cpu`), `POLICY` (`TQP_CACHE_POLICY`), `FLOOR_GRANULES` (`TQP_CACHE_FLOOR`, default 1), `PLAN_AT_COMPILE` (`TQP_CACHE_PLAN`, default on) and `CLEANING` (`TQP_CACHE_CLEAN`, default off), each read at its use so `switch_ab.py` can alternate it (`docs/cache-host-overhead-investigation.md`). |
| `cache_region.py` | `CacheRegion`, the arena's cache region as the engine holds it: its geometry (`base`, `block`, `num_blocks`; `boundary` and `nbytes` read live from the arena), `set_size` (moves the boundary through `tqpmemory.region_move` between queries, from no region too; a refused grow raises `the arena refused to grow the cache region`), `view` (one uint8 tensor over the region's bytes, zero-copy, valid until the boundary moves). |
| `granule_core.cc` | `TQPlib.tqpcache`, the cache's core in C++ (`CMakeLists.txt` builds it; no CUDA or torch at link time): `GranuleCore`, the granule, region and segment tables, the three policies and the run allocator of its slots (the region's arena blocks, `set_base` / `set_region`, handed out from the top), every call over numpy int64 arrays; home copies are *batches* the wrapper maps to events; the lend (`lend`, `reclaim_lent`, `set_arena`, `ctx_ptr`, `note_issued`, `lend_stats`) and the arena's C hooks `tqpcache_lend` / `tqpcache_lendable` (`lend_hooks()`); `plan_shrink` / `commit_shrink` / `lendable_slots` / `grow`, which the lend runs; `split_at_granules`, the producer's range cut. |
| `native_cache.py` | `NativeGranuleCache`, the granule cache: `GranuleCore` behind `register`, `admit`, `reclaim`, `victim_order`, `evictable_count`, `cleaning_targets` / `commit_cleaning` / `poll`, `replace_consume_pins` / `pin` / `unpin`, `granules`, `pieces`, `note_read`, `defer_read` / `settle_reads`, `release_dead`, `note_home_copy` / `home_events`, `drop`, `advance` / `advance_tick` / `reset_position`, `attach` / `detach` / `attached` / `lend_stats` / `take_lend_stats`, `clear`, `cleaning_open`, keeping every CUDA object (events, the copies home); `NativeRegion`; `Admission`, `Reclaimed`, `Pieces`, `Cleaning`; `CLEAN_PERCENT`, `CLEAN_GATE_ARM_B` and `CLEAN_GATE_WINDOW`. |

## Tests

`tests/page_cache/` has seven test files, the reference model
(`reference_cache.py`, `reference_policies.py`: the same cache in Python,
its policies as key functions over numpy arrays, over its own slot pool,
`reference_slot_pool.py`) and `regions.py` (a cache region over a tensor of
its own, for the unit tests):

- `test_granule_cache.py`: the granule cache on the CPU (reclaim order,
  pins, prefix admission and tie-keep, each policy's victim order against a
  reference order written out in the test, cleaning, the dead rule, the
  release of dead granules under a pipe's rolling pins, pieces, slot reuse
  behind events, slots filled from the region's top, a `clear` following a
  moved boundary with an earlier region's free harmless);
- `test_cache_region.py`: `CacheRegion` on the test arena (CUDA): `set_size`
  moves the boundary and the operator space follows, a refused grow raises,
  and a cache over it fills its top slots, read through the region's view;
- `test_granule_ranges.py`: the range calls against their one-range,
  one-granule forms (pieces, consume pins, the reader's cached spans) and
  the cleaning gate, on the CPU;
- `test_slot_runs.py`: the reference model's slot pool's runs, blocks and
  pending releases;
- `test_granule_producer.py`: the region producer with the granule cache on
  CUDA: a record cut between its slots and its home read back exactly
  (compressed and RAW, earlier ticks evicted home), whole-admitted ticks
  filled into their slots by one kernel, a slot fill on the writer's stream
  behind its victims' copies home (the producer's stream not waiting), a
  tick's fully cached stages bound with one `pieces` call, an uncached
  output kept a host region, and the slab-to-home copies when a dense
  buffer does not fit (with and without the cache);
- `test_native_core.py`: the cache against the reference model, on the CPU:
  the same random sequence of registrations, admissions, reads, settles,
  drops, reclaims and cleaning on both must make the same decisions at
  every step (admissions, victims, cleaning picks, pieces, states and
  residency; not slot ids);
- `test_shrink_grow.py`: moving the region's boundary in the core, on the
  CPU: random sequences keep its slots consistent (no slot held twice, every
  resident in the region, the floor kept, a pinned or same-tick granule
  never moved or evicted, each shrink giving what `lendable_slots` said), a
  shrink stopping below such a granule, its residents moved in slot order,
  `lendable_slots` against a shrink of every slot.

Lending itself is `tests/allocator/test_arena_lend.py`'s (GPU, each check
in a process of its own on a small arena): a miss lending the region's
lowest blocks, its evictions (dirty copied home, clean not) and relocations
reading back, the blocks back at the next pipeline start or an admission
short of slots left lent (it evicts) and not while in use, a pinned or same-tick granule stopping
a lend, a lend inside a torch op, the whole region lent, and the engine's
own paths: a read outside the ticks pinning its granules, the producer's
copy plan surviving a lend inside its dense buffer's allocation.

The unit tests of the cache need `TQPlib.tqpcache`. `tests/e2e` runs the
SF1 ladder under `lru`, `dataflow`, `mru` and `cpu` (its cache rungs sit on
the sizing-search rung), and the `lru` and `dataflow` caches with the
compile-time read plan and sizing off (`test_native_core_rung`);
`tests/e2e/test_cache_lend.py` runs each policy on a 2 GiB arena whose 1.75
GiB region leaves the operators too little, so they borrow from it.
Run the unit tests with:

```bash
TQP_GPU=<idle> docker compose run --rm -T dev python3 test.py tests/page_cache
```
