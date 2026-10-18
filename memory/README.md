# `memory/` — the GPU arena allocator for TQP-Vortex

This directory holds the pooled CUDA allocator that PyTorch runs on inside the
engine, plus the pinned-host allocation primitive the scan path pins its data
with. One `cudaMalloc` at startup reserves the whole arena (`--cmp_pool_gb`);
every torch CUDA allocation for the rest of the process is carved out of that
slab in fixed 1 MiB blocks, so operators never call `cudaMalloc` and an
out-of-memory failure surfaces as a pool diagnostic rather than a driver error.
The granule cache's region (the arena's top blocks, `IO/cache/cache_region.py`),
nvCOMP manager scratch, every per-tick buffer and cuDF's buffers (through RMM)
are all tenants of this one arena (see "Memory: pools and sizing" in
`AGENTS.md`).

The pieces fit together like this:

1. **`cuda_memory_allocator.cc`** is compiled by `CMakeLists.txt` into
   `libcustom_allocator.so`, a plain shared library with a C ABI. It holds one
   process-wide `GPUMemoryPool` and exports the two entry points torch's
   `CUDAPluggableAllocator` needs, `pa_cuda_malloc` / `pa_cuda_free`, next to the
   pool-management, diagnostic and pinned-host functions.
2. **`bindings.cpp`** is compiled into the `TQPlib.tqpmemory` Python extension.
   It links against `libcustom_allocator.so` and exposes the C functions to
   Python; `main.py` imports it as `cmp`.
3. **`main.py`** installs the allocator at startup: it imports
   `TQPlib.tqpmemory` (whose `$ORIGIN` runpath loads `libcustom_allocator.so`
   from `TQPlib/tqpmemory/`), then builds
   `CUDAPluggableAllocator("libcustom_allocator.so", "pa_cuda_malloc",
   "pa_cuda_free")` and calls `change_current_allocator` **before any CUDA
   tensor exists**, and finally `cmp.init(cmp_pool_gb << 30, 1 MiB)` to reserve
   the arena. The bare filename resolves because the library is already loaded
   by the import. Right after, `utility/rmm_arena.install(1 MiB)` makes RMM's
   device resource, which cuDF allocates from, a callback resource over the
   same pair (`stream_malloc` / `stream_free`, each call ordered on the stream
   RMM names). During the scan, `rmm_arena.small_buffers()` puts RMM's
   binning resource in front of it, so cuDF's buffers of at most 1 MiB share
   arena blocks (slabs of one block each). `tests/conftest.py` does the same
   once per pytest process with a 22 GB arena of 1 MiB blocks, so no test
   swaps the allocator itself.

Consumers: `main.py` (install, per-query `free_all` / `reset_peak`, the
`get_arena_stats` callback handed to the sizing controllers in `adaptive/`),
`utility/rmm_arena.py` (RMM's device resource, so cuDF's),
`IO/pages/pinned_mem.py` (`pinned_host_alloc_tensor` for the pinned host pool),
`IO/pipeline/vortex_pipeline.py` (the per-pipeline peak window) and
`IO/pipeline/rearrange_pipeline.py` (`get_arena_stats` for a gather's
streamed-or-UVA choice).
`tests/allocator/test_force_drain.py` covers the force-drain guard,
`tests/allocator/test_arena_region.py` the cache region,
`tests/allocator/test_rmm_arena.py` the RMM resource and
`tests/allocator/test_managed_arena.py` the managed arena, the balloon, the
reset and prefetch, `tests/allocator/test_freelist_evict.py` the miss hook
and `tests/allocator/test_arena_lend.py` lending (each check in a
subprocess, `managed_arena_child.py`, `freelist_child.py`,
`lend_child.py`, since the allocator is installed once per process);
`tests/conftest.py`
bootstraps it for every GPU test, and the smoke and e2e
runs exercise it end to end, and `tests/page_cache/test_cache_region.py` the
engine's `CacheRegion` over it. `tests/allocator/test_slot_pool_blocks.py`
covers the reference model's slot pool, not this code.
The arena-stats getter (`get_arena_stats`) feeds the sizing plane's memory
model (`docs/adaptive_intermediate_sizing.md`).

## How the pool works

- **Layout.** `init(pool_size_bytes, block_size_bytes)` rounds the block size
  down to a multiple of 256 B (and asserts a multiple of 512 B), reserves the
  arena with one `cudaMalloc`, and seeds a single free run covering every
  block. `init` is idempotent: a second call on a live pool returns without
  changes. The block size is 1 MiB, one granule of the granule cache, and at
  most 1 MiB of rounding per allocation. It was 5 MiB, chosen because 20 MB
  blocks once fragmented the arena (`main.py` comment above `cmp.init`).
- **The cache region.** The arena's top blocks can be set aside for the
  granule cache, behind a boundary (`boundary_`, the region's lowest block
  index). The region's blocks are never in `free_blocks_`, `allocs_` or
  `pending_frees_`, so no allocation lands in them and no counter, peak or
  stat includes them; the arena below the boundary is the operator space.
  `region_init(nbytes)` takes the top `nbytes // block` blocks (only when
  there is no region yet, and only if every one is free, else it raises).
  `region_move(delta_blocks)` moves the boundary: a positive delta moves it
  up, so the region's lowest blocks join the free list, coalesced with the
  operator space's top free run; a negative delta moves it down, so the
  region takes the blocks just below the boundary, all or none, only when
  every one is free (completed pending frees are drained first; a block
  still pending, or force-drained with a guard whose free has not run, is
  not free). It returns the blocks moved, 0 for a refused downward move;
  a move past either end raises. `region_boundary()`, `region_blocks()`,
  `block_size()` and `arena_base()` give a caller the region's place: block
  `i` starts at `arena_base() + i * block_size()`. With a cache, `main.py`
  sets the region from `--slot_pool_gb` and moves the boundary between
  queries (`IO/cache/cache_region.py`); the granule cache's slots are the
  region's blocks. An explicit move (`region_init`, `region_move`) sets the
  region's size anew: no block counts as lent after it. Within a query only
  lending moves the boundary (below), except under the free-list baseline,
  which has no cache: its region is the reserve (`--freelist_reserve_gb`),
  free blocks it holds back through a query's timed run and hands out from
  its miss hook (`IO/pages/device_regions.py`).
- **Allocation** is first-fit over `free_blocks_`, an ordered map of
  `start_block -> run length`, rounding each request up to whole blocks:
  **contiguous operator allocations come from the bottom** of the arena (the
  lowest run that fits), while the granule cache hands out its region's
  slots **from the top** (`IO/cache/granule_core.cc`), so the two grow
  toward each other and the blocks next to the boundary are the last either
  side takes. Frees coalesce with both neighbours. All state sits under one
  mutex.
- **Stream-ordered frees.** `free(ptr, stream)` does not return the block to
  the free list. It records a CUDA event on the freeing stream (events come
  from a small reusable pool, timing disabled) and parks the block in
  `pending_frees_`. On the next `malloc(size, stream)` the pool first drains
  pending blocks whose event has completed (same-stream ones first, then the
  rest, both via a non-blocking `cudaEventQuery`), then tries the clean list.
  Only if that fails does it force-drain every pending block into the clean
  list and retry. A force-drained block whose event has not fired keeps it as
  a *guard* (`guards_`) until it fires: whichever stream other than the
  freeing one later takes any of its blocks gets a `cudaStreamWaitEvent` on
  it (`try_alloc`, `wait_guards`), so a free stays ordered on its stream for
  every taker, not only the one that force-drained.
  Cross-stream waits therefore happen only after memory pressure, and no
  stream writes a block while another might still be reading it
  (`tests/allocator/test_force_drain.py`).
- **Lending.** With a granule cache attached to the region
  (`set_lend_hook(lend, lendable, ctx, stream)`, which
  `NativeGranuleCache.attach` calls with the core's `tqpcache.lend_hooks()`
  and the home writer's stream), an allocation that still misses after the
  force-drain borrows the region's lowest blocks, all in C++ (no Python, no
  GIL). The pool counts the miss and takes the fewest blocks `k` that, with
  the free run ending at the boundary (`top_free_run_b`), hold the request
  (`k` is the request alone when no free run touches the boundary; more than
  the region holds is the OOM). It releases its mutex, synchronizes the
  device (no kernel or copy still reads or writes a slot, every copy into a
  home has ended) and calls `lend(ctx, k)`, the core's `tqpcache_lend`:
  with no floor, the core evicts its surplus residents in victim order and
  moves the granules left in the region's lowest `k` slots to the highest
  free slots above them, issuing the copies (dirty victims to their homes
  first, then the relocations, merged where they run on) on the lend stream
  through `arena_lend_copy`, and commits, or changes nothing and returns 0
  when it cannot empty all `k` (a pinned granule, or one admitted this tick,
  in the way, or a copy home it was not told was issued). Once the lend
  stream has synchronized, the pool retakes its mutex, moves the boundary up
  by `k` (the blocks join the top free run), folds the pending frees in and
  tries again; a lend of 0 goes on to the miss hook, if any, and the OOM.
  The lent blocks go back when they are free again: the core calls
  `arena_region_return` (also `region_return(max_blocks)`) at every
  pipeline start (`GranuleCore.advance`), and only there (an admission short
  of slots evicts as it would with nothing lent); it moves the boundary down
  over the
  clean free run just below it (completed pending frees drained first,
  stopping above a guarded block), up to the blocks still lent, and the core
  adds them as free slots. An explicit `region_init` or `region_move` (a
  move by 0 too, which `CacheRegion.set_size` makes when the size does not
  change) resets the lent count. Counters: `lend_stats(reset)` (`misses`
  that tried to lend, `lent` those the cache lent for, `short` those it
  could not, `failed` those still out of memory after a lend, `blocks_lent`,
  `blocks_returned`,
  `lent_now`, `ns` of host time); `get_arena_stats` carries `lendable_b`
  (the attached cache's lendable blocks now, the core's `lendable_slots(0)`)
  and `lent_b`. `lend_api()` gives the core the addresses of
  `arena_lend_copy` and `arena_region_return`. The arena holds the core's
  address while it is attached: `clear_lend_hook(ctx)` detaches that cache
  only if it is still the one attached, which the cache's detach and its
  finalizer call. A lend never nests (an
  allocation inside a hook skips both hooks), and the lend hook and the
  free-list baseline's miss hook have separate slots: only one is set in a
  run (the baseline has no cache region).
- **The miss hook.** `set_miss_hook(fn)` (the free-list baseline,
  `IO/pages/device_regions.py`) adds one step before the OOM: when the
  retry after the force-drain fails, the pool counts the miss (as
  *fragmented* when the clean list's total free blocks held the request),
  releases its mutex, synchronizes the device, and stalls the allocation in
  a loop: it calls `fn(need_bytes, first)` (`first` on the miss's first
  call) through a trampoline that takes the GIL (torch drops it around most
  ops), retakes the mutex, folds every pending free in (`fn`'s frees, whose
  events have fired, coalesce with their free neighbours) and tries again.
  It stops at the first fit, or when `fn` returns 0 or less (nothing left
  to free), which is the OOM. The hook never nests (an allocation inside it
  skips it); an exception in it is reported as unraisable and counts as
  nothing freed. `miss_stats(reset)` reads the counters (`calls`,
  `fragmented`, `failed`, `ns` of host time in the sync, the hook and the
  retry), and `stats_if_freed(ptrs)` gives the free bytes and largest free
  run the arena would have if the live allocations starting at `ptrs` were
  freed, without freeing them.
- **Out of memory.** When the retry also fails the pool prints an
  `=== OOM DIAGNOSTIC ===` block on stderr (request size, block size, pool
  size, the cache region's size and boundary, allocated, free bytes and
  region count, largest contiguous free run, pending frees, with a cache
  attached the blocks lent and lendable, peak) and throws `std::runtime_error("Pool out of memory")`,
  which torch surfaces as a Python exception. The "largest contiguous free"
  line is the one to read: a request can fail with plenty of total free space
  when the arena is fragmented.
- **Two high-water marks**, both of operator-space allocations only (the
  cache region is never an allocation). `peak_allocated_blocks_` feeds the OOM
  diagnostic and `get_peak_bytes`; `reset_peak` rebaselines it to the current
  level, and `main.py` calls it after the warmup and where each query's timer
  starts, so the dump reports the failing query's peak and, under
  `TQP_EXPERIMENT_TELEMETRY`, the perf report's `residency` section the timed
  run's (`IO/telemetry/residency.py`).
  `window_peak_blocks_` is an independent mark for per-pipeline measurement:
  `reset_window_peak` rebaselines it (the window's peak then includes whatever
  operator allocation is already resident, never the cache region) and
  `get_window_peak_bytes`
  reads the maximum since. `IO/pipeline/vortex_pipeline.py` brackets each pipeline run
  with the pair, which the sizing search reads, and under
  `TQP_EXPERIMENT_TELEMETRY` records the result as `peak_gpu_bytes` in the perf
  report.
- **Arena stats.** `get_arena_stats(drain_completed_first=True)` returns
  `{capacity_b, allocated_b, free_b, largest_free_run_b, pending_b, region_b,
  top_free_run_b, lendable_b, lent_b}` from the counters the pool already
  maintains: no device sync, no allocation. All but `region_b`,
  `lendable_b` and `lent_b` describe the operator space (`capacity_b` is the
  bytes below the boundary, lent blocks included); `region_b` is the cache
  region's size, `lendable_b` what a miss could borrow from it now and
  `lent_b` what is lent. `free_b`, `largest_free_run_b` and `top_free_run_b` cover the
  clean list only; blocks still parked in `pending_frees_` are reported as
  `pending_b`. `top_free_run_b` is the free run that ends at the boundary (0
  when the block just below it is allocated or pending): the region's lowest
  blocks join it when the boundary moves up, so the sizing search counts
  `lendable_b` onto it (`adaptive/sizing_controller.py`). With `drain_completed_first`
  the completed pending frees are folded into the clean list before reading,
  so a read at a synced pipe boundary does not misreport a just-freed
  transient as unavailable. This is what the sizing search and a gather's
  streamed-or-UVA choice gate their decisions on.
- **Managed arena.** `init(pool_size_bytes, block_size_bytes, managed=True)`
  reserves the arena with one `cudaMallocManaged(cudaMemAttachGlobal)`
  instead of `cudaMalloc`; everything else (first fit, stream-ordered frees,
  the counters) is unchanged, and torch and RMM allocate from it the same
  way. Its size is address space only: a page is backed when it is first
  touched, on the GPU or the host, and the driver migrates pages on access and
  evicts them to host memory when the GPU is full. `init` raises
  `RuntimeError` unless the block size is a multiple of 2 MiB (the driver's
  migration granule) and the range's base is 2 MiB aligned, so no two blocks
  share a granule. `is_managed()`
  says which kind the arena is. `init(pool, block)` (`managed=False`) is the
  device arena.
- **The device balloon.** `reserve_device(size_bytes)` holds `size_bytes` of
  device memory outside the arena, in one plain `cudaMalloc`, until process
  exit; managed pages cannot use it, so it caps how much of the card a
  managed arena gets. A second `reserve_device` while one is held, or a
  failed `cudaMalloc`, raises `RuntimeError`.
- **The arena reset.** `remake()` synchronizes the device, drains every
  pending free, and raises `RuntimeError` naming the live bytes and
  allocation count if any allocation remains, and when the arena has a cache
  region (its granules would not survive the new range). Otherwise it
  replaces the range with a fresh one of the same size and kind, with one
  free run, no pending
  frees or guards, and both peaks at zero, as `init` leaves them. A managed
  range made this way has no populated page. A managed range's replacement is
  reserved before the old range is freed, so a failed reservation raises and
  keeps the old arena; a device range is freed first, and a failed
  reservation raises and leaves no arena, after which every allocation
  raises `RuntimeError("no arena ...")` rather than return a null pointer,
  which torch's pluggable allocator would not catch. The balloon is not
  touched.
- **Prefetch.** `prefetch(ptr, size_bytes, host_node, stream=0)` is
  `cudaMemPrefetchAsync_v2` of managed memory to host NUMA node `host_node`
  (`cudaMemLocationTypeHostNuma`). It is ordered on `stream` (a
  `cudaStream_t` handle, 0 the legacy default stream) and asynchronous:
  synchronize before relying on the placement. It raises `ValueError` for a
  `host_node` below 0 and `RuntimeError` when CUDA refuses (memory that is
  not managed, for one). It touches no arena state.
- **`free_all`** returns every operator-space block, pending or not, to a
  single free run without waiting on any event, and leaves the cache region
  as it is. It is only safe when nothing is in flight;
  `main.py` calls it between the warmup and the real run after
  `mgr_shutdown()` has dropped the nvCOMP managers' arena-resident scratch.
- **Defensive `free`.** A pointer outside the arena, not block-aligned, or not
  currently allocated is ignored silently. `malloc` of 0 bytes allocates one
  block (or returns null with no arena).
- **Pinned host memory.** `pinned_host_malloc(size, node)` maps anonymous
  memory, binds it to NUMA node `node` with `mbind(MPOL_BIND)`, then pins it
  with `cudaHostRegister(..., cudaHostRegisterMapped)`, which faults every page
  in on that node; `pinned_host_free(ptr, size)` unregisters and unmaps it.
  Pinned pages never migrate, so the placement holds for the mapping's life.
  A node that cannot hold the range triggers the OOM killer rather than a
  spill, so callers check its capacity first (`utility/numa.py`). Mapped
  memory is what lets `IO/zero_copy_reader.cu` read host pages over UVA.
  Inside a container `mbind` needs `CAP_SYS_NICE`; a refused call throws with
  that hint.

## Files

| File | Purpose |
|------|---------|
| `CMakeLists.txt` | Builds two targets. `custom_allocator` is a `SHARED` library from `cuda_memory_allocator.cc` linked against Torch (C++ only: the file uses the host-side CUDA runtime API and needs no `nvcc`). `tqpmemory` is a `MODULE` library from `bindings.cpp`, built with `add_library` rather than `pybind11_add_module`, linked against `custom_allocator`, `pybind11::module`, `Python3::Module`, Torch and `torch_python` (for the `at::Tensor` caster); its `lib` prefix is stripped and the CPython extension suffix applied so the `.so` imports directly. Both targets are C++17, position-independent, with a `$ORIGIN` build and install rpath so `tqpmemory` finds `libcustom_allocator.so` next to itself. `PYBIND11_DETAILED_ERROR_MESSAGES` is on. The top-level `setup.py` stages `build/memory/*.so` into `TQPlib/tqpmemory/`. |
| `cuda_memory_allocator.cc` | The `GPUMemoryPool` class (one static instance, `g_mem_pool`) and its `extern "C"` surface. Pool lifecycle: `init_allocator` (with the `managed` flag), `remake_allocator`, `allocator_is_managed`, `destroy_allocator` (frees the arena and every pooled event), `free_all`. Outside the pool: the balloon (`reserve_device`) and `prefetch_managed` (to a host NUMA node). Torch entry points with the pluggable-allocator signature `(size, device, stream)`: `pa_cuda_malloc`, `pa_cuda_free`; null-stream variants `cuda_malloc`, `cuda_free`. Diagnostics: `dump_free`, `dump_allocs`, `dump_consumption` (free / pending / cache region / total MB on stderr), `reset_peak`. Measurement: `get_peak_bytes`, `get_window_peak_bytes`, `reset_window_peak`, `get_arena_stats`. The cache region: `region_init`, `region_move`, `region_boundary`, `region_blocks`, `arena_block_size`, `arena_base`. Lending: `arena_set_lend_hook`, `arena_clear_lend_hook`, `arena_lend_ctx`, `arena_lend_copy`, `arena_region_return`, `arena_lend_stats` (malloc's `lend_and_retry`, which lends through `lend_blocks`). The miss hook: `arena_set_miss_hook`, `arena_miss_stats`, `arena_stats_if_freed`. Host: `pinned_host_malloc`, `pinned_host_free`. Internals: first-fit `try_alloc`, `coalesce_into_free_list`, the three drains (`drain_same_stream`, `drain_completed`, `force_drain_all`) and the event pool. |
| `bindings.cpp` | `PYBIND11_MODULE(tqpmemory)`. Declares the C functions above and binds them as `init(pool_size_bytes, block_size_bytes, managed=False)`, `is_managed`, `remake`, `reserve_device`, `prefetch(ptr, size_bytes, host_node, stream=0)`, `destroy`, `malloc`, `free`, `stream_malloc`, `stream_free` (`pa_cuda_malloc` / `pa_cuda_free` with the stream as an integer handle; RMM's arena resource calls them), `free_all`, `dump_free`, `dump_allocs`, `dump_consumption`, `reset_peak`, `get_peak_bytes`, `get_window_peak_bytes`, `reset_window_peak`, `get_arena_stats(drain_completed_first=True)` returning a Python dict, `region_init`, `region_move`, `region_boundary`, `region_blocks`, `block_size` and `arena_base` (an integer device address); lending's `set_lend_hook(lend, lendable, ctx, stream)` (C function addresses as integers), `clear_lend_hook(ctx)`, `lend_ctx()` (the attached cache's ctx, 0 for none), `lend_api()`, `region_return(max_blocks)` and `lend_stats(reset=False)`; the miss hook's `set_miss_hook(fn)`, `miss_stats(reset=False)` and `stats_if_freed(ptrs)`. Adds `pinned_host_alloc_tensor(size_bytes, numa_node)`, which wraps a `pinned_host_malloc` region in a uint8 CPU tensor via `torch::from_blob` with a `pinned_host_free` deleter, so the memory is released when the last view of it is dropped; `IO/pages/pinned_mem.py` builds each 128 GiB segment of the pinned pool from it. |

## Rebuilding

Edits here take effect only after a rebuild and a copy into `TQPlib/`, inside
the `dev` container (see "Notes for agents editing the engine" in `AGENTS.md`):

```bash
docker compose run --rm -T dev bash -c '
  cmake --build build -j && cp build/memory/*.so TQPlib/tqpmemory/'
```

The entrypoint restores the image's `.so` files only when `TQPlib/tqpmemory/`
has none, so a stale copy in the working tree is never refreshed for you.
