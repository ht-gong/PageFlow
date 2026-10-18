#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

namespace py = pybind11;

extern "C" {
    void init_allocator(size_t pool_size_bytes, size_t block_size_bytes,
                        int managed);
    void remake_allocator();
    int allocator_is_managed();
    void reserve_device(size_t size);
    void prefetch_managed(void* ptr, size_t size, int host_node,
                          cudaStream_t stream);
    void cuda_free(void* ptr);
    void* cuda_malloc(ssize_t size);
    void* pa_cuda_malloc(ssize_t size, int device, cudaStream_t stream);
    void pa_cuda_free(void* ptr, ssize_t size, int device, cudaStream_t stream);
    void free_all();
    void dump_free();
    void dump_allocs();
    void dump_consumption();
    void destroy_allocator();
    void reset_peak();
    size_t get_peak_bytes();
    size_t get_window_peak_bytes();
    void reset_window_peak();
    void get_arena_stats(size_t* capacity_b, size_t* allocated_b,
                         size_t* free_b, size_t* largest_free_run_b,
                         size_t* pending_b, size_t* region_b,
                         size_t* top_free_run_b, size_t* lendable_b,
                         size_t* lent_b, int drain_completed_first);
    void arena_set_lend_hook(void* lend, void* lendable, void* ctx,
                             cudaStream_t stream);
    int arena_lend_copy(const int64_t* dst, const int64_t* src,
                        const int64_t* size, int64_t n);
    int64_t arena_region_return(int64_t max_blocks);
    void arena_lend_stats(uint64_t* misses, uint64_t* lent, uint64_t* shorts,
                          uint64_t* failed,
                          uint64_t* blocks_lent, uint64_t* blocks_returned,
                          uint64_t* lent_now, uint64_t* ns, int reset);
    void arena_clear_lend_hook(void* ctx);
    void* arena_lend_ctx();
    size_t region_init(size_t nbytes);
    size_t region_move(long long delta_blocks);
    size_t region_boundary();
    size_t region_blocks();
    size_t arena_block_size();
    void* arena_base();
    void* pinned_host_malloc(size_t size, int node);
    void pinned_host_free(void* ptr, size_t size);
    void arena_set_miss_hook(void* hook, void* ctx);
    void arena_miss_stats(uint64_t* calls, uint64_t* fragmented,
                          uint64_t* failed, uint64_t* ns, int reset);
    void arena_stats_if_freed(void* const* ptrs, size_t n, size_t* free_b,
                              size_t* largest_free_run_b);
} // extern "C"

// The Python miss hook (set_miss_hook), owned here and never destroyed at
// exit: it is released only when replaced, with the GIL held.
static py::object* g_miss_hook = nullptr;

// The allocator's miss hook: call the Python one with the bytes requested
// and whether it is the miss's first call, under the GIL (the allocating
// thread may or may not hold it: torch drops it around most ops).  An
// exception is reported as unraisable and counts as nothing freed, so the
// allocation goes on to its OOM.
static int64_t miss_trampoline(void* /*ctx*/, size_t need_bytes, int first) {
    py::gil_scoped_acquire gil;
    if (g_miss_hook == nullptr) return 0;
    try {
        return (*g_miss_hook)(need_bytes, first != 0).cast<int64_t>();
    } catch (py::error_already_set& e) {
        e.discard_as_unraisable("TQPlib.tqpmemory miss hook");
    } catch (const std::exception& e) {
        std::cerr << "miss hook: " << e.what() << std::endl;
    }
    return -1;
}

// A uint8 CPU tensor over pinned, device-mapped host memory whose pages all
// sit on NUMA node `numa_node`, or unbound for -1 (pinned_host_malloc).  Its
// deleter unpins and unmaps the memory once every reference (views and slices
// included) is gone.
static torch::Tensor pinned_host_alloc_tensor(int64_t size_bytes,
                                              int64_t numa_node) {
    const size_t size = static_cast<size_t>(size_bytes);
    void* ptr = pinned_host_malloc(size, static_cast<int>(numa_node));
    return torch::from_blob(
        ptr, {size_bytes}, [size](void* p) { pinned_host_free(p, size); },
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
}

PYBIND11_MODULE(tqpmemory, m) {
    m.def("init",
          [](size_t pool_size_bytes, size_t block_size_bytes, bool managed) {
              init_allocator(pool_size_bytes, block_size_bytes,
                             managed ? 1 : 0);
          },
          py::arg("pool_size_bytes"), py::arg("block_size_bytes"),
          py::arg("managed") = false,
          "Reserve the arena: pool_size_bytes in blocks of block_size_bytes, "
          "device memory, or with managed=True one cudaMallocManaged range "
          "(raises unless its base is 2 MiB aligned and block_size_bytes a "
          "multiple of 2 MiB).  A second call on a "
          "live arena changes nothing.");
    m.def("is_managed", []() { return allocator_is_managed() != 0; },
          "True when the arena is managed memory.");
    m.def("remake", &remake_allocator,
          "The arena reset: synchronize the device, then free the arena's "
          "range and reserve a fresh one of the same size and kind, with "
          "every block free and the peaks at zero.  Raises RuntimeError, "
          "naming the bytes, if any allocation is live, or if the arena "
          "has a cache region; a managed arena whose new range cannot be "
          "had keeps the old one.");
    m.def("reserve_device", &reserve_device, py::arg("size_bytes"),
          "Hold size_bytes of device memory outside the arena (one "
          "cudaMalloc) until process exit; remake leaves it alone.  Raises "
          "RuntimeError if it fails or one is already held.");
    m.def("prefetch",
          [](uintptr_t ptr, size_t size, int host_node, uintptr_t stream) {
              if (host_node < 0) {
                  throw std::invalid_argument(
                      "prefetch: host_node must be a NUMA node >= 0, not "
                      + std::to_string(host_node));
              }
              py::gil_scoped_release release;
              prefetch_managed(reinterpret_cast<void*>(ptr), size, host_node,
                               reinterpret_cast<cudaStream_t>(stream));
          },
          py::arg("ptr"), py::arg("size_bytes"), py::arg("host_node"),
          py::arg("stream") = 0,
          "Prefetch managed memory [ptr, ptr + size_bytes) to host NUMA node "
          "host_node, ordered on stream (a cudaStream_t handle, 0 the legacy "
          "default stream); asynchronous, so synchronize before relying on "
          "it.  Raises ValueError for a host_node below 0, RuntimeError if "
          "CUDA refuses (e.g. memory that is not managed).");
    m.def("destroy", &destroy_allocator, "Init GPU pooled allocator");
    m.def("malloc", &cuda_malloc, "Memory alloc");
    m.def("free", &cuda_free, "Free");
    m.def("free_all", &free_all, "Free all alloc'd mem");
    // The stream-ordered pair torch's pluggable allocator calls, with the
    // stream as an integer handle.  RMM's arena resource (utility/rmm_arena.py)
    // allocates through them; a failed allocation raises
    // RuntimeError("Pool out of memory").
    m.def("stream_malloc",
          [](size_t size, uintptr_t stream) {
              return reinterpret_cast<uintptr_t>(pa_cuda_malloc(
                  static_cast<ssize_t>(size), 0,
                  reinterpret_cast<cudaStream_t>(stream)));
          },
          py::arg("size"), py::arg("stream"),
          "Allocate size bytes from the arena, ordered on stream (a "
          "cudaStream_t handle); returns the device address.");
    m.def("stream_free",
          [](uintptr_t ptr, uintptr_t stream) {
              pa_cuda_free(reinterpret_cast<void*>(ptr), 0, 0,
                           reinterpret_cast<cudaStream_t>(stream));
          },
          py::arg("ptr"), py::arg("stream"),
          "Free an arena allocation, ordered on stream: its blocks are "
          "reused once the work queued on stream so far has finished.");
    m.def("dump_free", &dump_free, "Dump free list");
    m.def("dump_allocs", &dump_allocs, "Dump alloc'd list");
    m.def("dump_consumption", &dump_consumption, "Dump mem usage");
    m.def("reset_peak", &reset_peak, "Reset the peak counter (OOM diagnostic, get_peak_bytes) to current usage");
    m.def("get_peak_bytes", &get_peak_bytes,
          "The arena's peak allocation since the last reset_peak, in bytes (whole blocks)");
    m.def("get_window_peak_bytes", &get_window_peak_bytes,
          "Get the windowed peak GPU allocation in bytes (independent of reset_peak; for per-pipeline measurement)");
    m.def("reset_window_peak", &reset_window_peak,
          "Rebaseline the windowed peak watcher to current usage (call at the start of a pipeline window)");
    m.def("pinned_host_alloc_tensor", &pinned_host_alloc_tensor,
          "Pinned, device-mapped host memory as a uint8 tensor, every page on "
          "NUMA node numa_node (-1: not bound).",
          py::arg("size_bytes"), py::arg("numa_node"));
    m.def("set_miss_hook",
          [](py::object fn) {
              arena_set_miss_hook(nullptr, nullptr);
              delete g_miss_hook;
              g_miss_hook = nullptr;
              if (!fn.is_none()) {
                  g_miss_hook = new py::object(std::move(fn));
                  arena_set_miss_hook(
                      reinterpret_cast<void*>(&miss_trampoline), nullptr);
              }
          },
          py::arg("fn"),
          "Install fn(need_bytes, first) -> freed_bytes as the arena's miss "
          "hook (None: remove it).  An allocation that misses after draining "
          "every pending free synchronises the device, then until it fits "
          "calls fn with the allocator unlocked (first: the miss's first "
          "call), folds the frees fn caused in and tries again; fn returning "
          "0 or less (nothing left to free) ends in the OOM.  fn may free "
          "arena memory but must not count on allocating it (an allocation "
          "inside fn skips the hook).");
    m.def("miss_stats",
          [](bool reset) {
              uint64_t calls, fragmented, failed, ns;
              arena_miss_stats(&calls, &fragmented, &failed, &ns,
                               reset ? 1 : 0);
              py::dict d;
              d["calls"] = calls;
              d["fragmented"] = fragmented;
              d["failed"] = failed;
              d["ns"] = ns;
              return d;
          },
          py::arg("reset") = false,
          "The miss hook's counters since the last reset: {calls, "
          "fragmented (misses whose total free space held the request), "
          "failed (still out of memory after the hook), ns (host time in "
          "the sync, hook and retry)}; reset zeroes them after reading.");
    m.def("stats_if_freed",
          [](const std::vector<uintptr_t>& ptrs) {
              size_t free_b, largest;
              std::vector<void*> p(ptrs.size());
              for (size_t i = 0; i < ptrs.size(); ++i)
                  p[i] = reinterpret_cast<void*>(ptrs[i]);
              arena_stats_if_freed(p.data(), p.size(), &free_b, &largest);
              py::dict d;
              d["free_b"] = free_b;
              d["largest_free_run_b"] = largest;
              return d;
          },
          py::arg("ptrs"),
          "{free_b, largest_free_run_b} of the clean free list (completed "
          "pending frees folded in) if the live allocations starting at "
          "ptrs were free as well; other pointers are ignored.");
    m.def("get_arena_stats",
          [](bool drain_completed_first) {
              size_t cap, alloc, free_b, largest, pending, region, top,
                  lendable, lent;
              get_arena_stats(&cap, &alloc, &free_b, &largest, &pending,
                              &region, &top, &lendable, &lent,
                              drain_completed_first ? 1 : 0);
              py::dict d;
              d["capacity_b"] = cap;
              d["allocated_b"] = alloc;
              d["free_b"] = free_b;
              d["largest_free_run_b"] = largest;
              d["pending_b"] = pending;
              d["region_b"] = region;
              d["top_free_run_b"] = top;
              d["lendable_b"] = lendable;
              d["lent_b"] = lent;
              return d;
          },
          py::arg("drain_completed_first") = true,
          "Operator-space occupancy stats: {capacity_b, allocated_b, free_b, "
          "largest_free_run_b, pending_b, top_free_run_b} cover the arena "
          "below the cache region's boundary, and region_b is the region's "
          "size.  top_free_run_b is the clean free run that ends at the "
          "boundary (0 when the block just below it is allocated or "
          "pending), which the region's lowest blocks join when the "
          "boundary moves up.  lendable_b is what an allocation that misses "
          "could borrow from the region now (the attached cache's lendable "
          "blocks, 0 with none attached), lent_b the blocks lent and not yet "
          "returned (inside capacity_b).  Read under the allocator mutex "
          "with no sync and no allocation.  drain_completed_first folds "
          "already-completed pending frees into the clean free list first "
          "(non-blocking) so boundary-time reads see just-freed "
          "transients.");
    // Lending: the granule cache attached to the region.
    m.def("set_lend_hook",
          [](uintptr_t lend, uintptr_t lendable, uintptr_t ctx,
             uintptr_t stream) {
              arena_set_lend_hook(reinterpret_cast<void*>(lend),
                                  reinterpret_cast<void*>(lendable),
                                  reinterpret_cast<void*>(ctx),
                                  reinterpret_cast<cudaStream_t>(stream));
          },
          py::arg("lend"), py::arg("lendable"), py::arg("ctx"),
          py::arg("stream"),
          "Attach a cache to the region: lend (int64_t lend(void* ctx, "
          "int64_t blocks), empty the region's lowest blocks or none) and "
          "lendable (int64_t lendable(void* ctx)), C function addresses, "
          "with their ctx and the stream (a cudaStream_t handle) the lend's "
          "copies run on.  An allocation that misses after the force-drain "
          "then borrows the region's lowest blocks.  lend=lendable=0 "
          "detaches it.");
    m.def("clear_lend_hook",
          [](uintptr_t ctx) {
              arena_clear_lend_hook(reinterpret_cast<void*>(ctx));
          },
          py::arg("ctx"),
          "Detach the cache whose ctx is ctx, if it is the one attached "
          "(another cache attached since stays attached).");
    m.def("lend_ctx",
          []() { return reinterpret_cast<uintptr_t>(arena_lend_ctx()); },
          "The attached cache's ctx (0: none attached).");
    m.def("lend_api",
          []() {
              return py::make_tuple(
                  reinterpret_cast<uintptr_t>(&arena_lend_copy),
                  reinterpret_cast<uintptr_t>(&arena_region_return));
          },
          "(arena_lend_copy, arena_region_return): the addresses the "
          "granule cache's core calls, for its lend's copies and to give "
          "lent blocks back.");
    m.def("region_return",
          [](int64_t max_blocks) { return arena_region_return(max_blocks); },
          py::arg("max_blocks"),
          "Move the boundary down over the free blocks just below it, up to "
          "min(max_blocks, the blocks still lent); returns the blocks "
          "moved.  The attached cache calls it itself (its core grows by "
          "them); a direct call leaves a cache's slots behind the region.");
    m.def("lend_stats",
          [](bool reset) {
              uint64_t misses, lent, shorts, failed, blocks, returned, now,
                  ns;
              arena_lend_stats(&misses, &lent, &shorts, &failed, &blocks,
                               &returned, &now, &ns, reset ? 1 : 0);
              py::dict d;
              d["misses"] = misses;
              d["lent"] = lent;
              d["short"] = shorts;
              d["failed"] = failed;
              d["blocks_lent"] = blocks;
              d["blocks_returned"] = returned;
              d["lent_now"] = now;
              d["ns"] = ns;
              return d;
          },
          py::arg("reset") = false,
          "Lending's counters since the last reset: {misses (that tried to "
          "lend), lent (misses the cache lent for), short (it could not: "
          "the region too small, or a pinned or same-tick granule in the "
          "way), failed (still out of memory after a lend), blocks_lent, "
          "blocks_returned, lent_now (blocks lent and not yet back, not "
          "reset), ns (host time of the lend path)}; reset zeroes them after "
          "reading.");
    // The cache region: the arena's top blocks, behind a boundary.
    m.def("region_init", &region_init, py::arg("nbytes"),
          "Take the arena's top nbytes // block blocks for the cache region; "
          "the arena must have no region yet and the blocks must be free "
          "(else RuntimeError).  Returns the blocks taken.");
    m.def("region_move", &region_move, py::arg("delta_blocks"),
          "Move the region's boundary by delta_blocks.  Positive moves it "
          "up (the region shrinks; its lowest blocks join the operator "
          "space's free list).  Negative moves it down (the region grows "
          "by the blocks just below the boundary, all or none, only if all "
          "are free).  Returns the blocks moved, 0 when a downward move is "
          "refused; a move past either end raises RuntimeError.  A move "
          "sets the region's size anew: no block counts as lent after it "
          "(a move by 0 only does that).");
    m.def("region_boundary", &region_boundary,
          "The cache region's lowest block index (the arena's block count "
          "when there is no region).");
    m.def("region_blocks", &region_blocks,
          "The cache region's size in blocks.");
    m.def("block_size", &arena_block_size, "The arena block size in bytes.");
    m.def("arena_base",
          []() { return reinterpret_cast<uintptr_t>(arena_base()); },
          "The arena's base device address (0 before init); block i starts "
          "at arena_base() + i * block_size().");
}
