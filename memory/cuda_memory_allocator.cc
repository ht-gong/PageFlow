#include <cuda_runtime_api.h>      /* host-side CUDA API */

#include <map>
#include <mutex>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <cstdarg>
#include <cstdio>
#include <exception>
#include <cassert>
#include <cerrno>
#include <cstdint>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

#include <linux/mempolicy.h>        /* MPOL_BIND */
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

// -------- CUDA aliases --------
  using gpuStream_t = cudaStream_t;
  using gpuEvent_t  = cudaEvent_t;
  using gpuError_t  = cudaError_t;
  #define GPU_SUCCESS               cudaSuccess
  #define gpuMalloc                 cudaMalloc
  #define gpuFree                   cudaFree
  #define gpuEventCreateWithFlags   cudaEventCreateWithFlags
  #define gpuEventDisableTiming     cudaEventDisableTiming
  #define gpuEventRecord            cudaEventRecord
  #define gpuEventQuery             cudaEventQuery
  #define gpuEventDestroy           cudaEventDestroy
  #define gpuStreamWaitEvent        cudaStreamWaitEvent
  #define gpuGetErrorString         cudaGetErrorString

inline size_t byte_to_mb(size_t byte) {
    return byte >> 20;
}

// ─────────────────────────────────────────────────────────────────────
// Stream-ordered GPU memory pool with deferred pending frees.
//
// Design:
//   free_blocks_   – the "clean" free list.  Every block here is safe
//                    to hand out on ANY stream without event waits.
//   pending_frees_ – blocks recently freed on a specific stream.  They
//                    stay here until we know re-use is safe:
//                      • same-stream malloc  → drain instantly (GPU ordering)
//                      • event completed     → drain opportunistically
//                      • OOM                 → force-drain with event waits
//   guards_        – force-drained blocks whose free event had not fired.
//                    They sit in the clean list, but a later allocation
//                    overlapping one on another stream waits for its event
//                    (the free stays ordered on the freeing stream for
//                    every taker, not only the one that force-drained).
//
// This keeps cross-stream cudaStreamWaitEvent calls off the hot path.
//
// The cache region:
//   The blocks at and above the boundary (boundary_, a block index) belong
//   to the granule cache.  They are never in free_blocks_, allocs_ or
//   pending_frees_, so no allocation lands in them and no counter or peak
//   includes them.  Everything below the boundary is the operator space;
//   the stats describe it.  region_init takes the arena's top blocks for
//   the region; region_move moves the boundary (up: blocks to the operator
//   space; down: free blocks from just below it into the region).
//
// Contiguous allocations come from the bottom (try_alloc takes the lowest
// free run that fits); the cache region's granules are paged from the top
// (the granule cache's core hands out the region's highest free slots
// first), so the two grow toward each other and the blocks next to the
// boundary are the last either side takes.
//
// Lending (the granule cache, IO/cache/granule_core.cc): with a cache
// attached to the region (set_lend_hook), an allocation that still misses
// after the force-drain borrows the region's lowest blocks.  It counts the
// miss, releases the mutex, synchronises the device (no kernel or copy
// still reads or writes a slot, every copy into a home has ended) and asks
// the cache to empty the fewest blocks k at the bottom of the region that,
// with the free run ending at the boundary, hold the request (k alone when
// no free run touches it).  The cache evicts its surplus in victim order
// (a dirty granule copied home first) and moves the granules left in those
// blocks to free slots higher up, the copies on the lend stream through
// lend_copy, all or nothing (a pinned or same-tick granule stops it).
// Once that stream has synchronised the boundary moves up by k, the blocks
// join the top free run, and the allocation is tried again; a lend that
// cannot give k blocks is the OOM.  The lent blocks go back when they are
// free again (region_return, called by the cache at each pipeline's start,
// and only there), up to the blocks still lent.
//
// The miss hook (the free-list baseline, IO/pages/device_regions.py): an
// allocation that still misses after the force-drain synchronises the
// device and stalls in a loop: with the mutex released the hook evicts one
// spillable GPU region (the least recently used, copied to its host home),
// whose free lands in pending_frees_, and the allocation is tried again
// with every pending free folded in; it goes on until the allocation fits
// or the hook has nothing left to evict, then the OOM.  Neither hook nests:
// an allocation inside one skips both.  Only one is set in a run (the
// free-list baseline has no cache region).
// ─────────────────────────────────────────────────────────────────────

// The miss hook: called with the bytes requested and whether this is the
// miss's first call; evicts one region and returns its bytes (0: nothing
// left to evict, negative: it failed).  It must not throw.
using MissHook = int64_t (*)(void* ctx, size_t need_bytes, int first);

// The lend hook: empty the cache region's lowest ``blocks`` blocks for the
// operator space (the device is synchronised) and return ``blocks``, or 0
// with nothing changed when they cannot all be emptied.  The lendable hook:
// the blocks a lend could give now.  Neither may throw.
using LendHook = int64_t (*)(void* ctx, int64_t blocks);
using LendableHook = int64_t (*)(void* ctx);

class GPUMemoryPool {
public:
    GPUMemoryPool() : pool_size_bytes_(0), block_size_bytes_(0),
                      num_blocks_(0), boundary_(0), mem_pool_(nullptr),
                      allocated_blocks_(0), peak_allocated_blocks_(0),
                      window_peak_blocks_(0), managed_(false) {}

    ~GPUMemoryPool() {
        destroy_allocator();
    }

    // With ``managed`` the arena is one cudaMallocManaged range
    // (cudaMemAttachGlobal) instead of device memory: its pages live wherever
    // the driver puts them, and a block is backed only once it is touched.
    // A managed range must start on a 2 MiB boundary, the driver's migration
    // granule, and its blocks be a multiple of it, so that no two blocks
    // share a granule.
    void init(size_t pool_size_bytes, size_t block_size_bytes,
              bool managed = false) {

        block_size_bytes = (block_size_bytes >> 8) << 8;
        assert(block_size_bytes % 512 == 0);

        std::lock_guard<std::mutex> lock(mtx_);
        if (mem_pool_) return;

        if (managed && block_size_bytes % kGranuleBytes) {
            throw std::runtime_error(
                "a managed arena's block size must be a multiple of 2 MiB, not "
                + std::to_string(block_size_bytes) + " B");
        }

        pool_size_bytes_ = pool_size_bytes;
        block_size_bytes_ = block_size_bytes;
        if (pool_size_bytes_ < block_size_bytes_ || block_size_bytes_ == 0) {
            std::cerr << "Invalid pool/block size." << std::endl;
            return;
        }

        num_blocks_ = pool_size_bytes_ / block_size_bytes_;
        boundary_ = num_blocks_;
        managed_ = managed;

        void* range = reserve_range();
        if (!range) return;
        mem_pool_ = range;
        reset_lists(/*zero_peaks=*/true);
    }

    // The arena reset: once the device is idle and no allocation is live,
    // free the range and reserve a fresh one of the same size and kind, so
    // no page of the old range stays resident anywhere.  Raises if an
    // allocation is live (naming its bytes), the arena has a cache region
    // (its granules would not survive), or the new range cannot be had.
    // A managed range's replacement is reserved before the old range is
    // freed (address space only), so a failed reset keeps the old arena; a
    // device range is freed first, since the card holds only one.
    void remake() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!mem_pool_) throw std::runtime_error("remake: no arena");
        gpuError_t status = cudaDeviceSynchronize();
        if (status != GPU_SUCCESS) {
            throw std::runtime_error(std::string("remake: device sync failed: ")
                                     + gpuGetErrorString(status));
        }
        drain_completed();
        prune_guards();
        if (allocated_blocks_ != 0) {
            throw std::runtime_error(
                "remake: " + std::to_string(allocated_blocks_ * block_size_bytes_)
                + " B of the arena are still allocated (" +
                std::to_string(allocs_.size()) + " allocations)");
        }
        if (boundary_ != num_blocks_) {
            throw std::runtime_error(
                "remake: the arena has a cache region of " +
                std::to_string(num_blocks_ - boundary_) + " blocks");
        }
        void* range = nullptr;
        if (managed_) {
            range = reserve_range();
            if (range) gpuFree(mem_pool_);
        } else {
            gpuFree(mem_pool_);
            mem_pool_ = nullptr;
            range = reserve_range();
        }
        if (!range) {
            throw std::runtime_error("remake: could not reserve a new arena of "
                                     + std::to_string(pool_size_bytes_) + " B");
        }
        mem_pool_ = range;
        reset_lists(/*zero_peaks=*/true);
    }

    bool is_managed() {
        std::lock_guard<std::mutex> lock(mtx_);
        return managed_;
    }

    void destroy_allocator() {
        std::lock_guard<std::mutex> lock(mtx_);
        // Return all pending events
        for (auto& pb : pending_frees_) {
            event_pool_.push_back(pb.event);
        }
        pending_frees_.clear();
        for (auto& g : guards_) event_pool_.push_back(g.event);
        guards_.clear();
        free_blocks_.clear();
        allocs_.clear();
        // Destroy all pooled events
        for (auto e : event_pool_) gpuEventDestroy(e);
        event_pool_.clear();
        if (mem_pool_) {
            gpuFree(mem_pool_);
            mem_pool_ = nullptr;
        }
        pool_size_bytes_ = 0;
        block_size_bytes_ = 0;
        num_blocks_ = 0;
        boundary_ = 0;
        lent_ = 0;
        lend_hook_ = nullptr;
        lendable_hook_ = nullptr;
        lend_ctx_ = nullptr;
        lend_stream_ = nullptr;
    }

    void* malloc(size_t size, gpuStream_t stream) {
        if (size == 0 && !mem_pool_) return nullptr;
        if (size == 0) size = 1;

        std::unique_lock<std::mutex> lock(mtx_);
        // Torch's pluggable allocator does not check for a null pointer, so
        // an arena lost to a failed remake must not hand one out.
        if (!mem_pool_) {
            throw std::runtime_error("no arena to allocate " +
                                     std::to_string(size) + " B from");
        }

        // 1. Drain same-stream pending frees into the clean free list.
        //    Same-stream ops are GPU-ordered — no event wait needed.
        size_t drained_same = drain_same_stream(stream);

        // 2. Drain cross-stream pending frees whose events already completed.
        //    gpuEventQuery is non-blocking; completed events are common
        //    under the tick barrier, which syncs the host between ticks.
        size_t drained_cross = drain_completed();

        // 3. Try to allocate from the clean (event-free) free list.
        prune_guards();
        void* ptr = try_alloc(size, stream);
        if (ptr) return ptr;

        // 4. OOM on the clean list — force-drain ALL pending frees.
        //    A block whose free event has not fired stays a guard; the
        //    retry's taker waits on it only if the block was freed on
        //    another stream (wait_guards), so no stream waits otherwise.
        force_drain_all(stream);

        // 5. Retry.
        ptr = try_alloc(size, stream);
        if (ptr) return ptr;

        // 6. Lending: the cache region's lowest blocks, then one more try.
        if (lend_hook_ && !in_hook_ && boundary_ < num_blocks_) {
            ptr = lend_and_retry(size, stream, lock);
            if (ptr) return ptr;
        }

        // 7. The miss hook evicts, then one more try.
        if (miss_hook_ && !in_hook_) {
            ptr = evict_and_retry(size, stream, lock);
            if (ptr) return ptr;
        }

        // Diagnostic dump before OOM
        {
            size_t requested_blocks = (size + block_size_bytes_ - 1) / block_size_bytes_;
            size_t total_free_blocks = 0;
            size_t largest_free_run = 0;
            size_t num_free_regions = free_blocks_.size();
            for (auto& [start, len] : free_blocks_) {
                total_free_blocks += len;
                if (len > largest_free_run) largest_free_run = len;
            }
            size_t pending_blocks = 0;
            for (auto& pb : pending_frees_) pending_blocks += pb.num_blocks;
            std::cerr << "=== OOM DIAGNOSTIC ===" << std::endl;
            std::cerr << "  Requested: " << byte_to_mb(size) << " MB ("
                      << requested_blocks << " blocks)" << std::endl;
            std::cerr << "  Block size: " << byte_to_mb(block_size_bytes_) << " MB" << std::endl;
            std::cerr << "  Pool total: " << byte_to_mb(pool_size_bytes_) << " MB ("
                      << num_blocks_ << " blocks)" << std::endl;
            std::cerr << "  Cache region: "
                      << byte_to_mb((num_blocks_ - boundary_) * block_size_bytes_)
                      << " MB (" << (num_blocks_ - boundary_)
                      << " blocks, boundary at block " << boundary_ << ")"
                      << std::endl;
            std::cerr << "  Allocated: " << byte_to_mb(allocated_blocks_ * block_size_bytes_)
                      << " MB (" << allocated_blocks_ << " blocks)" << std::endl;
            std::cerr << "  Free: " << byte_to_mb(total_free_blocks * block_size_bytes_)
                      << " MB (" << total_free_blocks << " blocks in "
                      << num_free_regions << " regions)" << std::endl;
            std::cerr << "  Largest contiguous free: "
                      << byte_to_mb(largest_free_run * block_size_bytes_)
                      << " MB (" << largest_free_run << " blocks)" << std::endl;
            std::cerr << "  Pending frees: "
                      << byte_to_mb(pending_blocks * block_size_bytes_)
                      << " MB (" << pending_blocks << " blocks)" << std::endl;
            if (lend_hook_) {
                size_t lendable = static_cast<size_t>(
                    std::max<int64_t>(0, lendable_hook_(lend_ctx_)));
                std::cerr << "  Lent: " << byte_to_mb(lent_ * block_size_bytes_)
                          << " MB (" << lent_ << " blocks); lendable now "
                          << byte_to_mb(lendable * block_size_bytes_)
                          << " MB (" << lendable << " blocks)" << std::endl;
            }
            std::cerr << "  Peak: " << byte_to_mb(peak_allocated_blocks_ * block_size_bytes_)
                      << " MB" << std::endl;
            std::cerr << "======================" << std::endl;
        }

        throw std::runtime_error("Pool out of memory");

        return nullptr;
    }

    void free(void* ptr, gpuStream_t stream) {
        if (!ptr || !mem_pool_) return;

        std::lock_guard<std::mutex> lock(mtx_);

        if (ptr < mem_pool_ || ptr >= static_cast<char*>(mem_pool_) + pool_size_bytes_) {
            return;
        }

        size_t offset = static_cast<char*>(ptr) - static_cast<char*>(mem_pool_);
        if (offset % block_size_bytes_ != 0) {
            return;
        }
        size_t start_block = offset / block_size_bytes_;

        auto it_alloc = allocs_.find(start_block);
        if (it_alloc == allocs_.end()) {
            return;
        }

        size_t num_blocks = it_alloc->second;
        allocated_blocks_ -= num_blocks;
        allocs_.erase(it_alloc);

        // Record an event so we know when this stream is done with the
        // memory, then park the block in the pending list.
        gpuEvent_t evt = get_event();
        gpuEventRecord(evt, stream);

        pending_frees_.push_back({start_block, num_blocks, evt, stream});
    }

    // Non-stream overloads (use null/default stream).
    void* malloc(size_t size) { return malloc(size, gpuStream_t(0)); }
    void  free(void* ptr)     { free(ptr, gpuStream_t(0)); }

    void free_all() {
        if (!mem_pool_) return;

        std::lock_guard<std::mutex> lock(mtx_);
        reset_lists(/*zero_peaks=*/false);
    }

    void dump_consumption() {
        size_t free_bytes = 0;
        for (auto& [start, len] : free_blocks_) {
            free_bytes += len * block_size_bytes_;
        }
        size_t pending_bytes = 0;
        for (auto& pb : pending_frees_) {
            pending_bytes += pb.num_blocks * block_size_bytes_;
        }
        std::cerr << "Mem free: " << byte_to_mb(free_bytes)
                  << " MB  pending: " << byte_to_mb(pending_bytes)
                  << " MB  cache region: "
                  << byte_to_mb((num_blocks_ - boundary_) * block_size_bytes_)
                  << " MB  of " << byte_to_mb(pool_size_bytes_)
                  << " MB" << std::endl;
    }

    // ── Arena stats ──────────────────────────────────────────────────
    // The OOM-diagnostic quantities promoted to a getter: everything is a
    // counter already maintained under mtx_ — no sync, no allocation.  They
    // describe the operator space: capacity_b is the bytes below the
    // boundary, and region_b the cache region's bytes above it.  The
    // free/largest-run walk covers the CLEAN list only; blocks parked in
    // pending_frees_ are reported separately (pending_b).  top_free_run_b
    // is the clean run that ends at the boundary (0 when the block just
    // below it is allocated or pending): the run the region's lowest
    // blocks join when the boundary moves up.  When drain_completed_first
    // is set, pending frees whose events already fired are folded into the
    // clean list before reading (non-blocking gpuEventQuery, same as the
    // malloc path) — callers at synced pipe boundaries use this so a
    // just-freed transient is not misread as unavailable.  lendable_b is
    // what a miss could borrow from the cache region now (the attached
    // cache's lendable blocks; 0 without one), lent_b the blocks lent and
    // not yet returned (part of capacity_b).
    void get_arena_stats(size_t* capacity_b, size_t* allocated_b,
                         size_t* free_b, size_t* largest_free_run_b,
                         size_t* pending_b, size_t* region_b,
                         size_t* top_free_run_b, size_t* lendable_b,
                         size_t* lent_b, int drain_completed_first) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (drain_completed_first) drain_completed();
        *capacity_b  = boundary_ * block_size_bytes_;
        *region_b    = (num_blocks_ - boundary_) * block_size_bytes_;
        *allocated_b = allocated_blocks_ * block_size_bytes_;
        *lent_b      = lent_ * block_size_bytes_;
        size_t lendable = 0;
        if (lend_hook_ && boundary_ < num_blocks_)
            lendable = static_cast<size_t>(
                std::max<int64_t>(0, lendable_hook_(lend_ctx_)));
        *lendable_b = std::min(lendable, num_blocks_ - boundary_)
                      * block_size_bytes_;
        size_t total_free = 0, largest_run = 0;
        for (auto& [start, len] : free_blocks_) {
            total_free += len;
            if (len > largest_run) largest_run = len;
        }
        *free_b = total_free * block_size_bytes_;
        *largest_free_run_b = largest_run * block_size_bytes_;
        size_t top_run = 0;
        auto top = free_blocks_.lower_bound(boundary_);
        if (top != free_blocks_.begin()) {
            --top;
            if (top->first + top->second == boundary_) top_run = top->second;
        }
        *top_free_run_b = top_run * block_size_bytes_;
        size_t pending = 0;
        for (auto& pb : pending_frees_) pending += pb.num_blocks;
        *pending_b = pending * block_size_bytes_;
    }

    // Rebaseline the all-time high-water mark.  Its readers are the OOM
    // diagnostic dump (above) and get_peak_bytes; callers reset it per query
    // so both report the peak of that query, not an all-run value.
    void reset_peak() {
        std::lock_guard<std::mutex> lock(mtx_);
        peak_allocated_blocks_ = allocated_blocks_;
    }

    // ── The cache region ─────────────────────────────────────────────
    // Take the arena's top nbytes / block blocks (whole blocks, rounded
    // down) for the cache region.  Only an arena without a region can
    // take one, and the blocks must all be free: completed pending frees
    // are drained first, and a block still pending or guarded is not free.
    // Returns the number of blocks taken.
    size_t region_init(size_t nbytes) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!mem_pool_)
            throw std::runtime_error("region_init: the arena is not initialised");
        if (boundary_ != num_blocks_)
            throw std::runtime_error("region_init: the arena already has a "
                                     "cache region; move its boundary instead");
        size_t k = nbytes / block_size_bytes_;
        if (k == 0) return 0;
        if (k > num_blocks_)
            throw std::runtime_error("region_init: " + std::to_string(k) +
                                     " blocks requested, the arena has " +
                                     std::to_string(num_blocks_));
        if (!take_below_boundary(k))
            throw std::runtime_error("region_init: the arena's top " +
                                     std::to_string(k) +
                                     " blocks are not all free");
        lent_ = 0;
        return k;
    }

    // Move the boundary by delta_blocks.  Positive moves it up: the
    // region's lowest delta_blocks blocks join the operator space's free
    // list, coalesced with its top free run.  Negative moves it down: the
    // -delta_blocks blocks just below the boundary join the region, all or
    // none, only when every one is free (completed pending frees drained
    // first; pending or guarded blocks are not free).  Returns the number
    // of blocks moved: delta's magnitude, or 0 when a downward move is
    // refused.  Moving up by more than the region holds, or down by more
    // than the operator space holds, raises.  A move sets the region's size
    // anew: no block counts as lent after it (lent_ is 0), a move by 0
    // included, which only does that.
    size_t region_move(long long delta_blocks) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!mem_pool_)
            throw std::runtime_error("region_move: the arena is not initialised");
        if (delta_blocks >= 0) {
            size_t k = static_cast<size_t>(delta_blocks);
            if (k > num_blocks_ - boundary_)
                throw std::runtime_error(
                    "region_move: moving the boundary up by " +
                    std::to_string(k) + " blocks, the region holds " +
                    std::to_string(num_blocks_ - boundary_));
            lent_ = 0;
            if (k == 0) return 0;
            size_t start = boundary_;
            boundary_ += k;
            coalesce_into_free_list(start, k);
            return k;
        }
        // The range is checked before delta_blocks is negated: negating the
        // most negative value is undefined.
        if (delta_blocks < -static_cast<long long>(boundary_))
            throw std::runtime_error(
                "region_move: moving the boundary down by " +
                std::to_string(0ULL - static_cast<unsigned long long>(delta_blocks)) +
                " blocks, the operator space holds " +
                std::to_string(boundary_));
        size_t k = static_cast<size_t>(-delta_blocks);
        if (!take_below_boundary(k)) return 0;
        lent_ = 0;
        return k;
    }

    // ── Lending ──────────────────────────────────────────────────────
    // The cache attached to the region: its lend and lendable hooks, their
    // ctx, and the stream the lend's copies run on (the home writer's).
    // Null hooks detach it.
    void set_lend_hook(LendHook lend, LendableHook lendable, void* ctx,
                       gpuStream_t stream) {
        std::lock_guard<std::mutex> lock(mtx_);
        if ((lend == nullptr) != (lendable == nullptr))
            throw std::runtime_error("set_lend_hook: both hooks or neither");
        lend_hook_ = lend;
        lendable_hook_ = lendable;
        lend_ctx_ = lend ? ctx : nullptr;
        lend_stream_ = lend ? stream : nullptr;
    }

    // The attached cache's ctx (null: none).
    void* lend_ctx() {
        std::lock_guard<std::mutex> lock(mtx_);
        return lend_hook_ ? lend_ctx_ : nullptr;
    }

    // Detach the cache whose ctx is ``ctx``, if it is the one attached (a
    // cache's finalizer: another cache attached since stays attached).
    void clear_lend_hook(void* ctx) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (lend_ctx_ != ctx) return;
        lend_hook_ = nullptr;
        lendable_hook_ = nullptr;
        lend_ctx_ = nullptr;
        lend_stream_ = nullptr;
    }

    // The lend's copies, issued by the cache from inside its lend hook (the
    // mutex is released then): size[i] bytes from src[i] to dst[i] on the
    // lend stream, which the allocator synchronises before the blocks are
    // handed out.  Returns 0, or 1 when CUDA refused a copy.
    int lend_copy(const int64_t* dst, const int64_t* src, const int64_t* size,
                  int64_t n) {
        for (int64_t i = 0; i < n; ++i) {
            gpuError_t status = cudaMemcpyAsync(
                reinterpret_cast<void*>(dst[i]),
                reinterpret_cast<const void*>(src[i]),
                static_cast<size_t>(size[i]), cudaMemcpyDefault, lend_stream_);
            if (status != GPU_SUCCESS) {
                std::cerr << "lend: a copy failed: "
                          << gpuGetErrorString(status) << std::endl;
                return 1;
            }
        }
        return 0;
    }

    // Move the boundary down over the free blocks just below it, up to
    // min(max_blocks, the blocks still lent): lent blocks the operator space
    // no longer uses go back to the region.  Completed pending frees are
    // drained first; a block still pending, or guarded, is not free, and
    // the move stops above it.  Returns the blocks moved (the cache adds
    // them as free slots).
    size_t region_return(size_t max_blocks) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!mem_pool_ || lent_ == 0 || max_blocks == 0) return 0;
        drain_completed();
        prune_guards();
        auto it = free_blocks_.lower_bound(boundary_);
        if (it == free_blocks_.begin()) return 0;
        --it;
        if (it->first + it->second != boundary_) return 0;
        size_t r = std::min({lent_, max_blocks, it->second});
        size_t lo = boundary_ - r;
        for (auto& g : guards_) {
            size_t g_end = g.start_block + g.num_blocks;
            if (g.start_block < boundary_ && lo < g_end)
                lo = std::min(boundary_, g_end);
        }
        r = boundary_ - lo;
        if (r == 0) return 0;
        size_t run_start = it->first;
        free_blocks_.erase(it);
        if (run_start < lo) free_blocks_[run_start] = lo - run_start;
        boundary_ = lo;
        lent_ -= r;
        lend_returned_ += r;
        return r;
    }

    // Since the last reset: the misses that tried to lend, those the cache
    // lent for, those it could not (the region too small, or a pinned or
    // same-tick granule in the way), those still failing after a lend, the
    // blocks lent and returned, the blocks lent now, and the host ns of the
    // lend path (device sync, the cache's lend, its copies, the retry).
    void lend_stats(uint64_t* misses, uint64_t* lent, uint64_t* shorts,
                    uint64_t* failed, uint64_t* blocks_lent,
                    uint64_t* blocks_returned, uint64_t* lent_now,
                    uint64_t* ns, int reset) {
        std::lock_guard<std::mutex> lock(mtx_);
        *misses = lend_misses_;
        *lent = lend_ok_;
        *shorts = lend_short_;
        *failed = lend_failed_;
        *blocks_lent = lend_blocks_;
        *blocks_returned = lend_returned_;
        *lent_now = lent_;
        *ns = lend_ns_;
        if (reset)
            lend_misses_ = lend_ok_ = lend_short_ = lend_failed_ =
                lend_blocks_ = lend_returned_ = lend_ns_ = 0;
    }

    size_t region_boundary() {
        std::lock_guard<std::mutex> lock(mtx_);
        return boundary_;
    }

    size_t region_blocks() {
        std::lock_guard<std::mutex> lock(mtx_);
        return num_blocks_ - boundary_;
    }

    size_t block_size() {
        std::lock_guard<std::mutex> lock(mtx_);
        return block_size_bytes_;
    }

    void* arena_base() {
        std::lock_guard<std::mutex> lock(mtx_);
        return mem_pool_;
    }

    // The high-water mark since the last reset_peak, in bytes (whole
    // blocks): the residency report of a query (main.py, under
    // TQP_EXPERIMENT_TELEMETRY).
    size_t get_peak_bytes() {
        std::lock_guard<std::mutex> lock(mtx_);
        return peak_allocated_blocks_ * block_size_bytes_;
    }

    // ── Windowed peak ────────────────────────────────────────────────
    // A second, independent high-water mark used to measure peak arena
    // occupancy over a caller-defined window (e.g. one pipeline run) WITHOUT
    // disturbing peak_allocated_blocks_, which serves the OOM diagnostic dump.
    // reset_window_peak() rebaselines to the current live level (so the
    // window's peak includes whatever is already allocated at window start),
    // and get_window_peak_bytes() reads the max live occupancy seen since.
    // Both count the operator space only.
    size_t get_window_peak_bytes() {
        std::lock_guard<std::mutex> lock(mtx_);
        return window_peak_blocks_ * block_size_bytes_;
    }

    void reset_window_peak() {
        std::lock_guard<std::mutex> lock(mtx_);
        window_peak_blocks_ = allocated_blocks_;
    }

    void dump_free() {
        std::cerr << "{ ";
        for (auto& [start, len] : free_blocks_) {
            std::cerr << start << ": " << len << ", ";
        }
        std::cerr << "}\n";
    }

    void dump_allocs() {
        std::cerr << "{ ";
        for (auto& [start, num] : allocs_) {
            std::cerr << start << ": " << num << ", ";
        }
        std::cerr << "}\n";
    }

    // ── The miss hook ────────────────────────────────────────────────
    void set_miss_hook(MissHook hook, void* ctx) {
        std::lock_guard<std::mutex> lock(mtx_);
        miss_hook_ = hook;
        miss_ctx_ = ctx;
    }

    // Since the last reset: the misses that called the hook, those of them
    // whose total clean free space already held the request (a fragmented
    // arena, not a full one), those still failing after it, and the host
    // ns spent in the device sync, the hook and the retry.
    void miss_stats(uint64_t* calls, uint64_t* fragmented, uint64_t* failed,
                    uint64_t* ns, int reset) {
        std::lock_guard<std::mutex> lock(mtx_);
        *calls = miss_calls_;
        *fragmented = miss_fragmented_;
        *failed = miss_failed_;
        *ns = miss_ns_;
        if (reset) miss_calls_ = miss_fragmented_ = miss_failed_ = miss_ns_ = 0;
    }

    // Free bytes and the largest free run (clean list, after folding the
    // completed pending frees) if the allocations starting at `ptrs` were
    // free too: what an eviction of them would leave.  Pointers that are
    // not live allocations are ignored.
    void stats_if_freed(void* const* ptrs, size_t n, size_t* free_b,
                        size_t* largest_free_run_b) {
        std::lock_guard<std::mutex> lock(mtx_);
        drain_completed();
        std::map<size_t, size_t> runs(free_blocks_);
        for (size_t i = 0; i < n; ++i) {
            char* p = static_cast<char*>(ptrs[i]);
            char* base = static_cast<char*>(mem_pool_);
            if (!mem_pool_ || p < base || p >= base + pool_size_bytes_) continue;
            size_t off = static_cast<size_t>(p - base);
            if (off % block_size_bytes_) continue;
            auto it = allocs_.find(off / block_size_bytes_);
            if (it == allocs_.end()) continue;
            runs[it->first] = it->second;
        }
        size_t total = 0, largest = 0, run_start = 0, run_len = 0;
        bool open = false;
        for (auto& [start, len] : runs) {
            total += len;
            if (open && run_start + run_len == start) {
                run_len += len;
            } else {
                run_start = start;
                run_len = len;
                open = true;
            }
            largest = std::max(largest, run_len);
        }
        *free_b = total * block_size_bytes_;
        *largest_free_run_b = largest * block_size_bytes_;
    }

private:
    // A block waiting for its stream to finish before it can be reused
    // on a different stream.
    struct PendingBlock {
        size_t       start_block;
        size_t       num_blocks;
        gpuEvent_t   event;
        gpuStream_t  stream;      // stream the event was recorded on
    };

    size_t pool_size_bytes_;
    size_t block_size_bytes_;
    size_t num_blocks_;
    // Lowest block of the cache region; num_blocks_ when there is none.
    size_t boundary_;
    void*  mem_pool_;
    // Clean free list — no events attached; safe for any stream.
    std::map<size_t, size_t> free_blocks_;   // start_block -> num_blocks
    std::map<size_t, size_t> allocs_;        // start_block -> num_blocks

    // Deferred frees awaiting event completion before entering free_blocks_.
    std::vector<PendingBlock> pending_frees_;
    // Force-drained blocks whose free event had not fired (``guards_`` in
    // the design note above); dropped once their event fires.
    std::vector<PendingBlock> guards_;

    std::mutex mtx_;
    std::vector<gpuEvent_t> event_pool_;

    // ── Peak memory tracking ────────────────────────────────────────
    size_t      allocated_blocks_;       // currently allocated blocks
    size_t      peak_allocated_blocks_;  // high-water mark (OOM diagnostic, get_peak_bytes)
    size_t      window_peak_blocks_;     // windowed high-water (per-pipeline)
    bool        managed_;                // the range is cudaMallocManaged

    // ── The miss hook (null: none) and its counters ──────────────────
    MissHook    miss_hook_ = nullptr;
    void*       miss_ctx_ = nullptr;
    bool        in_hook_ = false;        // a hook call is running
    uint64_t    miss_calls_ = 0, miss_fragmented_ = 0, miss_failed_ = 0,
                miss_ns_ = 0;

    // ── Lending (null hooks: no cache attached) and its counters ────────
    LendHook     lend_hook_ = nullptr;
    LendableHook lendable_hook_ = nullptr;
    void*        lend_ctx_ = nullptr;
    gpuStream_t  lend_stream_ = nullptr;
    size_t       lent_ = 0;              // blocks lent, not yet returned
    uint64_t     lend_misses_ = 0, lend_ok_ = 0, lend_short_ = 0,
                 lend_failed_ = 0, lend_blocks_ = 0,
                 lend_returned_ = 0, lend_ns_ = 0;

    // The clean free run that ends at the boundary, in blocks (0: none).
    size_t top_free_run() const {
        auto top = free_blocks_.lower_bound(boundary_);
        if (top == free_blocks_.begin()) return 0;
        --top;
        return top->first + top->second == boundary_ ? top->second : 0;
    }

    // Lend k blocks (``lock`` held on entry and on return; the mutex is
    // released around the device work): synchronise the device, call the
    // cache's lend hook, synchronise the lend stream (the lend's copies, or
    // those of one a refused copy cut short), then move the boundary up by
    // what the cache gave, the blocks joining the top free run.  Returns
    // the blocks lent (k or 0).
    size_t lend_blocks(size_t k, std::unique_lock<std::mutex>& lock) {
        LendHook hook = lend_hook_;
        void* ctx = lend_ctx_;
        gpuStream_t lstream = lend_stream_;
        in_hook_ = true;
        lock.unlock();
        int64_t got = 0;
        gpuError_t status = cudaDeviceSynchronize();
        if (status != GPU_SUCCESS) {
            std::cerr << "lend: device sync failed: "
                      << gpuGetErrorString(status) << std::endl;
        } else {
            got = hook(ctx, static_cast<int64_t>(k));
            status = cudaStreamSynchronize(lstream);
            if (status != GPU_SUCCESS)
                std::cerr << "lend: the copies' stream failed: "
                          << gpuGetErrorString(status) << std::endl;
        }
        lock.lock();
        in_hook_ = false;
        if (got <= 0) return 0;
        size_t g = static_cast<size_t>(got);
        if (g > num_blocks_ - boundary_)
            throw std::runtime_error(
                "lend: the cache gave " + std::to_string(g) +
                " blocks, the region holds " +
                std::to_string(num_blocks_ - boundary_));
        size_t start = boundary_;
        boundary_ += g;
        coalesce_into_free_list(start, g);
        lent_ += g;
        lend_blocks_ += g;
        return g;
    }

    // Step 6 of malloc, with `lock` held on entry and on return: lend the
    // fewest of the region's lowest blocks that, with the free run ending at
    // the boundary, hold the request (the design note above), then try
    // again.  Returns null when the cache cannot lend them or the
    // allocation still does not fit.
    void* lend_and_retry(size_t size, gpuStream_t stream,
                         std::unique_lock<std::mutex>& lock) {
        auto t0 = std::chrono::steady_clock::now();
        auto account = [&]() {
            lend_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - t0).count());
        };
        size_t need = (size + block_size_bytes_ - 1) / block_size_bytes_;
        size_t top = top_free_run();
        size_t k = need > top ? need - top : 0;
        ++lend_misses_;
        if (k == 0 || k > num_blocks_ - boundary_) {
            ++lend_short_;
            account();
            return nullptr;
        }
        size_t got = lend_blocks(k, lock);
        if (got) ++lend_ok_;
        else ++lend_short_;
        drain_completed();
        force_drain_all(stream);
        prune_guards();
        void* ptr = try_alloc(size, stream);
        if (!ptr && got) ++lend_failed_;
        account();
        return ptr;
    }

    // Step 6 of malloc, with `lock` held on entry and on return: count the
    // miss, release the mutex and synchronise the device (every queued read
    // and write of what the hook may evict has ended), then until the
    // allocation fits: call the hook (one region evicted), retake the
    // mutex, fold every pending free in (the eviction's coalesce with their
    // neighbours) and try; stop when the hook evicts nothing.  Returns null
    // when it still does not fit.
    void* evict_and_retry(size_t size, gpuStream_t stream,
                          std::unique_lock<std::mutex>& lock) {
        auto t0 = std::chrono::steady_clock::now();
        size_t need = (size + block_size_bytes_ - 1) / block_size_bytes_;
        size_t total_free = 0;
        for (auto& [start, len] : free_blocks_) total_free += len;
        ++miss_calls_;
        if (total_free >= need) ++miss_fragmented_;
        MissHook hook = miss_hook_;
        void* ctx = miss_ctx_;
        in_hook_ = true;
        lock.unlock();
        gpuError_t status = cudaDeviceSynchronize();
        if (status != GPU_SUCCESS) {
            std::cerr << "miss hook: device sync failed: "
                      << gpuGetErrorString(status) << std::endl;
        }
        void* ptr = nullptr;
        for (int first = 1; status == GPU_SUCCESS; first = 0) {
            int64_t freed = hook(ctx, size, first);
            lock.lock();
            drain_completed();
            force_drain_all(stream);
            prune_guards();
            ptr = try_alloc(size, stream);
            if (ptr || freed <= 0) break;
            lock.unlock();
        }
        if (!lock.owns_lock()) lock.lock();
        in_hook_ = false;
        if (!ptr) ++miss_failed_;
        miss_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t0).count());
        return ptr;
    }

    static constexpr size_t kGranuleBytes = size_t(2) << 20;

    // Update both high-water marks (called with mtx_ held).
    void update_peak() {
        if (allocated_blocks_ > peak_allocated_blocks_)
            peak_allocated_blocks_ = allocated_blocks_;
        if (allocated_blocks_ > window_peak_blocks_)
            window_peak_blocks_ = allocated_blocks_;
    }

    // ── helpers (all called with mtx_ held) ──────────────────────────

    // Reserve a range of the arena's size, device or managed as init
    // chose, and return it.  On failure, report it on stderr and return
    // null.
    void* reserve_range() {
        void* range = nullptr;
        gpuError_t status = managed_
            ? cudaMallocManaged(&range, pool_size_bytes_, cudaMemAttachGlobal)
            : gpuMalloc(&range, pool_size_bytes_);
        if (status != GPU_SUCCESS || !range) {
            std::cerr << "Failed to allocate "
                      << (managed_ ? "managed" : "device") << " memory pool: "
                      << gpuGetErrorString(status) << std::endl;
            return nullptr;
        }
        if (managed_ && reinterpret_cast<uintptr_t>(range) % kGranuleBytes) {
            gpuFree(range);
            throw std::runtime_error("managed arena base is not 2 MiB aligned");
        }
        return range;
    }

    // One free run over the operator space (the blocks below the boundary;
    // the cache region stays as it is); no allocation, no pending free or
    // guard; with ``zero_peaks`` the peaks at zero too.
    void reset_lists(bool zero_peaks) {
        free_blocks_.clear();
        if (boundary_ > 0) free_blocks_[0] = boundary_;
        allocs_.clear();
        for (auto& pb : pending_frees_) event_pool_.push_back(pb.event);
        pending_frees_.clear();
        for (auto& g : guards_) event_pool_.push_back(g.event);
        guards_.clear();
        allocated_blocks_ = 0;
        if (zero_peaks) {
            peak_allocated_blocks_ = 0;
            window_peak_blocks_ = 0;
        }
    }

    gpuEvent_t get_event() {
        if (!event_pool_.empty()) {
            gpuEvent_t e = event_pool_.back();
            event_pool_.pop_back();
            return e;
        }
        gpuEvent_t e;
        gpuEventCreateWithFlags(&e, gpuEventDisableTiming);
        return e;
    }

    // Move same-stream pending frees into the clean free list, but ONLY
    // when the recorded event has already fired.  This is critical: a
    // block freed on stream S is safe for immediate reuse on S (GPU FIFO
    // ordering), but once it enters free_blocks_ it becomes available to
    // ANY stream via try_alloc().  If the event hasn't fired yet, a
    // cross-stream malloc could pick up the block while S is still
    // reading it — a WAR hazard.  By requiring the event to have fired
    // we guarantee the block is truly idle before it enters the shared
    // clean list.
    //
    // Returns the number of blocks drained.
    size_t drain_same_stream(gpuStream_t stream) {
        size_t count = 0;
        auto new_end = std::remove_if(
            pending_frees_.begin(), pending_frees_.end(),
            [&](PendingBlock& pb) {
                if (pb.stream == stream &&
                    gpuEventQuery(pb.event) == GPU_SUCCESS) {
                    event_pool_.push_back(pb.event);
                    coalesce_into_free_list(pb.start_block, pb.num_blocks);
                    ++count;
                    return true;
                }
                return false;
            });
        pending_frees_.erase(new_end, pending_frees_.end());
        return count;
    }

    // Move cross-stream pending frees whose events already fired.
    // Returns the number of blocks drained.
    size_t drain_completed() {
        size_t count = 0;
        auto new_end = std::remove_if(
            pending_frees_.begin(), pending_frees_.end(),
            [&](PendingBlock& pb) {
                if (gpuEventQuery(pb.event) == GPU_SUCCESS) {
                    event_pool_.push_back(pb.event);
                    coalesce_into_free_list(pb.start_block, pb.num_blocks);
                    ++count;
                    return true;
                }
                return false;
            });
        pending_frees_.erase(new_end, pending_frees_.end());
        return count;
    }

    // Force ALL pending frees into the free list.  No wait is inserted
    // here: a block whose event has not fired keeps it as a guard, and a
    // later taker of any of its blocks on another stream than the freeing
    // one waits for it (try_alloc, wait_guards), so the free stays ordered
    // on the freeing stream for every taker.
    void force_drain_all(gpuStream_t requesting_stream) {
        (void)requesting_stream;
        for (auto& pb : pending_frees_) {
            if (gpuEventQuery(pb.event) == GPU_SUCCESS) {
                event_pool_.push_back(pb.event);
            } else {
                guards_.push_back(pb);
            }
            coalesce_into_free_list(pb.start_block, pb.num_blocks);
        }
        pending_frees_.clear();
    }

    // Drop the guards whose event has fired.
    void prune_guards() {
        if (guards_.empty()) return;
        auto new_end = std::remove_if(
            guards_.begin(), guards_.end(), [&](PendingBlock& g) {
                if (gpuEventQuery(g.event) == GPU_SUCCESS) {
                    event_pool_.push_back(g.event);
                    return true;
                }
                return false;
            });
        guards_.erase(new_end, guards_.end());
    }

    // ``stream`` waits for every guard overlapping blocks
    // [start, start + n) that was freed on another stream.
    void wait_guards(size_t start, size_t n, gpuStream_t stream) {
        for (auto& g : guards_) {
            if (g.start_block < start + n && start < g.start_block + g.num_blocks
                    && g.stream != stream) {
                gpuStreamWaitEvent(stream, g.event, 0);
            }
        }
    }

    // Move the k blocks just below the boundary into the region if every
    // one is free: completed pending frees are drained first, then the
    // blocks must lie in one clean run ending at the boundary and overlap
    // no guard.  All or none; returns whether they moved.
    bool take_below_boundary(size_t k) {
        drain_completed();
        prune_guards();
        size_t lo = boundary_ - k;
        auto it = free_blocks_.upper_bound(lo);
        if (it == free_blocks_.begin()) return false;
        --it;
        size_t run_start = it->first, run_len = it->second;
        if (run_start + run_len != boundary_) return false;
        for (auto& g : guards_) {
            if (g.start_block < boundary_ && lo < g.start_block + g.num_blocks)
                return false;
        }
        free_blocks_.erase(it);
        if (run_start < lo) free_blocks_[run_start] = lo - run_start;
        boundary_ = lo;
        return true;
    }

    // Insert a block into the clean free list with coalescing.
    void coalesce_into_free_list(size_t start_block, size_t num_blocks) {
        size_t new_start = start_block;
        size_t new_len   = num_blocks;

        // Coalesce with next neighbour
        auto next_it = free_blocks_.upper_bound(start_block);
        if (next_it != free_blocks_.end() &&
            start_block + num_blocks == next_it->first) {
            new_len += next_it->second;
            free_blocks_.erase(next_it);
        }

        // Coalesce with previous neighbour
        auto prev_it = free_blocks_.lower_bound(start_block);
        if (prev_it != free_blocks_.begin()) {
            --prev_it;
            if (prev_it->first + prev_it->second == new_start) {
                new_start = prev_it->first;
                new_len  += prev_it->second;
                free_blocks_.erase(prev_it);
            }
        }

        free_blocks_[new_start] = new_len;
    }

    // First-fit allocation from the clean free list.
    void* try_alloc(size_t size, gpuStream_t stream) {
        size_t alloc_blocks = (size + block_size_bytes_ - 1) / block_size_bytes_;

        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->second >= alloc_blocks) {
                size_t start_block = it->first;
                size_t len = it->second;

                if (len == alloc_blocks) {
                    free_blocks_.erase(it);
                } else {
                    free_blocks_.erase(it);
                    free_blocks_[start_block + alloc_blocks] = len - alloc_blocks;
                }

                allocs_[start_block] = alloc_blocks;
                allocated_blocks_ += alloc_blocks;
                update_peak();
                wait_guards(start_block, alloc_blocks, stream);
                return static_cast<char*>(mem_pool_) + start_block * block_size_bytes_;
            }
        }
        return nullptr;
    }
};

static GPUMemoryPool g_mem_pool;

// The balloon: device memory held outside the arena, so that managed pages
// can use only what is left of the card.  It is never released.
static void* g_balloon = nullptr;
static size_t g_balloon_bytes = 0;
static std::mutex g_balloon_mtx;

} // namespace

extern "C" {

void init_allocator(size_t pool_size_bytes, size_t block_size_bytes,
                    int managed) {
    g_mem_pool.init(pool_size_bytes, block_size_bytes, managed != 0);
}

void remake_allocator() {
    g_mem_pool.remake();
}

int allocator_is_managed() {
    return g_mem_pool.is_managed() ? 1 : 0;
}

// One plain device allocation of `size` bytes, held until process exit.
// The arena reset does not touch it.
void reserve_device(size_t size) {
    std::lock_guard<std::mutex> lock(g_balloon_mtx);
    if (g_balloon) {
        throw std::runtime_error("reserve_device: " +
                                 std::to_string(g_balloon_bytes) +
                                 " B are already reserved");
    }
    void* ptr = nullptr;
    gpuError_t status = gpuMalloc(&ptr, size);
    if (status != GPU_SUCCESS) {
        throw std::runtime_error("reserve_device(" + std::to_string(size) +
                                 " B) failed: " + gpuGetErrorString(status));
    }
    g_balloon = ptr;
    g_balloon_bytes = size;
}

// cudaMemPrefetchAsync_v2 of managed memory [ptr, ptr + size), ordered on
// `stream`, to host NUMA node `host_node` (cudaMemLocationTypeHostNuma).
void prefetch_managed(void* ptr, size_t size, int host_node,
                      cudaStream_t stream) {
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeHostNuma;
    loc.id = host_node;
    gpuError_t status = cudaMemPrefetchAsync_v2(ptr, size, loc, 0, stream);
    if (status != GPU_SUCCESS) {
        throw std::runtime_error(std::string("prefetch failed: ") +
                                 gpuGetErrorString(status));
    }
}

    void pa_cuda_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
        g_mem_pool.free(ptr, stream);
    }

    void* pa_cuda_malloc(ssize_t size, int device, cudaStream_t stream) {
        return g_mem_pool.malloc(size, stream);
    }

void cuda_free(void* ptr) {
    g_mem_pool.free(ptr);
}

void* cuda_malloc(ssize_t size) {
    return g_mem_pool.malloc(size);
}

void free_all() {
    g_mem_pool.free_all();
}

void dump_consumption() {
    g_mem_pool.dump_consumption();
}

void dump_free() {
    g_mem_pool.dump_free();
}

void dump_allocs() {
    g_mem_pool.dump_allocs();
}

void reset_peak() {
    g_mem_pool.reset_peak();
}

size_t get_peak_bytes() {
    return g_mem_pool.get_peak_bytes();
}

size_t get_window_peak_bytes() {
    return g_mem_pool.get_window_peak_bytes();
}

void get_arena_stats(size_t* capacity_b, size_t* allocated_b, size_t* free_b,
                     size_t* largest_free_run_b, size_t* pending_b,
                     size_t* region_b, size_t* top_free_run_b,
                     size_t* lendable_b, size_t* lent_b,
                     int drain_completed_first) {
    g_mem_pool.get_arena_stats(capacity_b, allocated_b, free_b,
                               largest_free_run_b, pending_b, region_b,
                               top_free_run_b, lendable_b, lent_b,
                               drain_completed_first);
}

// Lending (the design note in GPUMemoryPool).  The granule cache's core
// registers its hooks (tqpcache_lend, tqpcache_lendable) with
// arena_set_lend_hook and calls arena_lend_copy and arena_region_return
// through the addresses tqpmemory.lend_api() gives it.
void arena_set_lend_hook(void* lend, void* lendable, void* ctx,
                         cudaStream_t stream) {
    g_mem_pool.set_lend_hook(reinterpret_cast<LendHook>(lend),
                             reinterpret_cast<LendableHook>(lendable), ctx,
                             stream);
}

int arena_lend_copy(const int64_t* dst, const int64_t* src,
                    const int64_t* size, int64_t n) {
    return g_mem_pool.lend_copy(dst, src, size, n);
}

int64_t arena_region_return(int64_t max_blocks) {
    if (max_blocks <= 0) return 0;
    try {
        return static_cast<int64_t>(
            g_mem_pool.region_return(static_cast<size_t>(max_blocks)));
    } catch (const std::exception& e) {
        std::cerr << "region_return: " << e.what() << std::endl;
        return 0;
    }
}

void arena_lend_stats(uint64_t* misses, uint64_t* lent, uint64_t* shorts,
                      uint64_t* failed, uint64_t* blocks_lent,
                      uint64_t* blocks_returned, uint64_t* lent_now,
                      uint64_t* ns, int reset) {
    g_mem_pool.lend_stats(misses, lent, shorts, failed, blocks_lent,
                          blocks_returned, lent_now, ns, reset);
}

void arena_clear_lend_hook(void* ctx) {
    g_mem_pool.clear_lend_hook(ctx);
}

void* arena_lend_ctx() {
    return g_mem_pool.lend_ctx();
}

size_t region_init(size_t nbytes) {
    return g_mem_pool.region_init(nbytes);
}

size_t region_move(long long delta_blocks) {
    return g_mem_pool.region_move(delta_blocks);
}

size_t region_boundary() {
    return g_mem_pool.region_boundary();
}

size_t region_blocks() {
    return g_mem_pool.region_blocks();
}

size_t arena_block_size() {
    return g_mem_pool.block_size();
}

void* arena_base() {
    return g_mem_pool.arena_base();
}

void reset_window_peak() {
    g_mem_pool.reset_window_peak();
}

void arena_set_miss_hook(void* hook, void* ctx) {
    g_mem_pool.set_miss_hook(reinterpret_cast<MissHook>(hook), ctx);
}

void arena_miss_stats(uint64_t* calls, uint64_t* fragmented, uint64_t* failed,
                      uint64_t* ns, int reset) {
    g_mem_pool.miss_stats(calls, fragmented, failed, ns, reset);
}

void arena_stats_if_freed(void* const* ptrs, size_t n, size_t* free_b,
                          size_t* largest_free_run_b) {
    g_mem_pool.stats_if_freed(ptrs, n, free_b, largest_free_run_b);
}

void destroy_allocator() {
    g_mem_pool.destroy_allocator();
}

// Pinned, device-mapped host memory with every page on NUMA node `node`, or
// with `node` -1 not bound (TQP_NUMA_BIND off, utility/numa.py): the kernel
// then places each page on the node of the thread that faults it in.
//
// The range is bound to `node` with MPOL_BIND before any page exists;
// cudaHostRegister then faults every page in, on `node` whichever thread does
// it, and pins it.  Pinned pages never migrate, so the placement holds for the
// life of the mapping and does not depend on which CPUs the process's threads
// run on.  MPOL_BIND makes the kernel reclaim the node's page cache rather
// than spill to another node; if the node cannot hold the range at all the
// OOM killer runs, so callers check the node's capacity first.  Inside a
// container, mbind needs CAP_SYS_NICE (Docker's default seccomp profile).
void* pinned_host_malloc(size_t size, int node) {
    if (node < -1 || node >= 64) {
        throw std::runtime_error("NUMA node " + std::to_string(node) +
                                 " out of range [-1, 64)");
    }
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::runtime_error(std::string("mmap failed: ") +
                                 std::strerror(errno));
    }
    unsigned long nodemask = node < 0 ? 0 : 1UL << node;
    if (node >= 0 && syscall(SYS_mbind, ptr, size, MPOL_BIND, &nodemask,
                             sizeof(nodemask) * 8, 0) != 0) {
        int err = errno;
        munmap(ptr, size);
        throw std::runtime_error(
            "mbind to NUMA node " + std::to_string(node) + " failed: " +
            std::strerror(err) +
            (err == EPERM ? " (a container needs CAP_SYS_NICE)" : ""));
    }
    cudaError_t status = cudaHostRegister(ptr, size, cudaHostRegisterMapped);
    if (status != cudaSuccess) {
        munmap(ptr, size);
        throw std::runtime_error(std::string("cudaHostRegister failed: ") +
                                 cudaGetErrorString(status));
    }
    return ptr;
}

void pinned_host_free(void* ptr, size_t size) {
    if (!ptr) return;
    cudaHostUnregister(ptr);
    munmap(ptr, size);
}

} // extern "C"
