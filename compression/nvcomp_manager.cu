// nvCOMP 5.2 high-level manager component: the engine's only compress path.
// Codecs: ANS, CASCADED, BITCOMP, LZ4, SNAPPY, GDEFLATE, DEFLATE, each with
// fixed options; the per-codec chunk size comes from the Python side
// (compression/format_policy.py, profiling/results/codec_map.json).
//
// One manager `compress()` call emits one self-describing contiguous
// NVCOMP_NATIVE object; nvCOMP owns internal chunking and scratch.  This file
// exposes the enqueue-only batched surface the Python layer drives:
//
//   mgr_compress_begin   -> batch configure_compression, host-known max sizes
//   mgr_compress_launch  -> ONE batched compress + the sizes into pinned memory
//                           by UVA stores (no copy engine),
//                           derives DecompressionConfigs (documented no-sync)
//   mgr_compress_finish  -> host-only status readback after caller-observed
//                           completion
//   mgr_decompress       -> ONE batched decompress from cached config handles
//                           (slices of at most 65,535 buffers: kMaxBatch)
//   mgr_ingest_configs   -> cold header-parse fallback (documented stream sync)
//
// Contract: no cudaStreamSynchronize / blocking size readback anywhere except
// mgr_ingest_configs (counted).  get_compressed_output_size() and the
// comp-buffer configure_decompression overload never appear on hot paths.
// Managers are cached per (codec, dtype, chunk, stream) and live for the
// process so every issued config's status storage stays valid.

#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "nvcomp/nvcompManager.hpp"
#include "nvcomp/ans.hpp"
#include "nvcomp/cascaded.hpp"
#include "nvcomp/bitcomp.hpp"
#include "nvcomp/lz4.hpp"
#include "nvcomp/snappy.hpp"
#include "nvcomp/gdeflate.hpp"
#include "nvcomp/deflate.hpp"
#include "nvcomp/version.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

#define MGR_CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    TORCH_CHECK(_err == cudaSuccess, "CUDA error in nvcomp_manager: ", \
                cudaGetErrorString(_err)); \
} while (0)

// Codec ids mirror compression_algorithms._ALGO_NAMES indices (0 = RAW).
constexpr int kCodecANS = 1;
constexpr int kCodecCascaded = 2;
constexpr int kCodecBitcomp = 3;
constexpr int kCodecLZ4 = 4;
constexpr int kCodecSnappy = 5;
constexpr int kCodecGdeflate = 6;
constexpr int kCodecDeflate = 7;

// A batched manager call's buffers ride one grid dimension of its kernels,
// which CUDA caps at 65,535: a longer batch fails to launch ("invalid
// configuration argument"; SF1000 Q1's aggregation pass 2 at chunks of 128
// MiB or less reads more records than that in one tick).  The batched calls
// go in slices of at most this many buffers, each slice configured on its own
// (configure_compression, configure_decompression) as well as launched, so a
// slice's configs come from one configure of just its buffers.
constexpr size_t kMaxBatch = 65535;

// Byte-oriented codecs ignore the element type: one manager per chunk size
// serves every element width.  CASCADED and BITCOMP are typed.
bool codec_is_untyped(int codec) {
    return codec == kCodecANS || codec == kCodecLZ4 || codec == kCodecSnappy
        || codec == kCodecGdeflate || codec == kCodecDeflate;
}

nvcompType_t elem_size_to_type(int64_t elem_size) {
    switch (elem_size) {
        case 1: return NVCOMP_TYPE_CHAR;
        case 2: return NVCOMP_TYPE_SHORT;
        case 4: return NVCOMP_TYPE_INT;
        case 8: return NVCOMP_TYPE_LONGLONG;
        default: return NVCOMP_TYPE_CHAR;
    }
}

// ---------------------------------------------------------------------------
// Counters (exported via mgr_counters)
// ---------------------------------------------------------------------------

std::atomic<int64_t> g_scratch_live{0};
std::atomic<int64_t> g_scratch_hwm{0};
std::atomic<int64_t> g_scratch_allocs{0};
std::atomic<int64_t> g_header_parse_syncs{0};

// ---------------------------------------------------------------------------
// Manager cache: (codec, nvcompType_t, chunk_bytes, stream) -> manager
// ---------------------------------------------------------------------------

struct MgrKey {
    int codec;
    int type;
    size_t chunk_b;
    cudaStream_t stream;
    bool operator==(const MgrKey& o) const {
        return codec == o.codec && type == o.type && chunk_b == o.chunk_b
            && stream == o.stream;
    }
};

struct MgrKeyHash {
    size_t operator()(const MgrKey& k) const {
        size_t h = std::hash<int>()(k.codec);
        h = h * 31 + std::hash<int>()(k.type);
        h = h * 31 + std::hash<size_t>()(k.chunk_b);
        h = h * 31 + std::hash<void*>()(reinterpret_cast<void*>(k.stream));
        return h;
    }
};

std::mutex g_mgr_mu;
std::unordered_map<MgrKey, std::unique_ptr<nvcomp::nvcompManagerBase>, MgrKeyHash>
    g_managers;

// Manager scratch allocator choice.  DEFAULT: torch's (pluggable) allocator
// so the engine's arena accounting sees the scratch and it can't OOM the
// slim outside-arena headroom.  
// CONTRACT: scratch lives in the arena, so managers MUST
// NOT survive an arena reset — main.py calls mgr_shutdown() before its
// post-warmup cmp.free_all() (retained scratch across that reset aliases
// the real run's tensors.
// TQP_MGR_SCRATCH=cuda keeps the internal-pool behavior as a lever.
bool use_torch_scratch() {
    static const bool v = [] {
        const char* e = std::getenv("TQP_MGR_SCRATCH");
        return !(e && std::string(e) == "cuda");
    }();
    return v;
}

std::unique_ptr<nvcomp::nvcompManagerBase> make_manager(
    int codec, nvcompType_t type, size_t chunk_b, cudaStream_t stream) {
    using namespace nvcomp;
    std::unique_ptr<nvcompManagerBase> mgr;
    switch (codec) {
        case kCodecANS: {
            // Byte-mode ANS, default opts.
            mgr = std::make_unique<ANSManager>(
                chunk_b,
                nvcompBatchedANSCompressDefaultOpts,
                nvcompBatchedANSDecompressDefaultOpts,
                stream, NoComputeNoVerify, BitstreamKind::NVCOMP_NATIVE);
            break;
        }
        case kCodecCascaded: {
            // rle=2, delta=1, bp=1 over 4 KiB internal chunks.
            nvcompBatchedCascadedCompressOpts_t copts = {};
            copts.internal_chunk_bytes = 4096;
            copts.type = type;
            copts.num_RLEs = 2;
            copts.num_deltas = 1;
            copts.use_bp = 1;
            mgr = std::make_unique<CascadedManager>(
                chunk_b, copts, nvcompBatchedCascadedDecompressDefaultOpts,
                stream, NoComputeNoVerify, BitstreamKind::NVCOMP_NATIVE);
            break;
        }
        case kCodecBitcomp: {
            // algorithm=0, typed by element size.
            nvcompBatchedBitcompCompressOpts_t copts = {};
            copts.algorithm = 0;
            copts.data_type = type;
            mgr = std::make_unique<BitcompManager>(
                chunk_b, copts, nvcompBatchedBitcompDecompressDefaultOpts,
                stream, NoComputeNoVerify, BitstreamKind::NVCOMP_NATIVE);
            break;
        }
        // The four codecs below run with nvCOMP's own default options; the
        // sweep varies only the chunk.
        case kCodecLZ4: {
            // NVCOMP_TYPE_CHAR, no bitshuffle.
            mgr = std::make_unique<LZ4Manager>(
                chunk_b,
                nvcompBatchedLZ4CompressDefaultOpts,
                nvcompBatchedLZ4DecompressDefaultOpts,
                stream, NoComputeNoVerify, BitstreamKind::NVCOMP_NATIVE);
            break;
        }
        case kCodecSnappy: {
            mgr = std::make_unique<SnappyManager>(
                chunk_b,
                nvcompBatchedSnappyCompressDefaultOpts,
                nvcompBatchedSnappyDecompressDefaultOpts,
                stream, NoComputeNoVerify, BitstreamKind::NVCOMP_NATIVE);
            break;
        }
        case kCodecGdeflate: {
            // algorithm 1.
            mgr = std::make_unique<GdeflateManager>(
                chunk_b,
                nvcompBatchedGdeflateCompressDefaultOpts,
                nvcompBatchedGdeflateDecompressDefaultOpts,
                stream, NoComputeNoVerify, BitstreamKind::NVCOMP_NATIVE);
            break;
        }
        case kCodecDeflate: {
            // algorithm 1.
            mgr = std::make_unique<DeflateManager>(
                chunk_b,
                nvcompBatchedDeflateCompressDefaultOpts,
                nvcompBatchedDeflateDecompressDefaultOpts,
                stream, NoComputeNoVerify, BitstreamKind::NVCOMP_NATIVE);
            break;
        }
        default:
            TORCH_CHECK(false, "nvcomp_manager: unknown codec id ", codec);
    }
    if (use_torch_scratch()) {
        // Route manager scratch through torch's caching allocator so the
        // engine's arena accounting sees it (the default cudaMallocAsync pool
        // is invisible to torch and competes for physical memory).
        mgr->set_scratch_allocators(
            [](size_t bytes) -> void* {
                void* p = c10::cuda::CUDACachingAllocator::raw_alloc(bytes);
                TORCH_CHECK(reinterpret_cast<uintptr_t>(p) % 256 == 0,
                            "nvcomp_manager: scratch block not 256B-aligned");
                int64_t live = g_scratch_live.fetch_add((int64_t)bytes) + (int64_t)bytes;
                g_scratch_allocs.fetch_add(1);
                int64_t hwm = g_scratch_hwm.load();
                while (live > hwm && !g_scratch_hwm.compare_exchange_weak(hwm, live)) {}
                return p;
            },
            [](void* ptr, size_t bytes) {
                // Managers cached in a static map can be destroyed during
                // process teardown, after the (pluggable) allocator has shut
                // down — never let that terminate the process.  Python-side
                // atexit calls mgr_shutdown() first, so this guard only
                // fires on abnormal exits.
                try {
                    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
                } catch (...) {
                }
                g_scratch_live.fetch_sub((int64_t)bytes);
            });
    }
    return mgr;
}

nvcomp::nvcompManagerBase& get_manager(
    int codec, int64_t elem_size, int64_t chunk_b, cudaStream_t stream) {
    // Byte-oriented codecs (ANS, LZ4, SNAPPY, GDEFLATE, DEFLATE) are untyped:
    // normalise so all elem sizes share one manager per chunk.
    nvcompType_t type = codec_is_untyped(codec) ? NVCOMP_TYPE_CHAR
                                                : elem_size_to_type(elem_size);
    MgrKey key{codec, (int)type, (size_t)chunk_b, stream};
    std::lock_guard<std::mutex> lk(g_mgr_mu);
    auto it = g_managers.find(key);
    if (it == g_managers.end()) {
        it = g_managers.emplace(key, make_manager(codec, type, (size_t)chunk_b,
                                                  stream)).first;
    }
    return *it->second;
}

// ---------------------------------------------------------------------------
// Ticket registry: in-flight batched compress (configs must outlive the op)
// ---------------------------------------------------------------------------

struct Ticket {
    int codec;
    int64_t elem_size;
    int64_t chunk_b;
    cudaStream_t stream;
    std::vector<nvcomp::CompressionConfig> configs;
    bool launched = false;
};

std::mutex g_ticket_mu;
std::unordered_map<int64_t, Ticket> g_tickets;
std::atomic<int64_t> g_next_ticket{1};

// ---------------------------------------------------------------------------
// Config registry (handles held by Python Compression objects).  Entries are
// process-lifetime: one per emitted record, never erased before mgr_shutdown.
// Growth is bounded by total bytes compressed in the run (one entry per
// TQP_MGR_RECORD_MB), so no per-page refcounting is done.
//
// Entries normally hold the RECORD'S CompressionConfig; every decompress call
// derives FRESH private DecompressionConfigs from them (the documented
// no-sync overload).  Sharing one mutable DecompressionConfig between
// overlapping operations is forbidden by nvCOMP's contract — and the engine
// DOES decompress the same records concurrently on two streams (e.g. a mask
// column read by a compute-stream DirectIn group and an h2d-stream
// ZeroCopyMaskIn in the same tick).  Ingest-fallback entries (header parse;
// producer config unavailable) store a DecompressionConfig directly and
// accept the residual sharing hazard on that rare path.
// ---------------------------------------------------------------------------

struct CfgEntry {
    std::optional<nvcomp::CompressionConfig> comp;
    std::optional<nvcomp::DecompressionConfig> decomp;
};

std::mutex g_cfg_mu;
std::unordered_map<int64_t, CfgEntry> g_configs;
std::atomic<int64_t> g_next_cfg{1};

torch::Tensor register_comp_configs(
        const std::vector<nvcomp::CompressionConfig>& cfgs) {
    auto handles = torch::empty({(int64_t)cfgs.size()},
                                torch::TensorOptions().dtype(torch::kInt64));
    int64_t* h = handles.data_ptr<int64_t>();
    std::lock_guard<std::mutex> lk(g_cfg_mu);
    for (size_t i = 0; i < cfgs.size(); ++i) {
        int64_t id = g_next_cfg.fetch_add(1);
        g_configs.emplace(id, CfgEntry{cfgs[i], std::nullopt});
        h[i] = id;
    }
    return handles;
}

torch::Tensor register_decomp_configs(
        std::vector<nvcomp::DecompressionConfig>&& cfgs) {
    auto handles = torch::empty({(int64_t)cfgs.size()},
                                torch::TensorOptions().dtype(torch::kInt64));
    int64_t* h = handles.data_ptr<int64_t>();
    std::lock_guard<std::mutex> lk(g_cfg_mu);
    for (size_t i = 0; i < cfgs.size(); ++i) {
        int64_t id = g_next_cfg.fetch_add(1);
        g_configs.emplace(id, CfgEntry{std::nullopt, std::move(cfgs[i])});
        h[i] = id;
    }
    return handles;
}

// One block's stores of n 4-byte words into pinned host memory over its UVA
// alias: a small readback that does not queue on the copy engine behind the
// copies home other streams have issued, as a cudaMemcpyAsync would, with
// its stream waiting there (mgr_compress_launch's sizes, and uva_store).
__global__ void k_uva_store(const unsigned int* __restrict__ src,
                            unsigned int* __restrict__ dst, long long n) {
    for (long long i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

void uva_store_bytes(const void* src, void* host, size_t bytes,
                     cudaStream_t stream) {
    TORCH_CHECK(bytes % 4 == 0, "uva_store: ", bytes,
                " bytes, not a multiple of 4");
    void* alias = nullptr;
    MGR_CUDA_CHECK(cudaHostGetDevicePointer(&alias, host, 0));
    k_uva_store<<<1, 128, 0, stream>>>((const unsigned int*)src,
                                       (unsigned int*)alias,
                                       (long long)(bytes / 4));
    MGR_CUDA_CHECK(cudaGetLastError());
}

void check_cpu_i64(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.device().is_cpu(), "nvcomp_manager: ", name, " must be CPU");
    TORCH_CHECK(t.scalar_type() == torch::kInt64,
                "nvcomp_manager: ", name, " must be int64");
    TORCH_CHECK(t.is_contiguous(), "nvcomp_manager: ", name,
                " must be contiguous");
}

}  // namespace

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

// Batch configure. Host-only, no GPU work.  Returns (ticket_id, max_sizes).
static std::tuple<int64_t, torch::Tensor> mgr_compress_begin(
    int64_t codec, int64_t elem_size, int64_t chunk_bytes,
    torch::Tensor uncomp_sizes) {
    check_cpu_i64(uncomp_sizes, "uncomp_sizes");
    const int64_t n = uncomp_sizes.numel();
    TORCH_CHECK(n > 0, "nvcomp_manager: empty batch");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto& mgr = get_manager((int)codec, elem_size, chunk_bytes, stream);

    const int64_t* sp = uncomp_sizes.data_ptr<int64_t>();
    std::vector<size_t> sizes((size_t)n);
    for (int64_t i = 0; i < n; ++i) {
        TORCH_CHECK(sp[i] > 0, "nvcomp_manager: zero/negative uncomp size at ", i);
        sizes[(size_t)i] = (size_t)sp[i];
    }

    // Configured in the launch's slices of at most kMaxBatch buffers.
    std::vector<nvcomp::CompressionConfig> configs;
    configs.reserve((size_t)n);
    for (size_t lo = 0; lo < (size_t)n; lo += kMaxBatch) {
        const size_t hi = std::min((size_t)n, lo + kMaxBatch);
        auto part = mgr.configure_compression(
            std::vector<size_t>(sizes.begin() + lo, sizes.begin() + hi));
        configs.insert(configs.end(), part.begin(), part.end());
    }
    TORCH_CHECK((int64_t)configs.size() == n,
                "nvcomp_manager: configure_compression size mismatch");

    auto max_sizes = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
    int64_t* mp = max_sizes.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        mp[i] = (int64_t)configs[(size_t)i].max_compressed_buffer_size;
    }

    int64_t tid = g_next_ticket.fetch_add(1);
    {
        std::lock_guard<std::mutex> lk(g_ticket_mu);
        g_tickets.emplace(tid, Ticket{(int)codec, elem_size, chunk_bytes,
                                      stream, std::move(configs), false});
    }
    return {tid, max_sizes};
}

// Enqueue ONE batched compress + the actual sizes into pinned memory by UVA
// stores (uva_store_bytes).  Derives
// the DecompressionConfigs via the documented no-sync overload and returns
// their registry handles.  No sync, no event — the caller owns completion.
static torch::Tensor mgr_compress_launch(
    int64_t ticket_id, torch::Tensor in_ptrs, torch::Tensor out_ptrs,
    torch::Tensor comp_sizes_dev, torch::Tensor comp_sizes_pin) {
    check_cpu_i64(in_ptrs, "in_ptrs");
    check_cpu_i64(out_ptrs, "out_ptrs");
    TORCH_CHECK(comp_sizes_dev.is_cuda() &&
                comp_sizes_dev.scalar_type() == torch::kInt64 &&
                comp_sizes_dev.is_contiguous(),
                "nvcomp_manager: comp_sizes_dev must be contiguous CUDA int64");
    TORCH_CHECK(comp_sizes_pin.device().is_cpu() &&
                comp_sizes_pin.scalar_type() == torch::kInt64 &&
                comp_sizes_pin.is_contiguous() && comp_sizes_pin.is_pinned(),
                "nvcomp_manager: comp_sizes_pin must be pinned CPU int64");

    Ticket* t;
    {
        std::lock_guard<std::mutex> lk(g_ticket_mu);
        auto it = g_tickets.find(ticket_id);
        TORCH_CHECK(it != g_tickets.end(), "nvcomp_manager: unknown ticket ",
                    ticket_id);
        TORCH_CHECK(!it->second.launched, "nvcomp_manager: ticket ", ticket_id,
                    " already launched");
        t = &it->second;
    }
    const int64_t n = (int64_t)t->configs.size();
    TORCH_CHECK(in_ptrs.numel() == n && out_ptrs.numel() == n,
                "nvcomp_manager: ptr array size mismatch");
    TORCH_CHECK(comp_sizes_dev.numel() >= n && comp_sizes_pin.numel() >= n,
                "nvcomp_manager: comp_sizes too small");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(stream == t->stream,
                "nvcomp_manager: launch must run on the begin stream");
    auto& mgr = get_manager(t->codec, t->elem_size, t->chunk_b, stream);

    const auto* in_p =
        reinterpret_cast<const uint8_t* const*>(in_ptrs.data_ptr<int64_t>());
    auto* out_p = reinterpret_cast<uint8_t* const*>(out_ptrs.data_ptr<int64_t>());
    auto* sizes_p = reinterpret_cast<size_t*>(comp_sizes_dev.data_ptr<int64_t>());
    if ((size_t)n <= kMaxBatch) {
        mgr.compress(in_p, out_p, t->configs, sizes_p);
    } else {
        for (size_t lo = 0; lo < (size_t)n; lo += kMaxBatch) {
            const size_t hi = std::min((size_t)n, lo + kMaxBatch);
            std::vector<nvcomp::CompressionConfig> slice(
                t->configs.begin() + lo, t->configs.begin() + hi);
            mgr.compress(in_p + lo, out_p + lo, slice, sizes_p + lo);
        }
    }

    uva_store_bytes(comp_sizes_dev.data_ptr<int64_t>(),
                    comp_sizes_pin.data_ptr<int64_t>(),
                    (size_t)n * sizeof(int64_t), stream);

    t->launched = true;
    // Register the (copyable, pimpl-shared) CompressionConfigs; decompress
    // derives fresh private DecompressionConfigs from them per call.
    return register_comp_configs(t->configs);
}

// A CUDA tensor's bytes (a multiple of 4) into a pinned CPU tensor of the
// same dtype (the bytes are not converted) by uva_store_bytes on the current
// stream.
static void uva_store(torch::Tensor src, torch::Tensor dst) {
    TORCH_CHECK(src.is_cuda() && src.is_contiguous(),
                "uva_store: src must be a contiguous CUDA tensor");
    TORCH_CHECK(dst.device().is_cpu() && dst.is_pinned() && dst.is_contiguous(),
                "uva_store: dst must be a contiguous pinned CPU tensor");
    TORCH_CHECK(src.scalar_type() == dst.scalar_type(),
                "uva_store: src is ", src.scalar_type(), ", dst ",
                dst.scalar_type(), " (bytes are not converted)");
    const size_t bytes = (size_t)src.numel() * src.element_size();
    TORCH_CHECK(bytes % 4 == 0 && bytes <= (size_t)dst.numel() * dst.element_size(),
                "uva_store: ", bytes, " bytes into ", dst.numel() * dst.element_size());
    uva_store_bytes(src.data_ptr(), dst.data_ptr(), bytes,
                    c10::cuda::getCurrentCUDAStream().stream());
}

// Host-only status readback.  Call only after the caller has observed
// completion (event query / tick sync).  Frees the ticket.
static torch::Tensor mgr_compress_finish(int64_t ticket_id) {
    Ticket t;
    {
        std::lock_guard<std::mutex> lk(g_ticket_mu);
        auto it = g_tickets.find(ticket_id);
        TORCH_CHECK(it != g_tickets.end(), "nvcomp_manager: unknown ticket ",
                    ticket_id);
        t = std::move(it->second);
        g_tickets.erase(it);
    }
    const int64_t n = (int64_t)t.configs.size();
    auto statuses = torch::empty({n}, torch::TensorOptions().dtype(torch::kInt64));
    int64_t* sp = statuses.data_ptr<int64_t>();
    for (int64_t i = 0; i < n; ++i) {
        sp[i] = (int64_t)(*t.configs[(size_t)i].get_status());
    }
    return statuses;
}

// Hot path: ONE batched decompress per (codec, dtype, chunk) group from
// cached config handles.  Enqueue-only; comp sizes come from the native
// headers (NVCOMP_NATIVE), so none are passed.
static void mgr_decompress(
    int64_t codec, int64_t elem_size, int64_t chunk_bytes,
    torch::Tensor comp_ptrs, torch::Tensor out_ptrs, torch::Tensor dc_handles) {
    check_cpu_i64(comp_ptrs, "comp_ptrs");
    check_cpu_i64(out_ptrs, "out_ptrs");
    check_cpu_i64(dc_handles, "dc_handles");
    const int64_t n = comp_ptrs.numel();
    TORCH_CHECK(out_ptrs.numel() == n && dc_handles.numel() == n,
                "nvcomp_manager: decompress array size mismatch");

    // Each record's CompressionConfig, or, for an ingest-fallback entry, its
    // stored DecompressionConfig, under the lock.
    std::vector<std::optional<nvcomp::CompressionConfig>> comp((size_t)n);
    std::vector<std::optional<nvcomp::DecompressionConfig>> pre((size_t)n);
    {
        const int64_t* hp = dc_handles.data_ptr<int64_t>();
        std::lock_guard<std::mutex> lk(g_cfg_mu);
        for (int64_t i = 0; i < n; ++i) {
            auto it = g_configs.find(hp[i]);
            TORCH_CHECK(it != g_configs.end(),
                        "nvcomp_manager: unknown config handle ", hp[i]);
            if (it->second.comp.has_value()) {
                comp[(size_t)i] = *it->second.comp;
            } else {
                pre[(size_t)i] = *it->second.decomp;
            }
        }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto& mgr = get_manager((int)codec, elem_size, chunk_bytes, stream);
    auto* out_p = reinterpret_cast<uint8_t* const*>(out_ptrs.data_ptr<int64_t>());
    const auto* comp_p =
        reinterpret_cast<const uint8_t* const*>(comp_ptrs.data_ptr<int64_t>());

    // One batch per slice of at most kMaxBatch records (one slice unless the
    // group is longer), each with fresh PRIVATE DecompressionConfigs from its
    // own configure (documented no-sync batch overload) — never shared with a
    // concurrent consumer of the same records on another stream.
    for (size_t lo = 0; lo < (size_t)n; lo += kMaxBatch) {
        const size_t hi = std::min((size_t)n, lo + kMaxBatch);
        std::vector<nvcomp::CompressionConfig> comp_cfgs;
        comp_cfgs.reserve(hi - lo);
        for (size_t i = lo; i < hi; ++i) {
            if (comp[i].has_value()) {
                comp_cfgs.push_back(*comp[i]);
            }
        }
        auto derived = comp_cfgs.empty()
            ? std::vector<nvcomp::DecompressionConfig>{}
            : mgr.configure_decompression(comp_cfgs);
        std::vector<nvcomp::DecompressionConfig> cfgs;
        cfgs.reserve(hi - lo);
        size_t di = 0;
        for (size_t i = lo; i < hi; ++i) {
            cfgs.push_back(comp[i].has_value() ? derived[di++] : *pre[i]);
        }
        mgr.decompress(out_p + lo, comp_p + lo, cfgs, nullptr);
    }
}

// Cold fallback: batch header parse for persisted objects whose configs were
// lost.  nvCOMP documents this overload as synchronizing the user stream —
// keep it off hot paths (counted).
static torch::Tensor mgr_ingest_configs(
    int64_t codec, int64_t elem_size, int64_t chunk_bytes,
    torch::Tensor comp_ptrs) {
    check_cpu_i64(comp_ptrs, "comp_ptrs");
    const int64_t n = comp_ptrs.numel();
    TORCH_CHECK(n > 0, "nvcomp_manager: empty ingest batch");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto& mgr = get_manager((int)codec, elem_size, chunk_bytes, stream);
    auto cfgs = mgr.configure_decompression(
        reinterpret_cast<const uint8_t* const*>(comp_ptrs.data_ptr<int64_t>()),
        (size_t)n, nullptr);
    g_header_parse_syncs.fetch_add(n);
    return register_decomp_configs(std::move(cfgs));
}

// Worst-case bound for one input size — registry-init checks; never hardcode
// nvCOMP's bound formulas.
static int64_t mgr_max_compressed_size(
    int64_t codec, int64_t elem_size, int64_t chunk_bytes, int64_t uncomp_size) {
    TORCH_CHECK(uncomp_size > 0, "nvcomp_manager: uncomp_size must be > 0");
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto& mgr = get_manager((int)codec, elem_size, chunk_bytes, stream);
    auto cfg = mgr.configure_compression((size_t)uncomp_size);
    return (int64_t)cfg.max_compressed_buffer_size;
}

static pybind11::dict mgr_counters() {
    pybind11::dict d;
    d["scratch_live_bytes"] = g_scratch_live.load();
    d["scratch_hwm_bytes"] = g_scratch_hwm.load();
    d["scratch_allocs"] = g_scratch_allocs.load();
    d["header_parse_syncs"] = g_header_parse_syncs.load();
    {
        std::lock_guard<std::mutex> lk(g_cfg_mu);
        d["live_configs"] = (int64_t)g_configs.size();
    }
    {
        std::lock_guard<std::mutex> lk(g_mgr_mu);
        d["live_managers"] = (int64_t)g_managers.size();
    }
    {
        std::lock_guard<std::mutex> lk(g_ticket_mu);
        d["tickets_live"] = (int64_t)g_tickets.size();
    }
    return d;
}

// Release retained manager scratch (safe between queries; managers reallocate
// lazily on next use).
static void mgr_release_scratch() {
    std::lock_guard<std::mutex> lk(g_mgr_mu);
    for (auto& kv : g_managers) kv.second->deallocate_gpu_mem();
}

// Test/exit hook: drop everything.  Configs must go before managers (their
// status storage belongs to the issuing manager).
static void mgr_shutdown() {
    {
        std::lock_guard<std::mutex> lk(g_ticket_mu);
        g_tickets.clear();
    }
    {
        std::lock_guard<std::mutex> lk(g_cfg_mu);
        g_configs.clear();
    }
    {
        std::lock_guard<std::mutex> lk(g_mgr_mu);
        g_managers.clear();
    }
}

void register_manager_bindings(pybind11::module_& m) {
    m.def("mgr_compress_begin", &mgr_compress_begin,
          "Batch configure_compression; returns (ticket_id, max_sizes). "
          "Host-only, no sync.");
    m.def("mgr_compress_launch", &mgr_compress_launch,
          "ONE batched manager compress + the actual sizes into pinned "
          "memory by UVA stores; returns DecompressionConfig handles. "
          "Enqueue-only.");
    m.def("uva_store", &uva_store,
          "A CUDA tensor's bytes into a pinned tensor by one block's UVA "
          "stores on the current stream (no copy engine).",
          pybind11::arg("src"), pybind11::arg("dst"));
    m.def("mgr_compress_finish", &mgr_compress_finish,
          "Status readback after caller-observed completion; frees the ticket.");
    m.def("mgr_decompress", &mgr_decompress,
          "ONE batched manager decompress from cached config handles. "
          "Enqueue-only.");
    m.def("mgr_ingest_configs", &mgr_ingest_configs,
          "Cold batch header parse (synchronizes the stream; counted).");
    m.def("mgr_max_compressed_size", &mgr_max_compressed_size,
          "Worst-case compressed bound for one input size.");
    m.def("mgr_counters", &mgr_counters, "Manager component counters.");
    m.def("mgr_release_scratch", &mgr_release_scratch,
          "deallocate_gpu_mem() on all cached managers.");
    m.def("mgr_shutdown", &mgr_shutdown,
          "Drop tickets, configs, and managers (test/exit hook).");
    m.attr("NVCOMP_MGR_VERSION") =
        std::to_string(NVCOMP_VER_MAJOR) + "." + std::to_string(NVCOMP_VER_MINOR)
        + "." + std::to_string(NVCOMP_VER_PATCH) + "."
        + std::to_string(NVCOMP_VER_BUILD);
}
