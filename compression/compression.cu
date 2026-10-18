#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

// nvCOMP headers
// =================================================================================
//   Copy / pack kernels for the IO layer.  Codec work lives in nvcomp_manager.cu.
// =================================================================================

// --- Error Checking Macros ---
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                  << ": " << cudaGetErrorString(err) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)


// =================================================================================
//                          BATCHED MEMCPY (for compress packing)
// =================================================================================

__global__ void batched_memcpy_kernel(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ dst_offsets,
    const int64_t* __restrict__ sizes,
    int64_t num_chunks
) {
    for (int64_t chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int64_t sz = sizes[chunk];
        const uint8_t* csrc = src + src_offsets[chunk];
        uint8_t* cdst = dst + dst_offsets[chunk];

        // Copy 8 bytes at a time for the bulk
        int64_t vec_count = sz / 8;
        const int64_t* vsrc = reinterpret_cast<const int64_t*>(csrc);
        int64_t* vdst = reinterpret_cast<int64_t*>(cdst);
        for (int64_t i = threadIdx.x; i < vec_count; i += blockDim.x) {
            vdst[i] = vsrc[i];
        }

        // Remaining bytes
        int64_t tail_start = vec_count * 8;
        for (int64_t i = tail_start + threadIdx.x; i < sz; i += blockDim.x) {
            cdst[i] = csrc[i];
        }
    }
}

void batched_memcpy(
    torch::Tensor src,
    torch::Tensor dst,
    torch::Tensor src_offsets,
    torch::Tensor dst_offsets,
    torch::Tensor sizes
) {
    TORCH_CHECK(src.is_cuda(), "src must be on CUDA");
    TORCH_CHECK(dst.is_cuda(), "dst must be on CUDA");
    TORCH_CHECK(src_offsets.is_cuda() && src_offsets.dtype() == torch::kInt64);
    TORCH_CHECK(dst_offsets.is_cuda() && dst_offsets.dtype() == torch::kInt64);
    TORCH_CHECK(sizes.is_cuda() && sizes.dtype() == torch::kInt64);

    int64_t num_chunks = sizes.numel();
    if (num_chunks == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int threads = 256;
    int blocks = std::min((int)num_chunks, 2048);

    batched_memcpy_kernel<<<blocks, threads, 0, stream>>>(
        src.data_ptr<uint8_t>(),
        dst.data_ptr<uint8_t>(),
        src_offsets.data_ptr<int64_t>(),
        dst_offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        num_chunks
    );

    CUDA_CHECK(cudaGetLastError());
}

// --- Batched memcpy between absolute device addresses ---
// The same copy as batched_memcpy, but each chunk names its own source and
// destination address, so one launch can fill a scattered set of buffers
// (the granule cache's slots and a dense buffer).  A chunk whose source or
// destination is not 8-byte aligned is copied byte by byte.  The grid's
// y dimension splits each chunk across that many blocks, so a launch of a
// few large chunks (merged slot pieces, multi-MiB records) still spreads
// over the device instead of copying each chunk on one SM.
__global__ void batched_memcpy_abs_kernel(
    const int64_t* __restrict__ src_ptrs,
    const int64_t* __restrict__ dst_ptrs,
    const int64_t* __restrict__ sizes,
    int64_t num_chunks,
    int64_t dst_base
) {
    const int64_t lane = (int64_t)blockIdx.y * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)gridDim.y * blockDim.x;
    for (int64_t chunk = blockIdx.x; chunk < num_chunks; chunk += gridDim.x) {
        int64_t sz = sizes[chunk];
        const int64_t s_addr = src_ptrs[chunk];
        const int64_t d_addr = dst_base + dst_ptrs[chunk];
        const uint8_t* csrc = reinterpret_cast<const uint8_t*>(s_addr);
        uint8_t* cdst = reinterpret_cast<uint8_t*>(d_addr);
        int64_t tail_start = 0;
        if (((s_addr | d_addr) & 7) == 0) {
            int64_t vec_count = sz / 8;
            const int64_t* vsrc = reinterpret_cast<const int64_t*>(csrc);
            int64_t* vdst = reinterpret_cast<int64_t*>(cdst);
            for (int64_t i = lane; i < vec_count; i += stride) {
                vdst[i] = vsrc[i];
            }
            tail_start = vec_count * 8;
        }
        for (int64_t i = tail_start + lane; i < sz; i += stride) {
            cdst[i] = csrc[i];
        }
    }
}

// The kernel reads an address array from the device, or straight from
// pinned host memory over UVA (no copy ahead of the kernel, so nothing
// queues behind other host-to-device copies on the copy engine).
static const int64_t* abs_array_ptr(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dtype() == torch::kInt64,
                "batched_memcpy_abs: ", name, " must be int64");
    if (t.is_cuda()) return t.data_ptr<int64_t>();
    TORCH_CHECK(t.is_pinned(), "batched_memcpy_abs: ", name,
                " must be on the GPU or in pinned host memory");
    void* dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev, t.data_ptr<int64_t>(), 0));
    return static_cast<const int64_t*>(dev);
}

void batched_memcpy_abs(
    torch::Tensor src_ptrs,
    torch::Tensor dst_ptrs,
    torch::Tensor sizes
) {
    TORCH_CHECK(src_ptrs.is_contiguous() && dst_ptrs.is_contiguous() &&
                sizes.is_contiguous(),
                "batched_memcpy_abs: arrays must be contiguous");
    TORCH_CHECK(src_ptrs.numel() == sizes.numel() &&
                dst_ptrs.numel() == sizes.numel(),
                "batched_memcpy_abs: arrays must have identical lengths");

    int64_t num_chunks = sizes.numel();
    if (num_chunks == 0) return;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    // About 2048 blocks in all: one per chunk when there are that many,
    // else each chunk split across up to 64 (the sizes live on the device,
    // so the split cannot follow them; a block past a small chunk's end
    // only reads its size).
    int blocks = (int)std::min<int64_t>(num_chunks, 2048);
    int split = (int)std::max<int64_t>(1, std::min<int64_t>(64, 2048 / num_chunks));
    const int64_t* src = abs_array_ptr(src_ptrs, "src_ptrs");
    const int64_t* dst = abs_array_ptr(dst_ptrs, "dst_ptrs");
    const int64_t* siz = abs_array_ptr(sizes, "sizes");
    batched_memcpy_abs_kernel<<<dim3(blocks, split), threads, 0, stream>>>(
        src, dst, siz, num_chunks, 0
    );
    CUDA_CHECK(cudaGetLastError());
}

// The same copy from one pinned staging buffer at host address ``staged``:
// sources [0, m), destination offsets [m, 2m) from ``dst_base``, sizes
// [2m, 3m), int64, read by the kernel over UVA; then ``event`` (a CUDA event
// handle, 0 for none) is recorded on the current stream.  One call and one
// device-pointer lookup in place of batched_memcpy_abs's tensor views, three
// lookups and a separate event record (IO/pages/abs_copy.py, ``launch``).
void batched_memcpy_abs_staged(int64_t staged, int64_t m, int64_t dst_base,
                               int64_t event) {
    if (m <= 0) return;
    void* dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&dev, reinterpret_cast<void*>(staged), 0));
    const int64_t* a = static_cast<const int64_t*>(dev);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int threads = 256;
    int blocks = (int)std::min<int64_t>(m, 2048);
    int split = (int)std::max<int64_t>(1, std::min<int64_t>(64, 2048 / m));
    batched_memcpy_abs_kernel<<<dim3(blocks, split), threads, 0, stream>>>(
        a, a + m, a + 2 * m, m, dst_base);
    CUDA_CHECK(cudaGetLastError());
    if (event) CUDA_CHECK(cudaEventRecord(reinterpret_cast<cudaEvent_t>(event), stream));
}

// --- Pointer-based batched D2H ---
// Copies from scattered GPU sources to scattered CPU (pinned) destinations.
// Each entry is one page payload or one cache-slot image.
void batched_d2h_to_ptrs(
    std::vector<int64_t> src_ptrs,              // data_ptr() of each GPU source
    std::vector<int64_t> dst_ptrs,              // data_ptr() of each CPU destination
    std::vector<int64_t> copy_sizes              // bytes to copy per entry
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (size_t i = 0; i < src_ptrs.size(); ++i) {
        const void* src_ptr = reinterpret_cast<const void*>(src_ptrs[i]);
        void* dst_ptr = reinterpret_cast<void*>(dst_ptrs[i]);
        CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, copy_sizes[i],
                                   cudaMemcpyDeviceToHost, stream));
    }
}

// --- Pointer-based batched copy into scattered destinations ---
// Copies from scattered sources to scattered destinations.  Originally
// CPU(pinned)→GPU for the page-cache fill path, but also used for
// GPU→GPU fills from the boundary/pagecache hit path in CachedDirectIn.
// Uses cudaMemcpyDefault so direction is resolved per-entry via UVA;
// passing a device pointer with cudaMemcpyHostToDevice would be UB.
void batched_h2d_to_ptrs(
    std::vector<int64_t> src_ptrs,              // data_ptr() of each source
    std::vector<int64_t> dst_ptrs,              // data_ptr() of each destination
    std::vector<int64_t> copy_sizes              // bytes to copy per entry
) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    for (size_t i = 0; i < src_ptrs.size(); ++i) {
        const void* src_ptr = reinterpret_cast<const void*>(src_ptrs[i]);
        void* dst_ptr = reinterpret_cast<void*>(dst_ptrs[i]);
        CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, copy_sizes[i],
                                   cudaMemcpyDefault, stream));
    }
}

// --- In-stage packers -----------------------------------------------------
//
// One copy per page payload into the tick's aggregate buffer.  Python
// compiles each page's destination offset once per tick layout and patches
// only the source pointers per launch; a zero source means the other lane
// (H2D or D2D) copies that page.
static void batched_pack(
    torch::Tensor agg_buf,
    const std::vector<int64_t>& page_src_ptrs,
    const std::vector<int64_t>& page_dst_offsets,
    const std::vector<int64_t>& page_byte_sizes,
    cudaMemcpyKind copy_kind,
    const char* op_name
) {
    TORCH_CHECK(agg_buf.is_cuda() && agg_buf.dtype() == torch::kUInt8 &&
                agg_buf.is_contiguous(), op_name,
                ": agg_buf must be a contiguous CUDA uint8 tensor");
    TORCH_CHECK(page_src_ptrs.size() == page_dst_offsets.size() &&
                page_src_ptrs.size() == page_byte_sizes.size(),
                op_name, ": page arrays must have identical lengths");

    // Validate every destination before submitting any copy, so a malformed
    // pybind call fails closed without a partial aggregate.
    const int64_t agg_bytes = agg_buf.numel();
    const size_t n_pages = page_src_ptrs.size();
    for (size_t i = 0; i < n_pages; ++i) {
        const int64_t off = page_dst_offsets[i];
        const int64_t nbytes = page_byte_sizes[i];
        TORCH_CHECK(nbytes >= 0, op_name, ": page sizes must be non-negative");
        TORCH_CHECK(0 <= off && off <= agg_bytes && nbytes <= agg_bytes - off,
                    op_name, ": page destination exceeds agg_buf");
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    uint8_t* dst_base = agg_buf.data_ptr<uint8_t>();
    for (size_t i = 0; i < n_pages; ++i) {
        if (page_src_ptrs[i] == 0 || page_byte_sizes[i] == 0) continue;
        CUDA_CHECK(cudaMemcpyAsync(
            dst_base + page_dst_offsets[i],
            reinterpret_cast<const void*>(page_src_ptrs[i]),
            page_byte_sizes[i], copy_kind, stream));
    }
}

void batched_h2d_pack(
    torch::Tensor agg_buf,
    std::vector<int64_t> page_src_ptrs,
    std::vector<int64_t> page_dst_offsets,
    std::vector<int64_t> page_byte_sizes
) {
    batched_pack(agg_buf, page_src_ptrs, page_dst_offsets, page_byte_sizes,
                 cudaMemcpyHostToDevice, "batched_h2d_pack");
}

// D2D retains an explicit cudaMemcpyDeviceToDevice direction: a host pointer
// cannot be silently accepted through UVA/default-kind inference.
void batched_d2d_pack(
    torch::Tensor agg_buf,
    std::vector<int64_t> page_src_ptrs,
    std::vector<int64_t> page_dst_offsets,
    std::vector<int64_t> page_byte_sizes
) {
    batched_pack(agg_buf, page_src_ptrs, page_dst_offsets, page_byte_sizes,
                 cudaMemcpyDeviceToDevice, "batched_d2d_pack");
}

// nvCOMP 5.2 manager component (nvcomp_manager.cu)
void register_manager_bindings(pybind11::module_& m);

// --- PYBIND11 Module Binding ---
PYBIND11_MODULE(tqpcompress, m) {

    m.def("batched_memcpy", &batched_memcpy, "Batched memcpy from src to dst using per-chunk offsets and sizes");
    m.def("batched_memcpy_abs", &batched_memcpy_abs,
          "Batched memcpy between absolute device addresses (one source, destination and size per chunk); the arrays on the GPU or in pinned host memory");
    m.def("batched_memcpy_abs_staged", &batched_memcpy_abs_staged,
          "batched_memcpy_abs from one pinned staging buffer (sources, destination offsets from dst_base, sizes) read over UVA, then an event recorded");
    m.def("batched_d2h_to_ptrs", &batched_d2h_to_ptrs,"Batched async D2H from scattered GPU to scattered CPU (pinned)");
    m.def("batched_h2d_to_ptrs", &batched_h2d_to_ptrs, "Batched async H2D from scattered CPU to scattered GPU cache slots");
    m.def("batched_h2d_pack", &batched_h2d_pack,
          "Batched async H2D of page payloads into agg_buf offsets.");
    m.def("batched_d2d_pack", &batched_d2d_pack,
          "D2D sibling of batched_h2d_pack for cache-hit / GPU-resident pages.");

    // nvCOMP 5.2 manager component
    register_manager_bindings(m);
}
