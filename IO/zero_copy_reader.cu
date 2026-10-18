#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <numeric>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#define CHECK_CUDA(x) do {                                         \
  cudaError_t _e = (x);                                             \
  if (_e != cudaSuccess) {                                          \
    TORCH_CHECK(false, "CUDA error ", __FILE__, ":", __LINE__,      \
                ": ", cudaGetErrorString(_e), " (", (int)_e, ")");  \
  }                                                                 \
} while (0)

#define CONST_NULL -64

namespace {

// ---------------------------------------------------------------------------
// A column read over UVA is a list of *segments*: runs of whole rows that sit
// back to back in pinned host memory (one region, or one page).  The caller
// passes, as int64 GPU tensors,
//   seg_ptrs  n entries: the device alias of each segment's first row;
//   seg_rows  n + 1 entries: each segment's first row in the column's row
//             space, ascending, then the end of the last segment.
// and the rows [row_start, row_stop) of that row space the read covers.  Row
// i of the read is row row_start + i of the row space.
// ---------------------------------------------------------------------------

// Device helper: the segment p with seg_rows[p] <= row < seg_rows[p + 1]
// (num_segs for a row past the last segment: the kernels trap on it, and on
// a row before the first, rather than read through seg_ptrs out of range).
__device__ __forceinline__
int find_segment(const int64_t* __restrict__ seg_rows, int num_segs,
                 int64_t row)
{
    int lo = 0, hi = num_segs;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (seg_rows[mid + 1] <= row)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// Kernel: zero-copy read of the rows a boolean mask keeps
// ---------------------------------------------------------------------------
template <typename T>
__global__ void zero_copy_mask_kernel(
    const int64_t* __restrict__ seg_ptrs,
    const int64_t* __restrict__ seg_rows,
    int         num_segs,
    int64_t     row_start,
    const int64_t* __restrict__ mask_indices,
    size_t      other_dims,
    size_t      n,
    T* __restrict__ result)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x)
    {
        size_t sel = i / other_dims;
        size_t off = i % other_dims;
        int64_t row = row_start + mask_indices[sel];
        int p = find_segment(seg_rows, num_segs, row);
        if (p >= num_segs || row < seg_rows[0]) __trap();
        const T* seg = reinterpret_cast<const T*>(seg_ptrs[p]);
        result[i] = seg[(row - seg_rows[p]) * other_dims + off];
    }
}

// ---------------------------------------------------------------------------
// Kernel: zero-copy gather (rearrange), in sorted index order for locality.
// Uses argsort permutation: iterates in sorted order via rearrange[perm[i]]
// so host reads are sequential, while result is scattered via perm[i].
// ---------------------------------------------------------------------------
template <typename T>
__global__ void zero_copy_rearrange_kernel(
    const int64_t* __restrict__ seg_ptrs,
    const int64_t* __restrict__ seg_rows,
    int         num_segs,
    int64_t     row_start,
    const int64_t* __restrict__ rearrange,
    const int64_t* __restrict__ perm,
    size_t      other_dims,
    size_t      N,
    T* __restrict__ result)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N;
         i += gridDim.x * blockDim.x)
    {
        size_t idx = i / other_dims;
        size_t off = i % other_dims;
        int64_t p_idx = perm[idx];
        int64_t src_row = rearrange[p_idx];
        if (src_row == CONST_NULL) {
            result[p_idx * other_dims + off] = CONST_NULL;
            continue;
        }
        int64_t row = row_start + src_row;
        int p = find_segment(seg_rows, num_segs, row);
        if (p >= num_segs || row < seg_rows[0]) __trap();
        const T* seg = reinterpret_cast<const T*>(seg_ptrs[p]);
        result[p_idx * other_dims + off] =
            seg[(row - seg_rows[p]) * other_dims + off];
    }
}

void check_segments(const torch::Tensor& seg_ptrs,
                    const torch::Tensor& seg_rows,
                    int64_t row_start, int64_t row_stop)
{
    TORCH_CHECK(seg_ptrs.is_cuda() && seg_rows.is_cuda(),
                "segment tensors must be on the GPU");
    TORCH_CHECK(seg_ptrs.scalar_type() == at::kLong
                && seg_rows.scalar_type() == at::kLong,
                "segment tensors must be int64");
    TORCH_CHECK(seg_ptrs.dim() == 1 && seg_ptrs.is_contiguous()
                && seg_rows.is_contiguous(),
                "segment tensors must be contiguous and 1-D");
    TORCH_CHECK(seg_ptrs.numel() > 0, "no segment to read");
    TORCH_CHECK(seg_rows.numel() == seg_ptrs.numel() + 1,
                "seg_rows needs one entry per segment plus the end (",
                seg_rows.numel(), " for ", seg_ptrs.numel(), " segments)");
    TORCH_CHECK(0 <= row_start && row_start <= row_stop,
                "rows [", row_start, ", ", row_stop, ") are not a range");
}

int launch_blocks(int64_t N, int threads)
{
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    return std::min<int>(static_cast<int>((N + threads - 1) / threads),
                         sms * 32);
}

// ---------------------------------------------------------------------------
// Type-dispatched launch helpers
// ---------------------------------------------------------------------------

template <typename T>
inline void launch_mask(const torch::Tensor& seg_ptrs,
                        const torch::Tensor& seg_rows,
                        torch::Tensor mask,
                        torch::Tensor result,
                        int64_t other_dims,
                        int64_t row_start)
{
    auto mask_indices = torch::nonzero(mask).squeeze(1);
    TORCH_CHECK(
        mask_indices.numel() == result.numel() / other_dims,
        "mask selected count (", mask_indices.numel(),
        ") vs result rows (", result.numel() / other_dims, ") mismatch");

    const int64_t N = mask_indices.numel() * other_dims;
    if (N == 0) return;

    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 1024;
    zero_copy_mask_kernel<T><<<launch_blocks(N, threads), threads, 0, stream>>>(
        seg_ptrs.data_ptr<int64_t>(), seg_rows.data_ptr<int64_t>(),
        static_cast<int>(seg_ptrs.numel()), row_start,
        mask_indices.data_ptr<int64_t>(),
        static_cast<size_t>(other_dims), static_cast<size_t>(N),
        static_cast<T*>(result.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
inline torch::Tensor launch_rearrange(const torch::Tensor& seg_ptrs,
                                      const torch::Tensor& seg_rows,
                                      torch::Tensor rearrange,
                                      int64_t row_start,
                                      at::ScalarType dtype,
                                      int64_t other_dims)
{
    const int64_t n_output = rearrange.numel();
    const int64_t N = n_output * other_dims;
    auto options = torch::TensorOptions().dtype(dtype).device(torch::kCUDA);
    if (N == 0) return torch::empty({n_output, other_dims}, options);

    auto stream = at::cuda::getCurrentCUDAStream();

    // Use argsort instead of sort: only perm is returned; the temporary
    // sorted-values tensor is freed when argsort returns, reclaiming a
    // full n_output*8-byte block before we allocate result.
    auto perm = torch::argsort(rearrange, -1, false);

    // Allocate result AFTER argsort so that the freed sorted-values
    // space is available, avoiding fragmentation OOM.
    auto result = torch::empty({n_output, other_dims}, options);

    // Kernel reads rearrange[perm[i]] for sorted-order host access,
    // writes to result[perm[i]] for scatter.
    const int threads = 1024;
    zero_copy_rearrange_kernel<T><<<launch_blocks(N, threads), threads, 0, stream>>>(
        seg_ptrs.data_ptr<int64_t>(), seg_rows.data_ptr<int64_t>(),
        static_cast<int>(seg_ptrs.numel()), row_start,
        rearrange.data_ptr<int64_t>(), perm.data_ptr<int64_t>(),
        static_cast<size_t>(other_dims), static_cast<size_t>(N),
        static_cast<T*>(result.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}

// ---------------------------------------------------------------------------
// Python entry points
// ---------------------------------------------------------------------------

// The device alias of a pinned host tensor's first byte
// (cudaHostGetDevicePointer), as an integer for seg_ptrs.
int64_t device_alias(torch::Tensor host)
{
    TORCH_CHECK(host.is_cpu(),    "device_alias needs a host tensor");
    TORCH_CHECK(host.is_pinned(), "device_alias needs pinned memory");
    (void)cudaGetLastError();  // clear stale errors
    void* alias = nullptr;
    CHECK_CUDA(cudaHostGetDevicePointer(&alias, host.data_ptr(), 0));
    return reinterpret_cast<int64_t>(alias);
}

void zero_copy_mask(
    torch::Tensor seg_ptrs,
    torch::Tensor seg_rows,
    torch::Tensor mask,
    torch::Tensor result,
    int64_t row_start,
    int64_t row_stop)
{
    check_segments(seg_ptrs, seg_rows, row_start, row_stop);
    TORCH_CHECK(mask.is_cuda(),         "mask must be on GPU");
    TORCH_CHECK(result.is_cuda(),       "result must be on GPU");
    TORCH_CHECK(mask.is_contiguous(),   "mask must be contiguous");
    TORCH_CHECK(result.is_contiguous(), "result must be contiguous");
    TORCH_CHECK(mask.scalar_type() == at::kBool, "mask must be bool");
    TORCH_CHECK(mask.dim() == 1, "mask must be 1-D");
    TORCH_CHECK(mask.numel() == row_stop - row_start,
        "mask length (", mask.numel(), ") != row_stop - row_start (",
        row_stop - row_start, ")");

    const int64_t other_dims = result.dim() > 1 ? result.size(1) : 1;

    switch (result.scalar_type()) {
        case at::kByte:   launch_mask<uint8_t >(seg_ptrs, seg_rows, mask, result, other_dims, row_start); break;
        case at::kChar:   launch_mask<int8_t  >(seg_ptrs, seg_rows, mask, result, other_dims, row_start); break;
        case at::kShort:  launch_mask<int16_t >(seg_ptrs, seg_rows, mask, result, other_dims, row_start); break;
        case at::kInt:    launch_mask<int32_t >(seg_ptrs, seg_rows, mask, result, other_dims, row_start); break;
        case at::kLong:   launch_mask<int64_t >(seg_ptrs, seg_rows, mask, result, other_dims, row_start); break;
        case at::kFloat:  launch_mask<float   >(seg_ptrs, seg_rows, mask, result, other_dims, row_start); break;
        case at::kDouble: launch_mask<double  >(seg_ptrs, seg_rows, mask, result, other_dims, row_start); break;
        default:
            TORCH_CHECK(false, "Unsupported dtype for zero_copy_mask");
    }
}

torch::Tensor zero_copy_rearrange(
    torch::Tensor seg_ptrs,
    torch::Tensor seg_rows,
    torch::Tensor rearrange,
    int64_t row_start,
    int64_t row_stop,
    at::ScalarType dtype,
    int64_t width)
{
    check_segments(seg_ptrs, seg_rows, row_start, row_stop);
    TORCH_CHECK(rearrange.is_cuda(),       "rearrange must be on GPU");
    TORCH_CHECK(rearrange.is_contiguous(), "rearrange must be contiguous");
    TORCH_CHECK(rearrange.scalar_type() == at::kLong, "rearrange must be int64");
    TORCH_CHECK(width >= 1, "width must be positive");

    switch (dtype) {
        case at::kByte:   return launch_rearrange<uint8_t >(seg_ptrs, seg_rows, rearrange, row_start, dtype, width);
        case at::kChar:   return launch_rearrange<int8_t  >(seg_ptrs, seg_rows, rearrange, row_start, dtype, width);
        case at::kShort:  return launch_rearrange<int16_t >(seg_ptrs, seg_rows, rearrange, row_start, dtype, width);
        case at::kInt:    return launch_rearrange<int32_t >(seg_ptrs, seg_rows, rearrange, row_start, dtype, width);
        case at::kLong:   return launch_rearrange<int64_t >(seg_ptrs, seg_rows, rearrange, row_start, dtype, width);
        case at::kFloat:  return launch_rearrange<float   >(seg_ptrs, seg_rows, rearrange, row_start, dtype, width);
        case at::kDouble: return launch_rearrange<double  >(seg_ptrs, seg_rows, rearrange, row_start, dtype, width);
        default:
            TORCH_CHECK(false, "Unsupported dtype for zero_copy_rearrange");
    }
}

} // namespace

PYBIND11_MODULE(zero_copy_reader, m) {
    m.def("device_alias",        &device_alias,
          "Device alias of a pinned host tensor's first byte");
    m.def("zero_copy_mask",      &zero_copy_mask,
          "Zero-copy masked read of host segments");
    m.def("zero_copy_rearrange", &zero_copy_rearrange,
          "Zero-copy rearrange/gather of host segments");
}
