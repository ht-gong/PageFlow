// What the operator kernels (partition_sort.cu, join_core.cu) share: the key
// hash, the one host wait a call makes, and the UVA store of a readback.
#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <map>
#include <utility>

namespace tqpkernels {

// The Murmur3 64-bit finalizer.  A join key's radix bucket is the top
// log2(B) bits of mix64(key) (partition_sort.cu); a join core table takes a
// key's slot from the bits below them (join_core.cu).
__device__ __forceinline__ unsigned long long mix64(unsigned long long k) {
  k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33; return k;
}

// One event per (device, stream) and host thread, reused across calls: a
// call records it and waits for it before it returns.
inline cudaEvent_t stream_event(int device, cudaStream_t st) {
  thread_local std::map<std::pair<int, cudaStream_t>, cudaEvent_t> events;
  auto it = events.find({device, st});
  if (it != events.end()) return it->second;
  cudaEvent_t ev;
  C10_CUDA_CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  events.emplace(std::make_pair(device, st), ev);
  return ev;
}

// The host waits for the event, with the GIL released.
inline void wait_event(cudaEvent_t ev) {
  pybind11::gil_scoped_release no_gil;
  C10_CUDA_CHECK(cudaEventSynchronize(ev));
}

// A small device-to-host readback is one block's stores into the pinned
// buffer's UVA alias, not a cudaMemcpyAsync: a copy queues on the copy
// engine behind every copy home already issued on other streams, and the
// stream that issued it waits there.
static __global__ void k_uva_store(const unsigned int* __restrict__ src,
                                   unsigned int* __restrict__ dst, long long n) {
  for (long long i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
}

// `bytes` (a multiple of 4) of device memory `src` into pinned host memory
// `host` by one block on `st`.
inline void uva_store(const void* src, void* host, size_t bytes, cudaStream_t st) {
  TORCH_CHECK(bytes % 4 == 0, "uva_store: ", bytes, " bytes, not a multiple of 4");
  void* alias = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer(&alias, host, 0));
  k_uva_store<<<1, 128, 0, st>>>((const unsigned int*)src, (unsigned int*)alias,
                                 (long long)(bytes / 4));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace tqpkernels
