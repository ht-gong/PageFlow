// The join core's kernels (operators/join_core.py): one tick of a join core
// pipe matches the two sides' rows of the tick's radix buckets.
//
//   expand       the pairs of the sort/search probe: each probe row's run of
//                equal build keys in the sorted build, written as (build row,
//                probe row) pairs in probe-row order
//   unique hash  an inner join whose build tuple (1-2 int64 keys) is proven
//                unique: an open-addressing table of build rows, one partner
//                per probe row, the matched probe rows compacted in order
//   semi / anti  a hash set of the build keys, the probe rows that have (semi)
//                or lack (anti) a match compacted in order; with a residual
//                that compares one build column with one probe column, the set
//                keeps the column's min and/or max per key, which answers
//                EXISTS for <>, <, <=, > and >=
//
// A table has twice the build rows in slots (load <= 0.5) and probes
// linearly.  A slot comes from the bits of mix64(key) below its top 9: the top
// log2(B) bits (B <= 512) are the key's radix bucket, shared by a bucket's keys.
//
// Each call queues all its kernels, copies its output row count into pinned
// host memory behind an event, and waits for that event once: the outputs are
// allocated at an upper bound (the probe rows), and the caller narrows them to
// the count.  expand waits for the pair count before it can size its outputs.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

#include <climits>
#include <vector>

#include "kernel_utils.cuh"

namespace {

using tqpkernels::mix64;

constexpr int T = 256;
constexpr long long EMPTY_KEY = LLONG_MIN;
// The probe rows of a call (and a unique-hash build's, whose row ids the table
// holds as int32).  CUB 2.5's select and scan count items in int, and its
// select runs the last tile's lanes past the end into that count, so the
// rows stay a tile (6528 items on sm90) below INT_MAX.
constexpr long long MAX_ROWS = (1LL << 31) - (1LL << 16);

// The residual operator a semi/anti summary answers, as build.v OP probe.v.
enum Op : int { OP_NONE = 0, OP_NE = 1, OP_LT = 2, OP_LE = 3, OP_GT = 4, OP_GE = 5 };

// The first slot of a hash: multiply-shift of its bits below the top 9 (the
// radix bucket, B <= 512, partitions.JOIN_B_MAX) onto [0, cap).
__device__ __forceinline__ unsigned long long slot_of(unsigned long long h,
                                                      unsigned long long cap) {
  return __umul64hi(h << 9, cap);
}

__device__ __forceinline__ unsigned long long next_slot(unsigned long long s,
                                                        unsigned long long cap) {
  return s + 1 == cap ? 0 : s + 1;
}

__device__ __forceinline__ unsigned long long tuple_hash(long long k0, long long k1, bool two) {
  unsigned long long h = mix64((unsigned long long)k0);
  if (two) {
    unsigned long long h1 = mix64((unsigned long long)k1);
    h ^= h1 + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h = mix64(h);
  }
  return h;
}

inline unsigned blocks(long long n) { return (unsigned)((n + T - 1) / T); }

// A tensor's int64 data as the long long the kernels and CUDA's atomics take.
inline long long* ll(const at::Tensor& t) { return (long long*)t.data_ptr(); }

// ---------------------------------------------------------------- expand --

// Pair k: its probe row i is the first with incl[i] > k (incl: the inclusive
// sum of the run lengths), and its build row sorted_ids[start[i] + k -
// (incl[i] - len[i])].  Zero-length runs are never picked.
__global__ void k_expand(const long long* __restrict__ sorted_ids,
                         const long long* __restrict__ start,
                         const long long* __restrict__ len,
                         const long long* __restrict__ incl, long long m, long long n_flat,
                         long long* __restrict__ cs, long long* __restrict__ cr) {
  const long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (k >= n_flat) return;
  long long lo = 0, hi = m;
  while (lo < hi) {
    const long long mid = (lo + hi) >> 1;
    if (incl[mid] <= k) lo = mid + 1;
    else hi = mid;
  }
  cs[k] = sorted_ids[start[lo] + (k - (incl[lo] - len[lo]))];
  cr[k] = lo;
}

// ----------------------------------------------------------- unique hash --

template <bool TWO>
__global__ void k_unique_build(const long long* __restrict__ b0, const long long* __restrict__ b1,
                               int* table, long long n, unsigned long long cap) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long s = slot_of(tuple_hash(b0[i], TWO ? b1[i] : 0, TWO), cap);
  while (atomicCAS(&table[s], -1, (int)i) != -1) s = next_slot(s, cap);
}

template <bool TWO>
__global__ void k_unique_probe(const long long* __restrict__ b0, const long long* __restrict__ b1,
                               const long long* __restrict__ p0, const long long* __restrict__ p1,
                               const int* __restrict__ table, int* __restrict__ partner,
                               long long n, unsigned long long cap) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long k0 = p0[i], k1 = TWO ? p1[i] : 0;
  unsigned long long s = slot_of(tuple_hash(k0, k1, TWO), cap);
  int r = -1;
  while (true) {
    const int b = table[s];
    if (b == -1) break;
    if (b0[b] == k0 && (!TWO || b1[b] == k1)) { r = b; break; }
    s = next_slot(s, cap);
  }
  partner[i] = r;
}

struct HasPartner {
  const int* partner;
  __device__ __forceinline__ bool operator()(const int& i) const { return partner[i] >= 0; }
};

// The matched probe rows' build rows, for the first *count of probe_idx.
__global__ void k_take_partner(const int* __restrict__ partner,
                               const long long* __restrict__ probe_idx,
                               const long long* __restrict__ count, long long n,
                               long long* __restrict__ build_idx) {
  const long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (j >= n || j >= *count) return;
  build_idx[j] = partner[probe_idx[j]];
}

// ------------------------------------------------------------ semi / anti --

// Insert the key; with a summary, fold the build value into its slot's min
// and/or max.  Each reads the slot before its atomic, so a key's repeats
// neither CAS nor contend once it is in.  A key equal to the empty-slot value
// cannot be stored: it sets *bad and the caller takes another path.
__global__ void k_set_build(const long long* __restrict__ key, const long long* __restrict__ val,
                            long long* table, long long* mn, long long* mx, long long n,
                            unsigned long long cap, long long* bad) {
  const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (i >= n) return;
  const long long k = key[i];
  if (k == EMPTY_KEY) { *bad = 1; return; }
  unsigned long long s = slot_of(mix64((unsigned long long)k), cap);
  while (true) {
    long long prev = table[s];
    if (prev == EMPTY_KEY)
      prev = (long long)atomicCAS((unsigned long long*)&table[s], (unsigned long long)EMPTY_KEY,
                                  (unsigned long long)k);
    if (prev == EMPTY_KEY || prev == k) break;
    s = next_slot(s, cap);
  }
  // A stale read is never past the atomic's result: min only falls, max only rises.
  const long long v = (mn || mx) ? val[i] : 0;
  if (mn && v < mn[s]) atomicMin(&mn[s], v);
  if (mx && v > mx[s]) atomicMax(&mx[s], v);
}

// Probe row i is kept when it has a match (semi) or none (anti): its key is in
// the set and, with a summary, some build row of the key passes build.v OP
// probe.v, which the key's min (<, <=), max (>, >=) or both (<>) decide.
struct SetProbe {
  const long long* key;
  const long long* val;
  const long long* table;
  const long long* mn;
  const long long* mx;
  unsigned long long cap;
  int op;
  bool anti;
  __device__ __forceinline__ bool operator()(const int& i) const {
    const long long k = key[i];
    bool hit = false;
    unsigned long long s = slot_of(mix64((unsigned long long)k), cap);
    if (k != EMPTY_KEY) {
      while (true) {
        const long long t = table[s];
        if (t == k) { hit = true; break; }
        if (t == EMPTY_KEY) break;
        s = next_slot(s, cap);
      }
    }
    if (hit && op != OP_NONE) {
      const long long v = val[i];
      switch (op) {
        case OP_NE: hit = mn[s] != v || mx[s] != v; break;
        case OP_LT: hit = mn[s] < v; break;
        case OP_LE: hit = mn[s] <= v; break;
        case OP_GT: hit = mx[s] > v; break;
        default:    hit = mx[s] >= v; break;
      }
    }
    return hit != anti;
  }
};

// ------------------------------------------------------------------ host --

// Store `bytes` of `dev` into host_out over UVA (tqpkernels::uva_store) behind
// an event; the caller waits on the returned event once its last kernel is
// queued.
cudaEvent_t read_back(const at::Tensor& dev, size_t bytes, const at::Tensor& host_out,
                      cudaStream_t st) {
  tqpkernels::uva_store(dev.data_ptr(), host_out.data_ptr(), bytes, st);
  cudaEvent_t ev = tqpkernels::stream_event(dev.get_device(), st);
  C10_CUDA_CHECK(cudaEventRecord(ev, st));
  return ev;
}

void check_column(const at::Tensor& t, const char* what, const at::Tensor& like) {
  TORCH_CHECK(t.is_cuda() && t.device() == like.device() && t.scalar_type() == at::kLong &&
              t.dim() == 1 && t.is_contiguous(),
              "join_core: ", what, " must be a contiguous 1-D int64 tensor on the key's device");
}

void check_rows(const at::Tensor& t, const char* what, const at::Tensor& like) {
  check_column(t, what, like);
  TORCH_CHECK(t.numel() <= MAX_ROWS, "join_core: ", what, " has ", t.numel(),
              " rows; a call takes at most ", MAX_ROWS);
}

void check_host(const at::Tensor& host_out) {
  TORCH_CHECK(host_out.is_cpu() && host_out.is_pinned() && host_out.scalar_type() == at::kLong &&
              host_out.is_contiguous() && host_out.numel() >= 2,
              "join_core: host_out must be pinned int64 with 2 elements");
}

// Keep the rows of [0, n) that `pred` selects, in order, into out (int64, n
// rows) and their count into *count.
template <typename Pred>
void select_rows(Pred pred, long long n, long long* out, long long* count,
                 const at::TensorOptions& opt, cudaStream_t st) {
  cub::CountingInputIterator<int> rows(0);
  size_t bytes = 0;
  C10_CUDA_CHECK(cub::DeviceSelect::If(nullptr, bytes, rows, out, count, (int)n, pred, st));
  auto temp = at::empty({(int64_t)std::max<size_t>(bytes, 1)}, opt.dtype(at::kByte));
  C10_CUDA_CHECK(cub::DeviceSelect::If(temp.data_ptr(), bytes, rows, out, count, (int)n, pred, st));
}

}  // namespace

// sorted_ids: [nL] the build rows in key order; start, len: [m] probe row i's
// run of equal keys in it (len 0: no match); host_out: pinned int64[2].
// Waits for the pair count (host_out[0]), then returns (build rows, probe rows)
// of every pair, in probe-row order and run order.
std::vector<at::Tensor> join_expand(at::Tensor sorted_ids, at::Tensor start, at::Tensor len,
                                    at::Tensor host_out) {
  check_column(sorted_ids, "sorted_ids", sorted_ids);
  check_rows(start, "start", sorted_ids);
  check_rows(len, "len", sorted_ids);
  check_host(host_out);
  TORCH_CHECK(start.numel() == len.numel(), "join_core: start and len differ in length");
  const at::cuda::OptionalCUDAGuard guard(sorted_ids.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const auto opt = sorted_ids.options();
  const long long m = len.numel();
  if (m == 0) return {at::empty({0}, opt), at::empty({0}, opt)};

  auto incl = at::empty({m}, opt);
  size_t bytes = 0;
  const long long* lp = ll(len);
  long long* ip = ll(incl);
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, lp, ip, (int)m, st));
  auto temp = at::empty({(int64_t)std::max<size_t>(bytes, 1)}, opt.dtype(at::kByte));
  C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp.data_ptr(), bytes, lp, ip, (int)m, st));
  // the one wait: the pair count sizes the outputs
  tqpkernels::wait_event(read_back(incl.narrow(0, m - 1, 1), sizeof(long long), host_out, st));
  const long long n_flat = ll(host_out)[0];

  auto cs = at::empty({n_flat}, opt);
  auto cr = at::empty({n_flat}, opt);
  if (n_flat > 0) {
    k_expand<<<blocks(n_flat), T, 0, st>>>(ll(sorted_ids), ll(start), lp, ip, m, n_flat, ll(cs),
                                           ll(cr));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {cs, cr};
}

// build_keys, probe_keys: 1 or 2 aligned int64 key columns per side, the build
// tuple unique, at least one row each; host_out: pinned int64[2].  Returns
// (build rows, probe rows) sized for every probe row; the first host_out[0]
// are the matches, in probe-row order.
std::vector<at::Tensor> join_unique_hash(std::vector<at::Tensor> build_keys,
                                         std::vector<at::Tensor> probe_keys,
                                         at::Tensor host_out) {
  TORCH_CHECK((build_keys.size() == 1 || build_keys.size() == 2) &&
              build_keys.size() == probe_keys.size(),
              "join_core: one or two aligned build/probe keys");
  const at::Tensor& key = build_keys[0];
  const long long nb = key.numel(), np = probe_keys[0].numel();
  for (auto& t : build_keys) {
    check_rows(t, "a build key", key);
    TORCH_CHECK(t.numel() == nb, "join_core: build keys differ in length");
  }
  for (auto& t : probe_keys) {
    check_rows(t, "a probe key", key);
    TORCH_CHECK(t.numel() == np, "join_core: probe keys differ in length");
  }
  check_host(host_out);
  TORCH_CHECK(nb > 0 && np > 0, "join_core: unique hash needs rows on both sides");
  const at::cuda::OptionalCUDAGuard guard(key.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const auto opt = key.options();
  const bool two = build_keys.size() == 2;
  const long long* b0 = ll(build_keys[0]);
  const long long* b1 = ll(build_keys[two ? 1 : 0]);
  const long long* p0 = ll(probe_keys[0]);
  const long long* p1 = ll(probe_keys[two ? 1 : 0]);

  const unsigned long long cap = 2ULL * nb;
  auto table = at::full({(int64_t)cap}, -1, opt.dtype(at::kInt));
  auto partner = at::empty({np}, opt.dtype(at::kInt));
  int* tp = table.data_ptr<int>();
  int* pp = partner.data_ptr<int>();
  if (two) {
    k_unique_build<true><<<blocks(nb), T, 0, st>>>(b0, b1, tp, nb, cap);
    k_unique_probe<true><<<blocks(np), T, 0, st>>>(b0, b1, p0, p1, tp, pp, np, cap);
  } else {
    k_unique_build<false><<<blocks(nb), T, 0, st>>>(b0, b1, tp, nb, cap);
    k_unique_probe<false><<<blocks(np), T, 0, st>>>(b0, b1, p0, p1, tp, pp, np, cap);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  auto probe_idx = at::empty({np}, opt);
  auto build_idx = at::empty({np}, opt);
  auto count = at::empty({1}, opt);
  select_rows(HasPartner{pp}, np, ll(probe_idx), ll(count), opt, st);
  cudaEvent_t ev = read_back(count, sizeof(long long), host_out, st);
  k_take_partner<<<blocks(np), T, 0, st>>>(pp, ll(probe_idx), ll(count), np, ll(build_idx));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tqpkernels::wait_event(ev);
  return {build_idx, probe_idx};
}

// build_key, probe_key: int64 keys, at least one row each; build_val,
// probe_val: the residual's int64 columns when op is not OP_NONE; host_out:
// pinned int64[2].  Returns the probe rows kept, sized for every probe row:
// the first host_out[0], in order.  host_out[1] is 1 when a build key equals
// the empty-slot value (the result is then not valid).
at::Tensor join_semi_anti(at::Tensor build_key, at::Tensor probe_key,
                          c10::optional<at::Tensor> build_val,
                          c10::optional<at::Tensor> probe_val, int64_t op, bool anti,
                          at::Tensor host_out) {
  check_column(build_key, "the build key", build_key);
  check_rows(probe_key, "the probe key", build_key);
  check_host(host_out);
  const long long nb = build_key.numel(), np = probe_key.numel();
  TORCH_CHECK(nb > 0 && np > 0, "join_core: semi/anti needs rows on both sides");
  TORCH_CHECK(op >= OP_NONE && op <= OP_GE, "join_core: unknown residual operator ", op);
  const bool summary = op != OP_NONE;
  if (summary) {
    TORCH_CHECK(build_val.has_value() && probe_val.has_value(),
                "join_core: a residual needs both value columns");
    check_column(*build_val, "the build value", build_key);
    check_column(*probe_val, "the probe value", build_key);
    TORCH_CHECK(build_val->numel() == nb && probe_val->numel() == np,
                "join_core: a value column differs in length from its key");
  }
  const at::cuda::OptionalCUDAGuard guard(build_key.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const auto opt = build_key.options();

  const unsigned long long cap = 2ULL * nb;
  auto table = at::full({(int64_t)cap}, EMPTY_KEY, opt);
  at::Tensor mn, mx;
  if (op == OP_NE || op == OP_LT || op == OP_LE) mn = at::full({(int64_t)cap}, LLONG_MAX, opt);
  if (op == OP_NE || op == OP_GT || op == OP_GE) mx = at::full({(int64_t)cap}, LLONG_MIN, opt);
  long long* mnp = mn.defined() ? ll(mn) : nullptr;
  long long* mxp = mx.defined() ? ll(mx) : nullptr;
  auto dev = at::zeros({2}, opt);  // [0] kept rows, [1] a build key was the empty-slot value
  long long* devp = ll(dev);

  k_set_build<<<blocks(nb), T, 0, st>>>(ll(build_key), summary ? ll(*build_val) : nullptr,
                                        ll(table), mnp, mxp, nb, cap, devp + 1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto out = at::empty({np}, opt);
  SetProbe pred{ll(probe_key), summary ? ll(*probe_val) : nullptr, ll(table), mnp, mxp, cap,
                (int)op, anti};
  select_rows(pred, np, ll(out), devp, opt, st);
  tqpkernels::wait_event(read_back(dev, 2 * sizeof(long long), host_out, st));
  return out;
}
