// The join sort pipe's hash partition (operators/partition_sort.py): one
// fused, stable, two-pass counting partition of a tick's transfer set into the
// padded per-bucket layout of IO/codec/batched_compress.aligned_partition_layout.
//
//   K1 k_hist     per-tile bucket histogram (per-warp shared-memory atomics)
//   K2 k_scan     per-bucket exclusive scan over the tiles; the last block
//                 writes the bucket counts and each row-bytes class's padded
//                 total into pinned host memory (through its UVA alias) and the
//                 padded bucket starts per class
//   K3 k_scatter  stable block-local rank (warp ballots), the tile staged in
//                 shared memory in bucket order, every column and the global
//                 index written as coalesced bucket runs
//
// A row's bucket is the top log2(B) bits of mix64(key) (bucket 0 when B is 1),
// the same on both sides of a join; rows keep their source order inside a bucket, so the outputs are
// deterministic and gidx runs ascend in each bucket.  Under a mask K1 and K3
// walk the pre-mask rows with the mask as a predicate, after one of two
// prologues when the tick takes it:
//   select-first     k_mcount, k_exscan1, k_mselect: the kept rows' pre-mask
//                    ids (int32), then K1-K3 over the kept rows instead
//   compacted-input  k_tilecount, k_exscan1: each pre-mask tile's offset into
//                    the columns already compacted by the mask
// An event is recorded after K2, so the host waits for the counts while K3
// runs.  One host wait per call; no other sync.
//
// The caller's kept-row count (n_valid) sizes the outputs and select-first's
// domain, so a wrong one must fail, not corrupt memory: a mask prologue checks
// the kept rows it finds against it, K2 checks that the counts fit the outputs,
// and either check's flag makes K1 and K3 skip their work (each reads only the
// flags its tick can raise); the host then sees the kept rows found
// (host_out[B + 4]) and raises.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_reduce.cuh>

#include "kernel_utils.cuh"

#include <algorithm>
#include <map>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

namespace {

using tqpkernels::mix64;

constexpr int MAXC = 6;      // data columns per call
constexpr int MAXCLS = 4;    // row-bytes classes (the gidx's 8 B included)
constexpr int BMAX = 512;    // buckets
constexpr int LOGBMAX = 9;
constexpr int BPL = BMAX / 32;   // buckets per lane where one warp scans the buckets
constexpr int BT = 512;      // threads per tile
constexpr int ITEMS = 4;     // rows per thread
constexpr int TILE = BT * ITEMS;
constexpr int W = BT / 32;
constexpr int SCAN_BT = 256;     // K2 threads per bucket
constexpr int EXSCAN_BT = 1024;  // k_exscan1 threads
constexpr int MSEL_BT = 256;     // select-first: 16 mask bytes per thread
constexpr unsigned FULLM = 0xffffffffu;

struct Cols {
  const unsigned char* in[MAXC];
  unsigned char* out[MAXC];
  int rb[MAXC];     // row bytes
  int comp[MAXC];   // 1: the column arrives compacted (n_valid rows)
  int cls[MAXC];    // row-bytes class (its row of bases)
  int ncols;
};

struct Rpa { long long v[MAXCLS]; };  // rows per 512-B unit, per class

struct Params {
  Cols c;
  long long* gout; int gcls; long long goff;
  const long long* key; int key_comp;
  const unsigned char* mask; long long ndom;  // ndom: the rows K1-K3 walk
  const int* blk_coff;     // [tiles] compact offset of each pre-mask tile
  int needs_j;             // compacted columns indexed through blk_coff
  int shift, B, nblk, ncls, logB;
  const int* within;       // [B][tiles] rows of bucket b in earlier tiles
  const long long* bases;  // [MAXCLS][B] padded start of bucket b per class
  const int* src;          // select-first: kept row i's pre-mask row
  const int* flags;        // [1] a prologue's kept rows disagree with n_valid,
                           // [2] the counts outgrow the outputs
  int check;               // which flags this tick can raise: bit 0 [1], bit 1 [2]
};

// Where domain row i of a column lives: a compacted column at its compact
// index, a raw (pre-mask) column at the pre-mask row.  The domain is the
// pre-mask rows, or the kept rows after select-first (SRC).
template <bool SRC>
__device__ __forceinline__ long long col_index(const Params& p, bool comp, long long i,
                                               int jl, long long coff) {
  if (SRC) return comp ? i : (long long)p.src[i];
  return comp ? coff + jl : i;
}

__device__ __forceinline__ unsigned lanemask_lt() {
  unsigned r; asm("mov.u32 %0, %%lanemask_lt;" : "=r"(r)); return r;
}

// A row's bucket: the top logB bits of mix64(key), none when B is 1.
__device__ __forceinline__ int bucket_of(long long key, const Params& p) {
  return p.logB ? (int)(mix64((unsigned long long)key) >> p.shift) : 0;
}

// Lanes of the warp holding the same bucket (b < 0: an invalid row): one
// ballot on validity and one per bucket bit (at most 9, B <= 512).
__device__ __forceinline__ unsigned peers_of(int b, int logB) {
  unsigned v = __ballot_sync(FULLM, b >= 0);
  unsigned m = (b >= 0) ? v : ~v;
#pragma unroll
  for (int bit = 0; bit < LOGBMAX; ++bit) {
    if (bit < logB) {
      unsigned bal = __ballot_sync(FULLM, (b >> bit) & 1);
      m &= ((b >> bit) & 1) ? bal : ~bal;
    }
  }
  return m;
}

// Tile-local compact index of every valid row (rows in warp-contiguous
// segments: row = tile_base + w*32*ITEMS + k*32 + lane).  Contains a
// __syncthreads.
__device__ __forceinline__ void tile_compact_index(const bool (&valid)[ITEMS], int (&jl)[ITEMS],
                                                   int* wv) {
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const unsigned lt = lanemask_lt();
  int run = 0;
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    unsigned bal = __ballot_sync(FULLM, valid[k]);
    jl[k] = run + __popc(bal & lt);
    run += __popc(bal);
  }
  if (lane == 0) wv[w] = run;
  __syncthreads();
  int pre = 0;
#pragma unroll
  for (int q = 0; q < W; ++q) pre += (q < w) ? wv[q] : 0;
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) jl[k] += pre;
}

// compacted-input prologue: kept rows per pre-mask tile
__global__ void __launch_bounds__(BT) k_tilecount(const unsigned char* __restrict__ mask,
                                                  long long ndom, int* __restrict__ cnt) {
  const long long base = (long long)blockIdx.x * TILE;
  int c = 0;
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    long long i = base + (long long)k * BT + threadIdx.x;
    c += (i < ndom && mask[i]) ? 1 : 0;
  }
  typedef cub::BlockReduce<int, BT> BR;
  __shared__ typename BR::TempStorage tmp;
  int s = BR(tmp).Sum(c);
  if (threadIdx.x == 0) cnt[blockIdx.x] = s;
}

// one-block exclusive scan in place over a mask prologue's n per-block kept
// rows; flags[3] gets their total and flags[1] is set when it is not expect
__global__ void __launch_bounds__(EXSCAN_BT) k_exscan1(int* __restrict__ a, int n,
                                                       int* flags, long long expect) {
  typedef cub::BlockScan<int, EXSCAN_BT> BS;
  __shared__ typename BS::TempStorage tmp;
  const int per = (n + EXSCAN_BT - 1) / EXSCAN_BT;
  const int st = threadIdx.x * per;
  int s = 0;
  for (int q = 0; q < per; ++q) { int i = st + q; if (i < n) s += a[i]; }
  int ex, tot;
  BS(tmp).ExclusiveSum(s, ex, tot);
  for (int q = 0; q < per; ++q) {
    int i = st + q;
    if (i < n) { int t = a[i]; a[i] = ex; ex += t; }
  }
  if (threadIdx.x == 0) {
    flags[3] = tot;
    if (tot != expect) flags[1] = 1;
  }
}

// select-first prologue: each thread reads 16 mask bytes, one uint4 load
// when the mask is 16-B aligned (vec)
__device__ __forceinline__ void load16(const unsigned char* __restrict__ m, long long r0,
                                       long long ndom, bool vec, unsigned char (&v)[16]) {
  if (vec && r0 + 16 <= ndom) {
    uint4 x = *reinterpret_cast<const uint4*>(m + r0);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(&x);
#pragma unroll
    for (int q = 0; q < 16; ++q) v[q] = b[q];
  } else {
#pragma unroll
    for (int q = 0; q < 16; ++q) v[q] = (r0 + q < ndom) ? m[r0 + q] : 0;
  }
}

// kept rows of each block's MSEL_BT * 16 mask rows
__global__ void __launch_bounds__(MSEL_BT) k_mcount(const unsigned char* __restrict__ mask,
                                                    long long ndom, bool vec, int* __restrict__ cnt) {
  const long long r0 = ((long long)blockIdx.x * MSEL_BT + threadIdx.x) * 16;
  unsigned char v[16];
  load16(mask, r0, ndom, vec, v);
  int c = 0;
#pragma unroll
  for (int q = 0; q < 16; ++q) c += v[q] != 0;
  typedef cub::BlockReduce<int, MSEL_BT> BR;
  __shared__ typename BR::TempStorage tmp;
  int s = BR(tmp).Sum(c);
  if (threadIdx.x == 0) cnt[blockIdx.x] = s;
}

// src[i] = the pre-mask row of kept row i, in row order (coff: each block's
// first kept row; at most cap rows written)
__global__ void __launch_bounds__(MSEL_BT) k_mselect(const unsigned char* __restrict__ mask,
                                                     long long ndom, bool vec,
                                                     const int* __restrict__ coff,
                                                     int* __restrict__ src, long long cap) {
  const long long r0 = ((long long)blockIdx.x * MSEL_BT + threadIdx.x) * 16;
  unsigned char v[16];
  load16(mask, r0, ndom, vec, v);
  int c = 0;
#pragma unroll
  for (int q = 0; q < 16; ++q) c += v[q] != 0;
  typedef cub::BlockScan<int, MSEL_BT> BS;
  __shared__ typename BS::TempStorage tmp;
  int ex;
  BS(tmp).ExclusiveSum(c, ex);
  long long o = (long long)coff[blockIdx.x] + ex;
#pragma unroll
  for (int q = 0; q < 16; ++q) {
    if (v[q]) {
      if (o < cap) src[o] = (int)(r0 + q);
      ++o;
    }
  }
}

// K1: per-tile bucket histogram, hist[b * nblk + tile]
template <bool SRC>
__global__ void __launch_bounds__(BT) k_hist(Params p, int* __restrict__ hist) {
  if ((p.check & 1) && p.flags[1]) return;  // a wrong count: the host raises
  __shared__ int whist[W * BMAX];  // [w * B + b]
  __shared__ int wv[W];
  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, blk = blockIdx.x;
  const int B = p.B;
  for (int x = tid; x < W * B; x += BT) whist[x] = 0;
  const long long seg = (long long)blk * TILE + (long long)w * 32 * ITEMS;
  bool valid[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    long long i = seg + k * 32 + lane;
    bool v = i < p.ndom;
    if (v && p.mask) v = p.mask[i] != 0;
    valid[k] = v;
  }
  int jl[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) jl[k] = 0;
  long long coff = 0;
  // either branch's barrier also ends the zeroing of whist
  if (!SRC && p.key_comp) {
    tile_compact_index(valid, jl, wv);
    coff = p.blk_coff[blk];
  } else {
    __syncthreads();
  }
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    if (valid[k]) {
      long long i = seg + k * 32 + lane;
      long long key = p.key[col_index<SRC>(p, p.key_comp, i, jl[k], coff)];
      atomicAdd(&whist[w * B + bucket_of(key, p)], 1);
    }
  }
  __syncthreads();
  for (int b = tid; b < B; b += BT) {
    int s = 0;
#pragma unroll
    for (int q = 0; q < W; ++q) s += whist[q * B + b];
    hist[(long long)b * p.nblk + blk] = s;
  }
}

// K2: block b exclusive-scans bucket b's row of hist in place (within[b][tile])
// and writes count[b].  The last block to finish (flags[0] counts the blocks)
// writes the counts into pinned host memory, and a mask prologue's kept rows
// (flags[3]) after them, sets flags[2] when the counts total more than the
// n_valid the outputs were sized for, and, one warp per class, writes the
// padded bucket starts (bases) and the class's padded total
// (host_out[B + class]).
__global__ void __launch_bounds__(SCAN_BT) k_scan(int* __restrict__ hist, int nblk, int B,
                                                  long long* counts, Rpa rpa, int ncls,
                                                  long long* __restrict__ bases,
                                                  long long* host_out, int* flags,
                                                  long long n_valid, bool prologue) {
  constexpr int IPT = 4, CH = SCAN_BT * IPT;
  typedef cub::BlockScan<int, SCAN_BT> BS;
  __shared__ typename BS::TempStorage tmp;
  __shared__ bool last;
  __shared__ long long scnt[BMAX];
  const int b = blockIdx.x, tid = threadIdx.x;
  int* row = hist + (long long)b * nblk;
  int running = 0;
  for (int base = 0; base < nblk; base += CH) {
    int v[IPT], ts = 0;
#pragma unroll
    for (int q = 0; q < IPT; ++q) {
      int i = base + tid * IPT + q;
      v[q] = i < nblk ? row[i] : 0;
      ts += v[q];
    }
    int ex, agg;
    BS(tmp).ExclusiveSum(ts, ex, agg);
    ex += running;
#pragma unroll
    for (int q = 0; q < IPT; ++q) {
      int i = base + tid * IPT + q;
      if (i < nblk) row[i] = ex;
      ex += v[q];
    }
    running += agg;
    __syncthreads();
  }
  if (tid == 0) {
    counts[b] = running;
    __threadfence();
    last = (atomicAdd((unsigned int*)flags, 1u) == gridDim.x - 1);
  }
  __syncthreads();
  if (!last) return;
  __threadfence();
  volatile long long* vc = counts;
  for (int bb = tid; bb < B; bb += SCAN_BT) {
    long long n = vc[bb];
    scnt[bb] = n;
    host_out[bb] = n;
  }
  __syncthreads();
  if (tid == 0) {
    long long total = 0;
    for (int bb = 0; bb < B; ++bb) total += scnt[bb];
    if (total > n_valid) flags[2] = 1;
    if (prologue) host_out[B + MAXCLS] = ((volatile int*)flags)[3];
  }
  const int w = tid >> 5, lane = tid & 31;
  if (w < ncls) {
    const long long r = rpa.v[w];
    long long v4[BPL], sacc = 0;
#pragma unroll
    for (int q = 0; q < BPL; ++q) {
      int bb = lane * BPL + q;
      long long n = bb < B ? scnt[bb] : 0;
      v4[q] = n ? ((n + r - 1) / r) * r : 0;
      sacc += v4[q];
    }
    long long incl = sacc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      long long n = __shfl_up_sync(FULLM, incl, o);
      if (lane >= o) incl += n;
    }
    long long ex = incl - sacc;
#pragma unroll
    for (int q = 0; q < BPL; ++q) {
      int bb = lane * BPL + q;
      if (bb < B) bases[w * B + bb] = ex;
      ex += v4[q];
    }
    if (lane == 31) host_out[B + w] = incl;
  }
}

// the unsigned type of an ES-byte row
template <int ES> struct VT;
template <> struct VT<1> { typedef unsigned char T; };
template <> struct VT<2> { typedef unsigned short T; };
template <> struct VT<4> { typedef unsigned int T; };
template <> struct VT<8> { typedef unsigned long long T; };

// K3's dynamic shared memory at B buckets: 37 KiB at B = 128, 76 KiB at 512
// (opted in past 48 KiB; sm_86 and sm_89 allow 99 KiB a block)
constexpr size_t scatter_smem(int B) {
  return TILE * 8                              // stage
         + (size_t)MAXCLS * B * 8              // gb
         + ((size_t)W * B + 2 * B + W + 4) * 4 // whist, tstart, tot, wv, misc
         + TILE * 4;                           // sidx (u16), sb (u16)
}
static_assert(scatter_smem(128) <= 48 * 1024, "k_scatter takes no opt-in through B = 128");
static_assert(scatter_smem(BMAX) <= 99 * 1024, "k_scatter at BMAX fits sm_86's opt-in");

// K3: stable rank, the tile staged in bucket order, coalesced bucket runs
template <bool SRC>
__global__ void __launch_bounds__(BT) k_scatter(Params p) {
  // a wrong count: the host raises
  if (((p.check & 1) && p.flags[1]) || ((p.check & 2) && p.flags[2])) return;
  extern __shared__ __align__(16) unsigned char smem[];
  const int B = p.B;
  unsigned long long* stage = (unsigned long long*)smem;            // TILE
  long long* gb = (long long*)(smem + TILE * 8);                    // MAXCLS*B
  int* whist = (int*)(gb + MAXCLS * B);                             // W*B
  int* tstart = whist + W * B;                                      // B
  int* tot = tstart + B;                                            // B
  int* wv = tot + B;                                                // W
  int* misc = wv + W;                                               // 4: [0] valid rows
  unsigned short* sidx = (unsigned short*)(misc + 4);               // TILE
  unsigned short* sb = sidx + TILE;                                 // TILE

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, blk = blockIdx.x;
  for (int x = tid; x < W * B; x += BT) whist[x] = 0;
  const long long tile_base = (long long)blk * TILE;
  const int segl = w * 32 * ITEMS;
  const long long seg = tile_base + segl;
  bool valid[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    long long i = seg + k * 32 + lane;
    bool v = i < p.ndom;
    if (v && p.mask) v = p.mask[i] != 0;
    valid[k] = v;
  }
  int jl[ITEMS];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) jl[k] = 0;
  long long coff = 0;
  if (!SRC && p.needs_j) {
    tile_compact_index(valid, jl, wv);
    coff = p.blk_coff[blk];
  } else {
    __syncthreads();
  }
  int bk[ITEMS];
  {
    long long kv[ITEMS];
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      long long i = seg + k * 32 + lane;
      kv[k] = valid[k] ? p.key[col_index<SRC>(p, p.key_comp, i, jl[k], coff)] : 0;
    }
#pragma unroll
    for (int k = 0; k < ITEMS; ++k)
      bk[k] = valid[k] ? bucket_of(kv[k], p) : -1;
  }
  // stable rank inside (warp, bucket): rows keep their source order
  int sk[ITEMS];
  {
    const unsigned lt = lanemask_lt();
#pragma unroll
    for (int k = 0; k < ITEMS; ++k) {
      unsigned peers = peers_of(bk[k], p.logB);
      int base = bk[k] >= 0 ? whist[w * B + bk[k]] : 0;
      sk[k] = base + __popc(peers & lt);
      __syncwarp();
      if (bk[k] >= 0 && (peers & lt) == 0) whist[w * B + bk[k]] = base + __popc(peers);
      __syncwarp();
    }
  }
  __syncthreads();
  // per bucket: each warp's start inside the tile's run of the bucket
  for (int b = tid; b < B; b += BT) {
    int run = 0;
#pragma unroll
    for (int q = 0; q < W; ++q) {
      int t = whist[q * B + b];
      whist[q * B + b] = run;
      run += t;
    }
    tot[b] = run;
  }
  __syncthreads();
  // the tile's bucket runs, in bucket order
  if (w == 0) {
    int v4[BPL], s = 0;
#pragma unroll
    for (int q = 0; q < BPL; ++q) {
      int b = lane * BPL + q;
      v4[q] = b < B ? tot[b] : 0;
      s += v4[q];
    }
    int incl = s;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int n = __shfl_up_sync(FULLM, incl, o);
      if (lane >= o) incl += n;
    }
    int ex = incl - s;
#pragma unroll
    for (int q = 0; q < BPL; ++q) {
      int b = lane * BPL + q;
      if (b < B) tstart[b] = ex;
      ex += v4[q];
    }
    if (lane == 31) misc[0] = incl;
  }
  __syncthreads();
  // slot s of bucket b lands on row gb[c][b] + s of class c's output
  for (int x = tid; x < p.ncls * B; x += BT) {
    int c = x / B, b = x - c * B;
    gb[c * B + b] = p.bases[c * B + b] + p.within[(long long)b * p.nblk + blk] - tstart[b];
  }
  const int tvalid = misc[0];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k)
    if (bk[k] >= 0) sk[k] += tstart[bk[k]] + whist[w * B + bk[k]];
#pragma unroll
  for (int k = 0; k < ITEMS; ++k) {
    if (bk[k] >= 0) {
      sb[sk[k]] = (unsigned short)bk[k];
      sidx[sk[k]] = (unsigned short)(segl + k * 32 + lane);
    }
  }
  __syncthreads();

  for (int c = 0; c < p.c.ncols; ++c) {
    const int rb = p.c.rb[c], cl = p.c.cls[c];
    const unsigned char* in = p.c.in[c];
    unsigned char* out = p.c.out[c];
    const bool comp = p.c.comp[c] != 0;
    const long long* gbc = gb + cl * B;
    if (rb == 8 || rb == 4 || rb == 2 || rb == 1) {
#define STAGE_CASE(ES)                                                                    \
  {                                                                                       \
    typedef VT<ES>::T T;                                                                  \
    _Pragma("unroll") for (int k = 0; k < ITEMS; ++k) if (bk[k] >= 0) {                  \
      long long i = col_index<SRC>(p, comp, seg + k * 32 + lane, jl[k], coff);            \
      ((T*)stage)[sk[k]] = ((const T*)in)[i];                                             \
    }                                                                                     \
    __syncthreads();                                                                      \
    _Pragma("unroll") for (int t = 0; t < ITEMS; ++t) {                                   \
      int s = t * BT + tid;                                                               \
      if (s < tvalid) ((T*)out)[gbc[sb[s]] + s] = ((T*)stage)[s];                         \
    }                                                                                     \
    __syncthreads();                                                                      \
  }
      if (rb == 8) STAGE_CASE(8)
      else if (rb == 4) STAGE_CASE(4)
      else if (rb == 2) STAGE_CASE(2)
      else STAGE_CASE(1)
#undef STAGE_CASE
    } else if (!(p.needs_j && comp)) {
      // wide or odd rows (25-B strings): the block copies its sorted rows
      // byte by byte, consecutive threads on consecutive output bytes
      const long long nbytes = (long long)tvalid * rb;
      for (long long e = tid; e < nbytes; e += BT) {
        int s = (int)(e / rb), q = (int)(e - (long long)s * rb);
        long long i = col_index<SRC>(p, comp, tile_base + sidx[s], 0, 0);
        out[(gbc[sb[s]] + s) * rb + q] = in[i * rb + q];
      }
    } else {
      // a wide compacted column read through the tile's compact offsets
#pragma unroll
      for (int k = 0; k < ITEMS; ++k) {
        if (bk[k] >= 0) {
          long long i = col_index<SRC>(p, comp, seg + k * 32 + lane, jl[k], coff);
          const unsigned char* in_row = in + i * rb;
          unsigned char* out_row = out + (gbc[bk[k]] + sk[k]) * rb;
          for (int q = 0; q < rb; ++q) out_row[q] = in_row[q];
        }
      }
    }
  }
  const long long* gbg = gb + p.gcls * B;
#pragma unroll
  for (int t = 0; t < ITEMS; ++t) {
    int s = t * BT + tid;
    if (s < tvalid) {
      long long i = tile_base + sidx[s];
      p.gout[gbg[sb[s]] + s] = p.goff + (SRC ? (long long)p.src[i] : i);
    }
  }
}

// lcm(rb, 512) / rb: the rows of a bucket's padding unit, so every bucket
// starts 512-B aligned (nvCOMP's input alignment, constants.COMPRESSION_ALIGNMENT)
long long rows_per_unit(long long rb) { return 512 / std::gcd(rb, 512LL); }

// Raise a kernel's dynamic shared memory limit past the default 48 KiB, once
// per (kernel, device) at the largest size asked for so far
template <typename K>
void opt_in_smem(K* kernel, size_t bytes, int device) {
  if (bytes <= 48 * 1024) return;
  static std::mutex mu;
  static std::map<std::pair<const void*, int>, size_t> done;
  std::lock_guard<std::mutex> lock(mu);
  size_t& have = done[{(const void*)kernel, device}];
  if (have >= bytes) return;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)bytes));
  have = bytes;
}

}  // namespace

// cols: the transfer set (1-D or 2-D contiguous CUDA), raw (ndom rows: the
// mask's, else the key's) or compacted (n_valid rows) per ``comp``; mask: the
// pre-mask mask (1 byte a row) or none; n_valid: the kept rows, or an upper
// bound when every column is raw and select_first is off; goff: the tick's
// first global row; host_out: pinned int64[B + 5], receives the B counts, each
// class's padded total and, in [B + 4], the kept rows a mask prologue found (-1
// when none ran).  Returns each column's output, then the int64
// gidx, every one sized for the worst case (n_valid + B*(rpa-1) rows); the
// caller narrows them to the padded totals once the call returns.
std::vector<at::Tensor> partition_sort(std::vector<at::Tensor> cols, std::vector<int64_t> comp,
                                       int64_t key_idx, c10::optional<at::Tensor> mask,
                                       int64_t n_valid, int64_t goff, int64_t B,
                                       bool select_first, at::Tensor host_out) {
  const int ncols = (int)cols.size();
  TORCH_CHECK(ncols >= 1 && ncols <= MAXC, "partition_sort: 1-", MAXC, " columns, got ", ncols);
  TORCH_CHECK((int64_t)comp.size() == ncols, "partition_sort: one comp flag per column");
  TORCH_CHECK(B >= 1 && B <= BMAX && (B & (B - 1)) == 0,
              "partition_sort: B must be a power of two in [1, ", BMAX, "], got ", B);
  TORCH_CHECK(key_idx >= 0 && key_idx < ncols, "partition_sort: key index out of range");
  const at::Tensor& key = cols[key_idx];
  TORCH_CHECK(key.scalar_type() == at::kLong && key.dim() == 1, "partition_sort: int64 1-D key");
  TORCH_CHECK(host_out.is_cpu() && host_out.is_pinned() &&
              host_out.scalar_type() == at::kLong && host_out.is_contiguous() &&
              host_out.numel() >= B + MAXCLS + 1,
              "partition_sort: host_out must be pinned int64 with B + 5 elements");
  const at::cuda::OptionalCUDAGuard guard(key.device());
  cudaStream_t st = at::cuda::getCurrentCUDAStream();
  const bool has_mask = mask.has_value();
  const long long ndom = has_mask ? mask->size(0) : key.size(0);
  if (has_mask) {
    TORCH_CHECK(mask->is_cuda() && mask->dim() == 1 && mask->is_contiguous() &&
                mask->element_size() == 1, "partition_sort: 1-D contiguous 1-byte mask");
  }
  TORCH_CHECK(n_valid >= 0 && n_valid <= ndom, "partition_sort: n_valid ", n_valid,
              " outside [0, ", ndom, "]");
  TORCH_CHECK(ndom < (1LL << 31), "partition_sort: a tick must hold fewer than 2^31 rows");

  Params p{};
  int rb_cls[MAXCLS];
  int ncls = 0;
  auto cls_of = [&](int rb) {
    for (int c = 0; c < ncls; ++c)
      if (rb_cls[c] == rb) return c;
    TORCH_CHECK(ncls < MAXCLS, "partition_sort: more than ", MAXCLS, " row widths");
    rb_cls[ncls] = rb;
    return ncls++;
  };
  bool any_comp = false;
  std::vector<at::Tensor> outs;
  outs.reserve(ncols + 1);
  for (int c = 0; c < ncols; ++c) {
    const at::Tensor& t = cols[c];
    TORCH_CHECK(t.is_cuda() && t.device() == key.device() && t.is_contiguous() &&
                (t.dim() == 1 || t.dim() == 2), "partition_sort: column ", c,
                " must be a contiguous 1-D or 2-D tensor on the key's device");
    TORCH_CHECK(!comp[c] || has_mask, "partition_sort: a compacted column needs the mask");
    TORCH_CHECK(t.size(0) == (comp[c] ? n_valid : ndom), "partition_sort: column ", c, " has ",
                t.size(0), " rows, expected ", (comp[c] ? n_valid : ndom));
    const int rb = (int)(t.element_size() * (t.dim() == 2 ? t.size(1) : 1));
    TORCH_CHECK(rb >= 1, "partition_sort: empty rows");
    const int cl = cls_of(rb);
    auto shape = t.sizes().vec();
    shape[0] = n_valid + B * (rows_per_unit(rb) - 1);
    auto o = at::empty(shape, t.options());
    p.c.in[c] = (const unsigned char*)t.data_ptr();
    p.c.out[c] = (unsigned char*)o.data_ptr();
    p.c.rb[c] = rb;
    p.c.comp[c] = (int)comp[c];
    p.c.cls[c] = cl;
    any_comp |= comp[c] != 0;
    outs.push_back(o);
  }
  p.c.ncols = ncols;
  p.gcls = cls_of(8);
  auto gout = at::empty({n_valid + B * (rows_per_unit(8) - 1)}, key.options());
  outs.push_back(gout);
  p.gout = (long long*)gout.data_ptr();
  p.goff = goff;
  p.key = (const long long*)key.data_ptr();
  p.key_comp = (int)comp[key_idx];
  p.mask = has_mask ? (const unsigned char*)mask->data_ptr() : nullptr;
  p.ndom = ndom;
  p.B = (int)B;
  p.ncls = ncls;
  int logB = 0;
  while ((1 << logB) < B) ++logB;
  p.logB = logB;
  p.shift = 64 - logB;  // unused when B is 1 (bucket_of)

  const bool use_sel = select_first && has_mask;
  p.needs_j = any_comp && !use_sel;
  auto iopt = key.options().dtype(at::kInt);
  auto flags = at::zeros({4}, iopt);
  int* flags_dev = flags.data_ptr<int>();  // the prologue and K2 write it
  p.flags = flags_dev;
  // A prologue can find another kept-row count; the counts can outgrow the
  // outputs only under a mask (a plain tick's n_valid is its rows).
  p.check = ((use_sel || p.needs_j) ? 1 : 0) | (has_mask ? 2 : 0);
  long long* host_dev = nullptr;
  C10_CUDA_CHECK(cudaHostGetDevicePointer((void**)&host_dev, host_out.data_ptr(), 0));
  // the kept rows a mask prologue finds; no kernel of an earlier call is
  // running, since that call waited for its K2
  ((long long*)host_out.data_ptr())[B + MAXCLS] = -1;
  at::Tensor blk_cnt, src;
  if (use_sel) {
    const long long mblk = std::max<long long>(1, (ndom + MSEL_BT * 16 - 1) / (MSEL_BT * 16));
    blk_cnt = at::empty({mblk}, iopt);
    src = at::empty({std::max<int64_t>(n_valid, 1)}, iopt);
    const bool vec = (reinterpret_cast<uintptr_t>(p.mask) & 15) == 0;
    k_mcount<<<(unsigned)mblk, MSEL_BT, 0, st>>>(p.mask, ndom, vec, blk_cnt.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    k_exscan1<<<1, EXSCAN_BT, 0, st>>>(blk_cnt.data_ptr<int>(), (int)mblk, flags_dev,
                                       n_valid);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    k_mselect<<<(unsigned)mblk, MSEL_BT, 0, st>>>(p.mask, ndom, vec, blk_cnt.data_ptr<int>(),
                                                  src.data_ptr<int>(), n_valid);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    // K1-K3 then walk the kept rows
    p.src = src.data_ptr<int>();
    p.mask = nullptr;
    p.ndom = n_valid;
  }
  const long long nblk = std::max<long long>(1, (p.ndom + TILE - 1) / TILE);
  p.nblk = (int)nblk;
  if (p.needs_j) {
    blk_cnt = at::empty({nblk}, iopt);
    k_tilecount<<<(unsigned)nblk, BT, 0, st>>>(p.mask, ndom, blk_cnt.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    k_exscan1<<<1, EXSCAN_BT, 0, st>>>(blk_cnt.data_ptr<int>(), (int)nblk, flags_dev,
                                       n_valid);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    p.blk_coff = blk_cnt.data_ptr<int>();
  }

  auto hist = at::empty({B * nblk}, iopt);
  auto small = at::empty({B + MAXCLS * B}, key.options());  // counts, then bases
  Rpa rpa;
  for (int c = 0; c < MAXCLS; ++c) rpa.v[c] = 1;
  for (int c = 0; c < ncls; ++c) rpa.v[c] = rows_per_unit(rb_cls[c]);

  if (use_sel) k_hist<true><<<(unsigned)nblk, BT, 0, st>>>(p, hist.data_ptr<int>());
  else k_hist<false><<<(unsigned)nblk, BT, 0, st>>>(p, hist.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  long long* counts = (long long*)small.data_ptr();
  long long* bases = counts + B;
  k_scan<<<(unsigned)B, SCAN_BT, 0, st>>>(hist.data_ptr<int>(), (int)nblk, (int)B, counts, rpa,
                                          ncls, bases, host_dev, flags_dev, n_valid,
                                          use_sel || p.needs_j);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  cudaEvent_t ev = tqpkernels::stream_event(key.get_device(), st);
  C10_CUDA_CHECK(cudaEventRecord(ev, st));

  p.within = hist.data_ptr<int>();
  p.bases = bases;
  const size_t smem = scatter_smem((int)B);
  if (use_sel) {
    opt_in_smem(k_scatter<true>, smem, key.get_device());
    k_scatter<true><<<(unsigned)nblk, BT, smem, st>>>(p);
  } else {
    opt_in_smem(k_scatter<false>, smem, key.get_device());
    k_scatter<false><<<(unsigned)nblk, BT, smem, st>>>(p);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  // the call's one host wait: K2's counts, while K3 runs
  tqpkernels::wait_event(ev);
  return outs;
}
