// The granule cache's core in C++ (``tqpcache.GranuleCore``): the per-granule,
// per-region and per-segment tables of the granule cache, its policies, and
// the run allocator of its slots, with the semantics of the reference model
// ``tests/page_cache/reference_cache.py`` (``test_native_core.py`` compares
// the two).  The Python wrapper (``IO/cache/native_cache.py``) keeps every CUDA object: the
// copies into homes are *batches* here, integers the wrapper maps to events.
// The wrapper tells the core which batches have fired (``mark_fired``,
// ``mark_fired_upto``); the core never asks.  Every call takes and returns
// numpy int64 arrays.
//
// Batches are created in order and the writer stream completes them in
// order, so where the reference model returns a list of events the core returns
// the newest live batch (``wait_batch``): waiting on it waits on them all.
//
// The slots are the blocks of the arena's cache region (``memory/``): slot
// ids are arena block indices, over the whole arena (``num_slots`` blocks),
// a slot's address is the arena's base plus id x the slot size, and only
// the region's blocks, ``[lo, num_slots)``, are ever free
// (``set_region``).  Slots are handed out from the top of the region down
// (the arena's operator allocations come from the bottom up).
// The region's lowest slot moves with the arena's boundary: a shrink
// (``plan_shrink``, ``commit_shrink``) evicts the surplus in victim order
// and relocates the residents of the slots it gives to free slots higher
// up; ``grow`` takes slots back below.  A slot id never changes meaning.
//
// Lending.  Attached to the arena (``set_arena`` with the arena's
// ``lend_api``, and the arena's ``set_lend_hook`` with ``lend_hooks``), the
// core is what an allocation that misses borrows from: the arena
// synchronises the device and calls ``tqpcache_lend`` (``lend``), which
// plans a shrink of the asked blocks with no floor, issues its copies (the
// dirty victims home, then the relocations, on the arena's lend stream,
// through ``arena_lend_copy``) and commits it, or changes nothing when it
// cannot give them all.  The lent blocks come back through the arena's
// ``arena_region_return`` (``reclaim_lent``) at every ``advance`` (a
// pipeline's start), and only there: an admission short of slots evicts as
// it would with nothing lent (taking blocks back mid-pipe would put this
// tick's granules, which cannot move, just above the blocks the next lend
// needs).
// Everything on that path is C++: no Python, no GIL.  The core is not
// thread-safe; the engine drives it from one thread, and no core call
// allocates arena memory, so a lend never runs inside one.
//
// No CUDA, no torch: the lend reaches the arena only through the function
// addresses it was given.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;
using i64 = int64_t;
using Arr = py::array_t<i64, py::array::c_style | py::array::forcecast>;

namespace {

constexpr int CLASS_SHIFT = 60;
constexpr int TB_SHIFT = 50;
constexpr int KEY_SHIFT = 40;
constexpr i64 NO_HINT = std::numeric_limits<i64>::max();
constexpr int DEAD = 0, CLEAN = 1, DIRTY = 2;
// Policy key layout (tests/page_cache/reference_policies.py).
constexpr int NU_BITS = 16, HINT_BITS = 16, TICK_BITS = 27;
constexpr i64 SEQ_LIMIT = i64(1) << 60;
constexpr i64 TICK_LIMIT = i64(1) << TICK_BITS;
constexpr i64 NU_MAX = (i64(1) << NU_BITS) - 1;
constexpr i64 HINT_MAX = (i64(1) << HINT_BITS) - 1;
constexpr i64 TICK_MAX = (i64(1) << TICK_BITS) - 1;
constexpr int COALESCE_THRESHOLD = 5;

enum Policy { LRU = 0, MRU = 1, DATAFLOW = 2 };

std::vector<i64> to_vec(const Arr& a) {
    auto r = a.unchecked<1>();
    std::vector<i64> v(r.shape(0));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) v[i] = r(i);
    return v;
}

Arr to_arr(const std::vector<i64>& v) {
    Arr a(v.size());
    auto r = a.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) r(i) = v[i];
    return a;
}

py::array_t<bool> to_bool(const std::vector<uint8_t>& v) {
    py::array_t<bool> a(v.size());
    auto r = a.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) r(i) = v[i] != 0;
    return a;
}

struct PendRun { i64 start, count, batch; };

// The arena's lend_api (memory/cuda_memory_allocator.cc): a lend's copies
// on its lend stream (0, or 1 when CUDA refused one), and the return of
// lent blocks (the blocks moved back below the boundary).
using CopyFn = int (*)(const int64_t* dst, const int64_t* src,
                       const int64_t* size, int64_t n);
using ReturnFn = int64_t (*)(int64_t max_blocks);

// One copy (dst, src, n), merged into the last when it continues it in
// both source and destination.
void push_copy(std::vector<i64>& dst, std::vector<i64>& src,
               std::vector<i64>& size, i64 d, i64 s, i64 n) {
    if (!size.empty() && dst.back() + size.back() == d &&
        src.back() + size.back() == s) {
        size.back() += n;
        return;
    }
    dst.push_back(d);
    src.push_back(s);
    size.push_back(n);
}

}  // namespace

class GranuleCore {
public:
    GranuleCore(i64 num_slots, i64 slot_size, const std::string& policy,
                int clean_percent, i64 gate_arm_b, i64 gate_window)
        : G(slot_size), num_slots_(num_slots), lo_(0),
          clean_percent_(clean_percent), gate_arm_(gate_arm_b),
          gate_window_(gate_window) {
        if (policy == "lru") policy_ = LRU;
        else if (policy == "mru") policy_ = MRU;
        else if (policy == "dataflow") policy_ = DATAFLOW;
        else throw std::invalid_argument("GranuleCore: unknown policy " + policy);
        if (clean_percent < 0 || clean_percent > 100)
            throw std::invalid_argument("clean_percent must be in [0, 100]");
        if (num_slots_ < 0) throw std::invalid_argument("num_slots must not be negative");
        owner_.assign(num_slots_, -1);
        used_.assign(num_slots_, 0);
        seed_runs();
    }

    // -- the slots' geometry --------------------------------------------------

    // The address of slot 0 (the arena's base).
    void set_base(i64 base) { base_ = base; }

    // -- lending -------------------------------------------------------------

    // The arena's lend_api: its arena_lend_copy and arena_region_return (0,
    // 0: detached; a lend then gives nothing and reclaim_lent takes
    // nothing back).
    void set_arena(i64 copy, i64 ret) {
        copy_fn_ = reinterpret_cast<CopyFn>(copy);
        return_fn_ = reinterpret_cast<ReturnFn>(ret);
    }

    i64 ctx_ptr() { return reinterpret_cast<i64>(this); }

    // The wrapper has issued batch ``b``'s copies (and holds its event).
    // A lend marks every batch fired after the arena's device sync, which
    // is right only for batches whose copies were issued before it.
    void note_issued(i64 b) {
        if (b < 0 || b >= (i64)fired_.size()) throw std::out_of_range("note_issued");
        issued_upto_ = std::max(issued_upto_, b);
    }

    // The lend hook's body (an allocation in the arena's operator space
    // missed; the arena synchronised the device, so every copy home has
    // ended and no kernel reads or writes a slot): empty the region's
    // lowest k slots, all of them or none.  A plan_shrink of k with no
    // floor, its copies issued on the arena's lend stream (the dirty
    // victims home, then the relocations, each merged with the previous
    // where it continues it), then committed: the arena synchronises the
    // stream before it hands the blocks out.  Returns k, or 0 with nothing
    // changed when a pinned or same-tick granule leaves fewer than k
    // slots below it, a batch's copies were never issued, or a copy is
    // refused.  Never allocates arena memory.
    i64 lend(i64 k) {
        if (k <= 0) return 0;
        ++ls_lends_;
        if (!copy_fn_ || k > num_slots_ - lo_ || newest_batch() > issued_upto_) {
            ++ls_refused_;
            return 0;
        }
        i64 newest = newest_batch();
        if (newest > fired_upto_) mark_fired_upto(newest);
        ShrinkPlan p = plan_shrink_impl(k, 0);
        if (p.given < k) {
            ++ls_refused_;
            return 0;
        }
        // The copies home in slot order (victim order runs a region's
        // granules backwards), so a run of slots merges into one copy.
        std::vector<size_t> by_slot(p.dsrc.size());
        std::iota(by_slot.begin(), by_slot.end(), 0);
        std::sort(by_slot.begin(), by_slot.end(),
                  [&](size_t x, size_t y) { return p.dsrc[x] < p.dsrc[y]; });
        std::vector<i64> dst, src, size;
        for (size_t i : by_slot)
            push_copy(dst, src, size, p.ddst[i], p.dsrc[i], p.dsize[i]);
        for (size_t i = 0; i < p.from.size(); ++i)
            push_copy(dst, src, size, slot_addr(p.to[i]), slot_addr(p.from[i]),
                      p.size[i]);
        if (!size.empty() &&
            copy_fn_(dst.data(), src.data(), size.data(), (i64)size.size()) != 0) {
            ++ls_refused_;
            return 0;
        }
        for (size_t i = 0; i < p.victims.size(); ++i) {
            i64 g = p.victims[i], c = p.cls[i];
            ls_evicted_[c]++;
            ls_logical_[c] += g_ledger[g];
            ls_wire_[c] += granule_bytes(g);
        }
        ls_relocated_ += (i64)p.from.size();
        ls_copies_ += (i64)size.size();
        ls_blocks_ += p.given;
        commit_shrink_impl(p.given, p.victims, p.from, p.to);
        return p.given;
    }

    // Lent blocks the operator space no longer uses back into the region
    // (the arena's arena_region_return: the free blocks just below the
    // boundary, up to every block still lent), as free slots below the
    // region's lowest.  Called at ``advance`` (a pipeline's start) only.
    // Returns the slots added.
    i64 reclaim_lent() {
        if (!return_fn_ || lo_ <= 0) return 0;
        i64 r = return_fn_(lo_);
        if (r <= 0) return 0;
        grow(r);
        ls_returned_ += r;
        return r;
    }

    // The lends since the last reset: (lends asked, refused, blocks given,
    // blocks taken back, granules relocated, copies issued, and per class
    // dead / clean / dirty the granules evicted, their ledger bytes and
    // their own bytes).
    py::tuple lend_stats(bool reset) {
        auto t = py::make_tuple(
            ls_lends_, ls_refused_, ls_blocks_, ls_returned_, ls_relocated_,
            ls_copies_,
            py::make_tuple(ls_evicted_[DEAD], ls_evicted_[CLEAN], ls_evicted_[DIRTY]),
            py::make_tuple(ls_logical_[DEAD], ls_logical_[CLEAN], ls_logical_[DIRTY]),
            py::make_tuple(ls_wire_[DEAD], ls_wire_[CLEAN], ls_wire_[DIRTY]));
        if (reset) {
            ls_lends_ = ls_refused_ = ls_blocks_ = ls_returned_ = ls_relocated_ =
                ls_copies_ = 0;
            for (int c = 0; c < 3; ++c) ls_evicted_[c] = ls_logical_[c] = ls_wire_[c] = 0;
        }
        return t;
    }

    // The region is slots [lo, num_slots): every one of them free.  Only
    // while no slot is in use or waits on a copy.
    void set_region(i64 lo) {
        if (lo < 0 || lo > num_slots_)
            throw std::invalid_argument("set_region: the region's lowest slot is out of range");
        reap();
        if (pending_n_ || free_n_ != num_slots_ - lo_)
            throw std::runtime_error("set_region: slots are in use");
        lo_ = lo;
        seed_runs();
    }

    i64 slot_addr(i64 slot) const {
        if (slot < lo_ || slot >= num_slots_)
            throw std::runtime_error("slot_addr: a slot outside the region");
        return base_ + slot * G;
    }

    Arr slot_addrs(const Arr& gids) {
        auto g = to_vec(gids);
        std::vector<i64> out(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            i64 s = g_slot[g[i]];
            if (s < 0) throw std::runtime_error("a granule packed into no slot");
            out[i] = slot_addr(s);
        }
        return to_arr(out);
    }

    Arr slot_of(const Arr& gids) {
        auto g = to_vec(gids);
        std::vector<i64> out(g.size());
        for (size_t i = 0; i < g.size(); ++i) out[i] = g_slot[g[i]];
        return to_arr(out);
    }

    i64 free_count() const { return free_n_ + pending_n_; }
    i64 num_slots() const { return num_slots_; }
    i64 region_lo() const { return lo_; }
    i64 granule_count() const { return (i64)g_slot.size(); }
    i64 region_count() const { return (i64)r_first.size(); }

    // -- batches --------------------------------------------------------------

    i64 fired_upto() const { return fired_upto_; }
    i64 newest_batch() const { return (i64)fired_.size() - 1; }

    void mark_fired(i64 b) {
        if (b < 0 || b >= (i64)fired_.size()) throw std::out_of_range("mark_fired");
        fired_[b] = 1;
        while (fired_upto_ + 1 < (i64)fired_.size() && fired_[fired_upto_ + 1]) ++fired_upto_;
    }

    void mark_fired_upto(i64 b) {
        if (b >= (i64)fired_.size()) throw std::out_of_range("mark_fired_upto");
        for (i64 x = fired_upto_ + 1; x <= b; ++x) fired_[x] = 1;
        if (b > fired_upto_) fired_upto_ = b;
        while (fired_upto_ + 1 < (i64)fired_.size() && fired_[fired_upto_ + 1]) ++fired_upto_;
    }

    Arr live_batches() const {
        std::vector<i64> out;
        for (i64 b = fired_upto_ + 1; b < (i64)fired_.size(); ++b)
            if (!fired_[b]) out.push_back(b);
        return to_arr(out);
    }

    // -- regions --------------------------------------------------------------

    // Returns (index, first, n, seg0, nseg, min_hint, seq).  ``ledger``, None
    // or one value per granule: the bytes the movement ledger counts as the
    // granule's decoded (logical) bytes; None counts its own bytes.
    py::tuple register_region(i64 nbytes, i64 home, const Arr& offsets,
                              const Arr& sizes, const Arr& streams,
                              py::object hints, i64 next_use_op,
                              i64 expected_reads, py::object ledger) {
        if (!(nbytes > 0 && nbytes < (i64(1) << KEY_SHIFT)))
            throw std::invalid_argument("register: region size out of range");
        auto off = to_vec(offsets);
        auto size = to_vec(sizes);
        std::vector<i64> st = to_vec(streams);
        if (st.size() == 1 && off.size() > 1) st.assign(off.size(), st[0]);
        if (off.empty() || size.size() != off.size() || st.size() != off.size())
            throw std::invalid_argument("register: records' arrays differ in length");
        size_t m = off.size();
        std::vector<size_t> ord(m);
        std::iota(ord.begin(), ord.end(), 0);
        bool sorted = true;
        for (size_t i = 1; i < m; ++i) if (off[i - 1] > off[i]) { sorted = false; break; }
        if (!sorted)
            std::stable_sort(ord.begin(), ord.end(),
                             [&](size_t a, size_t b) { return off[a] < off[b]; });
        std::vector<i64> seg_lo, seg_hi, seg_st;
        for (size_t j = 0; j < m; ++j) {
            size_t i = ord[j];
            i64 o = off[i], n = size[i], t = st[i];
            if (o < 0 || o + n > nbytes) throw std::invalid_argument("records overrun the region");
            if (!seg_st.empty() && t == seg_st.back()) {
                seg_hi.back() = std::max(seg_hi.back(), o + n);
            } else {
                if (!seg_hi.empty() && o < seg_hi.back())
                    throw std::invalid_argument("a region's stream runs overlap");
                seg_lo.push_back(o); seg_hi.push_back(o + n); seg_st.push_back(t);
            }
        }
        std::vector<i64> seg_hint(seg_st);
        if (!hints.is_none()) {
            auto h = to_vec(hints.cast<Arr>());
            for (size_t i = 0; i < seg_st.size(); ++i) seg_hint[i] = h[seg_st[i]];
        }
        i64 index = (i64)r_first.size();
        if (index >= (i64(1) << (63 - KEY_SHIFT))) throw std::runtime_error("too many regions");
        i64 n = (nbytes + G - 1) / G;
        std::vector<i64> led;
        if (!ledger.is_none()) {
            led = to_vec(ledger.cast<Arr>());
            if ((i64)led.size() != n)
                throw std::invalid_argument("register: one ledger value per granule");
        }
        i64 first = (i64)g_slot.size();
        i64 seq = seq_++;
        i64 min_hint = *std::min_element(seg_hint.begin(), seg_hint.end());
        // check_region
        if (!(seq >= 0 && seq < SEQ_LIMIT)) throw std::runtime_error("region order out of the key's range");
        if (policy_ == DATAFLOW) {
            if (!(next_use_op >= 0 && next_use_op <= NU_MAX))
                throw std::runtime_error("consumer position out of the key's range");
            i64 mx = *std::max_element(seg_hint.begin(), seg_hint.end());
            if (min_hint < 0 || mx > HINT_MAX) throw std::runtime_error("use_order_hint out of the key's range");
        }
        i64 seg0 = (i64)s_region.size();
        i64 nseg = (i64)seg_lo.size();
        i64 base = index << KEY_SHIFT;
        for (i64 j = 0; j < nseg; ++j) {
            s_region.push_back(index);
            s_stream.push_back(seg_st[j]);
            s_lo.push_back(seg_lo[j]);
            s_hi.push_back(seg_hi[j]);
            s_key.push_back(base + seg_hi[j]);
            s_hint.push_back(seg_hint[j]);
            s_reads.push_back(expected_reads);
            s_glo.push_back(first + seg_lo[j] / G);
            s_ghi.push_back(first + (seg_hi[j] - 1) / G);
        }
        r_first.push_back(first); r_n.push_back(n); r_nbytes.push_back(nbytes);
        r_home.push_back(home); r_seg0.push_back(seg0); r_nseg.push_back(nseg);
        r_nu.push_back(next_use_op); r_seq.push_back(seq); r_min_hint.push_back(min_hint);
        r_admitted.push_back(0); r_freed.push_back(0);
        // granules
        for (i64 k = 0; k < n; ++k) {
            g_slot.push_back(-1); g_clean.push_back(0); g_live.push_back(0);
            g_pins.push_back(0); g_nu.push_back(next_use_op); g_hint.push_back(0);
            g_tick.push_back(0); g_seq.push_back(seq); g_tb.push_back(first + n - 1 - k);
            g_region.push_back(index); g_cleaning.push_back(0); g_batch.push_back(-1);
            g_ledger.push_back(led.empty() ? std::min(G, nbytes - k * G) : led[k]);
            g_nlive.push_back(0);
        }
        if (expected_reads > 0) {
            std::vector<i64> low(n, NO_HINT);
            for (i64 j = 0; j < nseg; ++j)
                for (i64 k = s_glo[seg0 + j] - first; k <= s_ghi[seg0 + j] - first; ++k) {
                    low[k] = std::min(low[k], seg_hint[j]);
                    g_nlive[first + k]++;
                }
            for (i64 k = 0; k < n; ++k) {
                g_live[first + k] = low[k] != NO_HINT;
                g_hint[first + k] = low[k] == NO_HINT ? 0 : low[k];
            }
        }
        return py::make_tuple(index, first, n, seg0, nseg, min_hint, seq);
    }

    // (first, n, nbytes, admitted, freed, home)
    py::tuple region_fields(i64 ri) const {
        return py::make_tuple(r_first[ri], r_n[ri], r_nbytes[ri], r_admitted[ri],
                              (bool)r_freed[ri], r_home[ri]);
    }

    void drop(i64 ri) {
        if (r_freed[ri]) return;
        std::vector<i64> resident;
        for (i64 g = r_first[ri]; g < r_first[ri] + r_n[ri]; ++g)
            if (g_slot[g] >= 0) resident.push_back(g);
        for (i64 g : resident)
            if (g_pins[g] > 0) throw std::runtime_error("region freed with pinned granules");
        if (!resident.empty()) release(resident);
        for (i64 g = r_first[ri]; g < r_first[ri] + r_n[ri]; ++g) { g_live[g] = 0; g_nlive[g] = 0; }
        for (i64 j = r_seg0[ri]; j < r_seg0[ri] + r_nseg[ri]; ++j) s_reads[j] = 0;
        r_freed[ri] = 1;
    }

    // A new batch over the refused granules of each region; -1 when none.
    i64 note_home_copy(const Arr& regions) {
        auto rs = to_vec(regions);
        std::vector<i64> gids;
        for (i64 ri : rs)
            for (i64 g = r_first[ri] + r_admitted[ri]; g < r_first[ri] + r_n[ri]; ++g)
                gids.push_back(g);
        if (gids.empty()) return -1;
        return new_batch(gids);
    }

    // The newest live batch among the copies into the region's home, or -1.
    i64 home_wait_batch(i64 ri) const {
        i64 mx = -1;
        for (i64 g = r_first[ri]; g < r_first[ri] + r_n[ri]; ++g)
            if (g_batch[g] >= 0 && batch_live(g_batch[g])) mx = std::max(mx, g_batch[g]);
        return mx;
    }

    // -- positions and ticks ----------------------------------------------------

    // (released granules, their ledger bytes, their bytes): the dataflow
    // release at a pipeline start, after lent blocks no longer in use come
    // back.
    py::tuple advance(i64 node_position) {
        reclaim_lent();
        auto s = settle_reads_impl();
        cur_++;
        if (policy_ == DATAFLOW) {
            if (node_position >= 0 && node_position != cur_)
                throw std::runtime_error("GranuleCore.advance: node position " +
                                         std::to_string(node_position) +
                                         " but the counter is at " + std::to_string(cur_));
            maybe_dead_ = true;
            full_scan_ = true;      // every granule whose consumer has run
            s += release_dead_impl();
        }
        return s.tuple();
    }

    void reset_position() { cur_ = -1; }
    i64 position() const { return cur_; }
    i64 tick() const { return tick_; }

    py::tuple advance_tick() {
        auto s = settle_reads_impl();
        poll();
        tick_++;
        if (tick_ >= TICK_LIMIT) throw std::runtime_error("granule tick out of the key's range");
        return s.tuple();
    }

    // -- reads ----------------------------------------------------------------

    Arr granules(const Arr& regions, const Arr& lo, const Arr& hi) {
        auto ri = to_vec(regions); auto l = to_vec(lo); auto h = to_vec(hi);
        std::vector<i64> out;
        for (size_t i = 0; i < ri.size(); ++i) {
            if (h[i] <= l[i]) continue;
            i64 first = r_first[ri[i]];
            for (i64 g = first + l[i] / G; g <= first + (h[i] - 1) / G; ++g) out.push_back(g);
        }
        return to_arr(out);
    }

    // (run, offset, size, src, on_gpu, wait_batch)
    py::tuple pieces(const Arr& regions, const Arr& lo, const Arr& hi) {
        auto ri = to_vec(regions); auto l = to_vec(lo); auto h = to_vec(hi);
        std::vector<i64> run, off, size, src;
        std::vector<uint8_t> on_gpu;
        i64 wait = -1;
        for (size_t i = 0; i < ri.size(); ++i) {
            i64 k = ri[i], a = l[i], b = h[i];
            if (r_freed[k]) throw std::runtime_error("a cached region read after its free");
            if (!(0 <= a && a <= b && b <= r_nbytes[k])) throw std::runtime_error("pieces: range out of the region");
            if (b <= a) continue;
            i64 first = r_first[k], home = r_home[k];
            i64 prev = -2;
            bool fresh = true;
            for (i64 g = first + a / G; g <= first + (b - 1) / G; ++g) {
                i64 s = g_slot[g];
                i64 ga = (g - first) * G;
                i64 x0 = std::max(ga, a), x1 = std::min(ga + G, b);
                if (s < 0) {
                    i64 bt = g_batch[g];
                    if (bt >= 0 && batch_live(bt)) wait = std::max(wait, bt);
                }
                bool cont = !fresh && ((s < 0 && prev < 0) ||
                                       (s >= 0 && prev >= 0 && s == prev + 1));
                if (cont) {
                    size.back() += x1 - x0;
                } else {
                    run.push_back((i64)i); off.push_back(x0); size.push_back(x1 - x0);
                    src.push_back(s < 0 ? home + x0 : slot_addr(s) + (x0 - ga));
                    on_gpu.push_back(s >= 0);
                }
                fresh = false;
                prev = s;
            }
        }
        return py::make_tuple(to_arr(run), to_arr(off), to_arr(size), to_arr(src),
                              to_bool(on_gpu), wait);
    }

    void note_read(const Arr& regions, const Arr& lo, const Arr& hi) {
        note_read_impl(to_vec(regions), to_vec(lo), to_vec(hi));
    }

    void defer_read(const Arr& regions, const Arr& lo, const Arr& hi) {
        auto ri = to_vec(regions); auto l = to_vec(lo); auto h = to_vec(hi);
        d_ri.insert(d_ri.end(), ri.begin(), ri.end());
        d_lo.insert(d_lo.end(), l.begin(), l.end());
        d_hi.insert(d_hi.end(), h.begin(), h.end());
    }

    py::tuple settle_reads() { return settle_reads_impl().tuple(); }
    py::tuple release_dead() { return release_dead_impl().tuple(); }

    void pin(const Arr& gids) { for (i64 g : to_vec(gids)) g_pins[g]++; }

    void unpin(const Arr& gids) {
        for (i64 g : to_vec(gids)) {
            if (g_pins[g] <= 0) throw std::runtime_error("unpin of an unpinned granule");
            g_pins[g]--;
            if (dead(g)) dead_cand_.push_back(g);
        }
        maybe_dead_ = true;
    }

    Arr replace_consume_pins(const Arr& gids) {
        auto g = to_vec(gids);
        std::vector<i64> res;
        if (g.empty() && consume_.empty()) return to_arr(res);
        for (i64 x : g) if (g_slot[x] >= 0) res.push_back(x);
        for (i64 x : res) g_pins[x]++;
        if (!consume_.empty()) {
            for (i64 x : consume_) {
                if (g_pins[x] <= 0) throw std::runtime_error("unpin of an unpinned granule");
                g_pins[x]--;
                if (dead(x)) dead_cand_.push_back(x);
            }
            maybe_dead_ = true;
        }
        consume_ = res;
        return to_arr(res);
    }

    // -- victims ----------------------------------------------------------------

    Arr victim_order(i64 k) {
        poll();
        auto ev = evictable();
        std::vector<i64> cls;
        auto v = order(ev, k < 0 ? (i64)ev.size() : k, cls);
        return to_arr(v);
    }

    i64 evictable_count() { poll(); return (i64)evictable().size(); }

    // Victim info shared by reclaim and admit:
    // (slots, dead, clean, dirty, dirty_src, dirty_dst, dirty_size, dirty_batch,
    //  wait_batch, avoided_dead_b, avoided_clean_b, and the ledger bytes of
    //  the dead, clean and dirty ones)
    py::tuple reclaim(i64 nbytes) {
        poll();
        i64 k = (nbytes + G - 1) / G;
        std::vector<i64> cls;
        auto victims = order(evictable(), k, cls);
        Victims v = reclaim_victims(victims, cls);
        observe_reclaim((v.clean + v.dirty) * G);
        return victims_tuple(v, (i64)victims.size());
    }

    // -- admission --------------------------------------------------------------

    // (order, admitted, victims tuple as reclaim's)
    py::tuple admit(const Arr& regions) {
        poll();
        auto offered = to_vec(regions);
        if (policy_ == DATAFLOW)
            std::stable_sort(offered.begin(), offered.end(), [&](i64 a, i64 b) {
                return std::make_pair(r_nu[a], r_min_hint[a]) < std::make_pair(r_nu[b], r_min_hint[b]);
            });
        i64 free = free_count();
        i64 total = 0;
        for (i64 ri : offered) total += r_n[ri];
        std::vector<i64> victims, vcls;
        if (total > free) victims = order(evictable(), total - free, vcls);
        size_t vi = 0;
        std::vector<i64> counts;
        for (i64 ri : offered) {
            if (r_freed[ri] || r_admitted[ri] != 0)
                throw std::runtime_error("region offered twice or after its free");
            if (policy_ == DATAFLOW && !(r_nu[ri] > cur_))
                throw std::runtime_error("granule cache: region offered with its consumer already run");
            i64 f = std::min(free, r_n[ri]);
            free -= f;
            i64 a = f;
            i64 m = std::min(r_n[ri] - f, (i64)victims.size() - (i64)vi);
            if (m > 0) {
                i64 j = m;
                if (policy_ == DATAFLOW) {
                    j = 0;
                    for (i64 x = 0; x < m; ++x) {
                        i64 cand = r_first[ri] + f + x, vic = victims[vi + x];
                        if (!(g_nu[cand] > cur_))
                            throw std::runtime_error("granule cache: a granule offered with its consumer already run");
                        bool ok = vcls[vi + x] == DEAD || g_nu[cand] < g_nu[vic] ||
                                  (g_nu[cand] == g_nu[vic] && g_hint[cand] < g_hint[vic]);
                        if (!ok) break;
                        ++j;
                    }
                }
                vi += j;
                a += j;
            }
            counts.push_back(a);
        }
        std::vector<i64> taken(victims.begin(), victims.begin() + vi);
        std::vector<i64> taken_cls(vcls.begin(), vcls.begin() + vi);
        Victims v;
        if (vi) v = reclaim_victims(taken, taken_cls);
        if (total) observe_pressure((v.clean + v.dirty) * G);
        // As the reference model: the admission waits only on the copies still
        // on the slots it reuses, never on the victims' own copies home.
        i64 wait = -1;
        for (size_t i = 0; i < offered.size(); ++i) {
            i64 a = counts[i];
            if (!a) continue;
            i64 ri = offered[i];
            std::vector<i64> slots;
            i64 w = acquire_runs(a, slots);
            if ((i64)slots.size() != a) throw std::runtime_error("granule cache: admission ran out of slots");
            wait = std::max(wait, w);
            bool any_dead = false;
            for (i64 k = 0; k < a; ++k) {
                i64 g = r_first[ri] + k;
                g_slot[g] = slots[k];
                owner_[slots[k]] = g;
                g_clean[g] = 0;
                g_tick[g] = tick_;
                if (!g_live[g]) { any_dead = true; dead_cand_.push_back(g); }
            }
            if (any_dead) maybe_dead_ = true;
            r_admitted[ri] = a;
        }
        v.wait_batch = wait;
        return py::make_tuple(to_arr(offered), to_arr(counts), victims_tuple(v, (i64)vi));
    }

    // -- cleaning -----------------------------------------------------------------

    bool cleaning_open() const { return gate_arm_ <= 0 || gate_armed_; }

    // (gids, src, dst, size)
    py::tuple cleaning_targets() {
        std::vector<i64> pick;
        if (cleaning_open()) {
            poll();
            std::vector<i64> gids;
            for (i64 s = lo_; s < num_slots_; ++s) {
                i64 g = owner_[s];
                if (g >= 0 && g_pins[g] == 0 && !dead(g)) gids.push_back(g);
            }
            // owner order is slot order; the reference model's gids are in slot
            // order too (owner[owner >= 0]).
            i64 n = (i64)((gids.size() * clean_percent_ + 99) / 100);
            if (n > 0) {
                std::vector<uint8_t> want(gids.size());
                bool any = false;
                for (size_t i = 0; i < gids.size(); ++i) {
                    want[i] = !(g_clean[gids[i]] || g_cleaning[gids[i]]);
                    any |= want[i];
                }
                if (n < (i64)gids.size() && any) {
                    std::vector<std::pair<i64, i64>> kt(gids.size());
                    for (size_t i = 0; i < gids.size(); ++i)
                        kt[i] = {policy_key(gids[i], false), g_tb[gids[i]]};
                    std::vector<size_t> idx(gids.size());
                    std::iota(idx.begin(), idx.end(), 0);
                    std::nth_element(idx.begin(), idx.begin() + (n - 1), idx.end(),
                                     [&](size_t a, size_t b) { return kt[a] < kt[b]; });
                    std::vector<uint8_t> front(gids.size(), 0);
                    for (i64 i = 0; i < n; ++i) front[idx[i]] = 1;
                    for (size_t i = 0; i < gids.size(); ++i) want[i] = want[i] && front[i];
                }
                for (size_t i = 0; i < gids.size(); ++i) if (want[i]) pick.push_back(gids[i]);
            }
        }
        std::vector<i64> src, dst, size;
        home_copies(pick, src, dst, size);
        return py::make_tuple(to_arr(pick), to_arr(src), to_arr(dst), to_arr(size));
    }

    i64 commit_cleaning(const Arr& gids) {
        auto g = to_vec(gids);
        if (g.empty()) return -1;
        for (i64 x : g) {
            if (g_slot[x] < 0) throw std::runtime_error("cleaning a granule not in a slot");
            g_cleaning[x] = 1;
        }
        i64 b = new_batch(g);
        cleaning_.emplace_back(g, b);
        return b;
    }

    void poll() {
        if (cleaning_.empty()) return;
        std::vector<std::pair<std::vector<i64>, i64>> keep;
        for (auto& c : cleaning_) {
            if (!batch_live(c.second)) {
                for (i64 g : c.first) { g_clean[g] = 1; g_cleaning[g] = 0; }
            } else keep.push_back(c);
        }
        cleaning_ = keep;
    }

    // -- moving the region's lowest slot -------------------------------------------

    // A plan to give the region's lowest k slots, [lo, lo + k), to the arena,
    // with ``floor`` slots of the shrunk region left free:
    // (given, evicted, classes, dirty_src, dirty_dst, dirty_size, from, to, size).
    // The surplus of residents is evicted in victim order (``classes``: 0 dead,
    // 1 clean, 2 dirty; the dirty ones' copies home are ``dirty_*``); every
    // resident left in the given slots is relocated (slot ``from`` to slot
    // ``to``, ``size`` bytes), to the highest slots free above them, in slot
    // order (the lowest resident to the lowest of those slots, so a run of
    // residents moves as one run).  A pinned
    // or same-tick granule is neither evicted nor relocated: the plan stops
    // below the lowest slot holding one and gives fewer slots, or none
    // (``given`` 0) when it can give nothing.  Every copy home must have been
    // marked fired (the caller synchronized them): slots pending behind one
    // are then free.  Nothing changes until ``commit_shrink``.
    py::tuple plan_shrink(i64 k, i64 floor) {
        ShrinkPlan p = plan_shrink_impl(k, floor);
        return py::make_tuple(p.given, to_arr(p.victims), to_arr(p.cls), to_arr(p.dsrc),
                              to_arr(p.ddst), to_arr(p.dsize), to_arr(p.from),
                              to_arr(p.to), to_arr(p.size));
    }

    // The slots ``plan_shrink`` of every slot would give now, with ``floor``
    // slots of the shrunk region left free, once the copies home are marked
    // fired (a slot pending behind one counts as free; nothing changes):
    // the slots below the lowest one holding a pinned or same-tick granule,
    // at most the region less the floor and those granules.
    i64 lendable_slots(i64 floor) const {
        if (floor < 0) throw std::invalid_argument("lendable_slots: a negative floor");
        i64 total = num_slots_ - lo_;
        i64 give = total, fixed = 0;
        for (i64 s = lo_; s < num_slots_; ++s) {
            i64 g = owner_[s];
            if (g >= 0 && !movable(g)) {
                if (!fixed) give = s - lo_;
                ++fixed;
            }
        }
        return std::max<i64>(0, std::min(give, total - floor - fixed));
    }

    // Apply a plan of ``plan_shrink`` once its copies have ended (the dirty
    // victims' copies home and the relocations): drop the evicted granules,
    // move each relocated one's slot, and raise the region's lowest slot by
    // ``given``.  The plan must still hold: it is checked, not recomputed.
    void commit_shrink(i64 given, const Arr& evicted, const Arr& from, const Arr& to) {
        commit_shrink_impl(given, to_vec(evicted), to_vec(from), to_vec(to));
    }

    // Lower the region's lowest slot by k: slots [lo - k, lo) join the free
    // slots (the arena gave the blocks).
    void grow(i64 k) {
        if (k < 0 || k > lo_) throw std::invalid_argument("grow: out of the slots' range");
        if (!k) return;
        lo_ -= k;
        add_free(lo_, k);
    }

    // -- teardown ------------------------------------------------------------------

    // Every batch must have been marked fired (the wrapper synchronized them).
    void clear() {
        for (i64 b = fired_upto_ + 1; b < (i64)fired_.size(); ++b)
            if (!fired_[b]) throw std::runtime_error("clear: a copy home has not been marked fired");
        poll();
        d_ri.clear(); d_lo.clear(); d_hi.clear();
        for (i64 g : consume_) g_pins[g] = 0;
        consume_.clear();
        for (i64 ri = 0; ri < (i64)r_first.size(); ++ri) {
            if (r_freed[ri]) continue;
            for (i64 g = r_first[ri]; g < r_first[ri] + r_n[ri]; ++g) g_pins[g] = 0;
            drop(ri);
        }
        reap();
        g_slot.clear(); g_clean.clear(); g_live.clear(); g_pins.clear(); g_nu.clear();
        g_nlive.clear();
        g_hint.clear(); g_tick.clear(); g_seq.clear(); g_tb.clear(); g_region.clear();
        g_cleaning.clear(); g_batch.clear(); g_ledger.clear();
        r_first.clear(); r_n.clear(); r_nbytes.clear(); r_home.clear(); r_seg0.clear();
        r_nseg.clear(); r_nu.clear(); r_seq.clear(); r_min_hint.clear(); r_admitted.clear();
        r_freed.clear();
        s_region.clear(); s_stream.clear(); s_lo.clear(); s_hi.clear(); s_key.clear();
        s_hint.clear(); s_reads.clear(); s_glo.clear(); s_ghi.clear();
        std::fill(owner_.begin(), owner_.end(), -1);
        cleaning_.clear();
        fired_.clear();
        fired_upto_ = -1;
        issued_upto_ = -1;
        maybe_dead_ = false;
        dead_cand_.clear();
        cand_stamp_.clear();
        full_scan_ = false;
        gate_history_.clear();
        gate_bytes_ = 0;
        gate_armed_ = false;
        tick_ = 0;
        seq_ = 0;
    }

    // -- state, for tests and telemetry ------------------------------------------

    // 0 dirty, 1 clean, 2 evicted, 3 dead
    Arr state(const Arr& gids) {
        auto g = to_vec(gids);
        std::vector<i64> out(g.size());
        for (size_t i = 0; i < g.size(); ++i) {
            i64 x = g[i];
            out[i] = dead(x) ? 3 : g_slot[x] < 0 ? 2 : g_clean[x] ? 1 : 0;
        }
        return to_arr(out);
    }

    i64 resident_bytes() const {
        i64 n = 0;
        for (i64 s = lo_; s < num_slots_; ++s) n += owner_[s] >= 0;
        return n * G;
    }

    // (ledger bytes, own bytes) of gids: what the movement ledger counts for
    // them as logical_b and wire_b.
    py::tuple ledger_bytes(const Arr& gids) const {
        auto g = to_vec(gids);
        i64 lb = 0, wb = 0;
        for (i64 x : g) { lb += g_ledger[x]; wb += granule_bytes(x); }
        return py::make_tuple(lb, wb);
    }

    // Slots in use (handed out, not yet released), and the most at once
    // since ``reset_peak``: the perf report's residency section.
    i64 used_slots() const { return used_n_; }
    i64 peak_used_slots() const { return peak_used_; }
    void reset_peak() { peak_used_ = used_n_; }

    // Each granule's bytes (the last granule of a region may be short).
    Arr granule_sizes(const Arr& gids) const {
        auto g = to_vec(gids);
        std::vector<i64> out(g.size());
        for (size_t i = 0; i < g.size(); ++i) out[i] = granule_bytes(g[i]);
        return to_arr(out);
    }

    Arr owner() const { return to_arr(owner_); }
    Arr seg_reads() const { return to_arr(s_reads); }
    Arr pins_of(const Arr& gids) {
        auto g = to_vec(gids);
        std::vector<i64> out(g.size());
        for (size_t i = 0; i < g.size(); ++i) out[i] = g_pins[g[i]];
        return to_arr(out);
    }

private:
    struct Victims {
        i64 dead = 0, clean = 0, dirty = 0;
        std::vector<i64> dirty_src, dirty_dst, dirty_size;
        i64 dirty_batch = -1;
        i64 wait_batch = -1;
        i64 avoided_dead_b = 0, avoided_clean_b = 0;
        i64 dead_lb = 0, clean_lb = 0, dirty_lb = 0;
    };

    // Granules released with no copy: their count, ledger and own bytes.
    struct Released {
        i64 n = 0, logical_b = 0, wire_b = 0;
        Released& operator+=(const Released& o) {
            n += o.n; logical_b += o.logical_b; wire_b += o.wire_b;
            return *this;
        }
        py::tuple tuple() const { return py::make_tuple(n, logical_b, wire_b); }
    };

    py::tuple victims_tuple(const Victims& v, i64 slots) const {
        return py::make_tuple(slots, v.dead, v.clean, v.dirty, to_arr(v.dirty_src),
                              to_arr(v.dirty_dst), to_arr(v.dirty_size), v.dirty_batch,
                              v.wait_batch, v.avoided_dead_b, v.avoided_clean_b,
                              v.dead_lb, v.clean_lb, v.dirty_lb);
    }

    // A shrink's plan (``plan_shrink``).
    struct ShrinkPlan {
        i64 given = 0;
        std::vector<i64> victims, cls, dsrc, ddst, dsize, from, to, size;
    };

    ShrinkPlan plan_shrink_impl(i64 k, i64 floor) {
        if (k < 0 || floor < 0) throw std::invalid_argument("plan_shrink: a negative count");
        settle_copies("plan_shrink");
        ShrinkPlan p;
        i64 total = num_slots_ - lo_;
        i64 give = std::min(k, total);
        // Stop below the lowest slot that must stay where it is.
        for (i64 s = lo_; s < lo_ + give; ++s) {
            i64 g = owner_[s];
            if (g >= 0 && !movable(g)) { give = s - lo_; break; }
        }
        // Fit the residents into the shrunk region with the floor free.
        auto ev = evictable();
        i64 resident = 0;
        for (i64 s = lo_; s < num_slots_; ++s) resident += owner_[s] >= 0;
        i64 fit = total - floor - (resident - (i64)ev.size());
        give = std::max<i64>(0, std::min(give, fit));
        p.given = give;
        if (give == 0) return p;
        i64 keep = total - give - floor;          // residents the shrunk region holds
        i64 surplus = std::max<i64>(0, resident - keep);
        p.victims = order(ev, surplus, p.cls);
        std::vector<i64> dirty;
        for (size_t i = 0; i < p.victims.size(); ++i)
            if (p.cls[i] == DIRTY) dirty.push_back(p.victims[i]);
        home_copies(dirty, p.dsrc, p.ddst, p.dsize);
        // The residents left in the given slots, and as many of the highest
        // slots above them free after the evictions, in slot order.
        std::vector<uint8_t> gone(num_slots_ - lo_, 0);
        for (i64 g : p.victims) gone[g_slot[g] - lo_] = 1;
        std::vector<i64> movers;
        for (i64 s = lo_; s < lo_ + give; ++s)
            if (owner_[s] >= 0 && !gone[s - lo_]) movers.push_back(s);
        std::vector<i64> dest;
        for (i64 s = num_slots_ - 1; s >= lo_ + give && dest.size() < movers.size(); --s)
            if (owner_[s] < 0 || gone[s - lo_]) dest.push_back(s);
        if (dest.size() < movers.size())
            throw std::runtime_error("plan_shrink: no free slot to relocate to");
        std::reverse(dest.begin(), dest.end());
        for (size_t i = 0; i < movers.size(); ++i) {
            p.from.push_back(movers[i]);
            p.to.push_back(dest[i]);
            p.size.push_back(granule_bytes(owner_[movers[i]]));
        }
        return p;
    }

    void commit_shrink_impl(i64 given, const std::vector<i64>& ev,
                            const std::vector<i64>& f, const std::vector<i64>& t) {
        if (given < 0 || given > num_slots_ - lo_ || f.size() != t.size())
            throw std::invalid_argument("commit_shrink: a malformed plan");
        settle_copies("commit_shrink");
        for (i64 g : ev)
            if (g_slot[g] < 0 || !movable(g))
                throw std::runtime_error("commit_shrink: an evicted granule is not evictable");
        if (!ev.empty()) {
            i64 live = 0;
            for (i64 g : ev) live += !dead(g);
            observe_reclaim(live * G);
            release(ev);
        }
        for (size_t i = 0; i < f.size(); ++i) {
            i64 s = f[i], d = t[i], g = owner_[s];
            if (g < 0 || !movable(g))
                throw std::runtime_error("commit_shrink: a relocated slot holds no movable granule");
            if (d < lo_ + given || d >= num_slots_ || used_[d])
                throw std::runtime_error("commit_shrink: a relocation's slot is not free");
            owner_[d] = g; used_[d] = 1;
            owner_[s] = -1; used_[s] = 0;
            g_slot[g] = d;
        }
        for (i64 s = lo_; s < lo_ + given; ++s)
            if (used_[s]) throw std::runtime_error("commit_shrink: a given slot is still in use");
        lo_ += given;
        rebuild_runs();
    }

    bool batch_live(i64 b) const { return b > fired_upto_ && !fired_[b]; }

    // Neither pinned nor admitted this tick: its slot may change.
    bool movable(i64 g) const { return g_pins[g] == 0 && g_tick[g] < tick_; }

    // Before the region's lowest slot moves: every copy home has been
    // marked fired, so no slot is pending and every cleaning has ended.
    void settle_copies(const char* who) {
        for (i64 b = fired_upto_ + 1; b < (i64)fired_.size(); ++b)
            if (!fired_[b])
                throw std::runtime_error(std::string(who) +
                                         ": a copy home has not been marked fired");
        reap();
        poll();
    }

    i64 new_batch(const std::vector<i64>& gids) {
        i64 b = (i64)fired_.size();
        fired_.push_back(0);
        for (i64 g : gids) g_batch[g] = b;
        return b;
    }

    bool dead(i64 g) const {
        return !g_live[g] || (policy_ == DATAFLOW && g_nu[g] < cur_);
    }

    i64 policy_key(i64 g, bool is_dead) const {
        switch (policy_) {
        case LRU: return g_seq[g];
        case MRU: return (SEQ_LIMIT - 1) - g_seq[g];
        default: {
            i64 newest = TICK_MAX - g_tick[g];
            if (is_dead) return newest;
            i64 consumed_now = g_nu[g] == cur_ ? 1 : 0;
            return (consumed_now << (NU_BITS + HINT_BITS + TICK_BITS)) |
                   ((NU_MAX - g_nu[g]) << (HINT_BITS + TICK_BITS)) |
                   ((HINT_MAX - g_hint[g]) << TICK_BITS) | newest;
        }
        }
    }

    // The first k of gids in reclaim order, with their classes.
    std::vector<i64> order(const std::vector<i64>& gids, i64 k, std::vector<i64>& cls_out) {
        cls_out.clear();
        if (gids.empty() || k <= 0) return {};
        size_t n = gids.size();
        std::vector<i64> key(n), tb(n), cls(n);
        for (size_t i = 0; i < n; ++i) {
            i64 g = gids[i];
            bool d = dead(g);
            cls[i] = d ? DEAD : (g_clean[g] || g_cleaning[g]) ? CLEAN : DIRTY;
            key[i] = ((i64)(!d) << CLASS_SHIFT) | policy_key(g, d);
            tb[i] = ((i64)(cls[i] == DIRTY) << TB_SHIFT) | g_tb[g];
        }
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        auto less = [&](size_t a, size_t b) {
            return key[a] != key[b] ? key[a] < key[b] : tb[a] < tb[b];
        };
        size_t kk = std::min<size_t>(n, (size_t)k);
        if (kk < n) std::nth_element(idx.begin(), idx.begin() + kk, idx.end(), less);
        std::sort(idx.begin(), idx.begin() + kk, less);
        std::vector<i64> out(kk);
        cls_out.resize(kk);
        for (size_t i = 0; i < kk; ++i) { out[i] = gids[idx[i]]; cls_out[i] = cls[idx[i]]; }
        return out;
    }

    std::vector<i64> evictable() const {
        std::vector<i64> out;
        for (i64 s = lo_; s < num_slots_; ++s) {
            i64 g = owner_[s];
            if (g >= 0 && g_pins[g] == 0 && g_tick[g] < tick_) out.push_back(g);
        }
        return out;
    }

    void home_copies(const std::vector<i64>& gids, std::vector<i64>& src,
                     std::vector<i64>& dst, std::vector<i64>& size) const {
        src.resize(gids.size()); dst.resize(gids.size()); size.resize(gids.size());
        for (size_t i = 0; i < gids.size(); ++i) {
            i64 g = gids[i], ri = g_region[g];
            i64 off = (g - r_first[ri]) * G;
            src[i] = slot_addr(g_slot[g]);
            dst[i] = r_home[ri] + off;
            size[i] = std::min(G, r_nbytes[ri] - off);
        }
    }

    i64 granule_bytes(i64 g) const {
        i64 ri = g_region[g];
        i64 off = (g - r_first[ri]) * G;
        return std::min(G, r_nbytes[ri] - off);
    }

    // Unbind gids from their slots; each slot goes back behind its granule's
    // live copy home if any.  Returns the newest live batch waited on, or -1.
    i64 release(const std::vector<i64>& gids) {
        // group by batch (live batches pend; everything else frees)
        std::vector<std::pair<i64, i64>> bs;   // (batch or -1, slot)
        bs.reserve(gids.size());
        i64 wait = -1;
        for (i64 g : gids) {
            i64 b = g_batch[g];
            if (!(b >= 0 && batch_live(b))) b = -1; else wait = std::max(wait, b);
            bs.emplace_back(b, g_slot[g]);
        }
        std::sort(bs.begin(), bs.end());
        size_t i = 0;
        while (i < bs.size()) {
            size_t j = i;
            std::vector<i64> slots;
            while (j < bs.size() && bs[j].first == bs[i].first) slots.push_back(bs[j++].second);
            release_slots(slots, bs[i].first);
            i = j;
        }
        for (i64 g : gids) {
            owner_[g_slot[g]] = -1;
            g_slot[g] = -1;
            g_cleaning[g] = 0;
        }
        return wait;
    }

    Victims reclaim_victims(const std::vector<i64>& victims, const std::vector<i64>& cls) {
        Victims v;
        if (victims.empty()) return v;
        std::vector<i64> dirty, rest;
        for (size_t i = 0; i < victims.size(); ++i) {
            i64 g = victims[i];
            if (cls[i] == DIRTY) { dirty.push_back(g); v.dirty++; v.dirty_lb += g_ledger[g]; }
            else {
                rest.push_back(g);
                if (cls[i] == DEAD) {
                    v.dead++; v.avoided_dead_b += granule_bytes(g); v.dead_lb += g_ledger[g];
                } else {
                    v.clean++; v.avoided_clean_b += granule_bytes(g); v.clean_lb += g_ledger[g];
                }
            }
        }
        if (!dirty.empty()) {
            home_copies(dirty, v.dirty_src, v.dirty_dst, v.dirty_size);
            v.dirty_batch = new_batch(dirty);
            v.wait_batch = std::max(v.wait_batch, release(dirty));
        }
        if (!rest.empty()) v.wait_batch = std::max(v.wait_batch, release(rest));
        return v;
    }

    void note_read_impl(const std::vector<i64>& ri, const std::vector<i64>& lo,
                        const std::vector<i64>& hi) {
        std::vector<i64> ended;
        for (size_t i = 0; i < ri.size(); ++i) {
            i64 base = ri[i] << KEY_SHIFT;
            auto a = std::upper_bound(s_key.begin(), s_key.end(), base + lo[i]) - s_key.begin();
            auto b = std::upper_bound(s_key.begin(), s_key.end(), base + hi[i]) - s_key.begin();
            for (auto j = a; j < b; ++j) {
                i64 before = s_reads[j];
                i64 after = std::max<i64>(before - 1, 0);
                s_reads[j] = after;
                if (before > 0 && after == 0) ended.push_back(j);
            }
        }
        if (ended.empty()) return;
        maybe_dead_ = true;
        for (i64 j : ended)
            for (i64 g = s_glo[j]; g <= s_ghi[j]; ++g) g_nlive[g]--;
        if (policy_ != DATAFLOW) {
            // A granule is live while one of its segments has a read to
            // come (its count); the hint is a dataflow key only.
            for (i64 j : ended)
                for (i64 g = s_glo[j]; g <= s_ghi[j]; ++g)
                    if (g_nlive[g] == 0 && g_live[g]) {
                        g_live[g] = 0;
                        g_hint[g] = 0;
                        dead_cand_.push_back(g);
                    }
            return;
        }
        std::vector<i64> gids;
        for (i64 j : ended)
            for (i64 g = s_glo[j]; g <= s_ghi[j]; ++g) gids.push_back(g);
        std::sort(gids.begin(), gids.end());
        gids.erase(std::unique(gids.begin(), gids.end()), gids.end());
        refresh(gids);
        for (i64 g : gids) if (!g_live[g]) dead_cand_.push_back(g);
    }

    void refresh(const std::vector<i64>& gids) {
        for (i64 g : gids) {
            auto a = std::lower_bound(s_ghi.begin(), s_ghi.end(), g) - s_ghi.begin();
            auto b = std::upper_bound(s_glo.begin(), s_glo.end(), g) - s_glo.begin();
            bool live = false;
            i64 low = NO_HINT;
            for (auto j = a; j < b; ++j) {
                if (s_reads[j] > 0) { live = true; low = std::min(low, s_hint[j]); }
            }
            g_live[g] = live;
            g_hint[g] = live ? low : 0;
        }
    }

    Released settle_reads_impl() {
        if (d_ri.empty()) return {};
        std::vector<i64> ri, lo, hi;
        ri.swap(d_ri); lo.swap(d_lo); hi.swap(d_hi);
        note_read_impl(ri, lo, hi);
        return release_dead_impl();
    }

    // The dead granules a release may free: the candidates (granules that
    // died, were unpinned or were admitted dead since the last release, and
    // the dead ones it kept, pinned or admitted this tick), deduplicated by a
    // stamp per granule; every slot of the region after a dataflow advance
    // (full_scan_).  The ones still pinned or admitted this tick stay
    // candidates.  release() orders the slots it frees itself.
    Released release_dead_impl() {
        if (!maybe_dead_) return {};
        std::vector<i64> gids, keep;
        bool all_free = true;
        auto visit = [&](i64 g) {
            if (g_pins[g] == 0 && g_tick[g] < tick_) gids.push_back(g);
            else { all_free = false; keep.push_back(g); }
        };
        if (full_scan_) {
            for (i64 s = lo_; s < num_slots_; ++s) {
                i64 g = owner_[s];
                if (g >= 0 && dead(g)) visit(g);
            }
            full_scan_ = false;
        } else {
            if (cand_stamp_.size() < g_slot.size()) cand_stamp_.resize(g_slot.size(), 0);
            const i64 stamp = ++stamp_;
            for (i64 g : dead_cand_) {
                if (cand_stamp_[g] == stamp) continue;
                cand_stamp_[g] = stamp;
                i64 s = g_slot[g];
                if (s >= lo_ && owner_[s] == g && dead(g)) visit(g);
            }
        }
        dead_cand_.swap(keep);
        maybe_dead_ = !all_free;
        if (gids.empty()) return {};
        poll();
        Released r;
        r.n = (i64)gids.size();
        for (i64 g : gids) { r.logical_b += g_ledger[g]; r.wire_b += granule_bytes(g); }
        release(gids);
        return r;
    }

    // -- the cleaning gate
    void observe_pressure(i64 live_b) {
        if (gate_arm_ <= 0) return;
        if ((i64)gate_history_.size() >= gate_window_) {
            gate_bytes_ -= gate_history_.front();
            gate_history_.pop_front();
        }
        gate_history_.push_back(live_b);
        gate_bytes_ += live_b;
        if (!gate_armed_ && gate_bytes_ >= gate_arm_) gate_armed_ = true;
        else if (gate_armed_ && gate_bytes_ == 0) gate_armed_ = false;
    }

    void observe_reclaim(i64 live_b) {
        if (gate_arm_ <= 0 || !live_b) return;
        if (gate_history_.empty()) gate_history_.push_back(0);
        gate_history_.back() += live_b;
        gate_bytes_ += live_b;
        if (!gate_armed_ && gate_bytes_ >= gate_arm_) gate_armed_ = true;
    }

    // -- the slots' run allocator
    void seed_runs() {
        runs_.clear();
        if (num_slots_ > lo_) runs_.emplace_back(lo_, num_slots_ - lo_);
        free_n_ = num_slots_ - lo_;
        frees_ = 0;
    }

    // The free runs again from the slots: every unused slot of the region
    // (none is pending: ``settle_copies``).
    void rebuild_runs() {
        runs_.clear();
        free_n_ = 0;
        frees_ = 0;
        for (i64 s = lo_; s < num_slots_; ++s) {
            if (used_[s]) continue;
            if (!runs_.empty() && runs_.back().first + runs_.back().second == s)
                runs_.back().second++;
            else runs_.emplace_back(s, 1);
            free_n_++;
        }
    }

    void coalesce() {
        frees_ = 0;
        if (runs_.size() < 2) return;
        std::stable_sort(runs_.begin(), runs_.end());
        std::vector<std::pair<i64, i64>> out;
        for (auto& r : runs_) {
            if (!out.empty() && out.back().first + out.back().second == r.first)
                out.back().second += r.second;
            else out.push_back(r);
        }
        runs_ = out;
    }

    void add_free(i64 start, i64 count) {
        runs_.emplace_back(start, count);
        free_n_ += count;
        if (++frees_ > COALESCE_THRESHOLD) coalesce();
    }

    void reap() {
        if (!pending_n_) return;
        std::deque<PendRun> keep;
        i64 freed = 0;
        for (auto& p : pending_) {
            if (!batch_live(p.batch)) {
                add_free(p.start, p.count);
                freed += p.count;
            } else keep.push_back(p);
        }
        pending_ = keep;
        pending_n_ -= freed;
    }

    // The highest free run that holds count slots gives its top count.
    bool top_fit(i64 count, std::pair<i64, i64>& got) {
        size_t best = runs_.size();
        for (size_t i = 0; i < runs_.size(); ++i)
            if (runs_[i].second >= count && (best == runs_.size() || runs_[i].first > runs_[best].first))
                best = i;
        if (best == runs_.size()) return false;
        auto [s, n] = runs_[best];
        if (n == count) runs_.erase(runs_.begin() + best);
        else runs_[best] = {s, n - count};
        free_n_ -= count;
        got = {s + n - count, count};
        return true;
    }

    // count free slots: the top of one run that holds them if there is one,
    // else the top runs, highest first (each run's slots in ascending order).
    void take_from_runs(i64 count, std::vector<std::pair<i64, i64>>& taken) {
        std::pair<i64, i64> got;
        if (top_fit(count, got) || (frees_ && (coalesce(), top_fit(count, got)))) {
            taken.push_back(got);
            return;
        }
        coalesce();
        i64 left = count;
        while (left) {
            auto [s, n] = runs_.back();
            i64 k = std::min(n, left);
            if (k == n) runs_.pop_back();
            else runs_.back() = {s, n - k};
            taken.emplace_back(s + n - k, k);
            left -= k;
        }
        free_n_ -= count;
    }

    // count slots into ``slots`` (run by run); returns the newest live batch
    // among the pending slots taken, or -1.
    i64 acquire_runs(i64 count, std::vector<i64>& slots) {
        if (count <= 0) return -1;
        reap();
        if (free_n_ + pending_n_ < count) return -1;
        i64 from_free = std::min(count, free_n_);
        std::vector<std::pair<i64, i64>> runs;
        if (from_free) take_from_runs(from_free, runs);
        i64 wait = -1;
        i64 left = count - from_free;
        while (left) {
            PendRun p = pending_.front();
            pending_.pop_front();
            i64 k = std::min(p.count, left);
            if (k < p.count) pending_.push_front({p.start + k, p.count - k, p.batch});
            pending_n_ -= k;
            runs.emplace_back(p.start, k);
            wait = std::max(wait, p.batch);
            left -= k;
        }
        for (auto [s, n] : runs) {
            for (i64 x = s; x < s + n; ++x) {
                if (used_[x]) throw std::runtime_error("slots handed out twice");
                used_[x] = 1;
                slots.push_back(x);
            }
            used_n_ += n;
        }
        peak_used_ = std::max(peak_used_, used_n_);
        return wait;
    }

    void release_slots(std::vector<i64>& slots, i64 batch) {
        if (slots.empty()) return;
        std::sort(slots.begin(), slots.end());
        for (size_t i = 0; i < slots.size(); ++i) {
            if (!used_[slots[i]] || (i && slots[i] == slots[i - 1]))
                throw std::runtime_error("release_slots: a slot is not allocated");
        }
        std::vector<std::pair<i64, i64>> runs;
        for (size_t i = 0; i < slots.size(); ++i) {
            i64 s = slots[i];
            used_[s] = 0;
            used_n_--;
            if (!runs.empty() && runs.back().first + runs.back().second == s)
                runs.back().second++;
            else runs.emplace_back(s, 1);
        }
        if (batch >= 0 && batch_live(batch)) {
            for (auto [s, n] : runs) pending_.push_back({s, n, batch});
            pending_n_ += (i64)slots.size();
        } else {
            for (auto r : runs) runs_.push_back(r);
            free_n_ += (i64)slots.size();
            frees_ += (i64)runs.size();
            if (frees_ > COALESCE_THRESHOLD) coalesce();
        }
    }

    // -- state
    const i64 G;
    i64 num_slots_, lo_;
    i64 base_ = 0;
    Policy policy_;
    int clean_percent_;
    i64 gate_arm_, gate_window_;
    std::deque<i64> gate_history_;
    i64 gate_bytes_ = 0;
    bool gate_armed_ = false;

    std::vector<i64> g_slot, g_pins, g_nu, g_hint, g_tick, g_seq, g_tb, g_region, g_batch,
        g_ledger;
    std::vector<uint8_t> g_clean, g_live, g_cleaning;
    std::vector<i64> r_first, r_n, r_nbytes, r_home, r_seg0, r_nseg, r_nu, r_seq, r_min_hint,
        r_admitted;
    std::vector<uint8_t> r_freed;
    std::vector<i64> s_region, s_stream, s_lo, s_hi, s_key, s_hint, s_reads, s_glo, s_ghi;
    std::vector<i64> owner_;
    std::vector<i64> consume_;
    std::vector<i64> d_ri, d_lo, d_hi;
    std::vector<std::pair<std::vector<i64>, i64>> cleaning_;
    std::vector<uint8_t> fired_;
    i64 fired_upto_ = -1;
    bool maybe_dead_ = false;
    // release_dead_impl's candidates (granules that may be dead and
    // resident), its stamps, and whether its next run scans every slot.
    std::vector<i64> dead_cand_, cand_stamp_;
    i64 stamp_ = 0;
    bool full_scan_ = false;
    // Per granule, its segments with a read still to come.
    std::vector<i64> g_nlive;
    i64 tick_ = 0, seq_ = 0, cur_ = -1;

    // the slots
    std::vector<uint8_t> used_;
    i64 used_n_ = 0, peak_used_ = 0;
    std::vector<std::pair<i64, i64>> runs_;
    i64 free_n_ = 0, frees_ = 0;
    std::deque<PendRun> pending_;
    i64 pending_n_ = 0;

    // lending
    CopyFn copy_fn_ = nullptr;
    ReturnFn return_fn_ = nullptr;
    i64 issued_upto_ = -1;        // newest batch whose copies were issued
    i64 ls_lends_ = 0, ls_refused_ = 0, ls_blocks_ = 0, ls_returned_ = 0,
        ls_relocated_ = 0, ls_copies_ = 0;
    i64 ls_evicted_[3] = {0, 0, 0}, ls_logical_[3] = {0, 0, 0}, ls_wire_[3] = {0, 0, 0};
};

// The arena's lend hooks (memory/cuda_memory_allocator.cc, set_lend_hook):
// ``ctx`` is a GranuleCore (``ctx_ptr``).  Neither throws into the
// allocator: a failure gives nothing.
extern "C" int64_t tqpcache_lend(void* ctx, int64_t blocks) noexcept {
    try {
        return static_cast<GranuleCore*>(ctx)->lend(blocks);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[lend] the granule cache's lend failed: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[lend] the granule cache's lend failed\n");
    }
    return 0;
}

extern "C" int64_t tqpcache_lendable(void* ctx) noexcept {
    try {
        return static_cast<GranuleCore*>(ctx)->lendable_slots(0);
    } catch (...) {
        return 0;
    }
}

// Byte ranges [off, off + size) cut at multiples of G: per part its range
// index, granule, offset and size (``region_producer._split_at_granules``).
static py::tuple split_at_granules(const Arr& off_a, const Arr& size_a, i64 G) {
    auto off = to_vec(off_a);
    auto size = to_vec(size_a);
    std::vector<i64> idx, k, lo, n;
    for (size_t i = 0; i < off.size(); ++i) {
        if (size[i] <= 0) continue;
        i64 a = off[i], b = off[i] + size[i];
        for (i64 g = a / G; g <= (b - 1) / G; ++g) {
            i64 x0 = std::max(a, g * G), x1 = std::min(b, (g + 1) * G);
            idx.push_back((i64)i); k.push_back(g); lo.push_back(x0); n.push_back(x1 - x0);
        }
    }
    return py::make_tuple(to_arr(idx), to_arr(k), to_arr(lo), to_arr(n));
}

PYBIND11_MODULE(tqpcache, m) {
    m.doc() = "The granule cache's core (IO/cache/granule_core.cc).";
    m.def("split_at_granules", &split_at_granules, py::arg("off"), py::arg("size"),
          py::arg("G"));
    m.def("lend_hooks",
          []() {
              return py::make_tuple(reinterpret_cast<i64>(&tqpcache_lend),
                                    reinterpret_cast<i64>(&tqpcache_lendable));
          },
          "(tqpcache_lend, tqpcache_lendable): the addresses of the arena's "
          "lend hooks (ctx: a GranuleCore's ctx_ptr()), for "
          "TQPlib.tqpmemory.set_lend_hook.");
    py::class_<GranuleCore>(m, "GranuleCore")
        .def(py::init<i64, i64, const std::string&, int, i64, i64>(),
             py::arg("num_slots"), py::arg("slot_size"), py::arg("policy"), py::arg("clean_percent"), py::arg("gate_arm_b"),
             py::arg("gate_window"))
        .def("set_base", &GranuleCore::set_base)
        .def("set_arena", &GranuleCore::set_arena, py::arg("copy"), py::arg("ret"))
        .def("ctx_ptr", &GranuleCore::ctx_ptr)
        .def("note_issued", &GranuleCore::note_issued)
        .def("reclaim_lent", &GranuleCore::reclaim_lent)
        .def("lend_stats", &GranuleCore::lend_stats, py::arg("reset") = false)
        .def("set_region", &GranuleCore::set_region)
        .def("slot_addrs", &GranuleCore::slot_addrs)
        .def("slot_of", &GranuleCore::slot_of)
        .def("free_count", &GranuleCore::free_count)
        .def("num_slots", &GranuleCore::num_slots)
        .def("region_lo", &GranuleCore::region_lo)
        .def("granule_count", &GranuleCore::granule_count)
        .def("region_count", &GranuleCore::region_count)
        .def("fired_upto", &GranuleCore::fired_upto)
        .def("newest_batch", &GranuleCore::newest_batch)
        .def("mark_fired", &GranuleCore::mark_fired)
        .def("mark_fired_upto", &GranuleCore::mark_fired_upto)
        .def("live_batches", &GranuleCore::live_batches)
        .def("register", &GranuleCore::register_region, py::arg("nbytes"),
             py::arg("home"), py::arg("offsets"), py::arg("sizes"), py::arg("streams"),
             py::arg("hints"), py::arg("next_use_op"), py::arg("expected_reads"),
             py::arg("ledger") = py::none())
        .def("region_fields", &GranuleCore::region_fields)
        .def("drop", &GranuleCore::drop)
        .def("note_home_copy", &GranuleCore::note_home_copy)
        .def("home_wait_batch", &GranuleCore::home_wait_batch)
        .def("advance", &GranuleCore::advance, py::arg("node_position"))
        .def("reset_position", &GranuleCore::reset_position)
        .def("position", &GranuleCore::position)
        .def("tick", &GranuleCore::tick)
        .def("advance_tick", &GranuleCore::advance_tick)
        .def("granules", &GranuleCore::granules)
        .def("pieces", &GranuleCore::pieces)
        .def("note_read", &GranuleCore::note_read)
        .def("defer_read", &GranuleCore::defer_read)
        .def("settle_reads", &GranuleCore::settle_reads)
        .def("release_dead", &GranuleCore::release_dead)
        .def("pin", &GranuleCore::pin)
        .def("unpin", &GranuleCore::unpin)
        .def("replace_consume_pins", &GranuleCore::replace_consume_pins)
        .def("victim_order", &GranuleCore::victim_order)
        .def("evictable_count", &GranuleCore::evictable_count)
        .def("reclaim", &GranuleCore::reclaim)
        .def("admit", &GranuleCore::admit)
        .def("cleaning_open", &GranuleCore::cleaning_open)
        .def("cleaning_targets", &GranuleCore::cleaning_targets)
        .def("commit_cleaning", &GranuleCore::commit_cleaning)
        .def("poll", &GranuleCore::poll)
        .def("plan_shrink", &GranuleCore::plan_shrink, py::arg("k"), py::arg("floor"))
        .def("lendable_slots", &GranuleCore::lendable_slots, py::arg("floor"))
        .def("commit_shrink", &GranuleCore::commit_shrink, py::arg("given"),
             py::arg("evicted"), py::arg("from"), py::arg("to"))
        .def("grow", &GranuleCore::grow)
        .def("clear", &GranuleCore::clear)
        .def("state", &GranuleCore::state)
        .def("resident_bytes", &GranuleCore::resident_bytes)
        .def("granule_sizes", &GranuleCore::granule_sizes)
        .def("ledger_bytes", &GranuleCore::ledger_bytes)
        .def("used_slots", &GranuleCore::used_slots)
        .def("peak_used_slots", &GranuleCore::peak_used_slots)
        .def("reset_peak", &GranuleCore::reset_peak)
        .def("owner", &GranuleCore::owner)
        .def("seg_reads", &GranuleCore::seg_reads)
        .def("pins_of", &GranuleCore::pins_of);
}
