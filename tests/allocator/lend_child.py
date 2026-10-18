"""One lending check, in a process of its own (tests/allocator/test_arena_lend.py).

The allocator can be installed once per process and the pytest process's
arena is 22 GB, so each check runs here on a small arena of 1 MiB blocks:
install the pluggable allocator as main.py does, give the arena's top
blocks to a cache region, build a granule cache over it and attach it
(``NativeGranuleCache.attach``), fill the region with granules, then
allocate past what the operator space holds and print one JSON line of
what happened.

    python3 tests/allocator/lend_child.py <case>

The arena cases run on a 64 MiB arena whose top 32 MiB are the region
(blocks 32..63) and whose operator space is blocks 0..31; region A (16
granules, older) takes slots 48..63 and region B (16 granules, newer)
slots 32..47 (the core hands out the top slots first), both dirty.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.cuda.memory import (CUDAPluggableAllocator,  # noqa: E402
                               change_current_allocator)

import TQPlib.tqpmemory as cmp  # noqa: E402

MIB = 1024 ** 2
SHORT = 1000               # each test region's last granule is short by it


def install(arena_b: int) -> None:
    change_current_allocator(CUDAPluggableAllocator(
        path_to_so_file="libcustom_allocator.so",
        alloc_fn_name="pa_cuda_malloc", free_fn_name="pa_cuda_free"))
    cmp.init(arena_b, MIB)


def device(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, device="cuda")


def try_alloc(nbytes: int):
    try:
        return device(nbytes), None
    except RuntimeError as exc:
        return None, str(exc).splitlines()[0]


def offset(t) -> int:
    """MiB from the arena's start."""
    return (t.data_ptr() - cmp.arena_base()) // MIB


def pattern(nbytes: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)


def setup(region_mb: int, policy: str = "lru", **kw):
    from IO.cache.cache_region import CacheRegion
    from IO.cache.native_cache import NativeGranuleCache
    from IO.codec.home_writer import copy_home
    region = CacheRegion()
    region.set_size(region_mb * MIB)
    cache = NativeGranuleCache(region, policy, copy_home=copy_home, **kw)
    cache.attach()
    return region, cache


def fill(cache, region, n: int, seed: int):
    """Register a region of ``n`` granules (the last short), admit it whole
    and write ``pattern(.., seed)`` into its slots; its home stays zero (the
    granules are dirty)."""
    nb = n * MIB - SHORT
    home = torch.zeros(nb, dtype=torch.uint8, pin_memory=True)
    data = pattern(nb, seed)
    cr = cache.register(nb, home.data_ptr(), [0], [nb], [0],
                        next_use_op=1, expected_reads=1)
    assert cache.admit([cr]).admitted == [n]
    view, lo = region.view(), region.boundary
    for k, s in enumerate(cache.slot_of(range(cr.first, cr.first + n)).tolist()):
        a, b = k * MIB, min(nb, (k + 1) * MIB)
        view[(s - lo) * MIB:(s - lo) * MIB + (b - a)].copy_(data[a:b])
    torch.cuda.synchronize()
    return SimpleNamespace(cr=cr, home=home, data=data, n=n)


def slots(cache, x) -> list:
    return cache.slot_of(range(x.cr.first, x.cr.first + x.n)).tolist()


def intact(cache, x) -> bool:
    """The region's bytes as a reader sees them (slot pieces from the
    slots, the rest from its home) are its data."""
    from IO.cache.cache_region import _DeviceBytes
    p = cache.pieces(x.cr, 0, x.cr.nbytes)
    for e in p.events:
        e.synchronize()
    out = torch.empty(x.cr.nbytes, dtype=torch.uint8)
    for off, n, src, on in zip(p.offset.tolist(), p.size.tolist(),
                               p.src.tolist(), p.on_gpu.tolist()):
        if on:
            out[off:off + n] = torch.as_tensor(
                _DeviceBytes(src, n), device="cuda").cpu()
        else:
            out[off:off + n] = x.home[src - x.home.data_ptr():][:n]
    return torch.equal(out, x.data)


def home_holds(x, first: int) -> bool:
    """Granules ``first``.. of ``x`` are in its home."""
    return torch.equal(x.home[first * MIB:], x.data[first * MIB:])


def fill_operator_space() -> list:
    """Allocate every free block of the operator space."""
    held = []
    torch.cuda.synchronize()
    st = cmp.get_arena_stats()
    while st["free_b"]:
        held.append(device(st["largest_free_run_b"]))
        st = cmp.get_arena_stats()
    return held


def stats(cache=None) -> dict:
    st = cmp.get_arena_stats()
    out = {"boundary": cmp.region_boundary(), "region_b": st["region_b"],
           "lendable_b": st["lendable_b"], "lent_b": st["lent_b"],
           "arena": cmp.lend_stats(False)}
    if cache is not None:
        out["core"] = cache.lend_stats()
        out["core_lo"] = cache._core.region_lo()
        out["free_slots"] = cache._core.free_count()
    return out


def report(**kw):
    print(json.dumps(kw))


def two_regions(cache, region, tick=True):
    a = fill(cache, region, 16, 0)
    b = fill(cache, region, 16, 1)
    if tick:
        cache.advance_tick()
    return a, b


# -- the arena cases (64 MiB, the top 32 MiB the region) -----------------------

def case_lend():
    """A temporary at [0, 24), so the free run at the boundary is 8 blocks: a
    12 MiB request lends the region's lowest 4 (12 - 8).  The region is
    full: 4 granules are evicted in LRU order (A's last four, dirty, copied
    home) and B's first four, in the given blocks, move to the slots they
    freed (one copy each way).  The request fits at 24; then the lent blocks
    come back at the next pipeline start once the allocation is freed, but
    not while it holds them."""
    region, cache = setup(32)
    a, b = two_regions(cache, region)
    before = stats(cache)
    t0 = device(24 * MIB)
    big, err = try_alloc(12 * MIB)
    after = stats(cache)
    out = dict(before=before, after=after, err=err, t0_at=offset(t0),
               big_at=offset(big) if big is not None else None,
               a_slots=slots(cache, a), b_slots=slots(cache, b),
               a_state=cache.state(range(a.cr.first, a.cr.first + 16)),
               intact=[intact(cache, a), intact(cache, b)],
               a_home_tail=home_holds(a, 12))
    cache.advance()                 # the lent blocks are still in use
    out["held"] = stats(cache)
    del big
    torch.cuda.synchronize()
    cache.advance()                 # a pipeline starts: they come back
    out["returned"] = stats(cache)
    out["intact_after"] = [intact(cache, a), intact(cache, b)]
    report(**out)


def case_clean():
    """As ``lend`` with every granule cleaned first (copied home, the copy
    ended): the four victims are clean, so the lend copies only the
    relocation."""
    from IO.codec.home_writer import copy_home
    region, cache = setup(32, clean_percent=100, clean_gate=None)
    a, b = two_regions(cache, region)
    targets = cache.cleaning_targets()
    cache.commit_cleaning(targets, copy_home(targets.src, targets.dst,
                                             targets.size))
    torch.cuda.synchronize()
    cache.poll()
    t0 = device(24 * MIB)
    big, err = try_alloc(12 * MIB)
    report(err=err, cleaned=len(targets), after=stats(cache),
           a_state=cache.state(range(a.cr.first, a.cr.first + 16)),
           intact=[intact(cache, a), intact(cache, b)], t0=offset(t0))


def case_admit_short():
    """After a lend whose allocation is freed, an admission that needs more
    slots than are free evicts as it would with nothing lent: the lent
    blocks stay with the operator space (the same tick's 12 MiB allocation
    fits there again), and they come back at the next pipeline start."""
    region, cache = setup(32)
    a, b = two_regions(cache, region)
    t0 = device(24 * MIB)
    big, err = try_alloc(12 * MIB)
    lent = stats(cache)
    del big
    torch.cuda.synchronize()
    nb = 6 * MIB - SHORT
    home = torch.zeros(nb, dtype=torch.uint8, pin_memory=True)
    cr = cache.register(nb, home.data_ptr(), [0], [nb], [0],
                        next_use_op=1, expected_reads=1)
    adm = cache.admit([cr])
    torch.cuda.synchronize()
    at_admission = stats(cache)
    again, err2 = try_alloc(12 * MIB)
    again_at = offset(again) if again is not None else None
    del again
    torch.cuda.synchronize()
    cache.advance_tick()
    cache.advance()
    report(err=err, lent=lent, admitted=adm.admitted, victims=adm.victims,
           at_admission=at_admission, err2=err2, again_at=again_at,
           after=stats(cache), t0=offset(t0),
           c_slots=cache.slot_of(range(cr.first, cr.first + 6)).tolist(),
           intact=[intact(cache, a), intact(cache, b)])


def case_pinned():
    """A pinned granule at the region's lowest block: nothing is lendable
    and the request is out of memory, nothing moved.  Pinned two blocks
    up: a lend of 4 is refused (all or nothing), one of 2 goes through."""
    region, cache = setup(32)
    a, b = two_regions(cache, region)
    t0 = device(24 * MIB)
    cache.pin([b.cr.first])                       # slot 32
    low = stats(cache)
    big, err_low = try_alloc(12 * MIB)
    cache.unpin([b.cr.first])
    cache.pin([b.cr.first + 2])                   # slot 34
    two = stats(cache)
    big, err_four = try_alloc(12 * MIB)
    four = stats(cache)
    ok, err_two = try_alloc(10 * MIB)
    report(low=low, err_low=err_low, two=two, err_four=err_four, four=four,
           err_two=err_two, ok_at=offset(ok) if ok is not None else None,
           after=stats(cache), b_slots=slots(cache, b), t0=offset(t0),
           intact=[intact(cache, a), intact(cache, b)])


def case_same_tick():
    """B admitted in the current tick, at the region's lowest blocks: it
    cannot move, so the request is out of memory; after the tick boundary
    it goes through."""
    region, cache = setup(32)
    a = fill(cache, region, 16, 0)
    cache.advance_tick()
    b = fill(cache, region, 16, 1)                # this tick's
    t0 = device(24 * MIB)
    first = stats(cache)
    big, err = try_alloc(12 * MIB)
    cache.advance_tick()
    big2, err2 = try_alloc(12 * MIB)
    report(first=first, err=err, err2=err2, t0=offset(t0),
           big2_at=offset(big2) if big2 is not None else None,
           after=stats(cache), intact=[intact(cache, a), intact(cache, b)])


def case_in_op():
    """The miss is inside a torch op's own allocation (``torch.cat``'s
    16 MiB output, 8 MiB free at the boundary): the lend runs inside it and
    the op completes with the right values."""
    region, cache = setup(32)
    a, b = two_regions(cache, region)
    filler = device(16 * MIB)                         # [0, 16)
    x = torch.arange(2 * MIB, dtype=torch.int32, device="cuda")  # [16, 24)
    out = torch.cat([x, x])
    host, xh = out.cpu(), x.cpu()
    ok = torch.equal(host, torch.cat([xh, xh]))
    report(ok=ok, out_at=offset(out), x_at=offset(x), filler=offset(filler),
           after=stats(cache), intact=[intact(cache, a), intact(cache, b)])


def case_whole():
    """No free run touches the boundary (its block below is allocated): a
    32 MiB request lends the whole region, every granule evicted home; a
    40 MiB one (more than the region) is out of memory with nothing moved."""
    region, cache = setup(32)
    a, b = two_regions(cache, region)
    t0 = device(31 * MIB)
    t1 = device(1 * MIB)                              # block 31
    del t0
    torch.cuda.synchronize()
    too, err_too = try_alloc(40 * MIB)
    first = stats(cache)
    big, err = try_alloc(32 * MIB)
    report(err_too=err_too, first=first, err=err, t1=offset(t1),
           big_at=offset(big) if big is not None else None,
           after=stats(cache), homes=[home_holds(a, 0), home_holds(b, 0)],
           intact=[intact(cache, a), intact(cache, b)])


# -- the engine's paths --------------------------------------------------------

def case_reader_pin():
    """A cached column (X, the LRU-oldest region, at the region's top) read
    outside a pipe's ticks (``TableVariable.col_to_gpu``) while the operator
    space is full: its slot pieces are resolved at the bind, then its
    buffer's allocation lends.  The read pins X's granules, so the lend
    evicts B's instead and moves A (in the lowest blocks) into their slots,
    not into X's: the column reads back exactly."""
    from IO.codec.region_producer import RegionProducer
    from IO.pages.pinned_mem import PinnedMemory
    from variable import TableVariable
    region, cache = setup(32)
    pool = PinnedMemory(capacity_gb=1, block_size_mb=0.5)
    producer = RegionProducer(pool, "t 0 sort pipeline", cache=cache)
    producer.cached_outputs = {0: (5, 1)}
    rows = MIB                                        # 8 MiB of int64
    vals = torch.arange(rows, dtype=torch.int64).unsqueeze(1)
    out = vals.cuda()
    cache.advance_tick()
    producer.compress([(0, out, [rows], "RAW")])
    torch.cuda.synchronize()
    cache.advance_tick()
    producer.drain()
    table = producer.tables()[0]
    del out
    xcr = table.regions[0].release[1]
    b = fill(cache, region, 16, 1)                    # slots 40..55
    a = fill(cache, region, 8, 0)                     # slots 32..39
    cache.advance_tick()
    x_slots = cache.slot_of(range(xcr.first, xcr.first + xcr.n)).tolist()
    held = fill_operator_space()
    got = TableVariable(table, "int").col_to_gpu()
    ok = torch.equal(got.cpu(), vals.squeeze(1))
    report(ok=ok, x_slots=x_slots,
           x_after=cache.slot_of(range(xcr.first, xcr.first + xcr.n)).tolist(),
           b_state=cache.state(range(b.cr.first, b.cr.first + 16)),
           a_slots=slots(cache, a), after=stats(cache), held=len(held),
           intact=[intact(cache, a), intact(cache, b)])


def case_producer_plan():
    """A tick with a cached and an uncached output takes the producer's copy
    plan (``_issue_copies``): it takes this tick's granules' slot addresses,
    then allocates the uncached output's dense buffer.  A lend of 24 blocks
    inside that allocation moves an older region's granules (the eight of
    them in the lowest 24 blocks) but not this tick's, so every output and
    the older region read back exactly, and the blocks come back at the
    next pipeline start."""
    import IO.codec.region_producer as region_producer
    from IO.cache import config as cache_config
    from IO.codec.batched_compress import aligned_partition_layout
    from IO.codec.region_producer import RegionProducer
    from IO.pages.pinned_mem import PinnedMemory
    from variable import TableVariable
    # The cached output's regions are under one granule: offer them anyway.
    cache_config.FLOOR_GRANULES = 0
    region, cache = setup(64)
    lo = region.boundary
    pool = PinnedMemory(capacity_gb=1, block_size_mb=0.5)
    first = fill(cache, region, 32, 9)                # slots lo+32..lo+63
    low = fill(cache, region, 16, 4)                  # slots lo+16..lo+31
    cache.drop(first.cr)
    cache.advance_tick()
    rng = np.random.default_rng(15)
    ticks = [[[torch.from_numpy(rng.integers(0, 50, 20_000).astype(np.int64))
               .unsqueeze(1) for _ in range(2)] for _ in range(2)]
             for _ in range(2)]
    feeds = []
    for t0, t1 in zip(*ticks):
        items = []
        for base, t in ((0, t0), (1, t1)):
            counts = [p.shape[0] for p in t]
            _, starts, total = aligned_partition_layout(counts, 8, 512)
            o = torch.full((total, 1), -1, dtype=torch.int64, device="cuda")
            for p, s in zip(t, starts):
                o[s // 8:s // 8 + p.shape[0]] = p.to("cuda")
            items.append((base, o, counts, "BITCOMP"))
        feeds.append(items)
    producer = RegionProducer(pool, "t 0 sort pipeline", cache=cache)
    producer.cached_outputs = {0: (5, 1)}
    lent = []
    real = RegionProducer._alloc_dense

    def alloc_dense(self, dense_b, device_):
        if not lent:
            # This tick's granules are admitted and their slot addresses
            # taken: a miss lends the region's lowest 24 blocks.
            held = fill_operator_space()
            probe = device(24 * MIB)
            lent.append(stats(cache))
            del probe, held
            torch.cuda.synchronize()
        return real(self, dense_b, device_)
    RegionProducer._alloc_dense = alloc_dense
    issued = []
    real_issue = RegionProducer._issue_copies

    def issue(self, *a, **k):
        issued.append(1)
        return real_issue(self, *a, **k)
    RegionProducer._issue_copies = issue
    for items in feeds:
        cache.advance_tick()
        if producer.has_pending():
            producer.pack()
        producer.compress(items)
        torch.cuda.current_stream().synchronize()
    cache.advance_tick()
    producer.drain()
    tables = producer.tables()
    # ticks[base][tick][bucket]; a table's rows are bucket-major.
    reads = [torch.equal(
        TableVariable(tables[base], "int").col_to_gpu().cpu(),
        torch.cat([t[b] for b in range(2) for t in ticks[base]]).squeeze(1))
        for base in (0, 1)]
    out = dict(issued=bool(issued), lent=lent[0] if lent else None, lo=lo,
               cached=[r.location for r in tables[0].regions],
               admitted=[bool(r.release[1].admitted) for r in tables[0].regions],
               reads=reads, low_intact=intact(cache, low),
               low_slots=slots(cache, low))
    torch.cuda.synchronize()
    cache.advance()
    out["returned"] = stats(cache)
    out["low_intact_after"] = intact(cache, low)
    report(**out)


CASES = {
    "lend": (64, case_lend), "clean": (64, case_clean),
    "admit_short": (64, case_admit_short), "pinned": (64, case_pinned),
    "same_tick": (64, case_same_tick), "in_op": (64, case_in_op),
    "whole": (64, case_whole), "reader_pin": (64, case_reader_pin),
    "producer_plan": (512, case_producer_plan),
}


if __name__ == "__main__":
    arena_mb, case = CASES[sys.argv[1]]
    install(arena_mb * MIB)
    from utility.logger import set_message_logger
    set_message_logger(None, False)
    case()
