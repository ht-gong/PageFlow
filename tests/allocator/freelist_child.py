"""One free-list baseline check, in a process of its own (tests/allocator/test_freelist_evict.py).

The allocator can be installed once per process and the pytest process's
arena is 22 GB, so each check runs here on a 64 MiB arena of 2 MiB blocks:
install the pluggable allocator as main.py does, arm the miss hook
(``IO.pages.device_regions.install``), lay out spillable regions and
operator temporaries, then allocate past what is free and print one JSON
line of what happened.

    python3 tests/allocator/freelist_child.py evict|lru|pinned|fragmented|in_op|hook_raises|reserve|reserve_short|no_hook
"""
from __future__ import annotations

import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from torch.cuda.memory import (CUDAPluggableAllocator,  # noqa: E402
                               change_current_allocator)

import TQPlib.tqpmemory as cmp  # noqa: E402
from IO.pages import device_regions  # noqa: E402
from IO.pages.region import DEVICE, HOST, Region  # noqa: E402

MIB = 1024 ** 2
ARENA_B = 64 * MIB
BLOCK_B = 2 * MIB


def install():
    change_current_allocator(CUDAPluggableAllocator(
        path_to_so_file="libcustom_allocator.so",
        alloc_fn_name="pa_cuda_malloc", free_fn_name="pa_cuda_free"))
    cmp.init(ARENA_B, BLOCK_B)


def device(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, device="cuda")


def region(nbytes: int, seed: int) -> Region:
    """A spillable region of ``nbytes`` holding ``pattern(nbytes, seed)``,
    registered."""
    storage = device(nbytes)
    storage.copy_(pattern(nbytes, seed))     # H2D, no GPU temporary
    home = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    r = Region(nbytes, DEVICE, storage=storage, home=home)
    device_regions.register(r)
    return r


def pattern(nbytes: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (nbytes,), dtype=torch.uint8, generator=g)


def intact(r: Region, seed: int) -> bool:
    return torch.equal(r.storage.cpu(), pattern(r.nbytes, seed))


def offset(t_or_r) -> int:
    """MiB from the arena's start (the first allocation sits at 0)."""
    p = t_or_r.data_ptr()
    return (p - BASE) // MIB


def arena():
    return cmp.get_arena_stats()


def report(**kw):
    kw["miss"] = cmp.miss_stats(False)
    kw["epoch"] = device_regions.epoch()
    print(json.dumps(kw))


def try_alloc(nbytes: int):
    try:
        return device(nbytes), None
    except RuntimeError as exc:
        return None, str(exc)


def case_evict():
    """A temporary at [0, 8), four 8 MiB regions at [8, 40), 24 MiB free:
    a 48 MiB request evicts them one at a time, least recently used first,
    and needs all four (only their 32 MiB together coalesce with the free
    tail into [8, 64)) to fit."""
    temp = device(8 * MIB)
    regions = [region(8 * MIB, s) for s in range(4)]
    layout = [offset(temp)] + [offset(r) for r in regions]
    before = arena()
    what_if = device_regions.arena_stats()
    big, err = try_alloc(48 * MIB)
    report(layout=layout, before_free_b=before["free_b"],
           what_if_free_b=what_if["free_b"],
           what_if_largest_b=what_if["largest_free_run_b"],
           spillable_b=what_if["spillable_b"],
           big_at=offset(big) if big is not None else None, err=err,
           locations=[r.location for r in regions],
           intact=[intact(r, s) for s, r in enumerate(regions)],
           live_b=device_regions.live_bytes(), stats=device_regions.take_stats(),
           temp_kept=temp.numel())


def case_lru():
    """As ``evict``, but the first three regions read since (pinned and
    unpinned), so the fourth, at [32, 40), is the least recently used: a 32
    MiB request evicts it alone ([32, 64)) and fits at 32; the others stay
    on the GPU."""
    temp = device(8 * MIB)
    regions = [region(8 * MIB, s) for s in range(4)]
    device_regions.unpin(device_regions.pin(regions[:3]))
    big, err = try_alloc(32 * MIB)
    report(big_at=offset(big) if big is not None else None, err=err,
           locations=[r.location for r in regions],
           intact=[intact(r, s) for s, r in enumerate(regions)],
           stats=device_regions.take_stats(), temp_kept=temp.numel())


def case_pinned():
    """As ``evict`` with the second region pinned: the other three are
    evicted one by one, leaving runs of 8 and 40 MiB (the pinned one splits
    [8, 64)), and none is left, so the 48 MiB request is out of memory; the
    pinned region is still on the GPU, intact."""
    temp = device(8 * MIB)
    regions = [region(8 * MIB, s) for s in range(4)]
    held = device_regions.pin([regions[1]])
    big, err = try_alloc(48 * MIB)
    after = arena()
    report(err=err, big=big is not None,
           locations=[r.location for r in regions],
           intact=[intact(r, s) for s, r in enumerate(regions)],
           largest_after_b=after["largest_free_run_b"],
           free_after_b=after["free_b"], pins=[r.pins for r in regions],
           temp_kept=temp.numel(), held=len(held))


def case_fragmented():
    """Regions and temporaries interleaved, 8 MiB each (R T R T R T), then
    the temporaries freed: 40 MiB free in runs of 8, 8 and 24.  A 32 MiB
    request is a fragmented miss (the free space holds it, no run does);
    evicting the two oldest regions leaves [0, 32), and it fits at 0; the
    third stays."""
    regions, temps = [], []
    for s in range(3):
        regions.append(region(8 * MIB, s))
        temps.append(device(8 * MIB))
    layout = [offset(x) for pair in zip(regions, temps) for x in pair]
    del temps
    torch.cuda.synchronize()
    before = arena()
    big, err = try_alloc(32 * MIB)
    report(layout=layout, before_free_b=before["free_b"],
           before_largest_b=before["largest_free_run_b"],
           big_at=offset(big) if big is not None else None, err=err,
           locations=[r.location for r in regions],
           intact=[intact(r, s) for s, r in enumerate(regions)])


def case_in_op():
    """The miss happens inside a torch op's own allocation (``torch.cat``'s
    output), not a bare ``torch.empty``: the hook runs inside it (evicting
    the two oldest regions, [0, 16)) and the op completes with the right
    values."""
    regions = [region(8 * MIB, s) for s in range(6)]
    a = torch.arange(2 * MIB, dtype=torch.int32, device="cuda")      # 8 MiB
    # 56 MiB used, 8 free: the 16 MiB output of cat misses.
    out = torch.cat([a, a])
    ok = bool((out[: a.numel()] == a).all().item()
              and (out[a.numel():] == a).all().item())
    report(ok=ok, locations=[r.location for r in regions],
           intact=[intact(r, s) for s, r in enumerate(regions)])


def case_hook_raises():
    """A hook that raises counts as nothing freed: the allocation is out of
    memory (no crash), and the process allocates normally afterwards."""
    keep = [device(8 * MIB) for _ in range(6)]

    def bad(need, first):
        raise ValueError("boom")
    cmp.set_miss_hook(bad)
    big, err = try_alloc(32 * MIB)
    cmp.set_miss_hook(None)
    del keep
    torch.cuda.synchronize()
    again, err2 = try_alloc(32 * MIB)
    report(err=err, again=again is not None, err2=err2)


def case_reserve():
    """As ``pinned``, with a 16 MiB reserve taken at [48, 64): the three
    unpinned regions are evicted, leaving [24, 48) beside the pinned one,
    and none is left, so the 32 MiB request draws the 8 MiB it lacks from
    the reserve ([48, 56) joins [24, 48)) and fits at 24.  Once it is freed
    and the pin dropped, the reserve refills to 16 MiB; ``release_all``
    gives it back."""
    device_regions.install(BLOCK_B, 16 * MIB)
    device_regions.begin_query()
    taken = dict(held=device_regions.reserve_b(),
                 capacity=arena()["capacity_b"])
    temp = device(8 * MIB)
    regions = [region(8 * MIB, s) for s in range(4)]
    held = device_regions.pin([regions[1]])
    big, err = try_alloc(32 * MIB)
    drawn = dict(at=offset(big) if big is not None else None, err=err,
                 held=device_regions.reserve_b())
    del big
    torch.cuda.synchronize()
    device_regions.unpin(held)
    refilled = device_regions.reserve_b()
    stats = device_regions.take_stats()
    locations = [r.location for r in regions]
    ok = [intact(r, s) for s, r in enumerate(regions)]
    device_regions.release_all()
    report(taken=taken, drawn=drawn, refilled_b=refilled, stats=stats,
           locations=locations, intact=ok,
           after_release=dict(held=device_regions.reserve_b(),
                              region_blocks=cmp.region_blocks()),
           temp_kept=temp.numel())


def case_reserve_short():
    """As ``reserve``, but a 48 MiB request: all of the reserve together
    with [24, 48) is 40 MiB, so the request is out of memory with the
    reserve drawn whole."""
    device_regions.install(BLOCK_B, 16 * MIB)
    device_regions.begin_query()
    temp = device(8 * MIB)
    regions = [region(8 * MIB, s) for s in range(4)]
    held = device_regions.pin([regions[1]])
    big, err = try_alloc(48 * MIB)
    after = dict(held=device_regions.reserve_b(),
                 largest_b=arena()["largest_free_run_b"])
    device_regions.unpin(held)
    report(err=err, big=big is not None, after=after,
           stats=device_regions.take_stats(), temp_kept=temp.numel())


def case_no_hook():
    """Without the hook a region is never moved: the request is out of
    memory and the miss counters stay at zero."""
    device_regions.uninstall()
    temp = device(8 * MIB)
    regions = [region(8 * MIB, s) for s in range(4)]
    big, err = try_alloc(48 * MIB)
    report(err=err, locations=[r.location for r in regions],
           temp_kept=temp.numel())


if __name__ == "__main__":
    install()
    # The arena's base: a 1-byte probe takes block 0 and is given back.
    probe = device(1)
    BASE = probe.data_ptr()
    del probe
    torch.cuda.synchronize()
    device_regions.install(BLOCK_B)
    {"evict": case_evict, "lru": case_lru, "pinned": case_pinned,
     "fragmented": case_fragmented, "in_op": case_in_op,
     "hook_raises": case_hook_raises, "reserve": case_reserve,
     "reserve_short": case_reserve_short,
     "no_hook": case_no_hook}[sys.argv[1]]()
