"""The free-list baseline's eviction (``IO/pages/device_regions.py``) through
the arena's miss hook (``TQPlib.tqpmemory.set_miss_hook``).

Each check runs ``freelist_child.py`` in a subprocess on a 64 MiB arena of
2 MiB blocks (the allocator is installed once per process; the pytest
process's is 22 GB), with 8 MiB regions and temporaries at known blocks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA")

MIB = 1024 ** 2
_CHILD = os.path.join(os.path.dirname(__file__), "freelist_child.py")


def run_child(case: str) -> dict:
    proc = subprocess.run([sys.executable, _CHILD, case], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_miss_evicts_until_it_fits(cuda_allocator):
    """A temporary at [0, 8) MiB and four regions at [8, 40): a 48 MiB
    request misses and evicts them one at a time, least recently used
    first, into their homes, intact; only after the fourth does it fit, at
    8 MiB, the regions' blocks coalesced with the free tail.  Before it,
    the what-if stats count the regions free (56 MiB in one run)."""
    out = run_child("evict")
    assert out["layout"] == [0, 8, 16, 24, 32], out
    assert out["before_free_b"] == 24 * MIB, out
    assert out["spillable_b"] == 32 * MIB, out
    assert out["what_if_free_b"] == out["what_if_largest_b"] == 56 * MIB, out
    assert out["err"] is None and out["big_at"] == 8, out
    assert out["locations"] == ["host"] * 4 and all(out["intact"]), out
    assert out["live_b"] == 0 and out["epoch"] == 4, out
    st = out["stats"]
    assert (st["evictions"], st["regions"], st["bytes"]) == (1, 4, 32 * MIB), st
    assert (st["miss_calls"], st["miss_fragmented"], st["miss_failed"]) \
        == (1, 0, 0), st


def test_lru_region_alone_is_evicted(cuda_allocator):
    """The least recently used region (the three others read since) goes
    first, and its eviction is enough for a 32 MiB request: it alone moves
    home."""
    out = run_child("lru")
    assert out["err"] is None and out["big_at"] == 32, out
    assert out["locations"] == ["device", "device", "device", "host"], out
    assert all(out["intact"]), out
    st = out["stats"]
    assert (st["evictions"], st["regions"], st["bytes"]) == (1, 1, 8 * MIB), st


def test_pinned_region_stays_and_the_miss_fails(cuda_allocator):
    """A pinned region is not evicted; the other three are, one by one, but
    the pinned one splits the free space, so the request is out of memory
    once none is left."""
    out = run_child("pinned")
    assert out["err"] and "Pool out of memory" in out["err"], out
    assert out["locations"] == ["host", "device", "host", "host"], out
    assert all(out["intact"]) and out["pins"] == [0, 1, 0, 0], out
    assert out["largest_after_b"] == 40 * MIB, out
    assert (out["miss"]["calls"], out["miss"]["failed"]) == (1, 1), out
    assert out["epoch"] == 3, out


def test_fragmented_miss_is_counted(cuda_allocator):
    """40 MiB free in runs of 8, 8 and 24: a 32 MiB request is a fragmented
    miss; evicting the two oldest regions leaves [0, 32) and the request
    fits at 0; the third region stays."""
    out = run_child("fragmented")
    assert out["layout"] == [0, 8, 16, 24, 32, 40], out
    assert out["before_free_b"] == 40 * MIB, out
    assert out["before_largest_b"] == 24 * MIB, out
    assert out["err"] is None and out["big_at"] == 0, out
    assert out["locations"] == ["host", "host", "device"], out
    assert all(out["intact"]), out
    assert (out["miss"]["calls"], out["miss"]["fragmented"]) == (1, 1), out


def test_miss_inside_a_torch_op(cuda_allocator):
    """The hook runs inside torch.cat's output allocation (dropping the
    two oldest regions' tensors from inside it) and the op completes."""
    out = run_child("in_op")
    assert out["ok"], out
    assert out["locations"] == ["host"] * 2 + ["device"] * 4, out
    assert all(out["intact"]), out


def test_hook_that_raises_is_an_oom(cuda_allocator):
    out = run_child("hook_raises")
    assert out["err"] and "Pool out of memory" in out["err"], out
    assert out["again"] and out["err2"] is None, out
    assert (out["miss"]["calls"], out["miss"]["failed"]) == (1, 1), out


def test_reserve_serves_what_eviction_cannot(cuda_allocator):
    """With a 16 MiB reserve, the miss the pinned region fails above fits:
    once no unpinned region is left, the request draws the 8 MiB it lacks
    beyond the free run below the reserve and fits at 24 MiB.  The reserve
    refills once that allocation is freed, and is given back at the end."""
    out = run_child("reserve")
    assert out["taken"] == {"held": 16 * MIB, "capacity": 48 * MIB}, out
    assert out["drawn"] == {"at": 24, "err": None, "held": 8 * MIB}, out
    assert out["refilled_b"] == 16 * MIB, out
    assert out["locations"] == ["host", "device", "host", "host"], out
    assert all(out["intact"]), out
    st = out["stats"]
    assert (st["reserve_b"], st["reserve_draws"], st["reserve_drawn_b"],
            st["reserve_refilled_b"], st["reserve_min_b"]) == (
        16 * MIB, 1, 8 * MIB, 8 * MIB, 8 * MIB), st
    assert (st["evictions"], st["regions"]) == (1, 3), st
    assert (st["miss_calls"], st["miss_failed"]) == (1, 0), st
    assert out["after_release"] == {"held": 0, "region_blocks": 0}, out


def test_reserve_too_small_is_an_oom(cuda_allocator):
    """A request the reserve and the free run below it cannot hold together
    draws all of the reserve and is still out of memory."""
    out = run_child("reserve_short")
    assert out["err"] and "Pool out of memory" in out["err"], out
    assert out["after"] == {"held": 0, "largest_b": 40 * MIB}, out
    st = out["stats"]
    assert (st["reserve_draws"], st["reserve_drawn_b"]) == (1, 16 * MIB), st
    assert (st["miss_calls"], st["miss_failed"]) == (1, 1), st


def test_no_hook_no_eviction(cuda_allocator):
    out = run_child("no_hook")
    assert out["err"] and "Pool out of memory" in out["err"], out
    assert out["locations"] == ["device"] * 4, out
    assert out["miss"]["calls"] == 0, out
