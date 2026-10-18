"""Lending: an allocation that misses borrows the cache region's lowest
blocks (``memory/cuda_memory_allocator.cc``, the lend hook; the granule
cache's core does the lend, ``IO/cache/granule_core.cc``), all in C++.

Each check runs ``lend_child.py`` in a subprocess on a small arena of 1 MiB
blocks (the allocator is installed once per process; the pytest process's
is 22 GB): 64 MiB, the top 32 MiB the region, filled by an older region A
(slots 48..63) and a newer region B (slots 32..47), both dirty, then
allocations past what the operator space holds.  The engine's own paths
(a read outside a pipe's ticks, the producer's copy plan) run there too.
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
_CHILD = os.path.join(os.path.dirname(__file__), "lend_child.py")


def run_child(case: str) -> dict:
    proc = subprocess.run([sys.executable, _CHILD, case], capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _core(st):
    """The core's lend counters as a dict."""
    (lends, refused, blocks, returned, relocated, copies, evicted, logical,
     wire) = st
    return dict(lends=lends, refused=refused, blocks=blocks,
                returned=returned, relocated=relocated, copies=copies,
                dead=evicted[0], clean=evicted[1], dirty=evicted[2],
                dirty_wire=wire[2])


def test_a_miss_lends_the_regions_lowest_blocks(cuda_allocator):
    """8 blocks free at the boundary, 12 asked: the region's lowest 4 are
    lent.  The full region evicts A's last four granules (LRU, dirty:
    copied home) and B's first four move up into their slots, one merged
    copy each way; everything reads back.  The blocks come back at the
    next pipeline start once free, not while the allocation holds them."""
    out = run_child("lend")
    before, after = out["before"], out["after"]
    assert before["boundary"] == 32 and before["lendable_b"] == 32 * MIB, out
    assert out["err"] is None and out["t0_at"] == 0 and out["big_at"] == 24
    assert after["boundary"] == 36 and after["core_lo"] == 36, after
    assert after["region_b"] == 28 * MIB and after["lent_b"] == 4 * MIB
    a = after["arena"]
    assert (a["misses"], a["lent"], a["short"], a["failed"],
            a["blocks_lent"]) == (1, 1, 0, 0, 4), a
    c = _core(after["core"])
    assert (c["lends"], c["refused"], c["blocks"], c["dirty"], c["clean"],
            c["dead"], c["relocated"], c["copies"]) == (1, 0, 4, 4, 0, 0, 4,
                                                        2), c
    assert c["dirty_wire"] == 4 * MIB - 1000, c
    assert out["b_slots"][:4] == [60, 61, 62, 63], out["b_slots"]
    assert out["b_slots"][4:] == list(range(36, 48))
    assert out["a_slots"][:12] == list(range(48, 60))
    assert out["a_slots"][12:] == [-1] * 4
    assert out["a_state"][12:] == ["evicted"] * 4
    assert out["a_home_tail"] and out["intact"] == [True, True], out
    held = out["held"]
    assert held["boundary"] == 36 and held["lent_b"] == 4 * MIB, held
    ret = out["returned"]
    assert ret["boundary"] == 32 and ret["core_lo"] == 32, ret
    assert ret["lent_b"] == 0 and ret["arena"]["blocks_returned"] == 4, ret
    assert _core(ret["core"])["returned"] == 4 and ret["free_slots"] == 4
    assert out["intact_after"] == [True, True]


def test_clean_victims_are_not_copied(cuda_allocator):
    """Every granule cleaned first: the lend's four victims are clean, so
    only the relocation is copied."""
    out = run_child("clean")
    assert out["err"] is None and out["cleaned"] == 32, out
    c = _core(out["after"]["core"])
    assert (c["clean"], c["dirty"], c["relocated"], c["copies"]) == (
        4, 0, 4, 1), c
    assert out["intact"] == [True, True]


def test_an_admission_short_of_slots_leaves_the_lent_blocks(cuda_allocator):
    """After a lend whose allocation is freed, an admission of 6 granules
    into a full region evicts 6 (LRU, dirty) as it would with nothing lent:
    the 4 lent blocks stay with the operator space, where the same tick's
    12 MiB allocation fits again with no lend, and they come back at the
    next pipeline start."""
    out = run_child("admit_short")
    assert out["err"] is None and out["lent"]["boundary"] == 36, out
    assert out["admitted"] == [6] and out["victims"] == [0, 0, 6], out
    adm = out["at_admission"]
    assert adm["boundary"] == 36 and adm["lent_b"] == 4 * MIB, adm
    assert adm["arena"]["blocks_returned"] == 0, adm
    assert sorted(out["c_slots"]) == list(range(54, 60)), out["c_slots"]
    assert out["err2"] is None and out["again_at"] == 24, out
    after = out["after"]
    assert after["arena"]["misses"] == 1, after
    assert after["boundary"] == 32 and after["core_lo"] == 32, after
    assert after["lent_b"] == 0 and after["arena"]["blocks_returned"] == 4
    assert out["intact"] == [True, True]


def test_a_pinned_granule_stops_a_lend(cuda_allocator):
    """Pinned at the region's lowest block: nothing is lendable and the
    request is out of memory.  Pinned at the third: 2 blocks are lendable,
    a lend of 4 is refused (all or nothing, nothing moved) and one of 2 goes
    through, moving the two granules below the pin."""
    out = run_child("pinned")
    assert out["low"]["lendable_b"] == 0, out["low"]
    assert out["err_low"] and "Pool out of memory" in out["err_low"]
    assert out["two"]["lendable_b"] == 2 * MIB and out["two"]["boundary"] == 32
    assert out["err_four"] and "Pool out of memory" in out["err_four"]
    assert out["four"]["boundary"] == 32
    assert out["err_two"] is None and out["ok_at"] == 24, out
    after = out["after"]
    assert after["boundary"] == 34 and after["lent_b"] == 2 * MIB, after
    a = after["arena"]
    assert (a["misses"], a["lent"], a["short"], a["blocks_lent"]) == (
        3, 1, 2, 2), a
    c = _core(after["core"])
    assert (c["lends"], c["refused"], c["dirty"], c["relocated"]) == (
        3, 2, 2, 2), c
    assert out["b_slots"][:3] == [62, 63, 34], out["b_slots"]
    assert out["intact"] == [True, True]


def test_a_same_tick_granule_stops_a_lend(cuda_allocator):
    """B admitted this tick sits in the region's lowest blocks: nothing is
    lendable until the tick boundary, then the request goes through."""
    out = run_child("same_tick")
    assert out["first"]["lendable_b"] == 0, out["first"]
    assert out["err"] and "Pool out of memory" in out["err"]
    assert out["err2"] is None and out["big2_at"] == 24, out
    assert out["after"]["boundary"] == 36
    assert out["intact"] == [True, True]


def test_a_miss_inside_a_torch_op_lends(cuda_allocator):
    """``torch.cat``'s 16 MiB output misses with about 8 MiB free at the
    boundary: the rest is lent inside the op, which completes with the
    right values."""
    out = run_child("in_op")
    assert out["ok"] and out["filler"] == 0, out
    # The op's output follows its input (torch may take a block of its own
    # first) and ends at the boundary the lend moved up.
    x, o, after = out["x_at"], out["out_at"], out["after"]
    assert x in (16, 17) and o == x + 8, out
    assert after["boundary"] == o + 16, after
    assert after["lent_b"] == (o + 16 - 32) * MIB, after
    assert after["arena"]["lent"] == 1 and after["arena"]["misses"] == 1
    assert out["intact"] == [True, True]


def test_the_whole_region_is_lent_and_more_is_out_of_memory(cuda_allocator):
    """No free run at the boundary (the block below it allocated): 32 MiB
    asked lends the whole region, every granule evicted to its home; 40
    MiB, more than the region, is out of memory and moves nothing."""
    out = run_child("whole")
    assert out["err_too"] and "Pool out of memory" in out["err_too"]
    assert out["first"]["boundary"] == 32, out["first"]
    assert out["first"]["arena"]["short"] == 1
    assert _core(out["first"]["core"])["lends"] == 0
    assert out["err"] is None and out["big_at"] == 32 and out["t1"] == 31
    after = out["after"]
    assert after["boundary"] == 64 and after["region_b"] == 0, after
    c = _core(after["core"])
    assert (c["dirty"], c["relocated"]) == (32, 0), c
    assert out["homes"] == [True, True] and out["intact"] == [True, True]


def test_a_read_outside_the_ticks_pins_its_granules(cuda_allocator):
    """A cached column, the LRU-oldest region, read by ``col_to_gpu`` with
    the operator space full: its buffer's allocation lends after its slot
    pieces are resolved.  Its granules are pinned through the read, so the
    lend evicts another region's and moves the lowest region's into those
    slots, not into the column's: it reads back exactly."""
    out = run_child("reader_pin")
    assert out["held"] >= 1 and out["x_slots"] == list(range(56, 64)), out
    assert out["ok"], out
    assert out["b_state"][8:] == ["evicted"] * 8, out["b_state"]
    assert out["a_slots"] == list(range(48, 56)), out["a_slots"]
    assert out["intact"] == [True, True]


def test_the_producers_copy_plan_survives_a_lend_inside_it(cuda_allocator):
    """The producer's copy plan takes this tick's slot addresses, then
    allocates a dense buffer; a lend of 24 blocks there moves an older
    region's eight granules in those blocks but none of this tick's, so
    every output and the older region read back exactly, and the blocks
    come back at the next pipeline start."""
    out = run_child("producer_plan")
    lo = out["lo"]
    assert out["issued"] and out["lent"] is not None, out
    lent = out["lent"]
    assert lent["boundary"] == lo + 24 and lent["core_lo"] == lo + 24, lent
    c = _core(lent["core"])
    assert (c["blocks"], c["relocated"], c["dead"] + c["clean"]
            + c["dirty"]) == (24, 8, 0), c
    assert out["cached"] and set(out["cached"]) == {"cached"}, out
    assert all(out["admitted"]) and out["reads"] == [True, True], out
    assert out["low_intact"] and out["low_intact_after"], out
    assert out["returned"]["boundary"] == lo, out["returned"]
