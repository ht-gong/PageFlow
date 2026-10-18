"""The managed arena (``cmp.init(..., managed=True)``), the device balloon
(``reserve_device``), the arena reset (``remake``) and ``prefetch``.

The pytest process's own arena is device memory, and the allocator can be
installed once per process, so each check runs ``managed_arena_child.py`` in
a subprocess, which installs a managed arena as main.py installs its device
arena.  The parent's 22 GB arena sits on the same GPU: the child sizes its
balloon from what the card has free.
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

GIB = 1024 ** 3
_CHILD = os.path.join(os.path.dirname(__file__), "managed_arena_child.py")


def run_child(case: str) -> dict:
    proc = subprocess.run([sys.executable, _CHILD, case], capture_output=True,
                          text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_oversubscribed_arena_reads_back(cuda_allocator):
    """With a balloon leaving about 1 GiB of the card to managed pages, 3 GiB
    of tensors written on the GPU read back right."""
    out = run_child("oversubscribe")
    assert out["budget_b"] < 1.5 * GIB, out
    assert out["arena_allocated_b"] >= out["written_b"] == 3 * GIB, out
    assert out["ok"], out


def test_remake_refuses_a_live_allocation(cuda_allocator):
    """A reset with a live tensor raises and names its bytes, and leaves the
    tensor intact; with none it succeeds, every block is free and the arena
    allocates again."""
    out = run_child("remake")
    assert out["raised"] is not None and "8388608 B" in out["raised"], out
    assert out["live_read_back"], out
    assert out["after_reset"] == {"capacity_b": 8 * GIB, "allocated_b": 0}, out
    assert out["managed"] and out["allocates_again"], out


def test_prefetched_to_host_node_reads_back(cuda_allocator):
    """A 1 GiB tensor prefetched to GPU 0's host NUMA node (the driver records
    that node as its last prefetch location) reads back right on the GPU."""
    out = run_child("prefetch")
    assert out["to_node"] == ["cudaMemLocationTypeHostNuma", out["node"]], out
    assert out["from_node"], out


def test_refusals(cuda_allocator):
    """An allocation with no arena raises instead of returning a null pointer;
    a managed arena whose blocks are not a multiple of 2 MiB is refused (and
    a 4 MiB one is then made); a prefetch to host node -1 is refused."""
    out = run_child("refusals")
    assert out["no_arena"] and "no arena" in out["no_arena"], out
    assert out["block_5mib"] and "multiple of 2 MiB" in out["block_5mib"], out
    assert out["block_4mib"], out
    assert out["host_node_-1"], out
