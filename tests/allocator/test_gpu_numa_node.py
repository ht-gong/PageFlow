"""
Tests for utility.numa.gpu_numa_node: the pinned host pool is bound to the node
it returns, so it must raise rather than guess when the node is unknown.
CPU-only: the CUDA driver's bus id, nvidia-smi (asked only without a
driver) and sysfs are faked.  And for utility.numa.nvidia_smi_id,
which names CUDA's device to nvidia-smi: the last test runs on the box's GPUs.
"""
import io
import os
import shutil
import subprocess
import sys

import pytest

import utility.numa as numa

BUS_ID = b"00000000:8C:00.0\n"


def _fake_sysfs(monkeypatch, numa_node, n_nodes):
    monkeypatch.setattr(numa, "nvidia_smi_id", lambda i: str(i))
    monkeypatch.setattr(numa.subprocess, "check_output", lambda *a, **k: BUS_ID)

    def fake_open(path, *a, **k):
        assert path == "/sys/bus/pci/devices/0000:8c:00.0/numa_node", path
        return io.StringIO(f"{numa_node}\n")
    monkeypatch.setattr(numa, "open", fake_open, raising=False)
    monkeypatch.setattr(numa.os, "listdir",
                        lambda p: [f"node{i}" for i in range(n_nodes)] + ["possible"])


def test_returns_the_node_sysfs_reports(monkeypatch):
    # Without a CUDA driver nvidia_smi_id is the index, and nvidia-smi names
    # the bus.
    _fake_sysfs(monkeypatch, numa_node=1, n_nodes=2)
    assert numa.gpu_numa_node(0) == 1
    # With one, the bus id is the driver's, and nvidia-smi is not asked.
    monkeypatch.setattr(numa, "nvidia_smi_id", lambda i: "0000:8C:00.0")

    def no_smi(*a, **k):
        raise AssertionError("nvidia-smi asked for a bus id CUDA named")
    monkeypatch.setattr(numa.subprocess, "check_output", no_smi)
    assert numa.gpu_numa_node(0) == 1
    monkeypatch.undo()

    def no_driver(name):
        raise OSError(f"{name}: cannot open shared object file")
    monkeypatch.setattr("ctypes.CDLL", no_driver)
    assert numa.nvidia_smi_id(0) == "0"


def test_raises_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(numa, "nvidia_smi_id", lambda i: str(i))

    def fail(*a, **k):
        raise subprocess.CalledProcessError(6, "nvidia-smi")
    monkeypatch.setattr(numa.subprocess, "check_output", fail)
    with pytest.raises(RuntimeError, match="cannot resolve GPU 0's NUMA node"):
        numa.gpu_numa_node(0)


def test_no_affinity_means_node_0_on_a_single_node_host(monkeypatch):
    _fake_sysfs(monkeypatch, numa_node=-1, n_nodes=1)
    assert numa.gpu_numa_node(0) == 0


def test_no_affinity_raises_on_a_multi_node_host(monkeypatch):
    _fake_sysfs(monkeypatch, numa_node=-1, n_nodes=2)
    with pytest.raises(RuntimeError, match="no NUMA node"):
        numa.gpu_numa_node(0)


@pytest.mark.skipif(shutil.which("nvidia-smi") is None, reason="needs nvidia-smi")
def test_nvidia_smi_id_names_the_physical_gpu_cuda_sees():
    # Each GPU of the box, made CUDA's only device: its device 0 is that GPU
    # (cuInit reads CUDA_VISIBLE_DEVICES once per process, hence a process
    # per GPU).  No CUDA context is created.
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
        text=True).splitlines()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for line in smi:
        index, bus = (f.strip() for f in line.split(","))
        got = subprocess.check_output(
            [sys.executable, "-c",
             "from utility.numa import nvidia_smi_id; print(nvidia_smi_id(0))"],
            cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": index},
            text=True).strip()
        assert got.lower() == bus.lower()[-12:], (index, bus, got)
