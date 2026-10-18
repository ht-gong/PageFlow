#!/usr/bin/env python
"""polars_gpu's engine and its spill-pass meter and record builder (AGENTS.md,
"Spill pass"; plan §3.2, §4 F1/F2).

The engine (build_engine, every run, bench and spill pass): cudf-polars 26.06's
streaming executor on an explicit SPMDEngine, as upstream's own benchmarks
(--frontend spmd) build it. pl.GPUEngine(executor="streaming") is not used: its
default singleton watches the memory resource rapidsmpf's spill limit applies
to only while the singleton is built, so every later collect, the timed run
included, allocated on a cached CudaAsyncMemoryResource the limit never sees.
Its memory resource is the one rapidsmpf's spill limit watches, under an rmm
StatisticsResourceAdaptor for the exact per-pass device peak. Spill goes to
pinned host memory when RAPIDSMPF_PINNED_MEMORY=1 (polars/env.sh): the pinned
pool is built on the GPU's own NUMA node, since this rapidsmpf build reports
node 0 whatever the box, and it is unbounded (a capped pool aborts the query
instead of falling back to pageable memory). Hardware binding binds CPUs only:
its memory binding gets EPERM where set_mempolicy is not allowed, and its
fallback then moved the process's CPUs to GPU 0's node.

The meter (Meter, spill pass: RAPIDSMPF_STATISTICS=1): rapidsmpf's Statistics
for spill and restore copies, the adaptor's main record for device allocation,
the stats adaptor's counters for the device peak, and hooks on cudf-polars
internals for shuffle / allgather / operator output bytes, GPU callbacks and
single-partition fallbacks. Every hooked callable is pinned (check_versions).

  spill_stats.py record <q_dir> --sf SF --query N --evidence-dir DIR
  spill_stats.py settings <evidence_dir> --sf SF

`record` reads <q_dir>/meter.json, which run_tpch_polars.py writes after each
pass ({"warm": {...}, "timed": {...}}), and appends the query's record to
DIR/records.jsonl. That part imports no GPU library.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import spill_results as sr  # noqa: E402

ENGINE = "polars_gpu"
PINS = {"polars": "1.39.3", "cudf_polars": "26.06.00", "rapidsmpf": "26.06.00", "rmm": "26.06.00"}


# ---- the engine ----------------------------------------------------------------

def _hooked():
    """(object, attribute, expected parameter names) of everything the meter and
    build_engine wrap or replace."""
    import cudf_polars.callback as cb
    import cudf_polars.engine.core as ec
    import cudf_polars.engine.spmd as spmd
    from cudf_polars.streaming.actor_graph import tracing as tr
    from cudf_polars.streaming.actor_graph.collectives import allgather as ag
    from cudf_polars.streaming.actor_graph.collectives import shuffle as sh
    return (
        (cb, "_callback", None),
        (ec, "execute_ir_on_rank", None),
        (spmd, "Context", None),          # build_engine's NUMA-aware Context stands in for it
        (sh.ShuffleManager.Inserter, "insert_hash", ["self", "chunk", "columns_to_hash"]),
        (sh.ShuffleManager.Inserter, "insert_split", ["self", "chunk", "splits"]),
        (sh.ShuffleManager.Inserter, "insert_index", ["self", "chunk", "partition_map"]),
        (ag.AllGatherManager.Inserter, "insert", ["self", "sequence_number", "chunk"]),
        (tr.ActorTracer, "add_chunk", ["self", "table"]),
    )


def check_versions():
    """Raise unless the versions and every hooked callable are the pinned ones."""
    import inspect
    import cudf_polars
    import polars as pl
    import rapidsmpf
    import rmm
    got = {"polars": pl.__version__, "cudf_polars": cudf_polars.__version__,
           "rapidsmpf": rapidsmpf.__version__, "rmm": rmm.__version__}
    if got != PINS:
        raise RuntimeError(f"polars/spill_stats.py is pinned to {PINS}, found {got}")
    for obj, attr, params in _hooked():
        fn = getattr(obj, attr, None)
        if fn is None:
            raise RuntimeError(f"polars/spill_stats.py: {obj!r}.{attr} is gone")
        if params is not None and list(inspect.signature(fn).parameters) != params:
            raise RuntimeError(f"polars/spill_stats.py: {obj!r}.{attr} signature changed")


def gpu_numa_node():
    """The NUMA node of the current CUDA device, from its PCI bus id (0 if the
    box reports none)."""
    from cuda.bindings import runtime as rt
    _, dev = rt.cudaGetDevice()
    _, bus = rt.cudaDeviceGetPCIBusId(32, dev)
    bus = bytes(bus).split(b"\0")[0].decode().lower()
    with open(f"/sys/bus/pci/devices/{bus}/numa_node") as f:
        return max(int(f.read()), 0)


class _NumaContext:
    """Stands in for rapidsmpf's Context in cudf_polars.engine.spmd, which calls
    only Context.from_options: the same Context, with the pinned pool (when
    options enable it) on the GPU's NUMA node."""
    numa = 0

    @staticmethod
    def from_options(logger, mr, options):
        from rapidsmpf.memory.buffer_resource import (AvailableMemoryMap, BufferResource,
                                                      periodic_spill_check_from_options,
                                                      stream_pool_from_options)
        from rapidsmpf.memory.pinned_memory_resource import PinnedMemoryResource
        from rapidsmpf.statistics import Statistics
        from rapidsmpf.streaming.core.context import Context
        pinned = options.get_or_default("pinned_memory", default_value=False)
        br = BufferResource(mr,
                            pinned_mr=PinnedMemoryResource(numa_id=_NumaContext.numa) if pinned else None,
                            memory_available=AvailableMemoryMap.from_options(mr, options),
                            periodic_spill_check=periodic_spill_check_from_options(options),
                            stream_pool=stream_pool_from_options(options),
                            statistics=Statistics.from_options(options))
        return Context(logger, br, options)


def build_engine(part_mb, mem_mode="async"):
    """The runner's streaming executor on an explicit SPMDEngine (module
    docstring). Base memory resource: async with a release threshold of 90 % of
    the card (the singleton's default) or UVM ("managed"), under an rmm
    StatisticsResourceAdaptor; rapidsmpf's options from RAPIDSMPF_*. Shut it
    down on the thread that built it (engine.shutdown())."""
    check_versions()
    import cudf_polars.engine.spmd as spmd
    from cudf_polars.engine.hardware_binding import HardwareBindingPolicy
    from cudf_polars.engine.spmd import SPMDEngine
    from cudf_polars.utils.config import MemoryResourceConfig, get_total_device_memory
    if mem_mode == "managed":
        base = {"qualname": "rmm.mr.PrefetchResourceAdaptor",
                "options": {"upstream_mr": {"qualname": "rmm.mr.ManagedMemoryResource"}}}
    else:
        base = {"qualname": "rmm.mr.CudaAsyncMemoryResource",
                "options": {"release_threshold": int(0.9 * get_total_device_memory())}}
    _NumaContext.numa = gpu_numa_node()
    spmd.Context = _NumaContext
    engine = SPMDEngine(
        executor_options={"fallback_mode": "warn", "target_partition_size": part_mb << 20},
        engine_options={
            "memory_resource_config": MemoryResourceConfig(
                qualname="rmm.mr.StatisticsResourceAdaptor", options={"upstream_mr": base}),
            "hardware_binding": HardwareBindingPolicy(memory=False),
        },
    )
    engine._spill_stats_mem_mode = mem_mode
    return engine


# ---- the meter -----------------------------------------------------------------

_HOST_TIERS = ("host", "pinned_host")


class Meter:
    """Per-pass movement of an engine from build_engine(): begin() before the
    pass, end() after it returns the pass's native record."""

    def __init__(self, engine):
        import rmm
        self.engine = engine
        self.ctx = engine.context
        self.br = self.ctx.br()
        self.adaptor = self.br.device_mr               # rapidsmpf RmmResourceAdaptor
        self.dev_stats = self.adaptor.upstream_mr       # rmm StatisticsResourceAdaptor
        if not isinstance(self.dev_stats, rmm.mr.StatisticsResourceAdaptor):
            raise RuntimeError("polars/spill_stats.py: engine not built by build_engine()")
        if not self.ctx.statistics().enabled:
            raise RuntimeError("polars/spill_stats.py: set RAPIDSMPF_STATISTICS=1")
        self.mem_mode = getattr(engine, "_spill_stats_mem_mode", "?")
        self.spill_limit = self.ctx.options().get_strings().get("spill_device_limit", "") or "80%"
        import threading
        self._lock = threading.Lock()
        self._zero()
        self._hook()

    def _zero(self):
        import collections
        self.n_gpu = 0
        self.mr_bad = 0
        self.shuffle = 0
        self.allgather = 0
        self.out = collections.Counter()

    def _hook(self):
        import rmm
        import cudf_polars.callback as _cb
        import cudf_polars.engine.core as _ec
        from cudf_polars.streaming.actor_graph import tracing as _tr
        from cudf_polars.streaming.actor_graph.collectives import allgather as _ag
        from cudf_polars.streaming.actor_graph.collectives import shuffle as _sh
        from rapidsmpf.memory.buffer import MemoryType
        m = self
        cb0, ex0 = _cb._callback, _ec.execute_ir_on_rank
        ins = _sh.ShuffleManager.Inserter
        h0, s0, i0 = ins.insert_hash, ins.insert_split, ins.insert_index
        ag0, add0 = _ag.AllGatherManager.Inserter.insert, _tr.ActorTracer.add_chunk

        def cnt_shuffle(chunk):
            n = chunk.data_alloc_size(MemoryType.DEVICE)
            with m._lock:
                m.shuffle += n

        def _callback(*a, **k):
            with m._lock:
                m.n_gpu += 1
            return cb0(*a, **k)

        def execute_ir_on_rank(*a, **k):
            if rmm.mr.get_current_device_resource() is not m.engine.memory_resource:
                with m._lock:
                    m.mr_bad += 1
            return ex0(*a, **k)

        def insert_hash(self, chunk, columns_to_hash):
            cnt_shuffle(chunk)
            return h0(self, chunk, columns_to_hash)

        def insert_split(self, chunk, splits):
            cnt_shuffle(chunk)
            return s0(self, chunk, splits)

        def insert_index(self, chunk, partition_map):
            cnt_shuffle(chunk)
            return i0(self, chunk, partition_map)

        def ag_insert(self, sequence_number, chunk):
            n = chunk.data_alloc_size(MemoryType.DEVICE)
            with m._lock:
                m.allgather += n
            return ag0(self, sequence_number, chunk)

        def add_chunk(self, *, table=None):
            if table is not None:
                n = sum(c.device_buffer_size() for c in table.columns())
                with m._lock:
                    m.out[self.ir_type] += n
            return add0(self, table=table)

        _cb._callback, _ec.execute_ir_on_rank = _callback, execute_ir_on_rank
        ins.insert_hash, ins.insert_split, ins.insert_index = insert_hash, insert_split, insert_index
        _ag.AllGatherManager.Inserter.insert = ag_insert
        _tr.ActorTracer.add_chunk = add_chunk

    def begin(self):
        import warnings
        self.ctx.statistics().clear()
        rec = self.adaptor.get_main_record()
        self._t0 = (rec.total(), rec.num_total_allocs())
        self._zero()
        self._base = self.dev_stats.push_counters()
        self._w = warnings.catch_warnings(record=True)
        self._warns = self._w.__enter__()
        warnings.simplefilter("always")

    def end(self):
        import re
        self._w.__exit__(None, None, None)
        popped = self.dev_stats.pop_counters()
        st = self.ctx.statistics().to_dict()
        rec = self.adaptor.get_main_record()
        copies, other = {}, []
        for name, s in st.items():
            mt = re.match(r"^copy-(\w+)-to-(\w+)-bytes$", name)
            if mt:
                copies[(mt[1], mt[2])] = (int(s["value"]), int(s["count"]))
        d2h = [copies.get(("device", t), (0, 0)) for t in _HOST_TIERS]
        h2d = [copies.get((t, "device"), (0, 0)) for t in _HOST_TIERS]
        for (src, dst), (b, _) in copies.items():
            if (src, dst) != ("device", "device") and (
                    "device" not in (src, dst) or not ({src, dst} - {"device"}) <= set(_HOST_TIERS)):
                other.append(f"{src}>{dst}:{b}")

        def stat(name):
            return st.get(name, {}).get("value", 0)
        pinned_peak = None
        if self.br.pinned_mr is not None:  # a lifetime peak, from the report text only
            for line in self.ctx.statistics().report(mr=self.adaptor, pinned_mr=self.br.pinned_mr).splitlines():
                if "using PinnedMemoryResource" in line:
                    v, u = line.split()[1:3]
                    pinned_peak = int(float(v) * {"B": 1, "KiB": 1 << 10, "MiB": 1 << 20,
                                                  "GiB": 1 << 30, "TiB": 1 << 40}[u])
        scan = self.out.pop("Scan", 0)
        return {
            "gpu_callbacks": self.n_gpu,
            "mr_ok": self.mr_bad == 0,
            "mem_mode": self.mem_mode,
            "spill_limit": self.spill_limit,
            "pinned": self.br.pinned_mr is not None,
            "spill_d2h_bytes": sum(b for b, _ in d2h),
            "spill_d2h_ops": sum(n for _, n in d2h),
            "spill_d2pinned_bytes": d2h[1][0],
            "restore_h2d_bytes": sum(b for b, _ in h2d),
            "restore_h2d_ops": sum(n for _, n in h2d),
            "spill_copy_s": round(sum(stat(f"copy-device-to-{t}-time") for t in _HOST_TIERS), 4),
            "restore_copy_s": round(sum(stat(f"copy-{t}-to-device-time") for t in _HOST_TIERS), 4),
            "host_alloc_bytes": int(sum(stat(f"alloc-{t}-bytes") for t in _HOST_TIERS)),
            "dev_peak_bytes": self._base.current_bytes + popped.peak_bytes,
            "dev_alloc_bytes": rec.total() - self._t0[0],
            "dev_allocs": rec.num_total_allocs() - self._t0[1],
            "pinned_peak_lifetime_bytes": pinned_peak,
            "d2d_copy_bytes": copies.get(("device", "device"), (0, 0))[0],  # fanout deep copies
            "shuffle_bytes": self.shuffle,
            "allgather_bytes": self.allgather,
            "scan_out_bytes": scan,
            "op_out_bytes": sum(self.out.values()),
            "single_partition_fallbacks": sum("partition" in str(w.message) for w in self._warns),
            "other_copies": ";".join(other),
        }


# ---- the record ----------------------------------------------------------------

def rows(timed, status):
    """The contract rows and identity checks of the timed pass's meter record."""
    q = "OK" if status not in ("TIMEOUT", "KILLED_DISK") else "PARTIAL"
    g = lambda k: timed.get(k) or 0
    rs = "rapidsmpf:"
    both = lambda kind, key, src, **kw: sr.row(kind, g(key), g(key), quality=q, source=src, **kw)
    pinned_peak = timed.get("pinned_peak_lifetime_bytes")
    out = [
        sr.row("scan", g("scan_out_bytes"), quality=q, source="ActorTracer.add_chunk of Scan"),
        both("spill_d2h", "spill_d2h_bytes", rs + "copy-device-to-{host,pinned_host}-bytes",
             count=g("spill_d2h_ops"), seconds=g("spill_copy_s")),
        both("restore_h2d", "restore_h2d_bytes", rs + "copy-{host,pinned_host}-to-device-bytes",
             count=g("restore_h2d_ops"), seconds=g("restore_copy_s")),
        both("shuffle", "shuffle_bytes", "ShuffleManager.Inserter"),
        both("allgather", "allgather_bytes", "AllGatherManager.Inserter"),
        both("fanout_d2d", "d2d_copy_bytes", rs + "copy-device-to-device-bytes"),
        both("materialized", "op_out_bytes", "ActorTracer.add_chunk of other operators"),
        sr.row("dev_alloc", g("dev_alloc_bytes"), g("dev_alloc_bytes"), g("dev_allocs"), quality=q,
               source="RmmResourceAdaptor main record"),
        both("peak_gpu", "dev_peak_bytes", "StatisticsResourceAdaptor pass peak"),
        sr.row("peak_host", pinned_peak, pinned_peak, quality="LIFETIME", source=rs + "pinned pool peak")
        if timed.get("pinned") and pinned_peak is not None
        else sr.na("peak_host", "pageable spill has no host counter"),
        sr.row("fallbacks", count=g("single_partition_fallbacks"), quality=q,
               source="single-partition fallback warnings"),
        sr.row("gpu_callbacks", count=g("gpu_callbacks"), quality=q, source="cudf_polars callbacks"),
    ]
    # Not among them: whether the timed pass spilled when its warm pass did.
    # Near the limit, spill is not deterministic (SF1 q19 under forced spill:
    # 4.4 MB in one warm pass, nothing in its timed pass); that the timed run
    # is under the spill limit at all (F1) is the resource check.
    checks = [
        sr.check("spill_d2h = host allocations", g("spill_d2h_bytes"), g("host_alloc_bytes")),
        sr.check("ran on the GPU", g("gpu_callbacks"), 1, ">="),
        sr.check("collects on the spill-limited resource", int(bool(timed.get("mr_ok"))), 1),
    ]
    return out, checks


def cmd_record(a):
    status, pt = sr.pass_time(os.path.join(a.evidence_dir, "pass.csv"), a.query)
    try:
        with open(os.path.join(a.q_dir, "meter.json")) as f:
            m = json.load(f)
        if "timed" not in m:
            raise LookupError("no timed pass in meter.json")
        rs, checks = rows(m["timed"], status)
        rec = sr.record(ENGINE, a.sf, a.query, status, rs + [pt], checks)
    except (OSError, LookupError, ValueError, KeyError) as e:
        rec = sr.stub(ENGINE, a.sf, a.query, a.evidence_dir, f"{type(e).__name__}: {e}")
    sr.append_record(a.evidence_dir, rec)


def cmd_settings(a):
    """The device spill limit and the host tier from the environment the pass
    ran with; the timed meter of a query for what rapidsmpf resolved."""
    m = {}
    for q_dir in sorted(os.listdir(a.evidence_dir)):
        p = os.path.join(a.evidence_dir, q_dir, "meter.json")
        if os.path.exists(p):
            with open(p) as f:
                m = json.load(f).get("timed", {})
            if m:
                break
    env = {k: v for k, v in os.environ.items() if k.startswith("RAPIDSMPF_") or k in ("GPU_MR", "GPU_PART_MB")}
    pinned = env.get("RAPIDSMPF_PINNED_MEMORY") == "1"
    sr.write_settings(a.evidence_dir, ENGINE, a.sf,
                      {"gpu_b": _bytes(env.get("RAPIDSMPF_SPILL_DEVICE_LIMIT")), "host_b": None, "disk_b": None},
                      [], versions=PINS, env=env, spill_limit_resolved=m.get("spill_limit"),
                      host_tier="pinned (unbounded pool on the GPU's NUMA node)" if pinned else "pageable",
                      smoke=os.environ.get("SPILL_SMOKE") == "1")


def _bytes(v):
    import re
    m = re.fullmatch(r"\s*([\d.]+)\s*([KMGT]i?B?)?\s*", str(v or ""))
    if not m:
        return None
    unit = (m.group(2) or "").rstrip("B")
    return int(float(m.group(1)) * {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12,
                                    "Ki": 2**10, "Mi": 2**20, "Gi": 2**30, "Ti": 2**40}[unit])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("q_dir")
    r.add_argument("--sf", type=int, required=True)
    r.add_argument("--query", type=int, required=True)
    r.add_argument("--evidence-dir", required=True)
    s = sub.add_parser("settings")
    s.add_argument("evidence_dir")
    s.add_argument("--sf", type=int, required=True)
    a = ap.parse_args()
    {"record": cmd_record, "settings": cmd_settings}[a.cmd](a)


if __name__ == "__main__":
    main()
