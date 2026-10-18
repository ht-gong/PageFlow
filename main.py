import argparse
import time
import json
import torch
import pandas as pd
import pickle
import os, gc
import sys
from collections import defaultdict

# TQP: Parsing Layer
import parsing
# TQP: Tensor Programs
from operators.scan import tqp_scan
from operators.filter import tqp_filter 
from operators.project import tqp_project
from operators.aggregate import tqp_hash_aggregate
from operators.sort import tqp_sort
from operators.output import host_frame, tqp_output_to_host
from operators.hashjoin import join_call_args, tqp_join
from tensor_group import TensorGroup
from uvm import executor as uvm_executor
from uvm.column import Payloads
from uvm.scan import scan_to_managed

from utility.logger import torch_profiler, perf_logger, message_logger, set_torch_profiler, set_perf_logger, set_message_logger
from utility.argparse_utils import enum_by_value
from utility.config import load_config
from utility.numa import pin_to_gpu_node, gpu_numa_node, nvidia_smi_id
from utility import numa
from utility import rmm_arena
from IO.transfer.data_exchange import set_exchange_to_naive
from IO.pages.pinned_mem import PinnedMemory
from IO.cache.config import set_global_buffer_strategy, BufferStrategy
from IO.cache.cache_region import CacheRegion
from IO.cache.native_cache import CLEAN_PERCENT
from adaptive.instage_optimization import set_fixed_compression, set_zero_copy_selectivity, get_fixed_compression, set_adaptive_compression, get_adaptive_compression
from compression.compression_algorithms import algo_names as _algo_names_fn
from IO.data_scanner import scan_to_ram
import constants
from torch.cuda.memory import CUDAPluggableAllocator, change_current_allocator
import torch.cuda.nvtx as nvtx
import TQPlib.tqpmemory as cmp

_is_warmup          = False
# qid -> (time to failure or None, error) of each query whose SF1 warm pass
# failed.  The real run records it FAIL and does not run it.
_warm_failures      = {}
# Armed by --sizing_search: holds the validated SizingSearchConfig, read by
# exec() to keep the search's converged budgets per cache-region size.
# None ⇒ the sizing plane is absent and every pool/rearrange path is
# byte-identical to the pre-sizing build.
_sizing_cfg         = None
# --uvm: every column and intermediate in CUDA managed memory, the operators
# run by the UVM executor (uvm/), no pinned pool, cache or compression.
_uvm                = False


def _gc_collection_counts():
    """Return CPython's process-lifetime collection count per generation."""
    return tuple(int(stats['collections']) for stats in gc.get_stats())


def _begin_timed_gc_guard():
    """Move cyclic collection out of the canonical query interval.

    Reference counting remains active.  When cyclic GC is enabled on entry,
    collect the live graph once at the pre-timing boundary and then disable
    automatic cyclic collection.  Collection counters are sampled *after* the
    forced collection so a non-zero timed delta also catches an explicit
    ``gc.collect()`` accidentally added to an operator later.
    """
    state = {
        'enabled_on_entry': gc.isenabled(),
        'disabled_by_guard': False,
        'disabled_during_timing': False,
        'restore_pending': False,
        'pre_collect_ns': 0,
        'pre_collected': 0,
        'post_collect_ns': 0,
        'post_collected': 0,
        'collections_start': (),
        'collections_end': (),
        'timed_collection_delta': (),
    }
    if state['enabled_on_entry']:
        started = time.perf_counter_ns()
        state['pre_collected'] = gc.collect(2)
        state['pre_collect_ns'] = time.perf_counter_ns() - started
        gc.disable()
        state['disabled_by_guard'] = True
        state['restore_pending'] = True

    state['disabled_during_timing'] = not gc.isenabled()
    state['collections_start'] = _gc_collection_counts()
    return state


def _end_timed_gc_guard(state):
    """Close the timed counter interval without re-enabling collection."""
    state['collections_end'] = _gc_collection_counts()
    state['timed_collection_delta'] = tuple(
        end - start
        for start, end in zip(
            state['collections_start'], state['collections_end'])
    )
    if any(state['timed_collection_delta']):
        print(
            '[variance] WARNING: cyclic collection entered the timed region: '
            f"{state['timed_collection_delta']}",
            flush=True,
        )


def _collect_and_restore_gc(state):
    """Collect released query objects, then restore the entry GC state."""
    if not state or not state['restore_pending']:
        return
    started = time.perf_counter_ns()
    try:
        state['post_collected'] = gc.collect(2)
        state['post_collect_ns'] = time.perf_counter_ns() - started
    finally:
        gc.enable()
        state['restore_pending'] = False


def _restore_gc_after_failure(state):
    """Exception-safety fallback; preserve a caller-disabled collector."""
    if state and state['restore_pending']:
        gc.enable()
        state['restore_pending'] = False


def _gpu_total_gib(gpu_index=0):
    """Total memory of CUDA device <gpu_index> in whole GiB (floor), via
    nvidia-smi.  Used to default --cmp_pool_gb to (capacity - 2 GB).  Returns
    None on failure."""
    import subprocess
    try:
        o = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total',
             '--format=csv,noheader,nounits', '-i', nvidia_smi_id(gpu_index)]).decode().strip()
        return int(o.splitlines()[0]) // 1024          # MiB -> GiB (floor)
    except Exception:
        return None


# Default pinned host pool, in GiB: max(1, ceil(1.05 x need x SF)).  need is the
# smallest pool, per SF, that runs every query under every --compression mode at
# SF1.  At SF10 (every mode) and SF100 (CASCADED, adaptive, LZ4, RAW) the
# largest need per SF is within 0.4% of it, but Q18's grows with SF (RAW: 0.37,
# 0.44, 0.53 GiB per SF at SF1, 10, 100), so re-measure at a large SF too.  need
# is in 1e-4 GiB per SF and the headroom in percent, so the rounding is exact.
_PINNED_NEED_PER_SF = 6524       # 1e-4 GiB per SF: CASCADED, Q19, SF1 (standard data)
_PINNED_HEADROOM_PCT = 105


def _pinned_capacity_gb(scale_factor):
    """Default --pinned_pool_gb: max(1, ceil(1.05 x need x SF)) GiB."""
    return max(1, -(-_PINNED_NEED_PER_SF * _PINNED_HEADROOM_PCT * scale_factor
                    // 1_000_000))


def _report_freelist(qid):
    """The free-list baseline's counters for the query just run (since its
    timer started; ``IO.pages.device_regions.take_stats``, which resets
    them) with the arena's peak: one ``[freelist]`` line, printed even under
    --no_debug (not for the warmup), and the perf report's ``freelist``
    section."""
    from IO.pages import device_regions
    from IO.telemetry import gates, movement
    st = device_regions.take_stats()
    st["arena_peak_b"] = cmp.get_peak_bytes()
    if gates.EXPERIMENT and st["regions"]:
        # The evictions' copies home: a page there is a region, its logical
        # bytes what its records decode to (a RAW one's: its stored ones).
        movement.record("spill_evict", st["regions"], st["logical_bytes"],
                        st["bytes"])
    if not globals().get('_is_warmup'):
        gib = 1024 ** 3
        print(f"[freelist] Q{qid}: {st['miss_calls']} misses "
              f"({st['miss_fragmented']} fragmented, {st['miss_failed']} "
              f"failed); {st['evictions']} evictions moved {st['regions']} "
              f"regions, {st['bytes'] / gib:.2f} GiB, in {st['seconds']:.3f} s"
              f" (miss path {st['miss_ns'] / 1e9:.3f} s); GPU-resident "
              f"regions peak {st['peak_live_b'] / gib:.2f} GiB of data in "
              f"{st['peak_live_arena_b'] / gib:.2f} GiB of arena blocks "
              f"(pinned at a miss {st['pinned_at_miss_b'] / gib:.2f} GiB); "
              f"arena peak {st['arena_peak_b'] / gib:.2f} GiB"
              + (f"; reserve {st['reserve_b'] / gib:.2f} GiB: "
                 f"{st['reserve_draws']} draws gave "
                 f"{st['reserve_drawn_b'] / gib:.2f} GiB, refilled "
                 f"{st['reserve_refilled_b'] / gib:.2f} GiB, held at least "
                 f"{st['reserve_min_b'] / gib:.2f} GiB"
                 if st['reserve_b'] else ""), flush=True)
    return st


def _report_lend(qid, cache):
    """The lending of the query just run (since its cache was attached;
    ``NativeGranuleCache.take_lend_stats``, which resets the counters and
    files the evictions in the movement ledger): one ``[lend]`` line,
    printed even under --no_debug (not for the warmup), and the perf
    report's ``lend`` section."""
    st = cache.take_lend_stats()
    st["block_b"] = cache.G
    if not globals().get('_is_warmup'):
        gib = 1024 ** 3
        print(f"[lend] Q{qid}: {st['misses']} misses lent for {st['lent']} "
              f"({st['short']} short, {st['failed']} failed); "
              f"{st['blocks_lent']} blocks lent "
              f"({st['blocks_lent'] * cache.G / gib:.2f} GiB), "
              f"{st['blocks_returned']} returned, {st['lent_now']} lent at "
              f"the end; evicted {st['evicted_dead']} dead, "
              f"{st['evicted_clean']} clean, {st['evicted_dirty']} dirty "
              f"({st['dirty_b'] / gib:.2f} GiB home), relocated "
              f"{st['relocated']} in {st['copies']} copies; "
              f"{st['ns'] / 1e6:.2f} ms", flush=True)
    return st


def _report_sizing(qid):
    """The tick sizing sessions of the query just run, one ``[sizing]``
    line (each session's sizes, measurements and steps), printed even under
    --no_debug (not for the warmup); none without the search."""
    ctl = globals().get('_tick_ctl')
    if ctl is None:
        return
    report = ctl.take_report()
    if report and not globals().get('_is_warmup'):
        print(f"[sizing] Q{qid}: " + " || ".join(report), flush=True)


def _record_run_metadata(perf_dir: str, args=None):
    """Snapshot argv into ``perf_dir`` so a perf_report folder records which
    command produced it.  Writes ``cmdline.json``, overwriting on repeat runs
    targeting the same directory, and the header of ``query_results.csv``.
    With ``args`` (main.py passes them under TQP_EXPERIMENT_TELEMETRY, and
    always under --uvm), ``cmdline.json`` also holds the settings as resolved (the pool sizes, the
    cache, the codecs, the partition budget, the NUMA binding) and the
    engine's environment variables; ``_note_run_metadata`` adds the codec
    map, the versions and the GPU.  The spill pass's settings read them
    (bench/tqp_spill_stats.py).  Returns the ``effective`` settings (None
    without ``args``), for a later ``_note_run_metadata`` to extend.
    """
    import shlex, datetime
    os.makedirs(perf_dir, exist_ok=True)

    info = {
        'argv': list(sys.argv),
        'command': shlex.join(sys.argv),
        'cwd': os.getcwd(),
        'timestamp_utc': datetime.datetime.utcnow().isoformat() + 'Z',
    }
    if args is not None:
        info['effective'] = {
            'cmp_pool_gb': args.cmp_pool_gb,
            # --uvm runs RAW with no pinned pool or cache region: what it
            # ignores is None.
            'pinned_pool_gb': args.pinned_pool_gb,
            'slot_pool_gb': args.slot_pool_gb,
            # Per query, in --q order, when set (the cache region resized per query).
            'slot_pool_gb_list': args.slot_pool_gb_list,
            'buffer_strategy': args.buffer_strategy.value,
            'freelist_reserve_gb': args.freelist_reserve_gb,
            'compression': args.compression,
            'compression_search': args.compression_search,
            'sizing_search': args.sizing_search,
            'chunk_size_mb': args.chunk_size_mb,
            'partition_budget_mb': args.partition_budget_mb,
            'mask_selectivity': args.mask_selectivity,
            'numa_bind': numa.NUMA_BIND,
            'warmup': not args.no_warmup,
        }
        info['env'] = {k: v for k, v in sorted(os.environ.items())
                       if k.startswith('TQP_') or k in (
                           'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
                           'KVIKIO_NTHREADS')}
    with open(os.path.join(perf_dir, 'cmdline.json'), 'w') as f:
        json.dump(info, f, indent=2)
        f.write('\n')
    # exec() appends one row per query occurrence.  Starting the file here
    # drops an earlier run's rows, but a run that exits before this point (a
    # flag error, no GPU) leaves them: clear a reused dir before the run.
    with open(os.path.join(perf_dir, 'query_results.csv'), 'w') as f:
        f.write('qid,status,approx_time,rows_or_error,load_time\n')
    return info.get('effective')

def _note_run_metadata(perf_dir: str, **items) -> None:
    """Add ``items`` to the ``cmdline.json`` that ``_record_run_metadata``
    wrote (what is known only later, such as the codec map)."""
    path = os.path.join(perf_dir, 'cmdline.json')
    with open(path) as f:
        info = json.load(f)
    info.update(items)
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
        f.write('\n')


def _uvm_metadata(arena_b, budget_b, balloon_b):
    """``cmdline.json``'s ``effective.uvm``: the managed arena, the GPU
    budget and the balloon, in bytes, the driver version and the
    ``nvidia_uvm`` module's parameters (those that can be read)."""
    import subprocess
    try:
        driver = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader',
             '-i', nvidia_smi_id(0)]).decode().strip()
    except Exception:
        driver = None
    params = {}
    pdir = '/sys/module/nvidia_uvm/parameters'
    for name in sorted(os.listdir(pdir)) if os.path.isdir(pdir) else ():
        try:
            with open(os.path.join(pdir, name)) as f:
                params[name] = f.read().strip()
        except OSError:
            pass
    return {'arena_b': arena_b, 'budget_b': budget_b, 'balloon_b': balloon_b,
            'driver': driver, 'nvidia_uvm': params}


def _uvm_cap_gpu_budget(budget_gb, arena_gb):
    """The GPU budget under --uvm: a balloon of plain device memory holds the
    rest of GPU 0 for the life of the process, so managed pages get at most
    ``budget_gb`` GiB (less what lives outside the arena).  Sized once the
    modules every query loads (torch's, cuDF's decode, the engine's kernels)
    are resident, after the warmup, so they do not come out of the budget.
    Exits when the card has less than the budget free (another tenant).
    Returns the balloon's bytes."""
    budget_b = budget_gb * 1024 ** 3
    torch.cuda.synchronize()
    free_b = torch.cuda.mem_get_info()[0]
    if free_b < budget_b:
        sys.exit(f"[uvm] GPU 0 has {free_b / 1024 ** 3:.2f} GiB free, less "
                 f"than the GPU budget (--cmp_pool_gb {budget_gb} GiB): use "
                 f"an idle GPU or a smaller --cmp_pool_gb")
    balloon_b = free_b - budget_b
    if balloon_b:
        cmp.reserve_device(balloon_b)
    print(f"[uvm] managed arena {arena_gb} GiB, GPU budget {budget_gb} GiB "
          f"(balloon {balloon_b / 1024 ** 3:.2f} GiB)", flush=True)
    return balloon_b


def _uvm_begin_occurrence() -> None:
    """The arena reset before an occurrence's scan under --uvm, so no page an
    earlier query touched stays resident anywhere and this query's
    intermediates are first touched on the GPU.  The earlier occurrence's
    columns must be gone first: the caller drops its tensor group, scan and
    payloads, and the collection here frees what a reference cycle may
    still hold.  A live allocation fails the occurrence."""
    gc.collect()
    cmp.remake()


def _write_failed_report(perf_dir, scale_factor, qid, occurrence, fail_s,
                         error, host_pool, cache):
    """Under TQP_EXPERIMENT_TELEMETRY, a failed query's partial perf report:
    ``wall`` with ``status`` FAIL (no ``approx_time``), and the ledger and
    the residency as far as the query got.  A report that cannot be written
    is skipped with a line; the query's row says FAIL either way."""
    from IO.telemetry import gates, movement, residency
    if not gates.EXPERIMENT:
        return
    try:
        report = {
            'wall': {'occurrence': occurrence, 'approx_time': None,
                     'status': 'FAIL', 'time_to_failure': fail_s,
                     'error': error[:300]},
            'movement': movement.report(),
            'residency': residency.report(host_pool, cache),
        }
        suffix = (f"_occ{occurrence}"
                  if os.environ.get('TQP_PERF_PER_OCCURRENCE') else "")
        with open(os.path.join(perf_dir, f"SF{scale_factor}_{qid}{suffix}.json"),
                  'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
    except Exception as e:
        print(f"[QFAIL] Q{qid}: no partial perf report ({type(e).__name__}: "
              f"{e})", flush=True)


def _materialize_deferred(tensor_group, nodes, cpu_mem_pool, applier, page_cache):
    """Gather deferred payload columns just before the op that first reads
    them.

    Each deferred column rode a row lineage through its join chain (see
    operators/hashjoin.py): the seed join's out_index, composed forward by
    each later join.  ``nodes`` are the op's ``deferred`` gather nodes, one
    per group: sources are the group's columns' pre-chain instances, the
    index is the lineage, and the outputs (typed as their columns) are what
    the op's own nodes read.  The lifetime rule frees the lineage
    after its last gather (a group split across two ops, as in Q22, keeps it
    until the second).
    """
    from IO.pipeline.rearrange_pipeline import rearrange_pipeline
    for node in nodes:
        rearrange_pipeline(node, cpu_mem_pool, 'inner',
                           applier=applier, page_cache=page_cache)
        for inst in node.outputs:
            tensor_group[inst.local] = inst.ccr


def warmup(qid):
    """SF=1 dry run to prime CUDA context, kernels, and memory pools.

    Always runs Q1 first (small aggregation — primes scan/filter/aggregate
    kernels reliably) followed by the requested qids.  Some queries — e.g.
    Q5 — fail with a CUDA index-out-of-bounds assert when run standalone
    at SF=1 without prior CUDA state, so Q1 acts as a universal primer.

    A query that fails here is recorded FAIL by the real run and not run
    again (``_warm_failures``).  A failed primer fails only Q1: each other
    query's warm pass is its own SF1 run, which is what warms it.
    """
    global _is_warmup
    _is_warmup = True
    # Warmup exclusion: sizing sessions are disabled so warmup's
    # differently-resident pipes contribute no fit samples and run the
    # static rearrange window.
    from adaptive.adaptive_driver import set_gather_sizing_enabled
    set_gather_sizing_enabled(False)
    try:
        warmup_qs = [1]
        for q in qid:
            if q not in warmup_qs:
                warmup_qs.append(q)
        exec(1, warmup_qs, None, None, True, None, False)
    finally:
        _is_warmup = False
        set_gather_sizing_enabled(True)

def exec(scale_factor, qids, perf_report, log, no_debug, torch_profile_dir,
         cuda_profile_markers=False, slot_pool_gb=4,
         slot_pool_gb_list=None, pinned_pool_gb=None):
    # load_config applies the TQP_TABLES_PATH override, so a container (or
    # any host whose data lives elsewhere) does not need a local edit to the
    # tracked config.json.
    config_json = load_config()
    with open('./queries/schema.json', 'r') as f:
        schema_json = json.load(f)

    block_mb = 0.5  # 0.5 MB blocks allow flexible page sizes for sort output
    # Pinning this pool can take MINUTES at large SF — it is NOT hung (see
    # AGENTS.md).  The pool is freed per query (free_all below), so it only
    # needs to hold the single heaviest query at this SF.  Every page of it is
    # bound to GPU 0's NUMA node; with TQP_NUMA_BIND off it is not bound, and
    # the real run's pages land there by first touch, since the threads are
    # pinned to the node after warmup (pin_to_gpu_node).
    # Under --uvm there is no pinned pool: the scan leaves the columns in
    # managed memory on the node (uvm/scan.py).
    pinned_pool = None
    if not _uvm:
        _pinned_gb = (
            _pinned_capacity_gb(scale_factor)
            if pinned_pool_gb is None else pinned_pool_gb
        )
        _numa_node = gpu_numa_node(0)
        print(f"Pinning {_pinned_gb} GB of CPU memory on NUMA node {_numa_node}"
              f"{'' if numa.NUMA_BIND else ' (not bound: TQP_NUMA_BIND off)'} "
              f"(can take minutes at large SF)...", flush=True)
        pinned_pool = PinnedMemory(capacity_gb=_pinned_gb, block_size_mb=block_mb,
                                   numa_node=_numa_node)
        print("done!", flush=True)
    
    print("Allocating GPU Memory...", flush=True)
    if slot_pool_gb_list is not None:
        assert len(slot_pool_gb_list) == len(qids), (
            f"slot_pool_gb_list length {len(slot_pool_gb_list)} must match "
            f"--q length {len(qids)}"
        )
    from IO.cache.config import buffer_strategy as _strategy, BufferStrategy as _BS
    from IO.pages import device_regions as _device_regions

    # The arena's cache region (IO/cache/cache_region.py): the granule
    # cache's slots, the arena's top blocks behind a boundary.  A run without
    # a cache has no region.  Its size is --slot_pool_gb, moved between
    # queries by --slot_pool_gb_list, and none for an occurrence without a
    # cache (IO.cache.config.CORE == "cpu").  It is sized for an
    # occurrence's operators only: after its scan (``_size_region``), and
    # emptied at its end (``_release_pools``).
    cache_region = (None if _strategy() in (_BS.CPU, _BS.FREELIST)
                    else CacheRegion())

    _native = {}      # policy name -> the granule cache with that policy

    def _native_cache(policy_name):
        # Granules of the region producer's regions (IO/cache/native_cache.py),
        # in the cache region; their homes are the producer's, and the home
        # writer copies granules into them.  Built once per policy (the
        # policy may change per occurrence), cleared between queries.
        if policy_name not in _native:
            from IO.cache.native_cache import NativeGranuleCache
            from IO.codec.home_writer import copy_home
            _native[policy_name] = NativeGranuleCache(
                cache_region, policy=policy_name,
                copy_home=copy_home, clean_percent=CLEAN_PERCENT)
        return _native[policy_name]

    def _set_region(gb):
        # Move the boundary so the region holds gb GB (0: no region); True
        # when it moved.  The caches are empty here (cleared at the end of
        # every query); they take the region's blocks as they are after the
        # move.  A grow the arena refuses raises.
        if not cache_region.set_size(int(gb * 1024**3)):
            return False
        for c in _native.values():
            c.clear()
        return True

    region_b_shown = None     # the region size the last [resize] line gave

    def _size_region(qid, gb):
        # Size the region for an occurrence's operators, after its scan and
        # before its timer.  The region is empty until then (it empties at
        # every query's end and the scan never uses it), so the scan had the
        # whole arena.  The scan's frees complete first, so its blocks are
        # free; a grow the arena still refuses raises, and the query fails.
        nonlocal region_b_shown
        torch.cuda.synchronize()
        _set_region(gb)
        _st = cmp.get_arena_stats()
        if _st['region_b'] != region_b_shown:
            region_b_shown = _st['region_b']
            print(f"[resize] Q{qid} cache region -> {gb:g} GB: "
                  f"region {_st['region_b']} B, operator space "
                  f"{_st['capacity_b']} B ({_st['free_b']} B free)",
                  flush=True)

    # None under --uvm, which runs with no cache.
    slot_gb = None if slot_pool_gb is None else float(slot_pool_gb)
    print("done!", flush=True)

    def _release_pools():
        """Return all per-query GPU/host memory to the pools.  Shared by the
        success tail and the failure handler so the two paths can't drift."""
        # No more lending, from any cache, before any is cleared: the
        # region is about to move.
        for c in _native.values():
            c.detach()
        for c in _native.values():
            c.clear()
        if cache_region is not None:
            # The region empties at every query's end and is sized again
            # after the next query's scan (``_size_region``), so a scan has
            # the whole arena.  Blocks still lent stay in the operator space
            # with the rest of the region.  The device syncs first, so no
            # copy still reads a slot once its block is operator space.
            torch.cuda.synchronize()
            _set_region(0)
        if _device_regions.armed():
            # Regions a query left registered: their GPU bytes go now, and
            # none can be evicted into a home the pinned pool hands out next.
            _device_regions.release_all()
        if pinned_pool is not None:
            pinned_pool.free_all()
        # Managers persist across queries (their cached configs must), but
        # their retained scratch — several hundred MB per manager, allocated
        # mid-query — would accumulate and fragment the arena over a suite
        # (Q13/Q17/Q18/Q21/Q22 OOMed at SF100 without this).
        # deallocate_gpu_mem() frees it; the next query's first use
        # re-allocates lazily.  This call MUST stay in _release_pools.
        from TQPlib.tqpcompress import mgr_release_scratch
        mgr_release_scratch()

    set_message_logger(log, not no_debug)
    
    with open("results.out", "w", encoding="utf-8") as f: # Clear outputs for this run
        pass
    # Keep the canonical timings next to the per-query JSON reports as well as
    # in the historical repo-root append-only file.  A report directory
    # describes one invocation, so truncate its copy at invocation start.
    # This happens before every timed query and therefore cannot perturb a
    # measured interval.
    _report_query_times = (
        os.path.join(perf_report, "query_times.csv") if perf_report else None
    )
    if _report_query_times is not None:
        with open(_report_query_times, "w", encoding="utf-8"):
            pass
    
    def _record_result(qid, status, seconds, rows_or_error, load_s):
        """One row of the report's query_results.csv; a time not measured
        stays empty."""
        if not perf_report:
            return
        secs, load = ('' if s is None else f'{s:.6f}' for s in (seconds, load_s))
        with open(os.path.join(perf_report, 'query_results.csv'), 'a',
                  encoding='utf-8') as f:
            f.write(f'{qid},{status},{secs},{rows_or_error},{load}\n')

    # Per-qid wall times across occurrences.  Repeating a query in --q (the
    # supported way to get repeated-run stability) accumulates here so a median
    # can be reported per repeated query at the end of the run.
    per_qid_times = defaultdict(list)
    # --uvm: the occurrence's payloads (uvm.column.Payloads), dropped where
    # the occurrence ends.
    _uvm_payloads = None
    for _q_idx, qid in enumerate(qids):
        _query_gc = q_page_cache = None
        q_start_time = load_time = None
        _recorded = False
        if qid in _warm_failures and not globals().get('_is_warmup'):
            # Its SF1 warm pass failed, so it is not timed.
            _fail_s, _err = _warm_failures[qid]
            print(f"[QFAIL] Q{qid}: SF1 warm pass: {_err} (not timed)", flush=True)
            _record_result(qid, "FAIL", _fail_s, f"SF1 warm pass: {_err}", None)
            continue
        try:
            # Bind (SF, qid) for the adaptive sizing plane's model keys/logs.
            constants.set_query_context(scale_factor, qid)

            # This occurrence's cache (IO.cache.config.CORE, read here so
            # switch_ab.py can alternate it per occurrence): the granule
            # cache in the region, or none, with no region (the boundary at
            # the arena's top).  --slot_pool_gb_list sets the region's size
            # per query.  A run without a cache has no region.
            import IO.cache.config as _cache_cfg
            q_page_cache = q_cache_region = None
            want_gb = 0.0
            if cache_region is not None:
                if _cache_cfg.CORE not in ("native", "cpu"):
                    raise ValueError(
                        f"IO.cache.config.CORE={_cache_cfg.CORE!r}: expected "
                        "native or cpu")
                resized = False
                if slot_pool_gb_list is not None:
                    new_slot_gb = float(slot_pool_gb_list[_q_idx])
                    resized = new_slot_gb != slot_gb
                    slot_gb = new_slot_gb
                want_gb = slot_gb if _cache_cfg.CORE == "native" else 0.0
                want_b = (int(want_gb * 1024**3) // cache_region.block
                          * cache_region.block)
                if _sizing_cfg is not None:
                    from adaptive.sizing_controller import (
                        reset_run_store, select_run_store)
                    if resized:
                        # A new region size is fresh evidence: re-open
                        # frozen per-instance convergence state.
                        reset_run_store()
                    # Converged budgets hold for the operator space they
                    # converged in: keep them per region size, so the
                    # occurrences with and without the cache (switch_ab.py
                    # alternates them) each reuse only their own.
                    select_run_store(want_b)
                if _cache_cfg.CORE == "native":
                    q_page_cache = _native_cache(
                        _cache_cfg.POLICY or _strategy().value)
                    q_cache_region = cache_region

            # Read SQL Statement

            set_perf_logger(log, True)
            set_torch_profiler(log, True, os.path.join(torch_profile_dir, str(qid)) if torch_profile_dir else None)

            # Per-query state: the sizing and compression-search records,
            # the movement ledger, and any copy log a failed query left open.
            from adaptive.run_stats import (
                reset_sizing_stats as _reset_szs,
                reset_compression_search as _reset_csr,
            )
            from IO.telemetry import copy_events as _copy_events
            from IO.telemetry import gates as _gates
            from IO.telemetry import movement as _movement
            from IO.telemetry import residency as _residency
            _reset_szs()
            _reset_csr()
            _movement.reset()
            _residency.reset()
            _copy_events.reset()

            print("======================", flush=True)
            print(f"Start Query {qid}", flush=True)
            print("======================", flush=True)

            # Load the physical plan, lowering the plan text if no pickle exists
            obj_name = f'./queries/pickles/planobj-q{qid}.pkl'
            if os.path.exists(obj_name):
                message_logger().info("Plan exists. Loading.")
                with open(obj_name, 'rb') as file:
                    query_plans = pickle.load(file)
            else:
                # Lower the checked-in Spark plan text; never start Spark here.
                # Plan column ids are Spark expression ids, which depend on the
                # JVM's history, so text is captured only by
                # parsing/regenerate.py (`make plans`), one fresh JVM per query.
                plan_text_path = f'./queries/plan-q{qid}.txt'
                if not os.path.exists(plan_text_path):
                    raise FileNotFoundError(
                        f"{plan_text_path} not found; capture it with `make plans Q={qid}`")
                message_logger().info("Plan object does not exist. Lowering %s.", plan_text_path)
                with open(plan_text_path, 'r', encoding='utf-8') as file:
                    plan_text = file.read()
                with perf_logger().time("Plan lowering"):
                    query_plans = parsing.ParsingLayer.lower(plan_text, qid)

                with open(obj_name, 'wb') as file:
                    pickle.dump(query_plans, file)

            # Re-run deterministic static analyses after every load. Cached plans
            # retain the stable SIMPLE representation; query corrections,
            # relational uniqueness proofs, and payload deferral are owned by
            # the parsing package and rebuilt in dependency order.
            parsing.prepare_query_plans(query_plans, qid)
            # The query's static dataflow graph, built here from the very
            # plans this occurrence executes (warmup included); nothing is
            # checked in.  Its nodes are attached to the operators (args['nodes']);
            # the applier resolves codecs off its instances.
            from dataflow.dataflow_gen import build_query_dataflow
            query_dataflow = build_query_dataflow(query_plans, schema_json, qid)

            for q in query_plans:
                if not no_debug:
                    print('=' * 20, flush=True)
                    print(q.ops_idx)
                    print('-' * 20, flush=True)
                    for k, v in q.ops_dict.items():
                        print(k, ':', v, flush=True)
                    print('-' * 20, flush=True)
                    print(q.query_parent_map)
                    print('-' * 20, flush=True)
                    print(q.query_child_map)
                    print('-' * 20, flush=True)

            if get_adaptive_compression() or get_fixed_compression() != "RAW":
                # Plan-driven compression: one applier over the query's
                # dataflow graph and the typed policy rule (each typed
                # instance resolves its (mapped_type, is_sorted) class
                # through the CompressionPolicy: adaptive = profiled
                # per-class best, --compression X = fixed(X)).  The builder
                # typed scan instances from the schema, all unsorted, exactly
                # as the scan path materializes reps, and intermediates by
                # static inference; untyped instances degrade to RAW.
                from dataflow.compression_applier import CompressionApplier
                from adaptive.instage_optimization import (
                    active_compression_policy, get_fixed_search_json)
                _pol = active_compression_policy()
                # --compression_search fixed: per-tag pins from the JSON,
                # validated against the graph (an untyped instance pinned
                # non-RAW is rejected, and a scan pin must name the rep the
                # scan path materializes).  A violation aborts the run
                # (SystemExit is not a [QFAIL]).
                _pins = None
                _fixed_json = get_fixed_search_json()
                if _fixed_json:
                    from dataflow.compression_applier import (
                        load_fixed_search_json, validate_pins)
                    _pins = load_fixed_search_json(_fixed_json, qid)
                    _viol = validate_pins(
                        query_dataflow, _pins,
                        scan_candidate=lambda inst: (
                            _pol.compressed_format(*inst.type_sorted)
                            if inst.type_sorted is not None else None))
                    if _viol:
                        raise SystemExit(
                            f"[compression_search] fixed: Q{qid}: {len(_viol)} "
                            f"invalid pin(s) in {_fixed_json}:\n  "
                            + "\n  ".join(_viol))
                    print(f"[compression_search] fixed: Q{qid}: {len(_pins)} "
                          f"pin(s) applied", flush=True)
                applier = CompressionApplier(query_dataflow, policy=_pol, pins=_pins)
            else:
                applier = None  # RAW mode: everything already RAW, no plan needed

            # Device Info
            message_logger().info("%s %s", "GPU available =", torch.cuda.is_available())
            message_logger().info("%s %s", "GPU count =", torch.cuda.device_count())
            message_logger().info("%s %s", "GPU name =", torch.cuda.get_device_name(0))
            message_logger().info("[CUDA set device]")
            torch.cuda.set_device(0)

            # [Query Execution]

            # A tensor group refers to a set of related Variables (= tensor + info) produced by an operator.
            # Each operator has its own tensor group as output.

            torch.cuda.empty_cache()

            subquery_result = None
        
            need_to_scan = [scanned_tuple for query_plan in query_plans for scanned_tuple in query_plan.disk_read_cols]
            if _uvm:
                tensor_group = scan_group = _uvm_payloads = None
                _uvm_begin_occurrence()
                _uvm_payloads = Payloads()
                load_start = time.time()
                scan_group = scan_to_managed(config_json, schema_json, scale_factor, need_to_scan)
            else:
                load_start = time.time()
                scan_group = scan_to_ram(config_json=config_json, schema_table=schema_json, SF=scale_factor, cpu_mem_pool=pinned_pool, to_scan=need_to_scan)
            load_time = time.time() - load_start

            # The cache region for the operators, now that the scan is done:
            # outside the load time and the query's timer, which starts
            # below.
            if cache_region is not None:
                _size_region(qid, want_gb)
                if q_page_cache is not None:
                    # From here on an allocation that misses borrows the
                    # region's lowest blocks (memory/README.md, "Lending");
                    # the counters start with the timed run.
                    q_page_cache.attach()
                    q_page_cache.take_lend_stats()

            # Rebaseline the allocator high-water mark per query so the OOM
            # diagnostic dump reports the peak of the query that failed.
            # The scan is deliberately outside the canonical timer.  Force a
            # full cyclic collection here, disable automatic cyclic GC for the
            # timed execution, and keep it disabled through pool release.  The
            # post-release collection/restoration below is outside timing too.
            _query_gc = _begin_timed_gc_guard()
            if cuda_profile_markers:
                nvtx.range_push("profiler")
                nvtx.range_push(f"Query {qid}")

            cmp.reset_peak()
            if _device_regions.armed():
                # The eviction counters are the timed run's.
                _device_regions.begin_query()
            if _gates.EXPERIMENT:
                # The residency peaks are the timed run's (the scan and the
                # warmup excluded).
                _residency.begin(pinned_pool, q_page_cache)
            perf_logger().start(f"Query {qid}")
            q_start_time = time.time()
            # The output steps' frame conversion and log, left out of the
            # timer (every plan's: a subquery plan has one too).
            output_time = 0.0

            tensor_group = TensorGroup()

            for plan_idx, query_plan in enumerate(query_plans):
                if q_page_cache is not None:
                    # Positions restart per plan: so does the counter.
                    q_page_cache.reset_position()
                ops, args, parent_tree = query_plan.ops_idx, query_plan.ops_dict, query_plan.query_parent_map

                for u in ops:
                    args_u = args[u]
                    name = args_u['name']
                    message_logger().info("%s %s", "\n", f"Executing operator {u} {name}")

                    # Payload deferral: gather the columns deferred to this op
                    # (via their row lineage) before it runs, so the operator
                    # sees normal materialized columns.
                    _deferred_nodes = (args_u.get('nodes') or {}).get('deferred')
                    if _deferred_nodes and not _uvm:
                        _materialize_deferred(tensor_group, _deferred_nodes, pinned_pool,
                                              applier, q_page_cache)

                    if cuda_profile_markers:
                        nvtx.range_push(f"{name} {u}")
                    if _uvm and name != 'AdaptiveSparkPlan':
                        # The UVM executor runs every operator but the
                        # output, its deferred gathers first.
                        uvm_executor.run(name, u, args_u, tensor_group,
                                         scan_group, subquery_result,
                                         _uvm_payloads)

                    elif name == 'Scan':
                        with perf_logger().time(f"scan {u}"):
                            tqp_scan(
                                scan_group,
                                tensor_group,
                                args_u,
                                config_json,
                                schema_json[args_u['table name']],
                                SF=scale_factor,
                                mem_pool=pinned_pool,
                            )
                        # The scan's instances carry the base reps.
                        for inst in args_u['nodes']['scan'].outputs:
                            inst.ccr = tensor_group[inst.local]

                    elif name == 'Filter':
                        with torch_profiler().profile(f"filter {u}"):
                            tqp_filter(
                                tensor_group,
                                args_u,
                                subquery_result,
                                cpu_mem_pool=pinned_pool,
                                cache_region=q_cache_region,
                                name=f"filter {u}",
                                applier=applier,
                                page_cache=q_page_cache,
                            )

                    elif name == 'Project':
                        tqp_project(
                            tensor_group,
                            args_u,
                            cache_region=q_cache_region,
                            cpu_mem_pool=pinned_pool,
                            name=f"project {u}",
                            applier=applier,
                            page_cache=q_page_cache,
                        )

                    elif name == 'Aggregate':
                        with torch_profiler().profile(f"aggregation {u}"):
                            tqp_hash_aggregate(
                                tensor_group,
                                args_u,
                                cpu_mem_pool=pinned_pool,
                                cache_region=q_cache_region,
                                name=f"aggregation {u}",
                                applier=applier,
                                page_cache=q_page_cache,
                            )
                    
                    elif name.endswith('Exchange'):
                        assert False, "Exchange is eliminated"

                    elif name == 'Sort':
                        tqp_sort(
                            tensor_group,
                            args_u,
                            with_limit=False,
                            name=f"sort {u}"
                        )

                    elif name == 'Join':
                        with torch_profiler().profile(f"join {u}"):
                            message_logger().debug(f"[Join Type] {args_u['join type']} JOIN")
                            message_logger().debug(f"[Build] {args_u['build']}")

                            (join_left_key, join_right_key, condition_list,
                             join_type) = join_call_args(args_u, tensor_group)
                            message_logger().debug(f"Join on {join_left_key} = {join_right_key}")
                            message_logger().debug(f"condition list = {condition_list}")
                            message_logger().debug(f"[left and right keys], {args_u['left keys']}, {args_u['right keys']}")

                            tqp_join(
                                tensor_group,
                                args_u,
                                join_left_key,
                                join_right_key,
                                condition_list,
                                join_type,
                                cpu_mem_pool=pinned_pool,
                                cache_region=q_cache_region,
                                name=f"join {u}",
                                applier=applier,
                                page_cache=q_page_cache,
                            )


                    elif name == 'take ordered': # LIMIT
                        tqp_sort(
                            tensor_group,
                            args_u,
                            with_limit=True,
                            name=f"sort lim {u}"
                        )

                    elif name == 'AdaptiveSparkPlan':       # Treated as output operator
                        with perf_logger().time(f"output {u}"):
                            # Timed: the result's columns onto the GPU and
                            # their copy to the host (the result's D2H).
                            message_logger().info(f"Output {u} {args_u['output ids']}")
                            # The ledger keeps the output step apart.
                            if _gates.EXPERIMENT:
                                _movement.set_phase("output")
                            if _uvm:
                                host_cols, out_types = uvm_executor.output_to_host(
                                    tensor_group, args_u)
                            else:
                                host_cols, out_types = tqp_output_to_host(
                                    tensor_group, args_u)
                            if _gates.EXPERIMENT:
                                _movement.set_phase("query")
                            if qid == 16 and scale_factor >= 100:
                                message_logger().debug("Abandon output.")
                                # No frame: the rows the final Sort wrote
                                # to the output columns.
                                result_rows = tensor_group[args_u['output ids'][0]].col_length()
                            else:
                                # Not timed: the host copies converted to
                                # the result frame (CPU only, after the
                                # D2H's sync), and its log.
                                _frame_start = time.time()
                                df_result = host_frame(host_cols, out_types, args_u)
                                result_rows = len(df_result)
                                message_logger().info(df_result)
                                output_time += time.time() - _frame_start
                            del host_cols

                            if plan_idx < len(query_plans) - 1:
                                # A subquery plan's scalar: every plan but the
                                # last, the main one, whose result may be empty.
                                # The plan that produced the value says whether
                                # it is scaled (Q11's subquery: 1/SF).
                                subquery_result = parsing.subquery_value(
                                    float(df_result.iloc[0, 0]), query_plan, scale_factor)
                                message_logger().info(f"subquery_result = %s", subquery_result)
                        
                    else:
                        raise Exception(f"Unsupported Operator: {name}")

                    torch.cuda.synchronize()
                    if cuda_profile_markers:
                        nvtx.range_pop()

        
            perf_logger().stop(f"Query {qid}")
            approx_time = time.time()-q_start_time-output_time
            if _gates.EXPERIMENT:
                _resid = _residency.report(pinned_pool, q_page_cache)
            _end_timed_gc_guard(_query_gc)
            if cuda_profile_markers:
                nvtx.range_pop()
                nvtx.range_pop()
            if (any(_query_gc['timed_collection_delta']) and
                    _gates.TIMED_GC_VALIDATE):
                raise RuntimeError(
                    'cyclic collection entered the canonical timed region: '
                    f"{_query_gc['timed_collection_delta']}"
                )

            message_logger().info(f"Approx Time = {approx_time:.6f} seconds")
            _freelist_stats = None
            if _device_regions.armed():
                _freelist_stats = _report_freelist(qid)
            _lend_stats = None
            if q_page_cache is not None:
                _lend_stats = _report_lend(qid, q_page_cache)
            _report_sizing(qid)
            if not globals().get('_is_warmup'):
                _timing_row = f'{qid},{approx_time:.6f}\n'
                with open('query_times.csv', 'a') as f:
                    f.write(_timing_row)
                if _report_query_times is not None:
                    with open(_report_query_times, 'a', encoding='utf-8') as f:
                        f.write(_timing_row)
                per_qid_times[qid].append(approx_time)
                _record_result(qid, "OK", approx_time, result_rows, load_time)
                _recorded = True
            # --uvm: the occurrence's payloads go with it.
            _uvm_payloads = None
            _release_pools()
            _collect_and_restore_gc(_query_gc)

            pd.options.display.float_format = '{:.6f}'.format
            with open("results.out", "a") as f:
                if qid == 16 and scale_factor >= 100:
                    f.write("Q16 Output Skipped\n")
                else:
                    # Every row: the smoke test and the e2e ladder compare
                    # them all with the reference.
                    f.write(df_result.to_string())
                    f.write("\n")

            if perf_report:
                # The occurrence and its canonical time, always; the rest
                # under TQP_EXPERIMENT_TELEMETRY.
                perf_logger().add_section(
                    'wall', {'occurrence': _q_idx, 'approx_time': approx_time})
                if _freelist_stats is not None:
                    perf_logger().add_section('freelist', _freelist_stats)
                if _lend_stats is not None:
                    perf_logger().add_section('lend', _lend_stats)
                if _gates.EXPERIMENT:
                    from adaptive.run_stats import (
                        get_sizing_stats as _get_szs,
                        snapshot_compression_choices as _snap_cmp,
                    )
                    perf_logger().add_section('movement', _movement.report())
                    # What the timed run held at most in each memory tier.
                    perf_logger().add_section('residency', _resid)
                    # Sizing-plane predicted-vs-observed transients (empty
                    # sample list when --sizing_search off).
                    perf_logger().add_section('sizing_model', _get_szs())
                    # Compression-search outcomes per searched pipe (empty
                    # when --compression_search off).
                    perf_logger().add_section('compression', _snap_cmp())
                # TQP_PERF_PER_OCCURRENCE: don't overwrite SF<sf>_<qid>.json across
                # repeated qids — keep one file per occurrence for fast-vs-slow diffs.
                _suffix = f"_occ{_q_idx}" if os.environ.get('TQP_PERF_PER_OCCURRENCE') else ""
                perf_logger().report(os.path.join(perf_report, f"SF{scale_factor}_{qid}{_suffix}.json"))
    
        except Exception as _qe:
            _fail_s = None if q_start_time is None else time.time() - q_start_time
            print(f"[QFAIL] Q{qid}: {type(_qe).__name__}: {str(_qe)[:200]}", flush=True)
            if _device_regions.armed() and q_start_time is not None:
                try:
                    _report_freelist(qid)
                except Exception as _fe:
                    print(f"[freelist] Q{qid}: no stats ({_fe})", flush=True)
            if (q_page_cache is not None and q_page_cache.attached
                    and q_start_time is not None):
                try:
                    _report_lend(qid, q_page_cache)
                except Exception as _le:
                    print(f"[lend] Q{qid}: no stats ({_le})", flush=True)
            _report_sizing(qid)
            # One line with no commas, so it stays one CSV field.
            _err = f"{type(_qe).__name__}: {_qe}".splitlines()[0].replace(",", ";")
            if globals().get('_is_warmup'):
                _warm_failures[qid] = (_fail_s, _err)
            elif not _recorded:
                _record_result(qid, "FAIL", _fail_s, _err, load_time)
                if perf_report:
                    _write_failed_report(perf_report, scale_factor, qid, _q_idx,
                                         _fail_s, _err, pinned_pool, q_page_cache)
            if os.environ.get('TQP_QFAIL_TRACE'):
                import traceback as _tb
                _tb.print_exc()
            if _uvm_payloads is not None:
                # Emptied, as a frame of the failure may still hold it.
                _uvm_payloads.clear()
                _uvm_payloads = None
            try:
                _release_pools()
            except Exception: pass
            gc.collect(); torch.cuda.empty_cache()
            print(f"[QDONE-OR-FAIL] continuing past Q{qid}", flush=True)
            continue
        finally:
            # No cache stays attached past its occurrence, whatever ended
            # it (a failed release, an exception out of the run): the
            # arena holds the core's address while one is attached.
            for c in _native.values():
                c.detach()
            _restore_gc_after_failure(_query_gc)

    # Median-of-repeats summary: when a qid appears multiple times in --q, report
    # the median of its per-occurrence wall times (plus min/max and the raw runs)
    # so one number characterizes a repeated-run stability check.
    if not globals().get('_is_warmup'):
        import statistics
        _repeated = {q: ts for q, ts in per_qid_times.items() if len(ts) > 1}
        if _repeated:
            print("====================== Repeated-run medians ======================", flush=True)
            for _q in sorted(_repeated):
                _ts = _repeated[_q]
                _med = statistics.median(_ts)
                _runs = ", ".join(f"{t:.4f}" for t in _ts)
                print(f"[median] Q{_q} over {len(_ts)} runs: {_med:.4f}s "
                      f"(min {min(_ts):.4f}, max {max(_ts):.4f}; runs: {_runs})", flush=True)
                message_logger().info(
                    f"[median] Q{_q} median over {len(_ts)} runs = {_med:.6f}s")

        # Preserve the logical-output artifact with the timings and telemetry.
        # The copy is post-suite, outside every canonical per-query interval.
        if perf_report:
            import shutil
            shutil.copyfile("results.out", os.path.join(perf_report, "results.out"))


if __name__ == "__main__":

    # ── Argparse first (before CUDA init) ──
    parser = argparse.ArgumentParser()
    parser.add_argument('--SF', type=int, default=1, help='Scale Factor')
    parser.add_argument('--q', type=int, nargs='+', default=list(range(1, 23)), help='List of Query IDs')
    parser.add_argument('--perf_report', type=str, required=False, help='Performance timers directory')
    parser.add_argument("--log", type=str, required=False, help='debug log directory')
    parser.add_argument('--no_debug', action='store_true', help='Disable debug logs')
    parser.add_argument('--torch_profile', type=str, required=False, help='Torch profiling files output directory')
    parser.add_argument('--cuda_profile', action='store_true', help='Flag for enabling cuda performance profiling')
    parser.add_argument('--smoke', action='store_true', help='Run the SF1 end-to-end correctness smoke test (pytest tests/smoke)')
    parser.add_argument("--buffer_strategy", type=enum_by_value(BufferStrategy), choices=list(BufferStrategy), default=BufferStrategy.CPU, help="Buffer strategy")
    parser.add_argument("--compression", type=str,
                        choices=["adaptive"] + _algo_names_fn(), default="adaptive",
                        help="Compression mode: 'adaptive' (default) selects the profiled "
                             "best codec per (type, sorted) column class "
                             "(the codec map, profiling/results/codec_map.json); naming a codec "
                             "(RAW = no compression) pins it for all reads and writes.")
    parser.add_argument("--compression_search", type=str,
                        choices=["off", "read-only", "read-and-write", "fixed"],
                        default="off",
                        help="Adaptive subset-selection search over which columns to "
                             "compress. "
                             "'off' (default) leaves every code path unchanged; "
                             "'read-only' probes read-leg codecs mid-run across a "
                             "pipe's ticks; 'read-and-write' adds the "
                             "write-leg probe windows; 'fixed' runs no "
                             "search and applies the per-column codecs of "
                             "--fixed_search_json instead. "
                             "Composes with --compression: the policy supplies each "
                             "column's single compressed candidate.")
    parser.add_argument("--fixed_search_json", type=str, default=None,
                        help="With --compression_search fixed: JSON of query id -> "
                             "{column tag: codec} pins, validated at load (a gather "
                             "source, deferred payload or lineage-seed index pinned "
                             "non-RAW is rejected). Unlisted tags keep the policy "
                             "choice.")
    parser.add_argument("--sizing_search", type=str,
                        choices=["off", "per_pipe", "mid_stream"], default="off",
                        help="Adaptive intermediate-sizing search "
                             "over chunk budgets. "
                             "'off' (default) leaves every code path unchanged. "
                             "'per_pipe' (sense, then march): every "
                             "tick-loop pipe (sort/core/agg/filter/project) "
                             "senses its per-pipe memory equation with small-"
                             "delta probes, then marches in constant additive "
                             "steps under the increment bound, accepting each "
                             "step only on measured >= epsilon per-byte gain "
                             "(knee-stop); long pipes walk in-pipe via "
                             "sections, short pipes across runs. 'mid_stream' "
                             "arms 'per_pipe' plus the rearrange gather-window "
                             "search (same sense->march machinery).  The "
                             "budgets count what the cache region could "
                             "lend; the allocator lends it on a miss.")
    parser.add_argument("--sizing_config", type=str, default=None,
                        help="Path to a sizing-search JSON config (c0/bounds/"
                             "gamma/epsilon per class + safety margins). "
                             "Defaults are used "
                             "when omitted; c0 defaults to the static "
                             "--chunk_size_mb value.")
    parser.add_argument("--mask_selectivity", type=float, default=0.1, help="Zero-copy mask selectivity threshold (0.0 = disabled)")
    parser.add_argument('--slot_pool_gb', type=float, default=4.0,
                        help='Size in GB of the granule cache\'s region at the top '
                             'of the arena (default 4); a run with a cache only.')
    parser.add_argument('--slot_pool_gb_list', type=float, nargs='+', default=None,
                        help='Per-query cache region sizes in GB; length must match --q. '
                             'Moves the region\'s boundary between queries, holding '
                             'the arena fixed while varying the operator space (e.g. '
                             'for baseline ablation sweeps). Overrides --slot_pool_gb.')
    parser.add_argument("--chunk_size_mb", type=int, default=256,
                        help="THE chunk budget, in MB (default 256). One uniform value "
                             "for every chunked pipeline — scan/filter/project, aggregate, "
                             "join core, the join's sort pipe, and the rearrange gather "
                             "window. There is no per-pipeline chunk knob. Raising it cuts "
                             "per-chunk barriers and deepens in-flight PCIe at a ~linear "
                             "cost in peak GPU (the rearrange leg dominates that peak), so "
                             "a larger chunk may need a larger --cmp_pool_gb or a smaller "
                             "--slot_pool_gb. See constants.py.")
    parser.add_argument("--partition_budget_mb", type=int, default=512,
                        help="The partition budget, in MiB (default 512): the "
                             "bytes of the largest unit an operator holds whole. "
                             "A join's unit is a bucket pair, its left and right "
                             "rows with their global index (B 1..512); a keyed "
                             "aggregation's is a pass-2 round, its partial rows "
                             "times the run's bytes per row (B 1..256). Each takes "
                             "the smallest power-of-two partition count B "
                             "predicted to fit, then packs whole buckets: into "
                             "the join core's ticks, whose input this bounds from "
                             "below, or into the aggregation's rounds.")
    parser.add_argument('--cmp_pool_gb', type=int, default=None,
                        help='Custom PyTorch CUDA allocator pool size in GB. Default: the '
                             'detected GPU memory capacity minus 2 GB.')
    parser.add_argument('--freelist_reserve_gb', type=float, default=0.0,
                        help='With --buffer_strategy freelist: GiB at the '
                             "arena's top kept free through each query's timed "
                             'run, for the allocations eviction cannot serve '
                             '(every region pinned, or the free space split): '
                             'one that still misses once no unpinned region is '
                             'left draws on it, and it refills as those blocks '
                             'free up (IO/pages/device_regions.py). Default 0, '
                             'none.')
    parser.add_argument(
        '--pinned_pool_gb', type=int, default=None,
        help='Override the scale-factor-derived pinned host-memory pool size '
             'in GiB for the real run. Warmup keeps its normal SF1 pool. '
             'Default: use the scale-factor sizing table.')
    parser.add_argument('--uvm', action='store_true',
                        help='UVM mode: every column and intermediate in CUDA '
                             'managed memory, the driver migrating pages to the '
                             'GPU on access and evicting them when the GPU budget '
                             '(--cmp_pool_gb) is full. No pinned pool, cache, '
                             'compression or search; the same plans, '
                             'kernels, partitioning and chunk budget (as morsels).')
    parser.add_argument('--uvm_pool_gb', type=int, default=None,
                        help='With --uvm: the managed arena in GiB (address '
                             'space; a page uses memory once touched). Default: '
                             "the GPU budget plus the GPU's NUMA node's free "
                             'memory and page cache, minus 16 GiB.')
    parser.add_argument('--no_warmup', action='store_true',
                        help='Skip the SF1 warmup pass over --q (CUDA context, kernels and '
                             'pools then prime during the real run). For correctness runs; '
                             'timing runs keep the warmup.')
    # ── Run-to-run variance / reproducibility controls ──
    args = parser.parse_args()
    if args.uvm:
        if args.smoke:
            parser.error("--smoke runs the streaming engine only: drop --uvm")
        for _bad, _flag in (
                (args.buffer_strategy != BufferStrategy.CPU,
                 "--buffer_strategy other than cpu"),
                (args.compression_search != "off",
                 "--compression_search other than off"),
                (args.sizing_search != "off", "--sizing_search other than off"),
                (args.sizing_config is not None, "--sizing_config"),
                (args.fixed_search_json is not None, "--fixed_search_json"),
                (args.slot_pool_gb_list is not None, "--slot_pool_gb_list"),
                (args.pinned_pool_gb is not None, "--pinned_pool_gb")):
            if _bad:
                parser.error(f"{_flag} does not apply with --uvm")
        if args.uvm_pool_gb is not None and args.uvm_pool_gb <= 0:
            parser.error("--uvm_pool_gb must be positive")
        # What UVM mode ignores: it runs RAW (no compression applier) with
        # no pinned pool and no cache region.
        args.compression = "RAW"
        args.slot_pool_gb = args.mask_selectivity = None
        args.pinned_pool_gb = None
        print("[uvm] RAW, no compression applier; --compression, "
              "--mask_selectivity and --slot_pool_gb do not apply",
              flush=True)
    elif args.uvm_pool_gb is not None:
        parser.error("--uvm_pool_gb is only valid with --uvm")
    if args.buffer_strategy == BufferStrategy.FREELIST:
        # The free-list baseline keeps intermediates on the GPU and has no
        # granule cache or slot pool for the searches to steer.
        for _bad, _flag in (
                (args.compression_search != "off",
                 "--compression_search other than off"),
                (args.sizing_search != "off", "--sizing_search other than off"),
                (args.slot_pool_gb_list is not None, "--slot_pool_gb_list")):
            if _bad:
                parser.error(f"{_flag} does not apply with "
                             "--buffer_strategy freelist")
        # When --cmp_pool_gb is not given, the reserve is checked against it
        # where it is resolved, below.
        if args.freelist_reserve_gb < 0 or (
                args.cmp_pool_gb is not None
                and args.freelist_reserve_gb >= args.cmp_pool_gb):
            parser.error("--freelist_reserve_gb must be at least 0 and less "
                         "than --cmp_pool_gb")
    elif args.freelist_reserve_gb:
        parser.error("--freelist_reserve_gb applies with --buffer_strategy "
                     "freelist only")
    from dataflow.compression_applier import RAW_INTERMEDIATES as _raw_int
    if _raw_int and args.compression_search != "off":
        # Every intermediate resolves RAW before any pin, and the search
        # menus have no compressed candidate left.
        parser.error("TQP_RAW_INTERMEDIATES does not apply with "
                     "--compression_search other than off")
    if args.compression_search == "fixed":
        # No search: the applier applies the JSON's per-tag pins.  Needs an
        # applier to override, so RAW mode (which builds none) is an error.
        if not args.fixed_search_json:
            parser.error("--compression_search fixed requires --fixed_search_json <path>")
        if not os.path.isfile(args.fixed_search_json):
            parser.error(f"--fixed_search_json {args.fixed_search_json}: no such file")
        if args.compression == "RAW":
            parser.error("--compression_search fixed needs a non-RAW --compression: "
                         "RAW mode builds no applier for the pins to override")
        from adaptive.instage_optimization import set_fixed_search_json
        set_fixed_search_json(args.fixed_search_json)
        print(f"[compression_search] fixed: pins from {args.fixed_search_json}; "
              "no search runs", flush=True)
    elif args.fixed_search_json:
        parser.error("--fixed_search_json is only valid with --compression_search fixed")
    if args.pinned_pool_gb is not None and args.pinned_pool_gb <= 0:
        parser.error("--pinned_pool_gb must be positive")
    if args.pinned_pool_gb is None and not args.uvm:
        # The real run's pool as resolved (the warmup's is SF1's).
        args.pinned_pool_gb = _pinned_capacity_gb(args.SF)
    if args.partition_budget_mb <= 0:
        parser.error("--partition_budget_mb must be positive")
    from IO.telemetry import gates as _gates
    if _gates.EXPERIMENT and not args.perf_report:
        parser.error("TQP_EXPERIMENT_TELEMETRY is set but --perf_report is "
                     "not: experiment telemetry is written only to the perf "
                     "report")

    # Resolve the allocator pool to (GPU capacity - 2 GB) unless given explicitly.
    if args.cmp_pool_gb is None:
        _cap_gib = _gpu_total_gib(0)
        if _cap_gib is None:
            parser.error("could not detect GPU memory; pass --cmp_pool_gb explicitly")
        args.cmp_pool_gb = max(1, _cap_gib - 2)
        if args.freelist_reserve_gb >= args.cmp_pool_gb:
            parser.error(f"--freelist_reserve_gb must be less than the "
                         f"arena ({args.cmp_pool_gb} GB, the GPU's capacity "
                         "less 2)")
    _uvm = args.uvm
    if _uvm and args.uvm_pool_gb is None:
        # The budget plus what the GPU's node can hand over (free memory and
        # page cache) less the headroom the pinned pool's check keeps.
        args.uvm_pool_gb = args.cmp_pool_gb + max(
            0, numa.node_reclaimable_b(gpu_numa_node(0))) // 1024 ** 3

    _effective = None
    if args.perf_report:
        # The resolved settings go with the experiment telemetry, and with
        # every --uvm run, which records the driver settings it ran under.
        _effective = _record_run_metadata(
            args.perf_report, args if _gates.EXPERIMENT or args.uvm else None)

    # ── CUDA allocator init ──
    print(torch.__version__)
    print(torch.version.cuda)

    alloc = CUDAPluggableAllocator(
        path_to_so_file="libcustom_allocator.so",
        alloc_fn_name="pa_cuda_malloc",
        free_fn_name="pa_cuda_free",
    )
    change_current_allocator(alloc)  # must be called before any CUDA tensors exist

    # The single chunk budget: every pipe class (tick pipes and the rearrange
    # gather window alike) reads constants.get_chunk_size_b().
    constants.set_chunk_size_b(args.chunk_size_mb * 1024 ** 2)
    # Each join and keyed aggregation derives its partition count B from this
    # (operators/partitions.py).
    constants.set_partition_budget_b(args.partition_budget_mb * 1024 ** 2)
    if _uvm:
        # The managed arena: 4 MiB blocks, two of the driver's 2 MiB
        # migration granules, so no block shares a granule.
        _block_b = 4 * 1024 ** 2
        cmp.init(args.uvm_pool_gb * 1024 ** 3, _block_b, managed=True)
    else:
        # 1 MiB arena blocks: a granule of the granule cache is one block of
        # the arena's cache region, and a small tensor wastes at most 1 MiB to rounding.
        # Blocks were 5 MiB before, chosen because 20 MB blocks fragmented a
        # 22.5 GB pool (SF100 Q20 ran out of memory at every compressed codec).
        # At SF100 with a 30 GiB pool, 1 MiB blocks ran every query that 5 MiB
        # ran, no pipe peaked higher, and the suite was no slower.
        # TQP_ARENA_BLOCK_B (bytes, a multiple of 512) overrides it, e.g. 512
        # for a byte-granular arena (the free-list baseline).
        _block_b = int(os.environ.get("TQP_ARENA_BLOCK_B") or 1024 ** 2)
        if _block_b <= 0 or _block_b % 512:
            parser.error(f"TQP_ARENA_BLOCK_B={_block_b}: a positive multiple "
                         "of 512 bytes")
        if _block_b != 1024 ** 2:
            print(f"[arena] {_block_b} B blocks (TQP_ARENA_BLOCK_B)", flush=True)
        cmp.init(args.cmp_pool_gb * 1024 ** 3, _block_b)
    # cuDF allocates through RMM: point it at the arena too, before any cuDF
    # call, so no device buffer lands outside the arena.
    rmm_arena.install(_block_b)

    set_message_logger(args.log, False)
    # The naive H2D/D2H exchange is the only transfer path; install it always.
    set_exchange_to_naive()

    set_global_buffer_strategy(args.buffer_strategy)
    if args.buffer_strategy == BufferStrategy.FREELIST:
        # Intermediates stay on the GPU as spillable regions; an allocation
        # that misses stalls while the least recently used are evicted to
        # their homes one at a time until it fits, or fails the query when
        # none is left (IO/pages/device_regions.py).
        from IO.pages import device_regions
        device_regions.install(
            _block_b, int(args.freelist_reserve_gb * 1024 ** 3))
        print("[freelist] intermediates stay on the GPU (in --compression's "
              "format); an arena miss evicts spillable regions, least "
              "recently used first, until it fits", flush=True)
        if args.freelist_reserve_gb:
            print(f"[freelist] reserve {args.freelist_reserve_gb:g} GiB at the "
                  "arena's top, kept free through each query's timed run: a "
                  "miss with no unpinned region left draws on it",
                  flush=True)
    if args.compression != "adaptive":
        # Explicit fixed-codec override (incl. RAW = no compression).
        set_fixed_compression(args.compression)
    else:
        # Adaptive per-(type, sorted) compression is the design default; it
        # drives codec selection via the applier.  Keep the global fixed codec
        # at RAW so incidental col_to_gpu materializations (sort keys, etc.)
        # always resolve to the always-present RAW rep.
        set_adaptive_compression(True)
        set_fixed_compression("RAW")
    # The codec map (profiling/results/codec_map.json, written by
    # `make profiling`) is loaded once per process, here: the class table the
    # adaptive policy reads and every codec's nvCOMP chunk.  A missing file
    # warns and uses the built-in defaults; a broken one raises and stops the
    # run before any pool is pinned.  The chunks are read back through
    # mgr_chunk_size so the line shows what the compress path will use.
    from compression.format_policy import codec_map as _codec_map
    from compression.compression_algorithms import mgr_chunk_size as _mgr_chunk
    _cm = _codec_map()
    print(f"[codec_map] {_cm.describe()}", flush=True)
    if args.perf_report and _gates.EXPERIMENT:
        import importlib.metadata as _md
        import platform
        _props = torch.cuda.get_device_properties(0)
        _versions = {'python': platform.python_version(),
                     'torch': torch.__version__, 'cuda': torch.version.cuda}
        for _dist in ('pylibcudf-cu12', 'nvidia-libnvcomp-cu12'):
            try:
                _versions[_dist] = _md.version(_dist)
            except _md.PackageNotFoundError:
                _versions[_dist] = None
        _note_run_metadata(
            args.perf_report, codec_map=_cm.describe(), versions=_versions,
            gpu={'name': _props.name, 'memory_b': _props.total_memory})
    print("[codec_map] codecs: " + " ".join(_algo_names_fn()) + " | mgr_chunk_size: "
          + " ".join(f"{c}={_mgr_chunk(c)}" for c in _algo_names_fn() if c != "RAW"),
          flush=True)

    # ── the searches' PCIe rate ──
    # Both searches price a byte's later crossing of PCIe (the write leg's
    # readback price, the sizing probe's) at this box's measured
    # host-to-device rate.
    if (args.compression_search in ("read-only", "read-and-write")
            or args.sizing_search != "off"):
        from adaptive import resource_monitor as _rm
        from IO.transfer.data_exchange import h2d_b_per_ms
        _rm.PCIE_B_PER_MS = h2d_b_per_ms()
        print(f"[search] PCIe rate {_rm.PCIE_B_PER_MS / 1e6:.1f} GB/s "
              "(the readback prices)", flush=True)

    # ── compression search mode ──
    if args.compression_search not in ("off", "fixed") and args.compression == "RAW":
        # Degenerate candidate set (candidate == RAW): degrade, don't error.
        # print (not message_logger) so the line survives --no_debug / no --log.
        print("[compression_search] --compression RAW makes every candidate "
              "degenerate; search degraded to off", flush=True)
        args.compression_search = "off"
    if args.compression_search == "read-and-write":
        # The read-leg search (as in read-only) composed with the write-leg
        # probe-window search.  Write picks take effect at t+1 (each tick
        # writes its own region); mixed-codec columns are legal downstream, so the
        # consumer-side acceptance switches on here and nowhere else.
        from adaptive.adaptive_driver import set_tick_observer_factory
        from adaptive.compression_controller import (
            CompressionController as _RWCompressionController)
        from adaptive.compression_search import (
            ReadWriteSearchDriver, min_searchable_ticks,
            min_write_searchable_ticks)
        from adaptive.instage_optimization import set_mixed_rep_acceptance

        set_mixed_rep_acceptance(True)    # mixed-codec singleton-rep acceptance

        def _rw_search_factory(pipe_name, menu):
            ctrl = _RWCompressionController(menu)
            drv = ReadWriteSearchDriver(ctrl, menu.total_ticks)
            if drv.inert:
                return None
            # Retain the driver for the per-query perf 'compression' section
            # (TQP_EXPERIMENT_TELEMETRY; the closure below is otherwise the
            # only reference).
            from IO.telemetry import gates as _gates
            if _gates.EXPERIMENT:
                from adaptive.run_stats import register_compression_search
                register_compression_search(pipe_name, drv, ctrl, menu)
            n_r, n_w = ctrl.decision_count(), ctrl.write_decision_count()
            message_logger().info(
                "[compression_search] %s: N_r=%d read / N_w=%d write "
                "decision cols, %d ticks — searching (read-and-write)",
                pipe_name, n_r, n_w, menu.total_ticks)

            def _observer(view):
                ret = drv(view)
                if (drv.read.converged
                        and not getattr(drv, "_rconv_logged", False)):
                    drv._rconv_logged = True
                    message_logger().info(
                        "[compression_search] %s: read leg converged "
                        "k*=%d/%d after %d samples", pipe_name,
                        drv.read.k_star, n_r, drv.read.probes_spent)
                if (drv.write is not None and drv.write.converged
                        and not getattr(drv, "_wconv_logged", False)):
                    drv._wconv_logged = True
                    message_logger().info(
                        "[compression_search] %s: write leg converged "
                        "k_w*=%d/%d (%s%s, %d window samples)", pipe_name,
                        drv.write.k_star, n_w, drv.write.mode,
                        (", pre-gated" if drv.write.pregate_skipped else
                         ", benefit-gated" if drv.write.benefit_gated else ""),
                        drv.write.probes_spent)
                return ret

            _observer.settled = drv.settled
            return _observer

        set_tick_observer_factory(
            _rw_search_factory,
            min_total_ticks=min(min_searchable_ticks(),
                                min_write_searchable_ticks()))
        print("[compression_search] read-and-write: read-leg search + "
              "write-leg probe windows live", flush=True)
    if args.compression_search == "read-only":
        # Per-pipe read-leg search.  One driver per pipe, created by the
        # process-wide observer factory at pipe construction (the
        # decision-menu publication); a budget-inert pipe gets NO observer,
        # so it pays nothing per tick.  All actuation flows through the
        # per-tick observer return + the pipeline's t+2 JIT in-stage rebuild.
        from adaptive.adaptive_driver import set_tick_observer_factory
        from adaptive.compression_controller import CompressionController
        from adaptive.compression_search import (
            CompressionSearchDriver, min_searchable_ticks)

        def _read_search_factory(pipe_name, menu):
            ctrl = CompressionController(menu)
            drv = CompressionSearchDriver(ctrl, menu.total_ticks)
            if drv.inert:
                return None
            from IO.telemetry import gates as _gates
            if _gates.EXPERIMENT:
                from adaptive.run_stats import register_compression_search
                register_compression_search(pipe_name, drv, ctrl, menu)
            n = ctrl.decision_count()
            message_logger().info(
                "[compression_search] %s: N=%d decision cols, %d ticks, "
                "probe budget %d ticks — searching", pipe_name,
                n, menu.total_ticks, drv._budget_ticks)

            def _observer(view):
                ret = drv(view)
                if drv.converged and not getattr(drv, "_conv_logged", False):
                    drv._conv_logged = True
                    message_logger().info(
                        "[compression_search] %s: converged k*=%d/%d after "
                        "%d samples", pipe_name, drv.k_star, n,
                        drv.probes_spent)
                return ret

            _observer.settled = drv.settled
            return _observer

        set_tick_observer_factory(_read_search_factory,
                                  min_total_ticks=min_searchable_ticks())
        print("[compression_search] read-only: per-chunk read-leg search "
              "live", flush=True)
    # ── sizing search mode (sizing plane) ──
    # per_pipe = tick-pipe sizing (sense, then march); mid_stream =
    # per_pipe + the rearrange gather-window search on top.
    if args.sizing_search != "off":
        from adaptive.sizing_controller import (
            install_gather_sizing, install_per_pipe_sizing,
            load_config as _load_sizing_config)
        _sizing_cfg = _load_sizing_config(args.sizing_config)
        # Run-level codec-policy fingerprint (part of the store's key).
        _codec_epoch = f"{args.compression}/{args.compression_search}"
        _tick_ctl = install_per_pipe_sizing(
            _sizing_cfg,
            arena_stats_fn=cmp.get_arena_stats,
            codec_epoch=_codec_epoch)
        print("[sizing_search] per-pipe learned jump live "
              f"(sort/core/agg/filter/project start at c0="
              f"{(_sizing_cfg.tick_c0_b or constants.get_chunk_size_b()) // 2**20} MB; "
              "sort/filter/project and agg pass 1 measure, walk x2 and "
              f"jump in [{_sizing_cfg.tick_c_min_b // 2**20}, "
              f"{_sizing_cfg.tick_c_max_b // 2**20}] MB), "
              f"{_sizing_cfg.tick_samples} chunks per size, "
              f"epsilon={_sizing_cfg.epsilon}", flush=True)
    if args.sizing_search == "mid_stream":
        install_gather_sizing(
            _sizing_cfg,
            arena_stats_fn=cmp.get_arena_stats)
        print("[sizing_search] mid_stream: rearrange gather-window search "
              "live; "
              f"c0={(_sizing_cfg.c0_b or constants.get_chunk_size_b()) // 2**20} MB, "
              f"bounds=[{_sizing_cfg.c_min_b // 2**20}, "
              f"{_sizing_cfg.c_max_b // 2**20}] MB, "
              f"delta={_sizing_cfg.march_delta_b // 2**20} MB, "
              f"sense_delta={_sizing_cfg.sense_delta_b // 2**20} MB, "
              f"epsilon={_sizing_cfg.epsilon}",
              flush=True)
    set_zero_copy_selectivity(mask=args.mask_selectivity)

    if args.smoke:
        if args.sizing_search != "off":
            # The in-process pytest gate imports `main` as a SECOND module
            # object whose _sizing_cfg is None — the armed factory would run
            # sessions the gate does not configure.  Disarm.
            from adaptive.sizing_controller import (
                uninstall_gather_sizing, uninstall_per_pipe_sizing)
            uninstall_gather_sizing()
            uninstall_per_pipe_sizing()
            print("[sizing_search] disarmed for --smoke (in-process pytest "
                  "gate)", flush=True)
        import pathlib
        import pytest
        smoke_dir = pathlib.Path(__file__).resolve().parent / "tests" / "smoke"
        sys.exit(pytest.main(["-q", str(smoke_dir)]))

    if not args.no_warmup:
        message_logger().info("Starting Warmup")
        warmup(args.q)
        message_logger().info("Warmup Finished")

    # Pin every thread to GPU0's NUMA node AFTER warmup: CUDA/allocator init
    # resets process affinity to all CPUs once during warmup, so pinning here
    # (post-init) holds for the real run.  The pinned host pool is bound to the
    # node on its own, unless TQP_NUMA_BIND is off: then the real run's pool
    # (made in exec, after this) lands where these threads first touch it.
    _node, _cpu0, _ncpu = pin_to_gpu_node(0)
    if _node is not None:
        print(f'[variance] pinned all threads to NUMA node{_node} '
              f'({_ncpu} CPUs, from cpu{_cpu0}) local to GPU0; '
              f'PyTorch intra-op threads={torch.get_num_threads()}',
              flush=True)
    else:
        print(f'[variance] NUMA pin failed ({_cpu0}); continuing unpinned'
              + ('' if numa.NUMA_BIND else ' (TQP_NUMA_BIND off: the pinned '
                 'pool lands where unpinned threads first touch it)'),
              flush=True)
    if args.perf_report and _gates.EXPERIMENT:
        # Where the run's threads were pinned (with TQP_NUMA_BIND off, also
        # where the pinned pool lands): node None when the pin failed.
        _note_run_metadata(args.perf_report, cpu_pin={
            'node': _node, 'n_cpus': _ncpu,
            'error': None if _node is not None else str(_cpu0)})

    # Force-free warmup's GPU memory pool before the real run allocates a new one.
    gc.collect()
    # nvCOMP managers created during warmup hold arena-allocated scratch;
    # they MUST NOT survive the arena reset below (retained scratch would
    # alias the real run's tensors — proven silent corruption).  The real
    # run lazily re-creates them.  mgr_shutdown() MUST always run
    # immediately BEFORE cmp.free_all() / any arena reset.
    if _uvm:
        # The arena reset, here too: no nvCOMP manager runs under --uvm, and
        # a live allocation is a bug that stops the run.  Then the GPU
        # budget, once the warmup has loaded the modules the queries use
        # (under --no_warmup, before the first occurrence), and its
        # settings in cmdline.json.
        cmp.remake()
        _balloon_b = _uvm_cap_gpu_budget(args.cmp_pool_gb, args.uvm_pool_gb)
        if _effective is not None:
            _effective['uvm'] = _uvm_metadata(
                args.uvm_pool_gb * 1024 ** 3, args.cmp_pool_gb * 1024 ** 3,
                _balloon_b)
            _note_run_metadata(args.perf_report, effective=_effective)
    else:
        from TQPlib.tqpcompress import mgr_shutdown
        mgr_shutdown()
        cmp.free_all()
    cmp.reset_peak()   # drop warmup's high-water mark from the OOM diagnostic

    # Nsight Systems: open the cudaProfilerApi capture range around ONLY the real
    # timed run, so an `nsys profile --capture-range=cudaProfilerApi` trace excludes
    # warmup and the multi-minute host pin/scan.
    if args.cuda_profile:
        torch.cuda.cudart().cudaProfilerStart()
    try:
        with torch.inference_mode():
            exec(args.SF, args.q, args.perf_report, args.log, args.no_debug,
                 args.torch_profile, args.cuda_profile,
                 slot_pool_gb=args.slot_pool_gb, slot_pool_gb_list=args.slot_pool_gb_list,
                 pinned_pool_gb=args.pinned_pool_gb)
    finally:
        if args.cuda_profile:
            torch.cuda.cudart().cudaProfilerStop()
