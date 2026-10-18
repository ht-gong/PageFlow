#!/usr/bin/env python3
"""NVTX ranges for every pipeline and tick phase, for an Nsight trace
(E5; docs/sf100-bottleneck-study.md §1.3).  Runs inside the ``dev``
container.

Runs ``main.py`` in this process after installing two kinds of ranges:

``P <tag>``  one range per perf-logger tag: every pipeline, operator and
             query, named as in the ``--perf_report`` JSON.
``<step>``   one range per call of the host steps a tick is made of
             (``STEP_HOOKS``).  A hook whose target this tree does not have
             (a tree without the region producer, say) is skipped.

With ``--cuda_profile`` and ``nsys --capture-range=cudaProfilerApi`` the trace
starts after the warmup.  ``trace_analyze.py`` reads the trace.
"""

from __future__ import annotations

import functools
import importlib
import os
import runpy
import sys

# (module, class or None for a module-level function, attribute, range name).
# The host steps of a tick (VortexPipeline.do_exchange), of the output path
# inside tick.post, and of the gather pipe.
STEP_HOOKS = (
    # Pipeline construction and the tick loop.
    ('IO.pipeline.vortex_pipeline', 'VortexPipeline', '__init__', 'pipe.init'),
    ('IO.pipeline.vortex_pipeline', 'VortexPipeline',
     '_execute_pre_comp_and_operator', 'tick.pre_op'),
    ('IO.pipeline.vortex_pipeline', 'VortexPipeline', '_execute_post_comp',
     'tick.post'),
    ('IO.pipeline.vortex_pipeline', 'VortexPipeline', '_jit_rebuild',
     'tick.rebuild'),
    ('IO.pipeline.vortex_pipeline', 'VortexPipeline',
     '_materialize_mask_counts', 'tick.mask_counts'),
    ('IO.pipeline.vortex_pipeline', 'VortexPipeline', '_tick_wait',
     'tick.wait'),
    ('IO.pipeline.vortex_pipeline', 'VortexPipeline', '_tick_collect',
     'tick.collect'),
    ('IO.transfer.data_exchange', 'NaiveExchange', 'bind', 'x.bind'),
    ('IO.transfer.data_exchange', 'NaiveExchange', 'launch', 'x.launch'),
    ('IO.transfer.data_exchange', 'NaiveExchange', 'sync', 'x.sync'),
    ('IO.transfer.tick_stages', 'TickStages', 'clean_intermediates',
     'tick.clean'),
    # Inside tick.pre_op: zero-copy reads and masks (decompress below); the
    # rest of tick.pre_op is the operator itself.
    ('IO.transfer.pre_comp_stage', 'PreCompStage', 'execute_zero_copy',
     'tick.zero_copy'),
    ('IO.transfer.pre_comp_stage', 'PreCompStage', 'apply_mask',
     'tick.apply_mask'),
    # The region producer: the pack before the operator (t+1), the home
    # writer's copy launches (right behind the pack) and the end of the pipe.
    ('IO.codec.region_producer', 'RegionProducer', 'pack', 'rp.pack'),
    ('IO.codec.home_writer', 'HomeWriter', 'ship', 'rp.ship'),
    ('IO.codec.region_producer', 'RegionProducer', 'drain', 'rp.drain'),
    # The gather pipe (rearrange / lineage / deferred gathers).
    ('IO.pipeline.rearrange_pipeline', None, '_gather', 'rg.gather'),
    ('IO.pipeline.rearrange_pipeline', None, 'col_to_gpu', 'col_to_gpu'),
    ('IO.pipeline.common', None, 'col_to_gpu', 'col_to_gpu'),
)
# Every class of this module with its own ``decompress_all`` gets a
# ``tick.decompress`` range (input decompression inside tick.pre_op).
DECOMPRESS_MODULE = 'IO.codec.batched_decompress'

_installed = False


def resolve(module, owner, attr):
    """``(owner object, its raw attribute)`` for one hook, or None when this
    tree has no such target."""
    try:
        obj = importlib.import_module(module)
    except ImportError:
        return None
    if owner is not None:
        obj = getattr(obj, owner, None)
    if obj is None or attr not in obj.__dict__:
        return None
    return obj, obj.__dict__[attr]


def _wrap(owner, attr, label):
    """Replace ``owner.attr`` by a version inside a ``label`` range."""
    import torch.cuda.nvtx as nvtx
    raw = owner.__dict__[attr]
    fn = raw.__func__ if isinstance(raw, (classmethod, staticmethod)) else raw

    @functools.wraps(fn)
    def wrapped(*a, **k):
        nvtx.range_push(label)
        try:
            return fn(*a, **k)
        finally:
            nvtx.range_pop()

    if isinstance(raw, classmethod):
        wrapped = classmethod(wrapped)
    elif isinstance(raw, staticmethod):
        wrapped = staticmethod(wrapped)
    setattr(owner, attr, wrapped)


def install():
    """Install the ``P <tag>`` and step ranges (once per process)."""
    global _installed
    if _installed:
        return
    import torch.cuda.nvtx as nvtx
    from utility.logger import PerformanceTracker

    # Perf-logger tags do not nest strictly (start/stop are keyed by tag), so
    # they use start/end ranges rather than the push/pop stack.
    open_ranges = {}
    orig_start, orig_stop = PerformanceTracker.start, PerformanceTracker.stop

    def start(self, tag):
        orig_start(self, tag)
        if self._enable:
            open_ranges[tag] = nvtx.range_start(f"P {tag}")

    def stop(self, tag):
        out = orig_stop(self, tag)
        rid = open_ranges.pop(tag, None)
        if rid is not None:
            nvtx.range_end(rid)
        return out

    hooks, missing = [], []
    for m, o, a, label in STEP_HOOKS:
        found = resolve(m, o, a)
        if found is None:
            missing.append(f"{m}.{o + '.' if o else ''}{a}")
        else:
            hooks.append((*found, a, label))
    if missing:
        print(f"[nvtx_ranges] not in this tree, no range: {', '.join(missing)}",
              flush=True)
    decompress = importlib.import_module(DECOMPRESS_MODULE)
    for name in dir(decompress):
        obj = getattr(decompress, name)
        if isinstance(obj, type) and 'decompress_all' in obj.__dict__:
            hooks.append((obj, obj.__dict__['decompress_all'], 'decompress_all',
                          'tick.decompress'))

    PerformanceTracker.start, PerformanceTracker.stop = start, stop
    for owner, _raw, attr, label in hooks:
        _wrap(owner, attr, label)
    _installed = True


def main():
    args = sys.argv[1:]
    if args and args[0] == '--':
        args = args[1:]
    root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))
    sys.path.insert(0, root)
    os.chdir(root)
    install()
    sys.argv = ['main.py'] + args
    runpy.run_path(os.path.join(root, 'main.py'), run_name='__main__')


if __name__ == '__main__':
    main()
