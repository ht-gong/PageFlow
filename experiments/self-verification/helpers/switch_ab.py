#!/usr/bin/env python3
"""In-process A/B over engine switches (E2's ``switches`` mode, E5's arms):
the arms alternate per query occurrence inside one ``main.py`` process, so
every arm is measured seconds from the others and the shared host's load
drift cancels (docs/sf100-bottleneck-study.md §1.6, §6).  Runs inside the
``dev`` container.

A switch is a module global that the engine sets once, at import, from its
environment variable and reads through its module at each use, e.g.
``IO.some.module.NEW_PATH`` from ``TQP_NEW_PATH``.  An arm is
``name:path=value[,path=value...]``; a value is a Python literal or a bare
string, and every path must exist when the run starts.

Schedule.  The warmup runs the first arm.  Each qid's first ``--warm``
occurrences cycle through the arms and are marked warm; the ones after form
cycles of one occurrence per arm, the order rotated by one each cycle, so a
qid needs ``warm + cycles * len(arms)`` occurrences.  Times are the rows of
``main.py``'s own ``<perf_report>/query_times.csv``.

Output, ``<out>/switch_ab.jsonl``: one line per occurrence (qid, occurrence,
cycle or None when warm, arm, ``approx_time`` or None when it failed, the
switch values it ended with).  ``--nvtx`` installs ``nvtx_ranges.py`` and
wraps each occurrence in an ``ARM <name>`` range for ``trace_analyze.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inprocess  # noqa: E402

OUT_NAME = 'switch_ab.jsonl'
CMDLINE_NAME = 'switch_ab_cmdline.json'


def parse_value(text):
    """A Python literal, or the text itself."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_arm(spec):
    """``name:path=value[,path=value...]`` -> ``(name, {path: value})``."""
    name, sep, body = spec.partition(':')
    if not sep or not name or not body:
        raise ValueError(f'arm {spec!r}: expected name:path=value[,path=value...]')
    values = {}
    for item in body.split(','):
        path, sep, value = item.partition('=')
        if not sep or not path:
            raise ValueError(f'arm {spec!r}: {item!r} is not path=value')
        values[path.strip()] = parse_value(value.strip())
    return name, values


def check_arms(arms):
    """Arms are named uniquely and set the same switches."""
    names = [n for n, _ in arms]
    if len(arms) < 2 or len(set(names)) != len(names):
        raise ValueError(f'need two or more uniquely named arms, got {names}')
    keys = [sorted(v) for _, v in arms]
    if any(k != keys[0] for k in keys):
        raise ValueError('every arm must set the same switches: '
                         + '; '.join(f'{n}: {k}' for n, k in zip(names, keys)))


def arm_for(k, names, n_warm):
    """``(arm, cycle)`` of a qid's k-th occurrence; cycle is None while warm."""
    n = len(names)
    if k < n_warm:
        return names[k % n], None
    c, p = divmod(k - n_warm, n)
    return names[(p + c) % n], c


class Driver:
    """The per-occurrence state: which arm runs, and what to record when the
    occurrence ends."""

    def __init__(self, arms, n_warm, out_path, times, nvtx=False):
        self.names = [n for n, _ in arms]
        self.values = dict(arms)
        self.n_warm = n_warm
        self.out_path = out_path
        self.times = times
        self.targets = {p: inprocess.resolve(p) for p in arms[0][1]}
        self.nvtx = nvtx
        self._range = None
        self.occ = defaultdict(int)
        self.pos = -1
        self.cur = None
        self.apply(self.names[0])

    def apply(self, name):
        for path, value in self.values[name].items():
            owner, attr = self.targets[path]
            setattr(owner, attr, value)

    def read_back(self):
        return {path: repr(getattr(owner, attr))
                for path, (owner, attr) in self.targets.items()}

    def query_start(self, sf, qid, warmup):
        self.finish()
        if warmup:
            self.apply(self.names[0])
            return
        k = self.occ[qid]
        self.occ[qid] = k + 1
        arm, cycle = arm_for(k, self.names, self.n_warm)
        self.apply(arm)
        self.pos += 1
        self.cur = {'qid': qid, 'occ': k, 'pos': self.pos, 'cycle': cycle,
                    'warm': cycle is None, 'arm': arm, 'sf': sf}
        if self.nvtx:
            import torch.cuda.nvtx as nvtx
            self._range = nvtx.range_start(f'ARM {arm}')

    def finish(self):
        """Record the occurrence that just ended (no-op outside one)."""
        if self._range is not None:
            import torch.cuda.nvtx as nvtx
            nvtx.range_end(self._range)
            self._range = None
        if self.cur is None:
            return
        rec = dict(self.cur, approx_time=self.times.take(self.cur['qid']),
                   switches=self.read_back())
        with open(self.out_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec) + '\n')
        self.cur = None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--arm', action='append', required=True,
                    help='name:path=value[,path=value...]; the first is the base')
    ap.add_argument('--warm', type=int, default=2,
                    help='warm occurrences per qid before the cycles (default 2)')
    ap.add_argument('--nvtx', action='store_true',
                    help="nvtx_ranges.py's ranges plus one ARM range per occurrence")
    ap.add_argument('main_args', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    main_args = inprocess.split_main_args(args.main_args)
    arms = [parse_arm(s) for s in args.arm]
    check_arms(arms)
    perf_dir = inprocess.perf_report_dir(main_args)
    if perf_dir is None:
        ap.error('main.py needs --perf_report: the times are its query_times.csv')

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUT_NAME)
    open(out_path, 'w').close()
    with open(os.path.join(out_dir, CMDLINE_NAME), 'w', encoding='utf-8') as f:
        json.dump({'driver': {'arms': [{'name': n, 'values': {p: repr(v) for p, v in vals.items()}}
                                       for n, vals in arms],
                              'warm': args.warm, 'nvtx': args.nvtx},
                   'main_args': main_args,
                   'env': {k: v for k, v in os.environ.items() if k.startswith('TQP_')}},
                  f, indent=1)

    inprocess.enter_repo()
    driver = Driver(arms, args.warm, out_path,
                    inprocess.QueryTimes(os.path.join(inprocess.ROOT, perf_dir)),
                    nvtx=args.nvtx)
    if args.nvtx:
        import nvtx_ranges
        nvtx_ranges.install()
    inprocess.on_query_start(driver.query_start)
    try:
        inprocess.run_main(main_args)
    finally:
        driver.finish()


if __name__ == '__main__':
    main()
