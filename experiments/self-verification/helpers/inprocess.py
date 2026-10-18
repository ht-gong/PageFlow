"""Running ``main.py`` inside an experiment driver's own process.

An in-process runner (``switch_ab.py``) installs its hooks, then hands the
process to ``main.py`` through :func:`run_main`.  :func:`on_query_start` calls back at
the start of every query occurrence, warmup included, and says whether it
belongs to the SF1 warmup.  :class:`QueryTimes` reads the row each finished
occurrence appended to ``<perf_report>/query_times.csv``: the engine's
canonical per-query time (AGENTS.md, "TIMING").

Nothing here imports the engine at module import time.
"""
from __future__ import annotations

import importlib
import os
import runpy
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    os.pardir, os.pardir, os.pardir))


def enter_repo():
    """Make the engine importable and the repository root the working
    directory, as ``main.py`` expects.  Call before installing hooks."""
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)


def split_main_args(rest):
    """The ``main.py`` arguments of an ``argparse.REMAINDER`` list, which
    starts with the ``--`` separator when one was given."""
    return list(rest[1:] if rest[:1] == ['--'] else rest)


def perf_report_dir(main_args):
    """The ``--perf_report`` directory of a ``main.py`` argument list, or
    None."""
    for i, a in enumerate(main_args):
        if a == '--perf_report' and i + 1 < len(main_args):
            return main_args[i + 1]
        if a.startswith('--perf_report='):
            return a.split('=', 1)[1]
    return None


def in_warmup() -> bool:
    """True while ``main.py`` runs its SF1 warmup.  :func:`run_main` runs
    ``main.py`` as the ``__main__`` module, whose ``_is_warmup`` global
    ``main.warmup`` sets for the warmup's duration."""
    return bool(getattr(sys.modules.get('__main__'), '_is_warmup', False))


def on_query_start(callback):
    """Call ``callback(sf, qid, warmup)`` at the start of every query
    occurrence.  ``main.py`` binds the query context
    (``constants.set_query_context``) once per occurrence: after the previous
    occurrence has written its ``query_times.csv`` row, its perf report and
    released its pools, and before this one's plan, scan and timed region."""
    import constants
    orig = constants.set_query_context

    def set_query_context(scale_factor, qid):
        callback(int(scale_factor), qid, in_warmup())
        return orig(scale_factor, qid)

    constants.set_query_context = set_query_context


class QueryTimes:
    """The ``qid,approx_time`` rows ``main.py`` appends to
    ``<perf_report>/query_times.csv``, read one finished occurrence at a time.
    ``main.py`` empties the file once, when its timed run starts, and appends
    one row per occurrence that completes; a failed occurrence appends none."""

    def __init__(self, perf_dir):
        self.path = (os.path.join(perf_dir, 'query_times.csv')
                     if perf_dir else None)
        self._seen = 0

    def take(self, qid):
        """The time of the one row appended since the last call, when it is
        ``qid``'s; None when there is no new row (the occurrence failed)."""
        if self.path is None or not os.path.exists(self.path):
            return None
        with open(self.path, encoding='utf-8') as f:
            rows = [r for r in f.read().splitlines() if r.strip()]
        if len(rows) < self._seen:
            self._seen = 0
        new, self._seen = rows[self._seen:], len(rows)
        if not new:
            return None
        q, t = new[-1].split(',')
        return float(t) if int(q) == int(qid) else None


def resolve(path):
    """``(owner, attribute name)`` for a dotted ``module[.Class].attr`` path:
    the longest importable module prefix, then attributes down to the last
    name, which must exist on its owner."""
    parts = path.split('.')
    for n in range(len(parts) - 1, 0, -1):
        try:
            owner = importlib.import_module('.'.join(parts[:n]))
        except ImportError:
            continue
        for name in parts[n:-1]:
            owner = getattr(owner, name)
        if not hasattr(owner, parts[-1]):
            raise AttributeError(f'{path}: no attribute {parts[-1]!r}')
        return owner, parts[-1]
    raise ImportError(f'{path}: no importable module prefix')


def run_main(main_args):
    """Run ``main.py`` with ``main_args`` in this process, as ``__main__``."""
    enter_repo()
    sys.argv = ['main.py'] + list(main_args)
    runpy.run_path(os.path.join(ROOT, 'main.py'), run_name='__main__')
