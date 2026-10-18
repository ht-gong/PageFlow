"""The telemetry and debug switches, each read once from the environment at
import.  Every one is off by default and turned on by ``1``, ``true``,
``yes`` or ``on`` (any case, surrounding spaces ignored); anything else,
``0`` included, leaves it off.

``TQP_EXPERIMENT_TELEMETRY`` (``EXPERIMENT``) turns on the numbers an
experiment reads from the perf report: the movement ledger (``movement.py``,
the ``movement`` section), the ``sizing_model`` and ``compression`` sections,
and each pipe's ``peak_gpu_bytes``.  ``main.py`` refuses it without
``--perf_report``.  Every hook tests it at its call site, before computing
its arguments::

    if gates.EXPERIMENT:
        movement.record(...)

The debug validators check the engine against itself and raise on a
mismatch; they cost host time, so a timing run leaves them off:

- ``TQP_VARCHK`` (``VARCHK``): a compressed read's record index (positive
  original sizes that sum to the decoded size, records that do not overlap)
  as its decode template compiles.
- ``TQP_PICK_DEBUG`` (``PICK_DEBUG``): each tick's in-stages against the
  codec picks recorded at its rebuild horizon, at launch.
- ``TQP_TIMED_GC_VALIDATE`` (``TIMED_GC_VALIDATE``): no cyclic garbage
  collection inside a query's timed region.

Import the module, never a name, so a test can set ``gates.<NAME>``.
"""
from __future__ import annotations

import os


def parse(environ, name: str) -> bool:
    """``environ[name]`` as a switch: on for ``1``, ``true``, ``yes`` or
    ``on`` in any case, off for anything else and when unset."""
    return (environ.get(name) or "").strip().lower() in ("1", "true", "yes", "on")


EXPERIMENT: bool = parse(os.environ, "TQP_EXPERIMENT_TELEMETRY")

VARCHK: bool = parse(os.environ, "TQP_VARCHK")
PICK_DEBUG: bool = parse(os.environ, "TQP_PICK_DEBUG")
TIMED_GC_VALIDATE: bool = parse(os.environ, "TQP_TIMED_GC_VALIDATE")
