"""The experiment scripts' one reader of TQP perf reports.

A perf-report directory (``main.py --perf_report <dir>``) holds one
``SF<sf>_<qid>.json`` per query.  Every accessor indexes strictly: a missing
file, section, kind or field raises instead of reading as 0, so a renamed key
fails loudly rather than turning into a silent zero.

What the experiments read, and nothing else:

* ``wall.approx_time``, always written (the same number as
  ``query_times.csv``);
* under ``TQP_EXPERIMENT_TELEMETRY=1``: the ``movement`` section (the
  movement ledger, ``IO/telemetry/movement.py``: per query and kind, the
  ``pages``, ``logical_b`` and ``wire_b`` moved; ``kinds`` without the
  output step, ``output`` the output step's), ``residency``
  (``IO/telemetry/residency.py``), ``sizing_model``, ``compression`` and
  each pipe's ``peak_gpu_bytes``.

A script that reads the ledger loads with ``movement=True``, which raises
:class:`PerfDataMissing` for a report written without the switch.

Standard library only, so it runs on the host.
"""
from __future__ import annotations

import glob
import json
import os

# The ledger's kinds, as IO/telemetry/movement.py writes them (a unit test
# keeps the lists equal).
KINDS = ("scan_h2d", "scan_uva",
         "spill_bypass", "spill_rejected", "spill_evict", "spill_mirror",
         "restore_h2d", "restore_evicted_h2d", "restore_uva",
         "cache_hit", "resident_d2d",
         "cache_fill", "avoided_dead", "avoided_clean",
         "result_d2h")
SPILL_KINDS = ("spill_bypass", "spill_rejected", "spill_evict",
               "spill_mirror")
# Where a pipe's output pages go (each output page is born exactly once).
OUT_KINDS = ("cache_fill", "spill_bypass", "spill_rejected")
# Kinds the ledger gained after reports were first written (2026-10-10): a
# report from before has none of them, and :func:`mv` reads them as 0 there.
# That keeps old and new reports' sums equal: a cached region's home pieces
# were ``restore_h2d`` before ``restore_evicted_h2d``, and the result copy
# was not counted.
LATER_KINDS = ("restore_evicted_h2d", "result_d2h")


class PerfDataMissing(LookupError):
    """A report lacks the data a script asked for."""


def report_path(perf_dir: str, sf: int, q: int) -> str:
    return os.path.join(perf_dir, f"SF{sf}_{q}.json")


def clear_run_outputs(perf_dir: str) -> None:
    """Remove what an earlier ``main.py --perf_report perf_dir`` wrote there,
    so a launch that fails early leaves nothing to be read as its own."""
    paths = glob.glob(os.path.join(glob.escape(perf_dir), "SF*_*.json"))
    paths += [os.path.join(perf_dir, name)
              for name in ("query_times.csv", "cmdline.json", "results.out")]
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


def load(perf_dir: str, sf: int, q: int, *, movement: bool = False) -> dict:
    """One query's report.  Raises on a missing or unreadable file, and, with
    ``movement=True``, on a report that has no ``movement`` section."""
    path = report_path(perf_dir, sf, q)
    with open(path, encoding="utf-8") as f:
        perf = json.load(f)
    if movement and "movement" not in perf:
        raise PerfDataMissing(
            f"{path}: no 'movement' section; rerun the engine with "
            "TQP_EXPERIMENT_TELEMETRY=1")
    return perf


def approx_time(perf: dict) -> float:
    """The canonical query time, ``wall.approx_time`` (seconds)."""
    return float(perf["wall"]["approx_time"])


def peak_gpu_bytes(perf: dict) -> int:
    """The query's high-water mark in the arena's operator space: the
    largest per-pipe ``totals.peak_gpu_bytes``.  The cache region is not in
    it; reports written while the cache was a slab inside the operator
    space count the slab too, so a cache run's peak is lower by the slot
    pool's size since then, with no change in operator memory."""
    peaks = [sec["totals"]["peak_gpu_bytes"] for sec in perf.values()
             if isinstance(sec, dict) and isinstance(sec.get("totals"), dict)
             and "peak_gpu_bytes" in sec["totals"]]
    if not peaks:
        raise PerfDataMissing(
            "perf report has no per-pipe peak_gpu_bytes: it was written "
            "without TQP_EXPERIMENT_TELEMETRY=1")
    return max(int(p) for p in peaks)


def mv(perf: dict, kind: str, field: str, phase: str = "kinds") -> int:
    """``movement.kinds[kind][field]``: ``pages``, ``logical_b`` or
    ``wire_b``; with ``phase="output"``, the output step's (a report from
    before the output phase has none).  0 for one of :data:`LATER_KINDS` in
    a report that predates it."""
    try:
        return perf["movement"][phase][kind][field]
    except KeyError:
        if "movement" not in perf:
            raise PerfDataMissing(
                "perf report has no 'movement' section: it was written "
                "without TQP_EXPERIMENT_TELEMETRY=1") from None
        at = perf["movement"].get(phase)
        if at is not None and kind in LATER_KINDS and kind not in at:
            return 0
        raise


def mv_sum(perf: dict, kinds, field: str) -> int:
    return sum(mv(perf, kind, field) for kind in kinds)


def born(perf: dict, field: str = "logical_b") -> int:
    """Intermediate pages produced: :data:`OUT_KINDS`."""
    return mv_sum(perf, OUT_KINDS, field)


def spill_total(perf: dict, field: str = "wire_b") -> int:
    """Every spill kind: all intermediate bytes written D2H."""
    return mv_sum(perf, SPILL_KINDS, field)
