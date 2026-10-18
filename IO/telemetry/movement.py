"""The movement ledger: every copy of page bytes between host and GPU memory,
and every copy the granule cache avoids, counted once at the site that issues
(or skips) it, per query.

A record has a *kind*, what moved and why, and three counts: ``pages``
(pages copied or served; 0 for zero-copy reads, which read rows in place),
``logical_b`` (decompressed bytes) and ``wire_b`` (stored bytes, the
payload).

A copied read (``bind_tick_sources``) counts what the tick selects of each
page, not the page: one run per page, its selected records' decoded bytes as
``logical_b`` (a RAW record read by rows: the rows copied) and the run's
bytes as ``wire_b``.

================== ===========================================================
kind               site
================== ===========================================================
``scan_h2d``       base-table run copied H2D: ``bind_tick_sources`` (for a
                   zero-copy stage's mask or index too, read through a
                   ``DirectIn``)
``scan_uva``       base-table rows read zero-copy over UVA: the zero-copy
                   stages and ``rearrange_pipeline._stream_gather``
``spill_bypass``   intermediate written home without being offered to the
                   cache: the home writer's copies of the region producer's
                   regions no cache holds (``IO/codec/region_producer.py``;
                   a page there is a region)
``spill_rejected`` a cached region's refused suffix, copied home by the
                   producer's pack
``spill_evict``    a dirty granule a reclaim copies home
                   (``NativeGranuleCache._victims``); under the free-list
                   baseline, the regions an arena miss evicts home
                   (``IO/pages/device_regions.evict_one``, filed per query
                   by ``main.py``; a page there is a region)
``spill_mirror``   a cleaning copy (the producer, after a tick's admissions)
``restore_h2d``    a run of a producer's record table copied H2D (one page
                   per run)
``restore_evicted_h2d``
                   a cached region's home piece copied H2D (one page per
                   piece): granules the cache holds no more, or never
                   admitted (``bind_tick_sources``)
``restore_uva``    intermediate rows read zero-copy (the ``scan_uva`` sites)
``cache_hit``      a cached region's slot piece read D2D (one page per
                   piece): ``bind_tick_sources``
``resident_d2d``   a GPU region outside the cache, read D2D
                   (a GPU region's run, one page per run):
                   ``bind_tick_sources`` and the zero-copy gather stage
``cache_fill``     the granules the producer's pack (or a RAW output) wrote
                   into their slots
``avoided_dead``   a reclaimed dead granule, released with no copy
                   (``NativeGranuleCache._victims``)
``avoided_clean``  a reclaimed clean granule (same)
``result_d2h``     the output step's copy of the result columns to the host
                   (``operators/output.py``; a page there is a column)
================== ===========================================================

A page of ``cache_fill``, ``spill_evict``, ``spill_mirror`` and the
``avoided`` kinds is a granule: ``wire_b`` its bytes (a region's last granule
is shorter), ``logical_b`` the decoded bytes of the records in it, each
record's decoded bytes spread evenly over its stored bytes (the producer's
pack for ``cache_fill``, ``NativeGranuleCache.ledger_bytes`` for the
others).

A record table's reads are scans when it holds a base table
(``RecordTable.base``, set by the scan), restores otherwise; a host page is a
base table's.

Records go to the current *phase*: ``query`` (the query's operators), or
``output`` while ``main.py`` runs the output step (``set_phase``): its reads
of the result columns and its ``result_d2h``.  The main plan's output step
is outside the query's timer (``approx_time``), so ``kinds`` is the query
without it; a subquery plan's output step (a scalar) is in ``output`` too.

Experiment telemetry: every hook is guarded by ``gates.EXPERIMENT`` at its
call site.  ``main.py`` calls :func:`reset` at the top of each query and
writes :func:`report` as the perf report's ``movement`` section.  The module
imports nothing from ``IO/``: pages and columns are read by attribute.
"""
from __future__ import annotations

KINDS = (
    "scan_h2d", "scan_uva",
    "spill_bypass", "spill_rejected", "spill_evict", "spill_mirror",
    "restore_h2d", "restore_evicted_h2d", "restore_uva",
    "cache_hit", "resident_d2d",
    "cache_fill", "avoided_dead", "avoided_clean",
    "result_d2h",
)
READ_KINDS = ("scan_h2d", "scan_uva", "restore_h2d", "restore_evicted_h2d",
              "restore_uva", "cache_hit", "resident_d2d")
SPILL_KINDS = ("spill_bypass", "spill_rejected", "spill_evict",
               "spill_mirror")
PHASES = ("query", "output")

# phase -> kind -> [pages, logical_b, wire_b]
_totals = {phase: {kind: [0, 0, 0] for kind in KINDS} for phase in PHASES}
_current = _totals["query"]


def record(kind, pages, logical_b, wire_b) -> None:
    """Add one site's counts to ``kind`` of the current phase."""
    t = _current[kind]
    t[0] += int(pages)
    t[1] += int(logical_b)
    t[2] += int(wire_b)


def set_phase(phase: str) -> None:
    """Send the records that follow to ``phase`` (``PHASES``)."""
    global _current
    _current = _totals[phase]


def reset() -> None:
    """Zero every count and go back to the query phase (top of each
    query)."""
    global _current
    for kinds in _totals.values():
        for t in kinds.values():
            t[:] = [0, 0, 0]
    _current = _totals["query"]


def report() -> dict:
    """The perf report's ``movement`` section: ``kinds``, the query phase,
    and ``output``, the output step; each lists every kind, zeros included,
    so a reader can index strictly."""
    def counts(kinds):
        return {k: {"pages": t[0], "logical_b": t[1], "wire_b": t[2]}
                for k, t in kinds.items()}
    return {"kinds": counts(_totals["query"]),
            "output": counts(_totals["output"])}


def table_read_kind(table) -> str:
    """``scan_h2d`` for a base table's record table (``RecordTable.base``),
    ``restore_h2d`` for a producer's."""
    return "scan_h2d" if table.base else "restore_h2d"


def uva_kind(col) -> str:
    """``scan_uva`` when ``col`` (a ``TableVariable``) is a base table's
    record table, ``restore_uva`` when it is a producer's."""
    return "scan_uva" if col.record_table().base else "restore_uva"


def record_uva_read(col, nbytes: int) -> None:
    """One zero-copy kernel that read ``nbytes`` of ``col``'s RAW rows over
    UVA (``logical_b == wire_b``)."""
    record(uva_kind(col), 0, nbytes, nbytes)
