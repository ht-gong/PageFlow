"""The paper scripts' one reader of the spill pass's table,
``baselines/results/spill_profile.csv``.

Its contract is ``baselines/AGENTS.md`` ("Spill pass"): one row per
``(engine, scale_factor, query, kind)``, written only by
``baselines/spill_results.py``, whose ``FIELDS``, ``KINDS``, ``COMMON_KINDS``,
``QUALITY``, ``RUN_STATUS`` and ``headline()`` are imported here,
so the file is checked against the writer's own definitions rather than a
copy of them.  Figures aggregate by category (``category()``) or take the
headline sums (``headlines()``), never a kind by name.

Read strictly, so a change on either side fails loudly instead of turning into
a gap in a figure: a header other than ``FIELDS``, an engine without ``KINDS``,
a kind outside its engine's lists, a query other than ``query1`` ..
``query22``, a repeated ``(engine, scale_factor, query, kind)``, a query
missing one of its engine's kinds, an unknown ``quality`` or ``run_status``,
or a value that is not a number raises.

Standard library only, so it runs on the host.
"""
from __future__ import annotations

import csv
import os
import sys

from common import QUERIES, ROOT

sys.path.insert(0, os.path.join(ROOT, "baselines"))
from spill_results import (COMMON_KINDS, FIELDS, KINDS, QUALITY, RUN_STATUS,  # noqa: E402
                           category, headline)

PATH = os.path.join(ROOT, "baselines", "results", "spill_profile.csv")
_NUM = ("logical_b", "wire_b", "count", "seconds")


def load(path: str | None = None) -> dict[tuple[str, int], dict[int, dict]]:
    """``{(engine, scale_factor): {query: {"run_status": str, "rows": {kind:
    row}}}}`` from ``path`` (default ``PATH``); a row has ``quality``,
    ``source`` and the numbers ``logical_b``, ``wire_b``, ``count``,
    ``seconds`` (float, or None where not measured).

    A pair with no rows is absent: not collected.  The engine keys are
    ``all_results.csv``'s.
    """
    path = path or PATH
    cells: dict[tuple[str, int], dict[int, dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        if rows.fieldnames != FIELDS:
            raise ValueError(f"{path}: header {rows.fieldnames}, expected "
                             f"spill_results.FIELDS {FIELDS}")
        for line, r in enumerate(rows, start=2):
            where = f"{path}:{line}"
            engine = r["engine"]
            if engine not in KINDS:
                raise ValueError(f"{where}: engine {engine!r} has no spill_results.KINDS")
            if r["kind"] not in {**KINDS[engine], **COMMON_KINDS}:
                raise ValueError(f"{where}: kind {r['kind']!r} is not one of {engine}'s")
            q = r["query"].removeprefix("query")
            if not q.isdigit() or int(q) not in QUERIES:
                raise ValueError(f"{where}: query {r['query']!r}")
            if r["quality"] not in QUALITY or r["run_status"] not in RUN_STATUS:
                raise ValueError(f"{where}: quality {r['quality']!r} / run_status {r['run_status']!r}")
            cell = cells.setdefault((engine, int(r["scale_factor"])), {}).setdefault(
                int(q), {"run_status": r["run_status"], "rows": {}})
            if r["kind"] in cell["rows"]:
                raise ValueError(f"{where}: repeated ({engine}, {r['scale_factor']}, "
                                 f"{r['query']}, {r['kind']})")
            if r["run_status"] != cell["run_status"]:
                raise ValueError(f"{where}: run_status differs within the query")
            row = {"quality": r["quality"], "source": r["source"]}
            for k in _NUM:
                try:
                    row[k] = float(r[k]) if r[k] != "" else None
                except ValueError:
                    raise ValueError(f"{where}: {k} {r[k]!r} is not a number") from None
            cell["rows"][r["kind"]] = row
    for (engine, sf), queries in cells.items():
        need = set(KINDS[engine]) | set(COMMON_KINDS)
        for q, cell in queries.items():
            missing = need - set(cell["rows"])
            if missing:
                raise ValueError(f"{path}: ({engine}, {sf}, query{q}) lacks {sorted(missing)}")
    return cells


def headlines(cells) -> dict[tuple[str, int], dict[int, dict]]:
    """Per query of ``load()``'s cells, spill_results.headline(): ``{name:
    (logical_b, wire_b, quality)}`` for off_gpu, pressure_spill, disk_spill,
    design_disk and read_back, None where the engine has no such tier."""
    out = {}
    for (engine, sf), queries in cells.items():
        out[(engine, sf)] = {q: headline(engine, [dict(v, kind=k) for k, v in c["rows"].items()])
                             for q, c in queries.items()}
    return out


def by_category(engine: str, cell: dict) -> dict[str, float]:
    """A query's logical bytes per category (peaks: the largest), without
    ``context``, whose kinds do not add up; unmeasured rows count
    nothing, so check each row's quality where it matters."""
    out: dict[str, float] = {}
    for kind, row in cell["rows"].items():
        cat, v = category(engine, kind), row["logical_b"]
        if v is None or cat == "context":
            continue
        out[cat] = max(out.get(cat, 0.0), v) if cat.startswith("peak_") else out.get(cat, 0.0) + v
    return out
