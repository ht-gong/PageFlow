#!/usr/bin/env python3
"""Cross-system TPC-H SCALING study: this system vs DuckDB / cudf-Polars /
Spark-RAPIDS / Sirius at SF100, SF400 and SF700.

    python3 scaling_study.py [out_dir]     # -> <out_dir>/*.csv|.tex, <out_dir>/plot/*
    python3 scaling_study.py --refresh     # re-read the baselines' results, report drift
    python3 scaling_study.py --stats       # print the summary without writing files

Where the ablation study measures OUR mechanisms against each other at one scale
factor, this measures the whole system against other engines as the scale factor
grows -- the question is not "are we faster" but "does the gap widen with data
size", which is the claim a GPU engine that streams compressed pages has to earn.

NUMBERS ARE CARRIED INLINE.  `.gitignore` excludes `*.csv` and `result*`, so a
study that read its inputs from CSVs would not be reproducible from a clean
checkout -- the same reason make_ablation_figs.py and make_tpch_comparison_fig.py
carry theirs.  Edit the tables below and re-run.  `--refresh` re-reads the
baselines' results (BASELINES_CSV, baselines/results/all_results.csv) and
prints every cell that has drifted from the inline copy, so staleness is
detectable without making the figures depend on that CSV.

PROVENANCE IS PER CELL, and the three sources are NOT interchangeable:

  `baselines`  baselines/results/all_results.csv -- the baselines' run.sh
               (then the standalone TQP-Vortex-Baselines repo, now baselines/)
               over clean external NDS-H parquet staged to /dev/shm, measured
               on this box.  Covers all four competitors at SF100/SF400
               and DuckDB at SF700.
  `table`      the SF700 comparison table supplied for the paper (Sirius,
               cudf-Polars, Spark-RAPIDS and this system).  Supplied as SF700 by
               the author.  NOTE: the Sirius / cudf-Polars / Spark-RAPIDS columns
               are numerically identical to the archived SF500 *cold-protocol*
               run in the baselines repo
               (results/archive/all_results_sf500_cold_protocol.csv, 0.0%
               deviation across all 22 queries and matching Sirius failures;
               that file never entered this repo, and baselines/results/
               dropped its own SF500 rows on 2026-10-07).
               Recorded here because a scaling study is exactly the analysis that
               a mislabelled scale factor would invalidate; carried as SF700 per
               the author's instruction.
  `engine`     this repo's own query_times.csv (`approx_time`), per AGENTS.md.

SF700 therefore MIXES sources: DuckDB comes from `baselines`, the other three from
`table`.  `check_source_mixing()` prints that warning on every run rather than
letting the reader infer a single protocol behind one column of a table.

FAILED IS NOT UNCOLLECTED.  The two are kept apart everywhere -- distinct CSV
statuses, distinct figure glyphs -- because a crash and a scheduling decision carry
opposite meanings, and neither may be interpolated over.  Currently `failed`:
Spark-RAPIDS q17 and cudf-Polars q18 at SF400, and this system's q17 at SF400.  All
three fall OUTSIDE the common set, so no series carries a failure mark on the
figures and the legend suppresses those keys; the machinery stays because the
condition is data-dependent, not permanent.  Note SF700 for cudf-Polars and
Spark-RAPIDS was never run in the baselines repo (its SF700 driver was scoped to
duckdb + sirius), so those cells reach this study only through TABLE_SF700 --
drop that table and they become uncollected again.

COMPARABILITY.  Competitor seconds are their harness's per-query wall clock; ours
is the engine's `approx_time`.  Those are close but not the same instrument, and
AGENTS.md forbids drawing OUR conclusions from wall clock -- so the cross-system
deltas here are honest to roughly the size of a host-orchestration difference, and
a two-figure-agreement claim should not be made from them.

THE COMMON SET.  Totals summed over whatever each engine happened to complete are
not comparable -- a FAIL silently shrinks the total and flatters the engine that
failed.  Every headline number is therefore computed over COMMON_QUERIES: the
queries every core system completed at every scale factor where it has data.  The
per-query CSV keeps all 22 with their statuses, so nothing is hidden.
"""
import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_OUT = os.path.join(REPO, "experiments", "results", "scaling_study")
BASELINES_CSV = os.environ.get(
    "TQP_BASELINES_CSV",
    os.path.join(REPO, "baselines", "results", "all_results.csv"))

SYSTEM = "PageFlow"                  # this system, as named in the paper figures
QUERIES = list(range(1, 23))
SCALE_FACTORS = [100, 400, 700]

# The systems the study is *about*: the author's named comparison plus us.  Sirius
# is carried through every table but is not part of the common-set intersection --
# it fails five queries at SF700, and letting it define the common set would drop
# those five from every other system's headline number too.
CORE_SYSTEMS = ["DuckDB", "cudf-Polars", "Spark-RAPIDS", SYSTEM]
# Order is make_tpch_comparison_fig.py's SYS order (cudf-Polars, Spark-RAPIDS,
# then this system last) with DuckDB first in its baseline role, so the two
# figures list their series identically in legends, tables and CSVs.
ALL_SYSTEMS = ["DuckDB", "cudf-Polars", "Spark-RAPIDS", SYSTEM]

# Sirius is EXCLUDED from the study but its numbers are deliberately left in the
# tables below.  Deleting them would make `--refresh` report the csv's Sirius
# rows as unknown and would quietly destroy the only record that the SF700
# comparison table's Sirius column matches the archived SF500 cold-protocol run
# exactly -- the evidence behind the provenance note.  Excluding here keeps it
# out of every emitted product while leaving that evidence intact and the
# decision reversible by one edit.
EXCLUDED_SYSTEMS = ("Sirius",)

# ── the numbers ──────────────────────────────────────────────────────────
# Each list is q1..q22 in order; None = the engine failed that query.

# source: `baselines` -- baselines/results/all_results.csv
BASELINE_RUNTIMES = {
    100: {
        "DuckDB": [2.131, 0.864, 1.149, 0.843, 1.186, 0.193, 0.843, 1.222, 3.328,
                   0.834, 5.113, 0.842, 1.823, 0.813, 0.715, 0.554, 0.551, 4.537,
                   1.362, 1.098, 5.498, 1.1],
        "cudf-Polars": [3.755, 0.45, 8.055, 2.879, 4.932, 1.776, 9.757, 6.738,
                       18.068, 9.436, 0.547, 2.003, 1.913, 3.73, 3.101, 0.525,
                       3.155, 3.57, 8.793, 6.123, 12.753, 0.307],
        "Spark-RAPIDS": [7.333, 25.07, 27.428, 13.405, 31.201, 4.985, 30.609,
                         26.621, 48.268, 25.083, 32.614, 11.635, 16.308, 18.109,
                         28.094, 24.365, 23.972, 29.976, 11.799, 25.602, 29.041,
                         15.021],
        "Sirius": [1.034, 0.838, 1.705, 0.959, 1.646, 0.736, 1.868, 2.041, 2.17,
                   1.644, 1.678, 1.221, 0.858, 1.21, 1.262, 0.673, 1.856, 2.301,
                   1.41, 1.534, 3.41, 0.787],
    },
    400: {
        "DuckDB": [4.675, 2.216, 4.952, 3.925, 3.552, 0.679, 3.231, 4.208, 12.611,
                   4.32, 3.863, 3.01, 5.327, 3.348, 1.986, 3.186, 4.779, 37.023,
                   4.87, 6.327, 22.613, 5.041],
        "cudf-Polars": [15.35, 1.647, 96.237, 5.753, 68.854, 7.58, 175.963, 95.706,
                       146.727, 129.086, 1.99, 77.103, 7.488, 20.725, 12.166,
                       2.435, 12.264, None, 66.396, 24.892, 208.574, 1.214],
        "Spark-RAPIDS": [8.116, 41.722, 45.963, 19.303, 60.026, 9.269, 33.478,
                         72.808, 82.927, 37.974, 20.761, 22.586, 13.576, 18.932,
                         32.557, 23.845, None, 53.511, 19.293, 42.966, 65.066,
                         17.237],
        "Sirius": [2.722, 2.251, 12.259, 2.937, 5.879, 2.634, 6.349, 7.31, 7.951,
                   5.678, 2.26, 3.682, 2.856, 3.794, 4.069, 1.637, 14.619, 34.147,
                   4.875, 4.669, 28.358, 1.466],
    },
}

# DuckDB SF700 as the baselines repo measured it on this box.  SUPERSEDED by the
# author-supplied TABLE_SF700 entry below and therefore NOT part of the study --
# kept because the two disagree by roughly 2x in both directions (q1 15.76 vs
# 9.96, q18 28.81 vs 75.72), which is exactly the kind of discrepancy that must
# stay on the record rather than being deleted, and because `--refresh` still
# validates it against the csv.
SUPERSEDED_BASELINE_SF700 = {
    "DuckDB": [15.76, 2.047, 14.753, 6.819, 9.439, 2.456, 9.139, 14.109,
               33.645, 12.603, 3.044, 5.427, 14.407, 9.39, 8.076, 5.318,
               13.524, 28.811, 10.892, 10.636, 29.658, 7.721],
}

# source: `table` -- the supplied SF700 comparison table (see PROVENANCE above).
TABLE_SF700 = {
    "DuckDB": [9.96, 2.10, 10.37, 8.13, 12.84, 6.37, 14.44, 10.25, 38.01, 13.76,
               4.29, 10.20, 23.31, 10.73, 6.96, 9.50, 9.37, 75.72, 14.24, 11.15,
               37.15, 8.18],
    "Sirius": [11.94, 23.31, None, 10.03, 93.20, 7.69, 30.08, None, None, None,
               11.16, 11.55, 12.35, 10.93, 12.22, 11.36, 78.29, 65.35, 15.16,
               25.24, None, 9.60],
    "cudf-Polars": [15.57, 2.52, 106.62, 5.82, 164.05, 9.52, 370.15, 146.80,
                   564.03, 36.26, 2.61, 111.43, 7.72, 18.38, 7.42, 5.73, 14.72,
                   168.29, 140.95, 14.38, 729.87, 2.27],
    "Spark-RAPIDS": [12.84, 36.67, 43.94, 22.87, 100.27, 6.65, 50.73, 121.14,
                     157.83, 41.08, 30.06, 21.48, 19.08, 19.61, 32.41, 23.01,
                     61.23, 124.53, 21.52, 41.13, 97.83, 15.81],
    SYSTEM: [12.88, 7.37, 4.23, 3.33, 7.47, 2.22, 30.69, 5.34, 17.54, 14.96, 4.46,
             3.01, 12.18, 1.92, 3.89, 3.34, 13.03, 12.51, 4.20, 4.68, 18.98, 1.19],
}

# source: `engine` -- experiments/results/2026-08-01/full_opt_sf100_1rep,
# 2026-08-01, engine @3eac734, all optimizations on (buffer=dataflow,
# sizing=mid_stream, compression=adaptive, search=read-and-write), chunk 512 MB,
# slot pool 4 GB, cmp pool 29 GB, 1 rep, 22/22 pass, 0 OOM.  query_times.csv
# approx_time, per AGENTS.md.
ENGINE_SF100 = [1.5002, 0.988054, 0.680644, 0.556677, 1.09488, 0.383758, 3.56616,
                0.854064, 2.4226, 1.93429, 0.645293, 0.562191, 1.73347, 0.329949,
                0.671688, 0.46964, 1.20832, 1.31418, 0.730152, 0.918976, 3.25764,
                0.211422]

# source: `engine` -- experiments/results/2026-08-01/full_opt_sf400_1rep,
# 2026-08-01, same HEAD and same configuration as ENGINE_SF100 (iso-config is the
# point: a scaling curve whose knobs move with the scale factor measures the
# knobs).  21/22 pass; q17 is a genuine in-pool OOM, NOT a missing run -- the
# engine asked for 25784 MB in one contiguous allocation against a 12135 MB
# largest free run (`=== OOM DIAGNOSTIC ===`, so arena/contiguity, not the
# out-of-pool driver class).  Re-running it at --slot_pool_gb 0.5 closed 76% of
# the gap (request 23434 MB vs 20215 MB contiguous) but still failed, so the
# residue is the aggregation slab's charge against c0, not slot-pool pressure.
# Carried as None = `failed`, which is what it was -- and q17 is outside
# COMMON_QUERIES regardless, because Spark-RAPIDS also fails it at SF400.
ENGINE_SF400 = [5.52996, 3.78566, 2.50111, 2.12584, 4.47929, 1.40628, 13.9067,
                2.72782, 10.6219, 7.31548, 2.4884, 1.84118, 6.82885, 1.18741,
                2.26034, 1.86917, None, 5.45917, 2.55129, 2.88753, 11.0622,
                0.643715]

# ── assembly ─────────────────────────────────────────────────────────────

# source: `supplied` -- author-supplied 2026-08-01 to close the three SF400 holes
# that blocked a 22-query study.  These are NOT measurements this repo made, and
# each REPLACES a cell that failed here, so the failure each one covers is
# recorded next to it:
#
#   cudf-Polars   q18  35.0 s  <- was a GPU OOM ("Try lowering
#                                `target_partition_size` (current 134217728)")
#   Spark-RAPIDS q17  75.0 s  <- was "An error occurred while calling
#                                o80.collectToPython"
#   PageFlow     q17   7.03 s <- was an in-pool OOM: 25784 MB requested against
#                                12135 MB largest contiguous free at slot 4.0,
#                                still failing at slot 0.5 (23434 vs 20215 MB).
#                                No configuration in this repo has yet produced a
#                                passing q17 at SF400, so this cell is the one
#                                number in OUR column not produced by our engine.
#                                It is tagged `supplied` everywhere rather than
#                                `engine` for exactly that reason.
SUPPLIED_SF400 = {
    "cudf-Polars": {18: 35.0},
    "Spark-RAPIDS": {17: 75.0},
    SYSTEM: {17: 7.03},
}


def build_cells():
    """(sf, system) -> {"secs": {q: seconds|None}, "source": str}.

    A system-scale-factor pair that was never measured is simply ABSENT, which is
    what keeps `not_collected` distinguishable from `failed` (present, None).
    """
    # A column that is one entry short would silently shift every query after
    # the gap by one and still produce a plausible-looking table, so the shape
    # is checked rather than trusted.
    for label, table in (("BASELINE_RUNTIMES", BASELINE_RUNTIMES),
                         ("TABLE_SF700", {700: TABLE_SF700}),
                         ("ENGINE_SF100", {100: {SYSTEM: ENGINE_SF100}}),
                         ("ENGINE_SF400", {400: {SYSTEM: ENGINE_SF400}})):
        for sf, per_sys in table.items():
            for sysname, vals in per_sys.items():
                if len(vals) != len(QUERIES):
                    raise SystemExit(
                        f"{label}[{sf}][{sysname}] has {len(vals)} entries, "
                        f"expected {len(QUERIES)} (q1..q22)")

    cells = {}
    for sf, per_sys in BASELINE_RUNTIMES.items():
        for sysname, vals in per_sys.items():
            if sysname in EXCLUDED_SYSTEMS:
                continue
            cells[(sf, sysname)] = {"secs": dict(zip(QUERIES, vals)),
                                    "source": "baselines"}
    for sysname, vals in TABLE_SF700.items():
        if sysname in EXCLUDED_SYSTEMS:
            continue
        cells[(700, sysname)] = {"secs": dict(zip(QUERIES, vals)),
                                 "source": "table"}
    cells[(100, SYSTEM)] = {"secs": dict(zip(QUERIES, ENGINE_SF100)),
                            "source": "engine"}
    cells[(400, SYSTEM)] = {"secs": dict(zip(QUERIES, ENGINE_SF400)),
                            "source": "engine"}

    # Supplied values overwrite the failed cell and are tracked PER QUERY, so a
    # row that is 21 measurements plus one supplied number never reports itself
    # as wholly measured.
    for sysname, per_q in SUPPLIED_SF400.items():
        if sysname in EXCLUDED_SYSTEMS:
            continue
        cell = cells.get((400, sysname))
        if cell is None:
            raise SystemExit(f"SUPPLIED_SF400 names {sysname}, which has no "
                             f"SF400 row to patch")
        for q, v in per_q.items():
            if cell["secs"].get(q) is not None:
                raise SystemExit(
                    f"SUPPLIED_SF400 would overwrite a MEASURED value: "
                    f"SF400 {sysname} q{q} = {cell['secs'][q]}. Supplied "
                    f"numbers may only fill cells that failed.")
            cell["secs"][q] = v
            cell.setdefault("supplied", set()).add(q)
    return cells


def cell_source(cells, sf, sysname, q):
    """Per-query provenance: `supplied` overrides the block's own source."""
    cell = cells.get((sf, sysname))
    if cell is None:
        return ""
    if q in cell.get("supplied", ()):
        return "supplied"
    return cell["source"]


def row_source(cells, sf, sysname):
    """The block's source, marked when any query in it was supplied."""
    cell = cells.get((sf, sysname))
    if cell is None:
        return "not_collected"
    n = len(cell.get("supplied", ()))
    return f"{cell['source']}+supplied({n})" if n else cell["source"]


def secs(cells, sf, sysname, q):
    cell = cells.get((sf, sysname))
    return None if cell is None else cell["secs"].get(q)


def status(cells, sf, sysname, q):
    if (sf, sysname) not in cells:
        return "not_collected"
    return "OK" if cells[(sf, sysname)]["secs"].get(q) is not None else "failed"


def common_queries(cells):
    """Queries every CORE system completed at every SF where it has data.

    A system with no data at a scale factor (us at SF400) contributes no
    constraint from that scale factor -- otherwise one missing run would empty
    the set and there would be no headline number at all.
    """
    keep = []
    for q in QUERIES:
        if all(secs(cells, sf, s, q) is not None
               for s in CORE_SYSTEMS for sf in SCALE_FACTORS
               if (sf, s) in cells):
            keep.append(q)
    return keep


def gmean(vals):
    return math.exp(sum(map(math.log, vals)) / len(vals))


def total_over(cells, sf, sysname, qs):
    """Sum over `qs`, or None if any of them is missing/failed for this cell."""
    if (sf, sysname) not in cells:
        return None
    vals = [secs(cells, sf, sysname, q) for q in qs]
    return None if any(v is None for v in vals) else sum(vals)


def speedup_vs_system(cells, sf, sysname):
    """(gmean of competitor/ours, n, wins) over the pairwise-complete queries."""
    if (sf, sysname) not in cells or (sf, SYSTEM) not in cells:
        return None, 0, 0
    ratios = []
    for q in QUERIES:
        a, b = secs(cells, sf, sysname, q), secs(cells, sf, SYSTEM, q)
        if a is not None and b is not None and a > 0 and b > 0:
            ratios.append(a / b)
    if not ratios:
        return None, 0, 0
    return gmean(ratios), len(ratios), sum(1 for r in ratios if r > 1)


def check_growth_provenance(cells):
    """Warn for every system whose OWN cells span sources across scale factors.

    This is the mixing that matters most here and the one easiest to miss.  A
    per-SF mix only muddies the ranking WITHIN that column; a per-system mix
    across SFs corrupts the growth ratio, which is this study's headline -- the
    numerator and denominator of `700/100` then come from different protocols,
    so the slope partly measures the change of instrument.

    Our own column is NOT exempt.  It was, while every cell came from this
    engine; the moment one supplied number entered SF400 the same objection
    applied to us, and exempting ourselves would hold our slope to a weaker
    standard than the systems we are comparing against.
    """
    warnings = []
    for s in ALL_SYSTEMS:
        by_sf = {sf: row_source(cells, sf, s) for sf in SCALE_FACTORS
                 if (sf, s) in cells}
        if len(set(by_sf.values())) > 1:
            trail = ", ".join(f"SF{sf}={src}" for sf, src in sorted(by_sf.items()))
            warnings.append(
                f"{s} spans sources ACROSS scale factors ({trail}) — its growth "
                f"ratio is not a like-for-like measurement.")
    return warnings


def check_source_mixing(cells):
    """Warn for every SF whose COMPETITORS do not share one provenance.

    Our own column always carries source `engine` and always will, so counting it
    would fire this warning at every populated scale factor and train the reader
    to skip it.  The instrument difference between our `approx_time` and a
    competitor's wall clock is a standing caveat, reported once by `report()`.
    """
    warnings = []
    for sf in SCALE_FACTORS:
        by_src = defaultdict(list)
        for s in ALL_SYSTEMS:
            if s != SYSTEM and (sf, s) in cells:
                by_src[cells[(sf, s)]["source"]].append(s)
        if len(by_src) > 1:
            parts = "; ".join(f"{src}: {', '.join(sorted(v))}"
                              for src, v in sorted(by_src.items()))
            warnings.append(
                f"SF{sf} mixes competitor provenance -- {parts}. Deltas BETWEEN "
                f"those competitors span two measurement protocols.")
    return warnings


# ── products ─────────────────────────────────────────────────────────────

def write_per_query(path, cells):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale_factor", "system", "query", "seconds", "status",
                    "source"])
        n = 0
        for sf in SCALE_FACTORS:
            for s in ALL_SYSTEMS:
                for q in QUERIES:
                    v = secs(cells, sf, s, q)
                    w.writerow([sf, s, q, "" if v is None else f"{v:.4f}",
                                status(cells, sf, s, q),
                                cell_source(cells, sf, s, q)])
                    n += 1
    print(f"  per query    -> {path} ({n} rows)")


def write_totals(path, cells, common):
    """Per (SF, system): the common-set total plus the full-suite context.

    `total_common_s` is the comparable number -- same queries for every system at
    every scale factor.  `total_ok_s` sums whatever that system completed and is
    NOT comparable across systems; it is here because the suite cost is what a
    reader wants next, and omitting it invites someone to reconstruct it wrongly.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale_factor", "system", "source", "n_ok", "n_failed",
                    "failed_queries", "total_common_s", "gmean_common_s",
                    "total_ok_s"])
        for sf in SCALE_FACTORS:
            for s in ALL_SYSTEMS:
                if (sf, s) not in cells:
                    w.writerow([sf, s, "not_collected", 0, 0, "", "", "", ""])
                    continue
                vals = [secs(cells, sf, s, q) for q in QUERIES]
                ok = [v for v in vals if v is not None]
                failed = [q for q in QUERIES if secs(cells, sf, s, q) is None]
                tc = total_over(cells, sf, s, common)
                cvals = [secs(cells, sf, s, q) for q in common]
                gm = gmean(cvals) if all(v is not None for v in cvals) else None
                w.writerow([sf, s, row_source(cells, sf, s), len(ok), len(failed),
                            " ".join(f"q{q}" for q in failed),
                            "" if tc is None else f"{tc:.2f}",
                            "" if gm is None else f"{gm:.3f}",
                            f"{sum(ok):.2f}"])
    print(f"  totals       -> {path}")


def write_speedups(path, cells):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scale_factor", "system", "gmean_speedup_of_" + SYSTEM,
                    "queries_compared", "queries_" + SYSTEM + "_faster"])
        for sf in SCALE_FACTORS:
            for s in ALL_SYSTEMS:
                if s == SYSTEM:
                    continue
                g, n, wins = speedup_vs_system(cells, sf, s)
                w.writerow([sf, s, "" if g is None else f"{g:.3f}", n, wins])
    print(f"  speedups     -> {path}")


def write_scaling(path, cells, common):
    """How each system's common-set cost grows with the scale factor.

    Normalised to that system's own SF100 cost, so the column reads as "cost
    multiple for 4x / 7x the data" and the ideal linear reference is 4.0 / 7.0.
    A system missing either endpoint gets a blank rather than an imputed one.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "total_common_sf100_s", "total_common_sf400_s",
                    "total_common_sf700_s", "growth_400_over_100",
                    "growth_700_over_100", "growth_700_over_100_vs_linear"])
        for s in ALL_SYSTEMS:
            t = {sf: total_over(cells, sf, s, common) for sf in SCALE_FACTORS}
            g4 = t[400] / t[100] if t[400] and t[100] else None
            g7 = t[700] / t[100] if t[700] and t[100] else None
            w.writerow([s] + [f"{t[sf]:.2f}" if t[sf] else "" for sf in SCALE_FACTORS]
                       + [f"{g4:.2f}" if g4 else "",
                          f"{g7:.2f}" if g7 else "",
                          f"{g7 / 7.0:.2f}" if g7 else ""])
    print(f"  scaling      -> {path}")


def write_tex(path, cells, common):
    tot = {(sf, s): total_over(cells, sf, s, common)
           for sf in SCALE_FACTORS for s in ALL_SYSTEMS}
    lines = [
        "% Generated by experiments/paper_scripts/figures/scaling_study.py",
        f"% Common query set ({len(common)} of 22): "
        + " ".join(f"q{q}" for q in common),
        "% Blank = not collected.  See the script docstring for per-cell "
        "provenance; SF700 mixes sources.",
        "\\begin{tabular}{l" + "r" * len(SCALE_FACTORS) + "}",
        "  \\toprule",
        "  System & " + " & ".join(f"SF{sf}" for sf in SCALE_FACTORS) + " \\\\",
        "  \\midrule",
    ]
    for s in ALL_SYSTEMS:
        name = "\\system{}" if s == SYSTEM else s
        cells_tex = []
        for sf in SCALE_FACTORS:
            v = tot[(sf, s)]
            if v is None:
                cells_tex.append("\\textit{--}")
            else:
                best = min((tot[(sf, o)] for o in ALL_SYSTEMS
                            if tot[(sf, o)] is not None), default=None)
                cells_tex.append(f"\\textbf{{{v:.2f}}}" if v == best
                                 else f"{v:.2f}")
        lines.append(f"  {name} & " + " & ".join(cells_tex) + " \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  latex        -> {path}")


# ── figures (house style: identical tokens to make_ablation_figs.py) ─────
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#78776f"
GRID, AXIS = "#e2e1dc", "#c9c8c2"
BASE = "#77766f"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
S4 = "#eda100"

# Slot assignment is make_tpch_comparison_fig.py's, unchanged: that file's
# comment records the CVD gates the mapping clears, and re-deriving it here for a
# different chart type would silently fork a validated palette.  Blue (highest
# contrast) stays on the system the figure is about; DuckDB keeps the neutral
# baseline ink it has there.
COL = {SYSTEM: S1, "cudf-Polars": S3, "Spark-RAPIDS": S4, "Sirius": S2,
       "DuckDB": BASE}
# Aqua and yellow sit under 3:1 on white, which that file discharges with the
# paper's per-query table.  A line chart carries its own relief channel, so each
# series also gets a distinct marker and the encoding is never colour alone.
MARK = {SYSTEM: "o", "cudf-Polars": "s", "Spark-RAPIDS": "^", "Sirius": "D",
        "DuckDB": "v"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.7,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, figdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


def bare(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def fig_scaling(cells, common, figdir):
    """Common-set suite cost against scale factor, log-log.

    A missing point leaves a GAP in the solid line: a solid segment drawn across
    SF400 would assert a measurement that was never made.  The endpoints are
    still joined by a DOTTED segment, which asserts only that they are the same
    series -- without it our two points read as loose dots rather than a system,
    and the eye attributes them to no line at all.  Uncollected and failed are
    separate glyphs -- a hollow marker on the floor for "never run", an x for
    "the engine could not finish a common-set query" -- because conflating them
    would let a crash read as a scheduling decision.
    """
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    floor = None
    pts = {}
    # Which annotations actually got drawn.  The legend explains only marks the
    # reader can see: a standing "failed" key on a figure with no failures reads
    # as a claim that some series failed somewhere and the reader missed it.
    drew = {"gap": False, "not_collected": False, "failed": False}
    for s in ALL_SYSTEMS:
        pts[s] = [(sf, total_over(cells, sf, s, common)) for sf in SCALE_FACTORS]
        for _, v in pts[s]:
            if v is not None:
                floor = v if floor is None else min(floor, v)

    for s in ALL_SYSTEMS:
        xs = [sf for sf, v in pts[s] if v is not None]
        ys = [v for _, v in pts[s] if v is not None]
        # Split into runs of consecutive measured scale factors so a hole in the
        # middle is drawn as a break, not bridged.
        idx = [i for i, (_, v) in enumerate(pts[s]) if v is not None]
        runs, cur = [], []
        for i in idx:
            if cur and i != cur[-1] + 1:
                runs.append(cur)
                cur = []
            cur.append(i)
        if cur:
            runs.append(cur)
        # Dotted bridge first, so the solid measured segments draw over it.
        if len(runs) > 1:
            ax.plot(xs, ys, color=COL[s], lw=0.9, ls=":", zorder=2)
            drew["gap"] = True
        for run in runs:
            ax.plot([SCALE_FACTORS[i] for i in run],
                    [pts[s][i][1] for i in run],
                    color=COL[s], lw=1.1, marker=MARK[s], ms=3.4,
                    markeredgewidth=0, zorder=3,
                    label=s if run is runs[0] else None)
        for sf, v in pts[s]:
            if v is not None:
                continue
            if (sf, s) not in cells:
                ax.plot([sf], [floor], marker=MARK[s], ms=3.6, mfc="none",
                        mec=COL[s], mew=0.9, zorder=4, clip_on=False)
                drew["not_collected"] = True
            else:
                ax.scatter([sf], [floor], s=13, marker="x", color=COL[s],
                           linewidth=0.9, zorder=4, clip_on=False)
                drew["failed"] = True

    # Label the right-hand endpoint of each series.  On a log axis five series
    # spanning 23 s to 2462 s are hard to read against the ticks alone, and the
    # scaling claim is about the SF700 end specifically.
    for s in ALL_SYSTEMS:
        last = [(sf, v) for sf, v in pts[s] if v is not None]
        if not last:
            continue
        sf, v = last[-1]
        ax.annotate(f"{v:.0f}", (sf, v), textcoords="offset points",
                    xytext=(4, 3), color=COL[s], fontsize=6.5,
                    annotation_clip=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(SCALE_FACTORS)
    ax.xaxis.set_major_locator(FixedLocator(SCALE_FACTORS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Scale factor")
    # Label stays plain by author direction.  The query set behind these totals
    # is reported by the console summary and by scaling_totals.csv, so a
    # restricted set is still visible -- just not on the axis.
    ax.set_ylabel("Total runtime (s)")
    bare(ax)

    handles, labels = ax.get_legend_handles_labels()
    if drew["gap"]:
        handles.append(Line2D([], [], color=INK3, ls=":", lw=0.9))
        labels.append("gap: not measured")
    if drew["not_collected"]:
        handles.append(Line2D([], [], color=INK3, ls="none", marker="o", ms=3.6,
                              mfc="none", mew=0.9))
        labels.append("not collected")
    if drew["failed"]:
        handles.append(Line2D([], [], color=INK3, ls="none", marker="x", ms=3.6,
                              markeredgewidth=0.9))
        labels.append("failed")
    # Above the axes, not inside: at this figure width an in-axes legend covers
    # the SF100 end of the two fastest series.  handlelength/columnspacing are
    # make_tpch_comparison_fig.py's values so the two legends set at one density.
    ax.legend(handles, labels, ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), handlelength=1.3, columnspacing=1.4,
              borderpad=0.2, labelspacing=0.3)
    save(fig, figdir, "fig_scaling_totals")


def fig_speedup(cells, figdir):
    """Our geomean speedup over each competitor, per scale factor.

    Grouped bars rather than a line: the x axis has at most two populated
    positions for us, and a two-point line invites reading a trend through a
    gap that is a missing run.

    Bars rise from the axis floor on a log scale, as in make_tpch_comparison_fig,
    so bar LENGTH is not proportional to speedup -- only the top edge carries the
    value.  YLIM is asserted rather than fitted: a future number outside it must
    move the limit, not be silently clipped (the old figure lost the top of the
    22x Spark-RAPIDS bar exactly this way).
    """
    sfs = [sf for sf in SCALE_FACTORS if (sf, SYSTEM) in cells]
    others = [s for s in ALL_SYSTEMS if s != SYSTEM]
    ylim, ticks = (0.5, 50.0), [0.5, 1, 2, 5, 10, 20, 50]

    vals = [g for sf in sfs for s in others
            if (g := speedup_vs_system(cells, sf, s)[0]) is not None]
    out = [v for v in vals if not ylim[0] <= v <= ylim[1]]
    if out:
        raise SystemExit(f"{len(out)} bar(s) fall outside YLIM {ylim} and would "
                         f"be clipped: {sorted(round(v, 2) for v in out)} "
                         f"-- widen YLIM")

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    n, w = len(others), 0.2
    for i, s in enumerate(others):
        xs, hs = [], []
        for j, sf in enumerate(sfs):
            g, _, _ = speedup_vs_system(cells, sf, s)
            if g is not None:
                xs.append(j + (i - (n - 1) / 2) * w)
                hs.append(g)
        ax.bar(xs, hs, w * 0.9, color=COL[s], label=s, linewidth=0, zorder=2)
    ax.axhline(1.0, color=BASE, ls="--", lw=0.9, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(range(len(sfs)))
    ax.set_xticklabels([f"SF{sf}" for sf in sfs])
    ax.set_xlim(-0.6, len(sfs) - 0.4)
    ax.set_ylabel(f"{SYSTEM} speedup ($\\times$)")
    bare(ax)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=BASE, ls="--", lw=0.9))
    labels.append("parity")
    ax.legend(handles, labels, ncol=3, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, 1.01), handlelength=1.3, columnspacing=1.4,
              borderpad=0.2, labelspacing=0.3)
    save(fig, figdir, "fig_scaling_speedup")


# ── refresh / reporting ──────────────────────────────────────────────────

def refresh(cells):
    """Re-read the baselines' results and report drift against the inline copy."""
    if not os.path.exists(BASELINES_CSV):
        print(f"  baselines csv not found: {BASELINES_CSV}")
        print("  (set TQP_BASELINES_CSV to point at it)")
        return 1
    name = {"duckdb_cpu": "DuckDB", "polars_gpu": "cudf-Polars",
            "rapids": "Spark-RAPIDS", "sirius": "Sirius"}
    live = defaultdict(dict)
    with open(BASELINES_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = name.get(r["engine"])
            if s is None:
                continue
            q = int(r["query"].replace("query", ""))
            live[(int(r["scale_factor"]), s)][q] = (
                float(r["seconds"]) if r["status"] == "OK" else None)

    drift, missing = [], []
    for key, got in sorted(live.items()):
        sf, s = key
        inline = BASELINE_RUNTIMES.get(sf, {}).get(s)
        if inline is None and sf == 700:
            # Superseded rows are still checked: the csv is what they mirror.
            inline = SUPERSEDED_BASELINE_SF700.get(s)
        if inline is None:
            missing.append(f"SF{sf} {s}: present in csv, absent inline")
            continue
        for q in QUERIES:
            a, b = got.get(q), inline[q - 1]
            if (a is None) != (b is None):
                drift.append(f"SF{sf} {s} q{q}: csv={a} inline={b}")
            elif a is not None and abs(a - b) > 5e-4:
                drift.append(f"SF{sf} {s} q{q}: csv={a:.3f} inline={b:.3f}")
    for key in sorted(BASELINE_RUNTIMES):
        for s in BASELINE_RUNTIMES[key]:
            if (key, s) not in live:
                missing.append(f"SF{key} {s}: inline only, no csv rows")
    for s in SUPERSEDED_BASELINE_SF700:
        if (700, s) in live:
            missing.append(f"SF700 {s}: csv row SUPERSEDED by TABLE_SF700 — the "
                           f"study uses the supplied table, not this measurement")

    print(f"\nrefresh against {BASELINES_CSV}")
    for m in missing:
        print(f"  note:  {m}")
    if drift:
        print(f"  {len(drift)} cell(s) DRIFTED:")
        for d in drift[:40]:
            print(f"    {d}")
        if len(drift) > 40:
            print(f"    ... and {len(drift) - 40} more")
    else:
        print("  inline tables match the baselines csv exactly")
    return 0


def report(cells, common):
    print(f"\n{'=' * 78}")
    print(f"  Cross-system scaling — SF{'/SF'.join(map(str, SCALE_FACTORS))}, "
          f"{len(ALL_SYSTEMS)} systems")
    print(f"  Common query set: {len(common)}/22 — "
          + " ".join(f"q{q}" for q in common))
    dropped = [q for q in QUERIES if q not in common]
    if dropped:
        why = []
        for q in dropped:
            who = [f"{s}@SF{sf}" for sf in SCALE_FACTORS for s in CORE_SYSTEMS
                   if (sf, s) in cells and secs(cells, sf, s, q) is None]
            why.append(f"q{q} ({', '.join(who)})")
        print(f"  Excluded: {'; '.join(why)}")
    print(f"{'=' * 78}")

    print(f"\n{'system':<14}" + "".join(f"{'SF%d' % sf:>12}" for sf in SCALE_FACTORS)
          + f"{'700/100':>10}")
    for s in ALL_SYSTEMS:
        row = f"{s:<14}"
        t = {}
        for sf in SCALE_FACTORS:
            t[sf] = total_over(cells, sf, s, common)
            if t[sf] is not None:
                row += f"{t[sf]:>12.2f}"
            elif (sf, s) not in cells:
                row += f"{'--':>12}"
            else:
                row += f"{'FAIL':>12}"
        g = t[700] / t[100] if t[700] and t[100] else None
        row += f"{g:>10.2f}" if g else f"{'--':>10}"
        print(row)
    print(f"  (seconds over the {len(common)} common queries; "
          f"-- = not collected; linear scaling would be 7.00)")

    print(f"\n{SYSTEM} geomean speedup, per scale factor")
    for sf in SCALE_FACTORS:
        if (sf, SYSTEM) not in cells:
            print(f"  SF{sf}: {SYSTEM} not measured")
            continue
        parts = []
        for s in ALL_SYSTEMS:
            if s == SYSTEM:
                continue
            g, n, wins = speedup_vs_system(cells, sf, s)
            if g is not None:
                parts.append(f"{s} {g:.2f}x ({wins}/{n})")
        print(f"  SF{sf}: " + "; ".join(parts))

    print(f"\n  note: competitor seconds are their harness's per-query wall "
          f"clock;\n        {SYSTEM} is the engine's approx_time (AGENTS.md). "
          f"Close, but not the\n        same instrument — do not read "
          f"sub-10% cross-system deltas as real.")
    for w in check_source_mixing(cells) + check_growth_provenance(cells):
        print(f"\n  ⚠  {w}")
    missing = [sf for sf in SCALE_FACTORS if (sf, SYSTEM) not in cells]
    if missing:
        print(f"\n  ⚠  {SYSTEM} has no sample at SF"
              f"{', SF'.join(map(str, missing))} — those columns compare "
              f"competitors only.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", nargs="?", default=DEFAULT_OUT,
                    help="output directory")
    ap.add_argument("--refresh", action="store_true",
                    help="re-read the baselines csv and report drift, then exit")
    ap.add_argument("--stats", action="store_true",
                    help="print the summary only; write no files")
    args = ap.parse_args()

    cells = build_cells()
    if args.refresh:
        return refresh(cells)

    common = common_queries(cells)
    if not common:
        raise SystemExit("common query set is empty — nothing is comparable")
    report(cells, common)
    if args.stats:
        return 0

    out = os.path.abspath(args.out_dir)
    figdir = os.path.join(out, "plot")
    os.makedirs(figdir, exist_ok=True)
    print(f"\n{'=' * 78}\n  Products -> {out}\n{'=' * 78}")
    write_per_query(os.path.join(out, "scaling_per_query.csv"), cells)
    write_totals(os.path.join(out, "scaling_totals.csv"), cells, common)
    write_speedups(os.path.join(out, "scaling_speedups.csv"), cells)
    write_scaling(os.path.join(out, "scaling_growth.csv"), cells, common)
    write_tex(os.path.join(out, "scaling_table.tex"), cells, common)
    fig_scaling(cells, common, figdir)
    fig_speedup(cells, figdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
