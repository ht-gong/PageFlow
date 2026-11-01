#!/usr/bin/env python3
"""fig_motivation_geomean -- geometric-mean TPC-H query time vs scale factor,
one ACM column wide.

    python3 make_motivation_geomean_fig.py [--baselines CSV] [--sf N ...] [--stats]
    -> experiments/paper_scripts/results/fig_motivation_geomean.pdf

The four baseline engines' per-query geometric mean against scale factor, in
the paper-figures style (sigmod_style.py: upright Linux Biolinum, 9 pt): log
time against log scale factor, at SF30 SF50 SF100 SF200 SF400 SF700 (``--sf``).
Every point is read from baselines/results/all_results.csv, whose rows
``e1_motivation.py`` (E1) writes on the box that ran it (``TQP_BASELINES_CSV``
or ``--baselines`` overrides the path); none is carried here.

THREE REGIONS.  SF30 and SF50 are cudf-Polars', Spark-RAPIDS' and DuckDB's
``*_resident`` rows (data in memory before the timed run); from SF100 their
parquet rows, and from SF200 the engines spill (shaded).  Sirius is the
``sirius_host`` rows at every scale factor: the second Sirius build
(``SIRIUS_HOST_REF``), the query's columns pinned in its host tier before the
timed run (baselines/AGENTS.md).

GEOMEAN OVER THE QUERIES EVERY ENGINE COMPLETED at that scale factor, so the
four points at one scale factor average the same queries.  A query one engine
failed leaves the set for all of them, and that engine's point is a red x
instead of its marker.  Sirius's red x's (COUNT_FAILS) also get a red leader to
a note under the lines with how many queries failed there ("1 query failed");
points with the same count share one note.  `--stats` lists the queries left
out and each engine's geomean over its own completed queries.
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from sigmod_style import COLUMN, FAIL_HALO, FAIL_MARK, INK2, INK3, LINES, save

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASELINES_CSV = os.environ.get("TQP_BASELINES_CSV") or os.path.join(
    ROOT, "baselines", "results", "all_results.csv")

# all_results.csv engine key -> the paper's name, in legend order.
ENGINES = {"sirius": "Sirius", "polars_gpu": "cudf-Polars",
           "rapids": "Spark-RAPIDS", "duckdb_cpu": "DuckDB"}
# An engine drawn from one key at every scale factor, instead of its
# ``*_resident`` key below RESIDENT_BELOW and its parquet key from it.
ONE_KEY = {"sirius": "sirius_host"}
QUERIES = list(range(1, 23))
SFS = [30, 50, 100, 200, 400, 700]
RESIDENT_BELOW = 100        # below it the *_resident rows
SPILL_SF = 200              # spilling from here: shaded
LADDER = [m * 10.0 ** k for k in range(-3, 4) for m in (1, 3)]
# Engines whose failed points also get a note of how many queries failed there.
COUNT_FAILS = {"Sirius"}
FAIL_NOTE_AT = (420, 0.6)   # (scale factor, s): where those notes sit, under the lines


def key(engine, sf):
    """The all_results.csv engine key ``engine`` is drawn from at ``sf``."""
    if engine in ONE_KEY:
        return ONE_KEY[engine]
    return f"{engine}_resident" if sf < RESIDENT_BELOW else engine


def load(path, sfs):
    """{(name, sf): {query: seconds or None}} over ``sfs``, each engine's row
    from ``key``.  A non-OK row is None."""
    want = {(key(e, sf), sf): name for e, name in ENGINES.items() for sf in sfs}
    cells = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        need = {"engine", "scale_factor", "query", "status", "seconds"}
        if not need <= set(rows.fieldnames or ()):
            raise SystemExit(f"{path}: header {rows.fieldnames} lacks {sorted(need)}")
        for r in rows:
            sf = int(r["scale_factor"])
            name = want.get((r["engine"], sf))
            if name is None:
                continue
            q = int(r["query"].removeprefix("query"))
            cell = cells.setdefault((name, sf), {})
            if q in cell:
                raise SystemExit(f"{path}: repeated ({r['engine']}, {sf}, {r['query']})")
            cell[q] = float(r["seconds"]) if r["status"] == "OK" else None
    for name in ENGINES.values():
        for sf in sfs:
            got = cells.get((name, sf), {})
            if sorted(got) != QUERIES:
                raise SystemExit(f"{path}: {name} SF{sf} has rows for "
                                 f"{len(got)}/22 queries")
    return cells


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def series(cells, sfs):
    """({name: [(sf, geomean_s, n_failed)]}, {sf: queries left out})."""
    out = {n: [] for n in ENGINES.values()}
    dropped = {}
    for sf in sfs:
        common = [q for q in QUERIES
                  if all(cells[(n, sf)][q] is not None for n in out)]
        dropped[sf] = [q for q in QUERIES if q not in common]
        for n in out:
            c = cells[(n, sf)]
            failed = sum(v is None for v in c.values())
            out[n].append((sf, geomean([c[q] for q in common]), failed))
    return out, dropped


def fig(s, sfs, path):
    ts = [t for v in s.values() for _, t, _ in v]
    xlim = (sfs[0] / 1.15, sfs[-1] * 1.15)
    # Headroom above the highest point: the region labels sit there, clear of lines.
    ylim = (max(v for v in LADDER if v <= min(ts)), max(ts) * 8)

    fig, ax = plt.subplots(figsize=(COLUMN, 2.4), layout="constrained")

    # Regions: a rule where the in-memory points end, a wash where spilling starts.
    split = (50 * 100) ** 0.5
    wash = SPILL_SF / 1.12
    ax.axvline(split, color=INK3, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.axvspan(wash, xlim[1], color="#000000", alpha=0.06, lw=0, zorder=0)
    top = ylim[1] / 1.15
    for x, label in (((xlim[0] * split) ** 0.5, "In memory"),
                     ((split * wash) ** 0.5, "Input\n> GPU"),
                     ((wash * xlim[1]) ** 0.5, "Spilling")):
        ax.text(x, top, label, ha="center", va="top", color=INK2)

    for name, run in s.items():
        st = LINES[name]
        z = st.get("zorder", 2)
        ax.plot([sf for sf, _, _ in run], [t for _, t, _ in run], color=st["color"],
                ls=st["linestyle"], lw=st.get("linewidth", 1.2), zorder=z)
        ok = [(sf, t) for sf, t, f in run if not f]
        bad = [(sf, t) for sf, t, f in run if f]
        if ok:
            ax.plot(*zip(*ok), ls="none", marker=st["marker"],
                    ms=st.get("markersize", 4), mew=0.5, color=st["color"],
                    mec="white", zorder=z + 1)
        if bad:
            ax.plot(*zip(*bad), **FAIL_HALO, zorder=7, clip_on=False)
            ax.plot(*zip(*bad), **FAIL_MARK, zorder=8, clip_on=False)
        if name in COUNT_FAILS:
            # One note per failure count, in the empty space under the lines,
            # with a thin red leader to each point that count applies to.
            by_count = {}
            for sf, t, f in run:
                if f:
                    by_count.setdefault(f, []).append((sf, t))
            for i, (k, pts) in enumerate(sorted(by_count.items())):
                at = (FAIL_NOTE_AT[0], FAIL_NOTE_AT[1] / 2.2 ** i)
                for p in pts:
                    ax.annotate(f"{k} {'query' if k == 1 else 'queries'} failed", p,
                                xytext=at, textcoords="data", ha="center", va="top",
                                color=FAIL_MARK["color"], zorder=9,
                                arrowprops=dict(arrowstyle="-", color=FAIL_MARK["color"],
                                                lw=0.6, shrinkA=1, shrinkB=4))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    for axis, ticks in ((ax.xaxis, sfs),
                        (ax.yaxis, [v for v in LADDER if ylim[0] <= v <= ylim[1]])):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Geomean query time (s)")

    handles = [Line2D([], [], color=LINES[n]["color"], ls=LINES[n]["linestyle"],
                      lw=LINES[n].get("linewidth", 1.2), marker=LINES[n]["marker"],
                      ms=LINES[n].get("markersize", 4), mew=0.5, mec="white")
               for n in s]
    labels = list(s)
    if any(f for v in s.values() for _, _, f in v):
        handles.append(Line2D([], [], **FAIL_MARK))
        labels.append("Query failed")
    ax.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=2.0, columnspacing=0.8)
    save(fig, path)


def stats(cells, s, dropped, sfs):
    print("\ngeomean query time (s) over the queries every engine completed"
          "   [own = over the engine's own completed queries]")
    print(f"  {'system':<14}" + "".join(f"{'SF' + str(sf):>16}" for sf in sfs))
    for name, run in s.items():
        row = ""
        for sf, t, _ in run:
            own = geomean([v for v in cells[(name, sf)].values() if v is not None])
            row += f"{t:>8.3f} ({own:6.3f})"
        print(f"  {name:<14}{row}")
    for sf in sfs:
        if dropped[sf]:
            who = {q: [n for n in s if cells[(n, sf)][q] is None] for q in dropped[sf]}
            print(f"  SF{sf}: left out " + ", ".join(
                f"Q{q} ({', '.join(w)} failed)" for q, w in who.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--baselines", default=BASELINES_CSV)
    ap.add_argument("--sf", type=int, nargs="+", default=SFS,
                    help=f"scale factors (default the figure's: {SFS})")
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_motivation_geomean.pdf"))
    ap.add_argument("--stats", action="store_true",
                    help="also print the geomeans and the queries left out")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    print(f"from {args.baselines}")
    sfs = sorted(set(args.sf))
    cells = load(args.baselines, sfs)
    s, dropped = series(cells, sfs)
    fig(s, sfs, args.out)
    if args.stats:
        stats(cells, s, dropped, sfs)


if __name__ == "__main__":
    main()
