#!/usr/bin/env python3
"""fig_scaling_geomean -- geometric-mean TPC-H query time vs scale factor,
every system, one ACM column wide.

    python3 make_scaling_fig.py [--e2 CSV] [--baselines CSV] [--sf N ...]
        [--stats] [--out PDF] -> experiments/paper_scripts/results/fig_scaling_geomean.pdf

Geomean query time against scale factor for every system (PageFlow,
Sirius, cudf-Polars, Spark-RAPIDS, DuckDB), in the paper-figures style
(sigmod_style.py: upright Linux Biolinum, 9 pt).  Log time against log
scale factor, at SF100 SF400 SF700 SF1000 (``--sf``).  None is carried here.
PageFlow is E2's full setting at each scale factor (``tqp_adaptive`` in
results/ablation_results.csv, ``e2_end_to_end_ablation.py --settings
tqp_adaptive --sf <SF>``; ``--e2`` overrides the path), its seconds the
engine's ``approx_time``.  The baselines are E1's rows in
baselines/results/all_results.csv (``e1_motivation.py``; ``TQP_BASELINES_CSV``
or ``--baselines`` overrides the path), as in fig_baselines: cudf-Polars',
Spark-RAPIDS' and DuckDB's parquet rows, and Sirius's ``sirius_host`` rows;
their seconds are their harness's wall clock.

GEOMEAN OVER THE COMMON SET: the queries every system completed at every
scale factor where it has rows, so every point averages the same queries.  A
query one system failed anywhere leaves the set for all of them, and that
system's point at that scale factor is a red x instead of its marker.
Sirius's red x's (COUNT_FAILS) also carry a small grey digit at their upper
left: how many queries it failed there (the legend's "N queries failed").  A
scale factor at which a system has no rows is not run: its line is dotted
across it, with no point.  `--stats` prints the totals and geomeans over that
set, the queries left out and PageFlow's speedups.
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from make_baselines_fig import E2_SETTING, ENGINES, OURS, load_e2
from sigmod_style import COLUMN, FAIL_HALO, FAIL_MARK, INK2, INK3, LINES, save

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASELINES_CSV = os.environ.get("TQP_BASELINES_CSV") or os.path.join(
    ROOT, "baselines", "results", "all_results.csv")

# The systems in legend order; the baselines' all_results.csv keys are
# fig_baselines' ENGINES.
NAMES = [OURS, *ENGINES.values()]
QUERIES = list(range(1, 23))
SFS = [100, 400, 700, 1000]
LADDER = [m * 10.0 ** k for k in range(-3, 6) for m in (1, 3)]
# Systems whose failed points also get a note of how many queries failed there.
COUNT_FAILS = {"Sirius"}


def load(path, e2, sfs):
    """{(name, sf): {query: seconds or None}} at ``sfs``: the baselines' rows
    of ENGINES in all_results.csv, PageFlow's E2 rows (E2_SETTING).  A non-OK
    row is None; a (system, sf) with rows has all 22."""
    cells = {(OURS, sf): c for sf, c in load_e2(e2, sfs).items()}
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        need = {"engine", "scale_factor", "query", "status", "seconds"}
        if not need <= set(rows.fieldnames or ()):
            raise SystemExit(f"{path}: header {rows.fieldnames} lacks {sorted(need)}")
        for r in rows:
            sf = int(r["scale_factor"])
            if r["engine"] not in ENGINES or sf not in sfs:
                continue
            q = int(r["query"].removeprefix("query"))
            cell = cells.setdefault((ENGINES[r["engine"]], sf), {})
            if q in cell:
                raise SystemExit(f"{path}: repeated ({r['engine']}, {sf}, {r['query']})")
            cell[q] = float(r["seconds"]) if r["status"] == "OK" else None
    for (name, sf), got in cells.items():
        if sorted(got) != QUERIES:
            raise SystemExit(f"{name} SF{sf} has rows for {len(got)}/22 queries")
    missing = [n for n in NAMES if not any(name == n for name, _ in cells)]
    if missing:
        raise SystemExit(f"no rows at SF{', SF'.join(map(str, sorted(sfs)))} for "
                         f"{missing} ({OURS}: {e2}, setting {E2_SETTING}; the "
                         f"baselines: {path}, keys {sorted(ENGINES)})")
    return cells


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def series(cells, agg=sum):
    """(scale factors, common queries, {name: [(sf, agg of seconds, queries failed)]})."""
    sfs = sorted({sf for _, sf in cells})
    common = [q for q in QUERIES if all(c[q] is not None for c in cells.values())]
    out = {}
    for name in NAMES:
        out[name] = [(sf, agg([cells[(name, sf)][q] for q in common]),
                      sum(v is None for v in cells[(name, sf)].values()))
                     for sf in sfs if (name, sf) in cells]
    return sfs, common, out


def fig(sfs, s, path):
    ts = [t for v in s.values() for _, t, _ in v]
    xlim = (sfs[0] / 1.15, sfs[-1] * 1.15)
    ylim = (max(v for v in LADDER if v <= min(ts) / 1.1),
            min(v for v in LADDER if v >= max(ts) * 1.1))

    fig, ax = plt.subplots(figsize=(COLUMN, 2.6), layout="constrained")
    for name, run in s.items():
        st = LINES[name]
        z = st.get("zorder", 2)
        lw = st.get("linewidth", 1.2)
        # Solid between neighbouring scale factors, dotted across one not run.
        for (a, ta, _), (b, tb, _) in zip(run, run[1:]):
            gap = sfs.index(b) - sfs.index(a) > 1
            ax.plot([a, b], [ta, tb], color=st["color"], lw=lw * (0.75 if gap else 1),
                    ls=(0, (1, 1.5)) if gap else st["linestyle"], zorder=z)
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
            # How many queries failed there: a grey digit at the x's upper left,
            # above the line coming in (lines only rise to the right).
            for sf, t, f in run:
                if f:
                    ax.annotate(str(f), (sf, t), xytext=(-3, 2),
                                textcoords="offset points", ha="right", va="bottom",
                                color=INK2, zorder=9)

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
        labels.append("N queries failed" if any(
            f for n in COUNT_FAILS if n in s for _, _, f in s[n]) else "Query failed")
    if any(sfs.index(b) - sfs.index(a) > 1
           for v in s.values() for (a, _, _), (b, _, _) in zip(v, v[1:])):
        handles.append(Line2D([], [], color=INK3, lw=0.9, ls=(0, (1, 1.5))))
        labels.append("Not run")
    ax.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=2.0, columnspacing=0.8)
    save(fig, path)


def stats(cells, sfs, common, s, gm):
    left = [q for q in QUERIES if q not in common]
    print(f"\ntotal query time (s) over the {len(common)} queries every system "
          f"completed at every scale factor")
    if left:
        print("  left out: " + ", ".join(
            f"Q{q} ({', '.join(f'{n} SF{sf}' for (n, sf), c in sorted(cells.items()) if c[q] is None)})"
            for q in left))
    print(f"  {'system':<14}" + "".join(f"{'SF' + str(sf):>10}" for sf in sfs)
          + f"{'growth':>9}")
    for name, run in s.items():
        t = {sf: tot for sf, tot, _ in run}
        row = "".join(f"{t[sf]:>10.1f}" if sf in t else f"{'--':>10}" for sf in sfs)
        g = t[run[-1][0]] / t[run[0][0]]
        print(f"  {name:<14}{row}{g:>8.1f}x")
    print(f"  (growth = last / first measured; the data grows {sfs[-1] / sfs[0]:g}x)")
    print(f"\ngeomean query time (s) over the same {len(common)} queries")
    print(f"  {'system':<14}" + "".join(f"{'SF' + str(sf):>10}" for sf in sfs)
          + f"{'growth':>9}")
    for name, run in gm.items():
        t = {sf: v for sf, v, _ in run}
        row = "".join(f"{t[sf]:>10.2f}" if sf in t else f"{'--':>10}" for sf in sfs)
        print(f"  {name:<14}{row}{t[run[-1][0]] / t[run[0][0]]:>8.1f}x")
    print(f"\n{OURS} speedup, total over the common set [geomean over the "
          f"queries both completed]")
    for sf in sfs:
        if (OURS, sf) not in cells:
            continue
        mine = {sf2: t for sf2, t, _ in s[OURS]}
        parts = []
        for name, run in s.items():
            if name == OURS or (name, sf) not in cells:
                continue
            tot = dict((sf2, t) for sf2, t, _ in run)[sf] / mine[sf]
            both = [q for q in QUERIES if cells[(name, sf)][q] is not None
                    and cells[(OURS, sf)][q] is not None]
            g = geomean([cells[(name, sf)][q] / cells[(OURS, sf)][q] for q in both])
            parts.append(f"{name} {tot:.1f}x [{g:.1f}x]")
        print(f"  SF{sf}: " + "; ".join(parts))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--e2", default=os.path.join(RESULTS, "ablation_results.csv"))
    ap.add_argument("--baselines", default=BASELINES_CSV)
    ap.add_argument("--sf", type=int, nargs="+", default=SFS)
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_scaling_geomean.pdf"))
    ap.add_argument("--stats", action="store_true",
                    help="also print the totals, the queries left out and "
                         f"{OURS}'s speedups")
    args = ap.parse_args()
    print(f"{OURS} = {E2_SETTING} in {args.e2}; baselines from {args.baselines}")
    cells = load(args.baselines, args.e2, set(args.sf))
    sfs, common, s = series(cells)
    if not common:
        raise SystemExit("no query every system completed")
    _, _, g = series(cells, geomean)
    fig(sfs, g, args.out)
    if args.stats:
        stats(cells, sfs, common, s, g)


if __name__ == "__main__":
    main()
