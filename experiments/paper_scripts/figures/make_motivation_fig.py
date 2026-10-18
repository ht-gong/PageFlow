#!/usr/bin/env python3
"""Motivation figure -- runtime vs scale factor, in the house style of make_ablation_figs.py.

    python3 make_motivation_fig.py [out_dir]    # -> <out_dir>/plot/*.{pdf,png}

The layout follows the slide-deck figure `fig-01-motivation.png` -- four engines,
log runtime against log scale factor, a shaded region from SF100 -- but as of the
2026-08-01 rebuild NONE of its numbers survive.  Every cell is now a measured
total from the baselines' results (then the standalone TQP-Vortex-Baselines
repo's results/, now baselines/results/).  The slide's own values were
checked against them and discarded; why, and what they got wrong, is in
"WHAT THE SLIDE GOT WRONG" at the bottom, which is worth reading before anyone
proposes restoring them.

THE PROTOCOL SEAM -- READ BEFORE MEASURING A SLOPE OFF THIS PLOT
----------------------------------------------------------------
The grid SF10/50/100/300/500 exists in no single run.  SF10 was only ever run
under the current protocol; SF300 and SF500 only under the archived one.  One
rule fills it, per the author's instruction of 2026-08-01:

    live where live has it (SF10 and SF50 only); archive otherwise.

        live     all_results.csv, as of 23:43 on 2026-08-01
        archive  archive/all_results_sf30-300_mixed_provenance.csv (SF100, SF300)
                 archive/all_results_sf500_cold_protocol.csv       (SF500)

None of these files holds those rows any more: baselines/results/ dropped the
archive and the SF500 rows on 2026-10-07 (the first file is in the git
history, `git log -- baselines/results`; the second never entered this repo).

The two protocols are NOT interchangeable, and the gap is not uniform across
engines.  Where both measured the same cell:

    at SF100        live    archive    live/archive
    duckdb_cpu     36.60      33.11          1.11
    polars_gpu    112.37     104.42          1.08
    rapids        506.54     503.98          1.01
    sirius         32.84      90.50          0.36     <-- 2.8x

So DuckDB, cudf-Polars and Spark-RAPIDS splice with roughly the noise of a
re-run, and SiriusDB does not: 2.4x-2.8x of its apparent SF50 -> SF100 rise is
the protocol changing underneath it, not the data growing.  That is larger than
several of the real effects on the plot, which is why the crossing segment is
DOTTED and carries its own legend key.  A dotted segment on this figure means
its two ends come from different runs and its slope is not a measured rate.  Only
SiriusDB's is badly affected, but styling only that one would read as a property
of SiriusDB rather than of the seam, so every series that crosses it is dotted.

`--stats` prints the per-cell provenance and the seam ratios; use it rather than
re-deriving any of this by eye.

PARTIAL TOTALS
--------------
Four cells sum fewer than 22 queries because the engine failed the rest, and are
therefore NOT comparable to a full-22 total in the same column: SiriusDB SF300
(21/22) and SF500 (17/22), cudf-Polars SF100 (21/22) and SF300 (21/22).  Per the
author's instruction the figure carries no failure glyphs, so this docstring and
`--stats` are the only record -- SiriusDB's SF500 point in particular is a
17-query sum drawn against three 22-query ones, and reads lower than a complete
run of it would.

Spark-RAPIDS has no SF10 measurement anywhere in the repo, so its line starts at
SF50; that is a gap in the data, not a fast result withheld.

WHAT THE SLIDE GOT WRONG (checked 2026-08-01, kept as the reason not to go back)
--------------------------------------------------------------------------------
The slide IS built on this repo -- two of the three values it prints are archived
SF500 totals to the last digit (`2645` vs polars_gpu 2645.09, `1102` vs rapids
1101.73) -- but its DuckDB series matched no run in it:

    SF          30      50     100     300     500      (total of 22 queries, s)
    slide     14.6    16.8    27.4    96.2   287.0
    measured  10.28   12.16   33.11  101.92  206.25

  * The printed `287s` matched nothing: no engine at any scale factor in the repo
    totals within 2% of it.  Measured DuckDB SF500 is 206.25.
  * There is no DuckDB spill knee.  Measured per-unit cost is flat -- 0.366 s/SF
    at SF100, 0.364 at SF400, 0.382 at SF700, +4% across a 7x data increase.  The
    slide's knee was an understated SF100 point plus an overstated SF500 one.
  * Its low-SF ordering was inverted.  It drew DuckDB slower than Polars and
    Sirius below SF100; measured, DuckDB is the fastest of the four at every
    scale factor from 10 to 700.

That last point is now visible in the figure itself: DuckDB is the bottom line
throughout, so this plot no longer shows a crossover, because there isn't one in
the data.  It shows a widening SPREAD -- which is the claim the measurements do
support.

Finally, one reproduction the slide does not get: its x axis is hand spaced and
is neither linear nor logarithmic (its columns fit no monotone law to better than
~70 px).  This one is a true log ladder.

Colour assignment is inherited from make_tpch_comparison_fig.py so a system keeps
one colour across the paper: Sirius S2, cudf-Polars S3, Spark-RAPIDS S4, and
DuckDB in BASE grey -- the role it already plays there as the reference system.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_OUT = os.path.join(REPO, "experiments", "results", "2026-08-01", "motivation")
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") \
    else DEFAULT_OUT
HERE = os.path.join(OUT, "plot")

# ---------------------------------------------------------------- style
# Identical tokens to make_ablation_figs.py so the two figures sit on one page.
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#78776f"
GRID, AXIS = "#e2e1dc", "#c9c8c2"
BASE = "#77766f"                                    # baseline / "none" role
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"        # validated categorical slots
S4 = "#eda100"                                      # slot 4, needed for a 4th system

# ---------------------------------------------------------------- data
# (scale_factor, seconds, source, n_queries) -- every cell is now a MEASURED
# total from the baselines repo; nothing is read off the source PNG any more.
# One rule fills the grid, applied uniformly:
#
#     live where live has it (SF10 and SF50 only); archive otherwise.
#
# `source` is not decoration.  The two protocols are not interchangeable, and
# for one series they differ by more than the effect the figure is about -- see
# "THE PROTOCOL SEAM" above.  It drives the seam styling in fig_motivation() and
# the provenance column in --stats, so a cell that changes source cannot change
# silently.  n_queries < 22 means the total omits failed queries and is NOT
# comparable to a full-22 total in the same column; --stats lists every one.
SPILL_SF = 100          # everything at and beyond this SF is shaded, as in the source
SEAM = (50, 100)        # the segment that crosses the live -> archive boundary
RUNTIMES = [
    ("SiriusDB", S2, [(10, 10.43, "live", 22), (50, 21.89, "live", 22),
                      (100, 90.50, "archive", 22), (300, 343.65, "archive", 21),
                      (500, 439.44, "archive", 17)]),
    ("cudf-Polars", S3, [(10, 8.41, "live", 22), (50, 43.98, "live", 22),
                         (100, 104.42, "archive", 21), (300, 773.82, "archive", 21),
                         (500, 2645.09, "archive", 22)]),
    # No live SF10/SF50 run exists for this engine anywhere in the repo, so the
    # rule above hands both cells to the archive -- and SF10 is absent entirely.
    # Safe here in a way it would not be for SiriusDB: rapids is the one engine
    # whose two protocols can be compared head to head (SF100, live 506.54 vs
    # archive 503.98) and they agree to 0.5%.
    ("Spark-RAPIDS", S4, [(50, 368.35, "archive", 22), (100, 503.98, "archive", 22),
                          (300, 870.42, "archive", 22), (500, 1101.73, "archive", 22)]),
    ("DuckDB", BASE, [(10, 3.76, "live", 22), (50, 12.25, "live", 22),
                      (100, 33.11, "archive", 22), (300, 101.92, "archive", 22),
                      (500, 206.25, "archive", 22)]),
]

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


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf / {name}.png")


def bare(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)


# ---------------------------------------------------------------- figure
# Both limits are set on the tick ladder's own steps rather than fitted to the
# data, so the axis reads as chosen.  ASSERTED below: a data swap that escapes
# these bounds must move the limit, not silently lose a point.
XLIM = (8.5, 580)
XTICKS = [10, 50, 100, 300, 500]
# 3 is ticked as well as bounded so the axis floor carries its own value, and so
# the smallest point (DuckDB SF10 at 3.76) has a gridline beneath it.
YLIM = (3, 4000)
YTICKS = [3, 10, 30, 100, 300, 1000, 3000]


def fig_motivation():
    pts = [(sf, s) for _, _, series in RUNTIMES for sf, s, _, _ in series]
    out = [p for p in pts if not (XLIM[0] <= p[0] <= XLIM[1] and YLIM[0] <= p[1] <= YLIM[1])]
    if out:
        raise SystemExit(f"{len(out)} point(s) fall outside the axes and would be "
                         f"clipped: {out} -- widen XLIM/YLIM")

    fig, ax = plt.subplots(figsize=(4.3, 3.0))

    # The spill region, recessive: it is the context the lines are read against,
    # not a mark competing with them.  Same wash weight as fig3's parity band.
    ax.axvspan(SPILL_SF, XLIM[1], color=INK3, alpha=0.075, linewidth=0, zorder=0)
    # A wash alone has no readable boundary, and this boundary is the threshold
    # the whole figure turns on -- so it carries a rule, as fig1's 1x line does.
    ax.axvline(SPILL_SF, color=BASE, ls="--", lw=0.9, zorder=1)

    for name, col, series in RUNTIMES:
        # Solid WITHIN a protocol, dotted ACROSS the seam.  The dotted segment is
        # the whole point: its slope is not a measured rate, because its two ends
        # come from different runs, and for SiriusDB roughly 2.4x of the rise is
        # the protocol rather than the data.  A continuous line would assert a
        # scaling rate that nothing measured.
        for src in ("live", "archive"):
            run = [(sf, s) for sf, s, t, _ in series if t == src]
            if run:
                ax.plot([p[0] for p in run], [p[1] for p in run], color=col,
                        lw=1.2, zorder=2)
        a = [(sf, s) for sf, s, t, _ in series if t == "live" and sf == SEAM[0]]
        b = [(sf, s) for sf, s, t, _ in series if t == "archive" and sf == SEAM[1]]
        if a and b:
            ax.plot([a[0][0], b[0][0]], [a[0][1], b[0][1]], color=col, lw=1.2,
                    ls=(0, (1.4, 1.6)), zorder=2)
        ax.scatter([sf for sf, _, _, _ in series], [s for _, s, _, _ in series],
                   s=17, marker="o", facecolor=col, edgecolor="white",
                   linewidth=0.7, zorder=3)
        # Endpoint magnitudes, direct-labelled: the SF500 spread is what the
        # figure is for, and reading four lines against a log ladder loses it.
        sf, s = series[-1][0], series[-1][1]
        ax.annotate(f"{s:,.0f} s", (sf, s), textcoords="offset points",
                    xytext=(7, 0), ha="left", va="center", fontsize=7,
                    color=col, annotation_clip=False)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    # matplotlib's log ticker falls back to minor ticks typeset as "4 x 10^2" at
    # this range.  Place the study's own ladder instead, as fig3 does.
    for axis, ticks in ((ax.xaxis, XTICKS), (ax.yaxis, YTICKS)):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("TPC-H scale factor")
    ax.set_ylabel("runtime (s)")

    # Sits inside the wash it names, at its top edge and hard against the divider
    # it starts at, clear of every line.
    ax.text(SPILL_SF * 1.09, YLIM[1] * 0.86, "spilling from SF100",
            ha="left", va="top", fontsize=7, color=INK2)

    bare(ax)
    handles = [Line2D([], [], color=c, lw=1.2, marker="o", markersize=3.4,
                      markeredgecolor="white", markeredgewidth=0.6)
               for _, c, _ in RUNTIMES]
    labels = [n for n, _, _ in RUNTIMES]
    # The seam earns a legend key.  Without one the dotted run reads as a style
    # choice; with one it states that those two ends are different runs, which is
    # the single thing a reader must know before measuring a slope off this plot.
    # INK2, not the INK3 that fig3 uses for recessive furniture: INK3 (#78776f)
    # is within one step of DuckDB's BASE (#77766f), so a grey key sitting next
    # to the grey series would read as belonging to it.  Carrying no marker is
    # the second separator -- every series key has one, this key does not.
    handles.append(Line2D([], [], color=INK2, lw=1.2, ls=(0, (1.4, 1.6))))
    labels.append("differing runs")
    ax.legend(handles, labels, ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.16), handlelength=1.25, columnspacing=0.8,
              handletextpad=0.4, fontsize=7)
    save(fig, "fig1_motivation")


def stats():
    sfs = sorted({sf for _, _, series in RUNTIMES for sf, _, _, _ in series})
    print("\nruntime (s), total of the queries that completed   "
          "[L = live, A = archive]")
    print(f"  {'system':<14}" + "".join(f"{'SF'+str(s):>12}" for s in sfs))
    for name, _, series in RUNTIMES:
        d = {sf: (s, t, n) for sf, s, t, n in series}
        row = ""
        for sf in sfs:
            if sf not in d:
                row += f"{'--':>12}"
            else:
                s, t, n = d[sf]
                row += f"{s:>10,.1f}" + (" " if t == "live" else "A")
                row += "!" if n < 22 else " "
        print(f"  {name:<14}{row}")
    print("  unmarked = live;  A = archive;  ! = total omits failed queries")

    part = [(name, sf, n) for name, _, series in RUNTIMES
            for sf, _, _, n in series if n < 22]
    if part:
        print("\nPARTIAL TOTALS -- not comparable to a full-22 total in the same column")
        for name, sf, n in part:
            print(f"  {name:<14} SF{sf:<5} {n}/22 queries")

    ref = {sf: s for name, _, series in RUNTIMES if name == "DuckDB"
           for sf, s, _, _ in series}
    print("\nslowdown vs DuckDB at the same scale factor")
    for name, _, series in RUNTIMES:
        if name == "DuckDB":
            continue
        d = {sf: s for sf, s, _, _ in series}
        row = "".join(f"{d[sf] / ref[sf]:>11.1f}x" if sf in d else f"{'--':>12}"
                      for sf in sfs)
        print(f"  {name:<14}{row}")

    print(f"\nseam SF{SEAM[0]} -> SF{SEAM[1]} (live -> archive), apparent growth"
          f" vs the {SEAM[1] / SEAM[0]:.0f}x data increase")
    for name, _, series in RUNTIMES:
        d = {sf: (s, t) for sf, s, t, _ in series}
        if SEAM[0] in d and SEAM[1] in d and d[SEAM[0]][1] != d[SEAM[1]][1]:
            print(f"  {name:<14}{d[SEAM[1]][0] / d[SEAM[0]][0]:>6.2f}x   "
                  f"<- spans two protocols; not a measured rate")


def main():
    os.makedirs(HERE, exist_ok=True)
    print("generating figures in", HERE)
    fig_motivation()
    if "--stats" in sys.argv:
        stats()


if __name__ == "__main__":
    main()
