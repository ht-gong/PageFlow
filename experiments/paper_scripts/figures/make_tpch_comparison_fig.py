#!/usr/bin/env python3
"""Cross-system TPC-H comparison figure, in the house style of make_ablation_figs.py.

    python3 make_tpch_comparison_fig.py [out_dir]   # -> <out_dir>/plot/*.{pdf,png}

Unlike the ablation figures this one has no CSV to read: its numbers are the
per-query wall-clock times of five *different systems*, which come from the
paper's own comparison table rather than from a harness run.  They are therefore
carried inline in RUNTIMES below -- edit that table and re-run.

Every runtime is plotted as a SPEEDUP over DuckDB on the same query
(DuckDB seconds / system seconds), so DuckDB itself is the 1.0 reference line
rather than a fifth bar that would be flat at 1.0 in all 22 groups.  Bars grow
upward from the axis floor exactly as in fig1_runtime_log, and the dashed line
at 1.0 is what separates faster from slower: a bar reaching past it beat DuckDB.

Because the axis is log and the floor is a chosen datum rather than zero, bar
LENGTH is not proportional to speedup -- only the bar's top edge carries the
value, read against the ticks.  That is true of fig1_runtime_log too; the shared
convention is the point, and YLIM's floor is therefore part of the figure's
meaning, not just its framing.

Sirius did not complete q3, q8, q9, q10 and q21.  Those five slots carry an `x`
on the axis floor -- the same "no measurement" marker fig4 uses -- and are
excluded from the geometric mean printed by --stats, which therefore covers only
the 17 queries Sirius completed and is not comparable to the other systems' 22.
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_OUT = os.path.join(REPO, "experiments", "results", "2026-08-01", "tpch_systems")
OUT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "--stats" \
    else DEFAULT_OUT
HERE = os.path.join(OUT, "plot")

# ---------------------------------------------------------------- style
# Identical tokens to make_ablation_figs.py so the two figures sit on one page.
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#78776f"
GRID, AXIS = "#e2e1dc", "#c9c8c2"
BASE = "#77766f"                                    # baseline / "none" role
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"        # validated categorical slots
S4 = "#eda100"                                      # slot 4, needed for a 4th system

# Slot order is the CVD-safety mechanism, so it is a validated choice and not a
# preference: both this order and the plain 1-2-3-4 sequence clear every gate
# (worst adjacent pair is aqua<->yellow, protan dE 9.1 vs the >=8 target, normal
# -vision 22.9 vs the >=15 floor).  Among the two passing orders this one puts
# the highest-contrast slot (blue, 4.3:1) on the system the figure is about and
# the weakest (yellow, 2.11:1) on a context series.  Aqua and yellow sit below
# 3:1 on white, which obligates a relief channel -- the paper's own per-query
# table is it.
SYS = [
    ("Sirius",       S2),
    ("cudf-Polars",   S3),
    ("Spark-RAPIDS", S4),
    ("PageFlow",     S1),
]
BASELINE = "DuckDB"

# query -> {system: wall-clock seconds}; None = the query failed on that system
COLS = ["Sirius", "DuckDB", "cudf-Polars", "Spark-RAPIDS", "PageFlow"]
RUNTIMES = [
    (1,  11.94,  9.96,  15.57,  12.84,  12.88),
    (2,  23.31,  2.10,   2.52,  36.67,   7.37),
    (3,  None,  10.37, 106.62,  43.94,   4.23),
    (4,  10.03,  8.13,   5.82,  22.87,   3.33),
    (5,  93.20, 12.84, 164.05, 100.27,   7.47),
    (6,   7.69,  6.37,   9.52,   6.65,   2.22),
    (7,  30.08, 14.44, 370.15,  50.73,  30.69),
    (8,  None,  10.25, 146.80, 121.14,   5.34),
    (9,  None,  38.01, 564.03, 157.83,  17.54),
    (10, None,  13.76,  36.26,  41.08,  14.96),
    (11, 11.16,  4.29,   2.61,  30.06,   4.46),
    (12, 11.55, 10.20, 111.43,  21.48,   3.01),
    (13, 12.35, 23.31,   7.72,  19.08,  12.18),
    (14, 10.93, 10.73,  18.38,  19.61,   1.92),
    (15, 12.22,  6.96,   7.42,  32.41,   3.89),
    (16, 11.36,  9.50,   5.73,  23.01,   3.34),
    (17, 78.29,  9.37,  14.72,  61.23,  13.03),
    (18, 65.35, 75.72, 168.29, 124.53,  12.51),
    (19, 15.16, 14.24, 140.95,  21.52,   4.20),
    (20, 25.24, 11.15,  14.38,  41.13,   4.68),
    (21, None,  37.15, 729.87,  97.83,  18.98),
    (22,  9.60,  8.18,   2.27,  15.81,   1.19),
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


def gmean(vals):
    return math.exp(sum(map(math.log, vals)) / len(vals))


# ---------------------------------------------------------------- data
def speedups():
    """{system: {query: DuckDB_s / system_s}}, absent where the query failed."""
    out = {c: {} for c in COLS}
    for row in RUNTIMES:
        q, times = row[0], dict(zip(COLS, row[1:]))
        for c in COLS:
            if times[c] is not None:
                out[c][q] = times[BASELINE] / times[c]
    return out


# ---------------------------------------------------------------- figure
# Round limits on the tick ladder's own 1-2-5 steps rather than bounds fitted to
# the data: 0.02 is the tightest round value that still clears the smallest bar
# (cudf-Polars Q7 at 0.039x), so nothing is clipped and the axis reads as chosen
# rather than as whatever the minimum happened to be.  ASSERTED below -- a future
# data swap that dips under 0.02x must move the limit, not silently lose a bar.
YLIM = (0.02, 10.0)
# 0.02 is ticked as well as bounded, so the origin line carries its own value.
# Without it the lowest label is 0.05 and the axis floor -- which the failure
# marks sit on -- is an unlabelled rule the reader has to infer.
YTICKS = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]


def fig_speedup(s, qs):
    live = [v for name, _ in SYS for v in s[name].values()]
    out = [v for v in live if not YLIM[0] <= v <= YLIM[1]]
    if out:
        raise SystemExit(f"{len(out)} bar(s) fall outside YLIM {YLIM} and would be "
                         f"clipped: {sorted(round(v, 4) for v in out)} -- widen YLIM")

    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    n, w = len(SYS), 0.20
    for i, (name, col) in enumerate(SYS):
        xs, hs, fx = [], [], []
        for j, q in enumerate(qs):
            x = j + (i - (n - 1) / 2) * w
            if q in s[name]:
                xs.append(x)
                hs.append(s[name][q])
            else:
                fx.append(x)
        # Default bottom=0, which a log axis clips to YLIM[0] -- bars rise from
        # the floor, as in fig1_runtime_log.
        ax.bar(xs, hs, w * 0.9, color=col, label=name, linewidth=0, zorder=2)
        if fx:
            # A failed query has no bar, and "no bar" must not read as "parity".
            # The marker sits ON the floor, clipped off, as in fig4 -- but in the
            # SERIES colour, not fig4's neutral ink: there the panel held one
            # series so there was nothing to attribute, whereas here four bars
            # share a group and a grey x at a sub-slot is ambiguous.
            ax.scatter(fx, [YLIM[0]] * len(fx), s=13, marker="x", color=col,
                       linewidth=0.9, zorder=3, clip_on=False)

    # Drawn OVER the bars, not under them: now that bars rise from the floor,
    # every bar past 1x crosses this line, and a threshold the marks paint over
    # in 16 of 22 groups cannot be read as a continuous rule.
    ax.axhline(1.0, color=BASE, ls="--", lw=0.9, zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(*YLIM)
    ax.yaxis.set_major_locator(FixedLocator(YTICKS))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"Q{q}" for q in qs])
    ax.set_xlim(-0.65, len(qs) - 0.35)
    ax.set_ylabel("Speedup vs DuckDB ($\\times$)")
    bare(ax)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=BASE, ls="--", lw=0.9))
    labels.append("DuckDB $=1\\times$")
    handles.append(Line2D([], [], color=INK3, ls="none", marker="x", markersize=3.6,
                          markeredgewidth=0.9))
    labels.append("failed")
    ax.legend(handles, labels, ncol=6, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.17), handlelength=1.3, columnspacing=1.4)
    save(fig, "fig1_tpch_speedup")


def stats(s, qs):
    print("\ngeometric-mean speedup vs DuckDB")
    for name, _ in SYS:
        v = [s[name][q] for q in qs if q in s[name]]
        wins = sum(1 for x in v if x > 1)
        print(f"  {name:<13} {gmean(v):5.2f}x   over {len(v):2d} queries, "
              f"faster on {wins:2d}")
    common = [q for q in qs if all(q in s[c] for c in COLS)]
    print(f"\nover the {len(common)} queries every system completed")
    for name, _ in SYS:
        print(f"  {name:<13} {gmean([s[name][q] for q in common]):5.2f}x")


def main():
    os.makedirs(HERE, exist_ok=True)
    qs = [r[0] for r in RUNTIMES]
    s = speedups()
    print("generating figures in", HERE)
    fig_speedup(s, qs)
    if "--stats" in sys.argv:
        stats(s, qs)


if __name__ == "__main__":
    main()
