#!/usr/bin/env python3
"""fig_regret — what one codec per (type, sortedness) costs against the
per-column best, in measured round-trip time.

Per class (type x run) three bars: the per-column best codec's time (each
column at its own cheapest of the seven codecs), the extra time the SHIPPED
codec spends on top of that, and the no-compression time (the measured raw
D2H + H2D).  Everything is read from the sweep's outputs; nothing is fitted
or re-derived here:

  profiling/results/sweep_per_column.csv   only rows with selected = 1, i.e.
                                           every codec at its shipped chunk
  profiling/results/codec_map.json         only "codec" (the shipped map)

Costs are ``profiling/objective.py``'s sum of the four measured legs; classes
are byte-weighted by scan frequency x raw bytes, like the sweep.  Styled to
match make_ablation_figs.py: the same serif stack, ink/grid neutrals,
validated categorical slots, bare() spines, legend above the axes, and
pdf+png at 300 dpi.  No title -- the caption carries it.

    make experiments SCRIPT=experiments/paper_scripts/figures/make_regret_fig.py
    -> experiments/paper_scripts/figures/plot/fig_regret.{pdf,png}
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from profiling.objective import cost_s, raw_cost_s   # noqa: E402

RESULTS = os.path.join(REPO, "profiling", "results")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot")
RUNS = ["unsorted", "sorted"]
TYPES = ["int", "float", "date", "string"]
GB = 1024 ** 3

# ---------------------------------------------------------------- style
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#78776f"
GRID, AXIS = "#e2e1dc", "#c9c8c2"
BASE = "#77766f"                                    # baseline / "none" role
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"        # validated categorical slots
TRACK = "#dfddd6"                                   # the un-compressed budget

CODEC_INSET = 6.0       # pt, codec column's right edge to the axis
COL_GAP = 14.0          # pt, clear space between the codec and class columns
FS = 20                 # figure-wide font size

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.linewidth": 0.9,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    plt.close(fig)


def bare(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)                             # value axis is x here
    ax.yaxis.grid(False)


# ---------------------------------------------------------------- data
def build():
    with open(os.path.join(RESULTS, "codec_map.json"), encoding="utf-8") as f:
        shipped = json.load(f)["codec"]
    # cols[(type, run)][(table, column)] = {"w", "raw", "raw_s", codec: cost_s}
    cols = defaultdict(dict)
    with open(os.path.join(RESULTS, "sweep_per_column.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["selected"] != "1":
                continue
            e = cols[(r["type"], r["run"])].setdefault((r["table"], r["column"]), {
                "w": int(r["scan_freq"]), "raw": int(r["raw_bytes"]),
                "raw_s": raw_cost_s(r), "cand": {}})
            e["cand"][r["codec"]] = cost_s(r)

    bars = []
    for ttype in TYPES:
        for run in RUNS:
            items = list(cols.get((ttype, run), {}).values())
            if not items:
                continue
            codec = shipped[ttype][run]
            den = sum(e["w"] * e["raw"] for e in items)
            t_o = sum(e["w"] * min(e["cand"].values()) for e in items) / den
            t_p = sum(e["w"] * (e["raw_s"] if codec == "RAW" else e["cand"][codec])
                      for e in items) / den
            t_r = sum(e["w"] * e["raw_s"] for e in items) / den
            bars.append((f"{ttype}, {run}", t_o * GB * 1e3, t_p * GB * 1e3,
                         t_r * GB * 1e3, codec))
    return bars


# ---------------------------------------------------------------- figure
def main():
    bars = build()
    raw_ms = max(b[3] for b in bars)

    fig, ax = plt.subplots(figsize=(11.6, 4.9))
    ypos, labels = [], []
    y = 0.0
    for label, t_o, t_p, t_r, codec in bars:
        ax.barh(y, t_r, height=0.62, color=TRACK, linewidth=0, zorder=1)
        ax.barh(y, t_o, height=0.62, color=S1, edgecolor="white", linewidth=0.7,
                zorder=2)
        if t_p > t_o:
            ax.barh(y, t_p - t_o, height=0.62, left=t_o, color=S2, linewidth=0,
                    zorder=3)
        # Annotate the excess only where it reaches 0.05 ms/GB; below that it
        # rounds to nothing and "+0.0" is noise.
        excess, pct = t_p - t_o, 100 * (t_p / t_o - 1)
        txt = f"{t_p:.1f}"
        if excess >= 0.05:
            txt += f"   +{excess:.1f} ({pct:.1f}%)" if pct < 10 else \
                   f"   +{excess:.1f} ({pct:.0f}%)"
        ax.annotate(txt, (max(t_p, t_o) + 1.2, y), va="center", fontsize=FS,
                    color=INK if excess >= 0.05 else INK2)
        ypos.append(y)
        labels.append((label, codec))
        y -= 1.0

    ax.axvline(raw_ms, color=BASE, ls="--", lw=0.9, zorder=4)
    # The row label is two columns, not one string: the CLASS is the row's name
    # and the CODEC is the finding -- which codec one rule per class picks -- so
    # the codec carries bold primary ink and the class stays secondary.
    # Deliberately NOT coloured: S1 blue is already the oracle series here.
    # Offsets are in POINTS, so the two columns keep their gap regardless of how
    # wide the axes end up.
    ax.set_yticks(ypos)
    ax.set_yticklabels([c for c, _ in labels])
    drawn = [ax.annotate(codec, xy=(0, yy), xycoords=("axes fraction", "data"),
                         xytext=(-CODEC_INSET, 0), textcoords="offset points",
                         ha="right", va="center", fontweight="bold", color=INK)
             for (_, codec), yy in zip(labels, ypos)]
    # MEASURE the codec column rather than estimating it from a per-character
    # guess: bold metrics are not a fixed multiple of the font size.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer=r).width for t in drawn)
    ax.tick_params(axis="y", length=0,
                   pad=widest * 72.0 / fig.dpi + CODEC_INSET + COL_GAP)

    ax.set_xlim(0, raw_ms * 1.015)
    ax.set_ylim(y + 0.35, 0.65)
    ax.set_xlabel("round-trip time per GB (ms)")
    bare(ax)

    ax.legend(handles=[Patch(facecolor=c, linewidth=0, label=l) for c, l in
                       ((S1, "Per-Column Best"),
                        (S2, "Extra Time Spent by Our Rule"),
                        (TRACK, "No Compression"))],
              ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              handlelength=1.2, columnspacing=1.6)

    os.makedirs(PLOT_DIR, exist_ok=True)
    save(fig, os.path.join(PLOT_DIR, "fig_regret"))


if __name__ == "__main__":
    main()
