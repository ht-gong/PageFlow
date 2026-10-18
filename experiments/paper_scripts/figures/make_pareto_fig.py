#!/usr/bin/env python3
"""fig_pareto — per-(type, sortedness) compression Pareto front: measured
round-trip cost against compression ratio, every codec at its shipped chunk.

Reads only ``profiling/results/sweep_by_class.csv`` rows with
``selected = 1`` (each codec at the chunk ``codec_map.json`` gives it, plus
RAW) and draws one panel per class: two rows (unsorted / sorted), one column
per mapped type.  The dashed line joins the Pareto front (``on_pareto``) and
the larger, ink-edged, labelled marker is the class winner (``is_best``).  Nothing is re-derived.

    make experiments SCRIPT=experiments/paper_scripts/figures/make_pareto_fig.py
    -> experiments/paper_scripts/figures/plot/fig_pareto.{pdf,png}
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(REPO, "profiling", "results")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot")
RUNS = ["unsorted", "sorted"]
TYPES = ["int", "float", "date", "string"]
GB = 1024 ** 3

# ---------------------------------------------------------------- style
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#78776f"
GRID, AXIS = "#e2e1dc", "#c9c8c2"
BASE = "#77766f"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
S4, S5, S6, S7 = "#8e5bd6", "#c9a227", "#1c9fbf", "#d64a86"
COL = {"RAW": BASE, "ANS": S1, "CASCADED": S2, "BITCOMP": S3, "LZ4": S4,
       "SNAPPY": S5, "GDEFLATE": S6, "DEFLATE": S7}
MARK = {"RAW": "s", "ANS": "o", "CASCADED": "^", "BITCOMP": "D", "LZ4": "v",
        "SNAPPY": "P", "GDEFLATE": "X", "DEFLATE": "h"}

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
    ax.yaxis.grid(True)
    ax.xaxis.grid(True)


def load():
    """{(type, run): {codec: row}} over selected rows."""
    out = defaultdict(dict)
    with open(os.path.join(RESULTS, "sweep_by_class.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["selected"] == "1":
                out[(r["type"], r["run"])][r["codec"]] = r
    return out


def main():
    data = load()
    types = [t for t in TYPES if any((t, run) in data for run in RUNS)]
    fig, axes = plt.subplots(2, len(types), figsize=(2.3 * len(types), 4.2),
                             squeeze=False)
    for ri, run in enumerate(RUNS):
        for ci, ttype in enumerate(types):
            ax = axes[ri][ci]
            rows = data.get((ttype, run), {})
            pts = {c: (float(r["cost_s_per_gb"]) * 1e3, float(r["ratio"]))
                   for c, r in rows.items()}
            front = sorted((c for c, r in rows.items() if r["on_pareto"] == "1"),
                           key=lambda c: pts[c][0])
            ax.plot([pts[c][0] for c in front], [pts[c][1] for c in front],
                    "--", color=GRID, lw=0.9, zorder=1)
            for c, (t, ratio) in pts.items():
                best = rows[c]["is_best"] == "1"
                ax.scatter(t, ratio, s=46 if best else 22, marker=MARK[c],
                           color=COL[c], edgecolors=INK if best else "white",
                           linewidths=0.9 if best else 0.4, zorder=3, label=c)
                if best:
                    ax.annotate(c, (t, ratio), textcoords="offset points",
                                xytext=(5, 3), fontsize=6.5, color=INK,
                                fontweight="bold")
            scanned = float(next(iter(rows.values()))["scanned_bytes"]) / GB if rows else 0
            ax.set_title(f"{ttype}, {run}  ({scanned:.1f} GB scanned)")
            ax.set_yscale("log")
            if ri == 1:
                ax.set_xlabel("round-trip time per GB (ms)")
            if ci == 0:
                ax.set_ylabel("compression ratio")
            bare(ax)
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), ncol=len(seen), frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.03), handletextpad=0.3,
               columnspacing=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(PLOT_DIR, exist_ok=True)
    save(fig, os.path.join(PLOT_DIR, "fig_pareto"))


if __name__ == "__main__":
    main()
