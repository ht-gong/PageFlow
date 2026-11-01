#!/usr/bin/env python3
"""fig_regret -- what one codec per (type, sortedness) costs against the
per-column best, in measured round-trip time; one ACM column wide.

    python3 make_regret_fig.py [--results DIR] [--out PDF]
    -> experiments/paper_scripts/results/fig_regret.pdf

Per class (type x run) one bar, as a share of the no-compression round trip
(100%, a dashed line): the per-column best codec's time (each column at its
own cheapest of the seven codecs), and the extra time the SHIPPED codec spends
on top of that, labelled as the shipped codec's time over the best, as a ratio.
The shipped codec is named in bold beside the class.  Everything is read from the sweep's outputs; nothing is fitted or
re-derived here:

  profiling/results/sweep_per_column.csv   only rows with selected = 1, i.e.
                                           every codec at its shipped chunk
  profiling/results/codec_map.json         only "codec" (the shipped map)

Costs are ``profiling/objective.py``'s sum of the four measured legs; classes
are byte-weighted by scan frequency x raw bytes, like the sweep.  Every bar
is labelled with the ratio; an extra under 0.05 ms/GB (noise) reads 1.0x.

NO COMPRESSION IS ONE RATE.  A raw round trip moves the same bytes whatever
the type or run, so every class gets the same no-compression time: the
measured raw D2H + H2D pooled over every column and both runs, weighted like
the classes.  Per class it would only show measurement noise (the raw H2D of
one column differs by up to 4 ms/GB between its sorted and unsorted runs).
So it is the axis's unit: every bar is divided by it.

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page one column wide).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sigmod_style import COLUMN, SYSTEMS, save

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
from profiling.objective import cost_s, raw_cost_s   # noqa: E402

RESULTS = os.path.join(REPO, "profiling", "results")
OUT = os.path.join(os.path.dirname(HERE), "results", "fig_regret.pdf")
RUNS = ["unsorted", "sorted"]
TYPES = ["int", "float", "date", "string"]
GB = 1024 ** 3

# Bar parts: the best in PageFlow's blue, the rule's extra a darker blue; two
# lightnesses, so greyscale holds.  No compression is the dashed 100% line.
EDGE = dict(edgecolor="black", linewidth=0.5)
BEST = dict(facecolor=SYSTEMS["PageFlow"][0], **EDGE)
EXTRA = dict(facecolor="#123f78", **EDGE)
RAW_LINE = dict(color="black", lw=0.8, ls=(0, (3, 2)))
CODEC_INSET = 3.0       # pt, codec column's right edge to the axis
COL_GAP = 6.0           # pt, clear space between the codec and class columns
TYPE_GAP = 0.35         # rows of extra space between two types


# ---------------------------------------------------------------- data
def build(results):
    with open(os.path.join(results, "codec_map.json"), encoding="utf-8") as f:
        shipped = json.load(f)["codec"]
    # cols[(type, run)][(table, column)] = {"w", "raw", "raw_s", codec: cost_s}
    cols = defaultdict(dict)
    with open(os.path.join(results, "sweep_per_column.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["selected"] != "1":
                continue
            e = cols[(r["type"], r["run"])].setdefault((r["table"], r["column"]), {
                "w": int(r["scan_freq"]), "raw": int(r["raw_bytes"]),
                "raw_s": raw_cost_s(r), "cand": {}})
            e["cand"][r["codec"]] = cost_s(r)

    # The no-compression rate, pooled over every class (see the docstring).
    pool = [e for c in cols.values() for e in c.values()]
    t_r = (sum(e["w"] * e["raw_s"] for e in pool)
           / sum(e["w"] * e["raw"] for e in pool))

    bars = []
    for ttype in TYPES:
        for run in RUNS:
            items = list(cols.get((ttype, run), {}).values())
            if not items:
                continue
            codec = shipped[ttype][run]
            den = sum(e["w"] * e["raw"] for e in items)
            t_o = sum(e["w"] * min(e["cand"].values()) for e in items) / den
            t_p = t_r if codec == "RAW" else \
                sum(e["w"] * e["cand"][codec] for e in items) / den
            bars.append((f"{ttype}, {run}", t_o * GB * 1e3, t_p * GB * 1e3,
                         t_r * GB * 1e3, codec))
    return bars


# ---------------------------------------------------------------- figure
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=RESULTS,
                    help="a `make profiling` output dir (codec_map.json, sweep_per_column.csv)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    bars = build(args.results)
    for label, t_o, t_p, t_r, codec in bars:
        print(f"  {label:<17} {codec:<9} best {t_o:6.2f}  rule {t_p:6.2f} "
              f"(+{100 * (t_p / t_o - 1):4.1f}%)  raw {t_r:6.2f} ms/GB")

    fig, ax = plt.subplots(figsize=(COLUMN, 2.3), layout="constrained")
    h = 0.66
    ypos, y, prev = [], 0.0, None
    for label, t_o, t_p, t_r, codec in bars:
        ttype = label.split(",")[0]
        if prev is not None and ttype != prev:
            y -= TYPE_GAP
        prev = ttype
        best, rule = 100 * t_o / t_r, 100 * t_p / t_r
        ax.barh(y, best, height=h, zorder=2, **BEST)
        if rule > best:
            ax.barh(y, rule - best, height=h, left=best, zorder=3, **EXTRA)
        # An extra under 0.05 ms/GB is noise: the rule picked the best, 1.0x.
        ratio = f"{t_p / t_o:.2f}×" if t_p - t_o >= 0.05 else "1.0×"
        # Beside the bar's end; inside it, in white, for a bar that would
        # carry it into the 100% line.
        inside = rule > 80
        ax.annotate(ratio, (rule, y), xytext=(-2 if inside else 2, 0),
                    textcoords="offset points", va="center", zorder=4,
                    ha="right" if inside else "left",
                    color="white" if inside else "black")
        ypos.append(y)
        y -= 1.0
    # No compression: the unit, one line (named in the legend).
    ax.axvline(100, zorder=4, **RAW_LINE)

    # Two label columns: the class, and the codec the rule ships for it in bold
    # (the finding).  Offsets are in points, so the gap holds at any axes width.
    ax.set_yticks(ypos)
    ax.set_yticklabels([b[0] for b in bars])
    drawn = [ax.annotate(b[4], xy=(0, yy), xycoords=("axes fraction", "data"),
                         xytext=(-CODEC_INSET, 0), textcoords="offset points",
                         ha="right", va="center", fontweight="bold")
             for b, yy in zip(bars, ypos)]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    widest = max(t.get_window_extent(renderer=r).width for t in drawn)
    ax.tick_params(axis="y", length=0,
                   pad=widest * 72.0 / fig.dpi + CODEC_INSET + COL_GAP)

    ax.set_xlim(0, 104)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(y + 1 - h / 2 - 0.25, h / 2 + 0.25)
    ax.set_xlabel("Time (% of no compression)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    fig.legend(handles=[Patch(label=l, **st) for st, l in
                        ((BEST, "Per-column best"), (EXTRA, "Extra by our rule"))]
               + [Line2D([], [], label="No compression", **RAW_LINE)],
               ncol=3, loc="outside upper center", handlelength=1.0,
               handletextpad=0.4, columnspacing=0.9)
    save(fig, args.out)


if __name__ == "__main__":
    main()
