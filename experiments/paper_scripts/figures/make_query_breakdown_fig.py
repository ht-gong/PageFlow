#!/usr/bin/env python3
"""fig_query_breakdown -- fig_ablation (b)'s breakdown of GPU time, for single
queries of PageFlow's adaptive setting at SF1000; one ACM column wide.

    python3 make_query_breakdown_fig.py [--queries N ...]
                                                         [--profile CSV] [--out PDF]
    -> experiments/paper_scripts/results/fig_query_breakdown.pdf

One bar per query (default Q1, Q6, Q8, Q18, Q21), stacked to 100%, splits
that query's traced window into fig_ablation (b)'s categories: Compute (a
kernel or a D2D copy, no PCIe copy), Overlap (a kernel and a PCIe copy at
once), PCIe (a PCIe copy only) and Host Work (nothing on the GPU), from E4's
per-query traces (results/ablation_profile.csv, e4_ablation_profile.py).  The
categories, colours and labelling are fig_ablation's (imported), so the bars
read against its (b).  stdout also gives each query's E2 time (``e2_mean_s``).

A query the profile has no OK row for stops the script.

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page one column wide).
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from make_ablation_fig import PARTS, RESULTS
from sigmod_style import COLUMN, save


SETTING = "tqp_adaptive"     # PageFlow's full setting (E2, E4)


def load(path, queries):
    """{q: (E2 seconds, {part: share 0..1})} for SETTING's ``queries``."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["setting"] == SETTING and r["status"] == "OK" and r["window_s"]:
                rows[int(r["query"])] = r
    missing = [q for q in queries if q not in rows]
    if missing:
        raise SystemExit(f"{path}: no OK {SETTING} row for Q{missing}")
    out = {}
    for q in queries:
        r, win = rows[q], float(rows[q]["window_s"])
        out[q] = (float(r["e2_mean_s"]),
                  {name: sum(float(r[c]) for c in cols) / win for name, cols, *_ in PARTS})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queries", type=int, nargs="+", default=[1, 6, 8, 18, 21])
    ap.add_argument("--profile", default=os.path.join(RESULTS, "ablation_profile.csv"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_query_breakdown.pdf"))
    args = ap.parse_args()

    prof = load(args.profile, args.queries)
    print(f"{SETTING}, share of each query's traced window:")
    for q, (t, sh) in prof.items():
        print(f"  Q{q:<3} {t:6.2f} s  " + "  ".join(f"{n} {100 * v:4.1f}%" for n, v in sh.items()))

    fig, bx = plt.subplots(figsize=(COLUMN, 0.55 + 0.27 * len(prof)), layout="constrained")
    # One 100% bar per query, top down.  Host Work (the hatched end) is not
    # numbered: it is what is left.
    for y, (q, (_, sh)) in enumerate(prof.items()):
        left = 0.0
        for pname, _, face, hatch, ink in PARTS:
            v = 100 * sh[pname]
            bx.barh(y, v, left=left, height=0.62, facecolor=face, hatch=hatch,
                    edgecolor="black", linewidth=0.5)
            if v >= 9 and pname != "Host Work":
                bx.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", color=ink)
            left += v
    bx.set_yticks(range(len(prof)))
    bx.set_yticklabels([f"Q{q}" for q in prof])
    bx.tick_params(axis="y", length=0)
    bx.set_ylim(len(prof) - 0.5, -0.5)
    bx.set_xlim(0, 100)
    bx.set_xticks([0, 25, 50, 75, 100])
    bx.set_xlabel("Share of GPU time (%)")
    bx.grid(axis="y", visible=False)
    bx.grid(axis="x", visible=True)
    fig.legend([Patch(facecolor=face, hatch=hatch, edgecolor="black", linewidth=0.5)
                for _, _, face, hatch, _ in PARTS], [p[0] for p in PARTS],
               loc="outside upper right", ncol=len(PARTS), handlelength=1.0,
               handletextpad=0.4, columnspacing=0.8)
    save(fig, args.out)


if __name__ == "__main__":
    main()
