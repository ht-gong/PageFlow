#!/usr/bin/env python3
"""fig4_buffer: the dataflow-vs-LRU buffer leg, as a share of each query's traffic.

    python3 buffer_traffic_fig.py [--data DIR] [--out DIR] [--verify]

RECONSTRUCTED GENERATOR.  The figure `fig4_buffer.{pdf,png}` exists under the
2026-07-29 study but no commit in this repo has ever contained the code that
made it -- it is not in HEAD and not in the history of make_ablation_figs.py,
whose `fig_buffer()` emits two *different* figures (fig4a/fig4b, ratio-to-LRU,
not share-of-traffic).  The two series below were therefore recovered by fitting
candidate formulas against the surviving PNG; both land within 0.36 percentage
points of every value that can be read off it, over 21 read-offs.  `--verify`
re-renders from the 2026-07-29 CSVs so the reconstruction can be diffed against
that PNG directly -- values should match, but the column COUNT deliberately no
longer does: the original dropped queries an arm had not completed, whereas this
spans all 22 (see `series`), so --verify shows 22 columns against the PNG's 19.

Both series are the SAME shape: a dataflow-minus-LRU delta, normalised by the
LRU arm's own total traffic on that leg.  A1 (+sizing, LRU) and A2 (+dataflow)
hold sizing_search=mid_stream, fixed BITCOMP, both compression searches off and
the 25% write-through mirror on, so the eviction policy is the only difference.

  extra served, % of total H2D
      (pcie_h2d_avoided[df] - pcie_h2d_avoided[lru]) / (pcie_h2d_avoided + h2d_miss)[lru]
      Signed on purpose: dataflow serves FEWER bytes than LRU on some queries
      (Q7 in both runs), and a magnitude-only view would hide it.

  critical eviction avoided, % of total D2H
      (spill_evict_d2h[lru] - spill_evict_d2h[df]) / spill_total[lru]
      The evict path is the critical one -- an overflow spill is already lost
      time, whereas an eviction D2H is what the buffer exists to prevent.

COLUMN MEANINGS.  ablation.py builds ablation_buffer_manager.csv from the
movement ledger (IO/telemetry/movement.py).  Against the 2026-07-29 and
2026-07-31 CSVs, which --verify and the default --data read, two denominators
changed meaning:
  * ``h2d_miss_bytes`` is the ledger's ``restore_h2d`` + ``restore_evicted_h2d``
    (a cached region's home pieces): every intermediate read
    back over PCIe, including the outputs never offered to the cache
    (rearrange, lineage and deferred-gather outputs).  The old column counted
    only pages offered to the cache, so the H2D total grows and "extra served"
    shrinks.  ``pcie_h2d_avoided_bytes`` (the ledger's ``cache_hit``) keeps its
    value.
  * ``spill_total_bytes`` is every ``spill_*`` kind, mirror D2H
    included (the old column was eviction + overflow), so the D2H total grows
    and "critical eviction avoided" shrinks.  ``spill_evict_d2h_bytes`` keeps
    its value.
A figure from a CSV written after the change needs a caption that says so,
e.g. "Total H2D counts every intermediate read back over PCIe, whether or not
it was offered to the buffer; total D2H counts every intermediate written to
the host, mirror copies included."

FLOOR: a query that moved essentially nothing on a leg has no meaningful
percentage, so the ratio is undefined below 1 MiB of LRU traffic and takes an x
rather than a bar.  The floor is itself part of the reconstruction -- it is what
reproduces the original's two x marks (Q6, Q19) while leaving Q4/12/13/14/16 as
true zeros -- but the figure is insensitive to its exact value: the largest
floored denominator is 0.5 MB and the smallest un-floored one is 0.52 GB, a
gap of three orders of magnitude.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(REPO, "experiments", "results")
NEW_DATA = os.path.join(RESULTS, "2026-07-31", "ablation_study_sf700_sizefix_pin500")
OLD_DATA = os.path.join(RESULTS, "2026-07-29", "ablation_study_sf700")

# ---------------------------------------------------------------- style
# Same tokens as make_ablation_figs.py so the figures sit on one page.
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#78776f"
GRID, AXIS = "#e2e1dc", "#c9c8c2"
BASE = "#77766f"
S1, S2 = "#2a78d6", "#eb6834"

FLOOR = 1 << 20     # 1 MiB of LRU traffic; below this the share is undefined

# A share this small is not a bar anybody can see: the y-axis spans ~113 points
# over ~2.2 in, so one 300-dpi pixel is ~0.17 points and anything under ~0.05
# cannot render.  Such a value takes the "no change" dot instead of a sub-pixel
# bar -- otherwise a query where both legs moved ~1e-4 % (Q11 in the 2026-07-31
# run) draws NOTHING and reads as "not measured", which is the exact confusion
# the dot exists to prevent.  Testing `== 0` instead, as a float test, leaves
# that hole open; the figure's own resolution is the honest threshold.
EPS = 0.05

# All text at 20 pt.  The canvas grows with it, but deliberately NOT in
# proportion (6.2 -> 11.0 in wide, a factor of 1.8 against the type's 2.5): a
# proportional scale-up would leave the text exactly as large *relative to the
# figure* as it was, which is invisible once LaTeX scales the PDF to column
# width -- the thing bigger type is meant to fix.  The canvas grows only as far
# as the layout needs, so type gains ~1.4x relative size.  Below ~11 in the
# legend's two labels no longer fit on one row at this size.
FONT = 20
SCALE = FONT / 8.0
FIGSIZE = (11.0, 4.6)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": FONT,
    "axes.labelsize": FONT,
    "axes.titlesize": FONT,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
    "legend.fontsize": FONT,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    # Chrome and marks scale with the type, or 20 pt text would sit on hairlines
    # and pinpoint markers that read as artefacts beside it.
    "axes.linewidth": 0.7 * SCALE,
    "grid.color": GRID,
    "grid.linewidth": 0.7 * SCALE,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def bare(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


# ---------------------------------------------------------------- data
def load(path):
    d = defaultdict(dict)
    with open(os.path.join(path, "ablation_buffer_manager.csv")) as f:
        for r in csv.DictReader(f):
            d[r["arm"]][int(r["query"])] = r
    return d


QUERIES = list(range(1, 23))    # the benchmark, not whatever this run completed


def series(d):
    """-> (queries, {q: (served, evict)}, filled).

    Spans all 22 queries.  A query an arm never completed is carried as 0% rather
    than dropped, so the x-axis is the benchmark and stays comparable with the
    other figures; `filled` names those queries.  Note what that 0 does and does
    not claim: it means "no measured difference to plot here", NOT that dataflow
    matched LRU -- for Q9 in the 2026-07-31 run, dataflow did not complete at all.
    Rendered, they are indistinguishable from a true measured zero (Q4, Q12,
    ...), so the caption has to name them.

    None (an x mark) is different again and still reserved for its own case: the
    query ran, but moved less than FLOOR on that leg, so the share is undefined.
    """
    def F(arm, q, col):
        return float(d[arm][q][col])

    out, filled = {}, []
    for q in QUERIES:
        if q not in d["sizing"] or q not in d["dataflow"]:
            out[q] = (0.0, 0.0)
            filled.append(q)
            continue
        h2d = F("sizing", q, "pcie_h2d_avoided_bytes") + F("sizing", q, "h2d_miss_bytes")
        d2h = F("sizing", q, "spill_total_bytes")
        served = None if h2d < FLOOR else 100 * (
            F("dataflow", q, "pcie_h2d_avoided_bytes")
            - F("sizing", q, "pcie_h2d_avoided_bytes")) / h2d
        evict = None if d2h < FLOOR else 100 * (
            F("sizing", q, "spill_evict_d2h_bytes")
            - F("dataflow", q, "spill_evict_d2h_bytes")) / d2h
        out[q] = (served, evict)
    return QUERIES, out, filled


# ---------------------------------------------------------------- figure
LEG = [(0, S1, "extra served, % of total H2D"),
       (1, S2, "critical eviction avoided, % of total D2H")]


def fig_buffer_traffic(data, outdir):
    qs, s, filled = series(load(data))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    w = 0.38

    for i, col, label in LEG:
        x = [j + (i - 0.5) * w for j in range(len(qs))]
        v = [s[q][i] for q in qs]
        drawn = [(xx, vv) for xx, vv in zip(x, v) if vv is not None and abs(vv) >= EPS]
        ax.bar([xx for xx, _ in drawn], [vv for _, vv in drawn], w * 0.92,
               color=col, label=label, linewidth=0, zorder=2)
        # An invisible bar must not read the same as "no measurement".  Both
        # markers sit ON the zero line: a dot for "measured, no change", an x
        # for "no traffic to measure".
        flat = [xx for xx, vv in zip(x, v) if vv is not None and abs(vv) < EPS]
        ax.scatter(flat, [0] * len(flat), s=7 * SCALE**2, marker="o", color=col, zorder=3)
        ax.scatter([xx for xx, vv in zip(x, v) if vv is None], [0] * v.count(None),
                   s=10 * SCALE**2, marker="x", color=INK3, linewidth=0.8 * SCALE, zorder=3)

    ax.axhline(0, color=BASE, lw=1.1 * SCALE, zorder=1)
    vals = [vv for q in qs for vv in s[q] if vv is not None]
    ax.set_ylim(min(-8.0, min(vals) * 1.25), 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(range(len(qs)))
    # Bare query numbers, unrotated -- the same call fig_buffer makes in
    # make_ablation_figs.py, and for the same reason: the "Q" prefix is what
    # makes 22 labels collide.  At 20 pt it stops being only a fit problem and
    # becomes a correctness one, because rotating to buy room puts each label
    # up-LEFT of its tick by more than half a group, so "Q9" reads as belonging
    # to Q8's bars.  Dropping two glyphs lets them sit centred under their own
    # group instead.
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_xlim(-0.7, len(qs) - 0.3)
    # Matches the legend's own wording ("% of total H2D" / "% of total D2H"),
    # where "total" has always meant that query's total on that leg.  The
    # normalisation is unchanged and still per-query -- see `series`.
    ax.set_ylabel("% of total traffic")
    bare(ax)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16),
              handlelength=1.3, columnspacing=1.8)

    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig4_buffer.{ext}"), dpi=300)
    plt.close(fig)
    print(f"  wrote fig4_buffer.pdf / fig4_buffer.png -> {outdir}")

    print(f"  {len(qs)} queries: {qs}")
    print(f"    carried as 0% (arm did not complete): {filled or 'none'}")
    for i, _, label in LEG:
        v = [s[q][i] for q in qs]
        live = [x for x in v if x is not None]
        print(f"    {label:44s} range {min(live):6.1f} .. {max(live):5.1f}"
              f"   flat: {[q for q, x in zip(qs, v) if x is not None and abs(x) < EPS]}"
              f"   floored: {[q for q, x in zip(qs, v) if x is None]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=NEW_DATA, help="dir holding ablation_buffer_manager.csv")
    p.add_argument("--out", default=None,
                   help="dir to write figs into (default: <data>/plot, so a figure is "
                        "always filed under the run that produced its numbers)")
    p.add_argument("--verify", action="store_true",
                   help="render from the 2026-07-29 CSVs into a scratch dir instead, "
                        "to diff the reconstruction against the surviving PNG")
    a = p.parse_args()
    if a.verify:
        out = os.path.join(REPO, "experiments", "results", "_verify_fig4_buffer", "plot")
        print(f"verifying reconstruction from {OLD_DATA}")
        fig_buffer_traffic(OLD_DATA, out)
        return
    print(f"data {a.data}")
    fig_buffer_traffic(a.data, a.out or os.path.join(a.data, "plot"))


if __name__ == "__main__":
    main()
