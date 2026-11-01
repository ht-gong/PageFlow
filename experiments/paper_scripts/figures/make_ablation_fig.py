#!/usr/bin/env python3
"""fig_ablation -- (a) E2's end-to-end ablation as geomean query time per
setting, each mechanism's gain marked against the setting before it, and
(b) E4's breakdown of each PageFlow setting's GPU time, side by side; the page wide.

    python3 make_ablation_fig.py [--csv CSV] [--sf N] [--profile CSV] [--out PDF]
    -> experiments/paper_scripts/results/fig_ablation.pdf

One bar per setting of e2_end_to_end_ablation.py, in its order: UVM, then
PageFlow with one mechanism added per bar (the GPU-native buffer, paging
without compression, then a fixed codec, then the adaptive codec choice and
sizing search: the full system).  Read from results/ablation_results.csv;
none is carried here.  A bar is the geometric mean of the per-query mean time
(``mean_s``) over the queries every setting completed, so all bars average the
same queries.

INCREMENTS ARE MARKED (a).  Above each bar but the first is its geomean
speedup over the bar to its left (previous / this), on a dashed line that
carries the left bar's top over this bar, so the gap between the line and the
bar is the step's gain.  A bar more than 2.5x the next tallest ends on a short
upper axis above a break in the y axis, its value written above it.

THE BREAKDOWN (b).  One bar per PageFlow setting, stacked to 100%, splits
that setting's GPU timeline over the same queries into Compute (a kernel or a D2D copy, no PCIe copy), Overlap (a
kernel and a PCIe copy at once), PCIe (a PCIe copy only) and Host Work (nothing on
the GPU: host work, launches, waits), from E4's per-query traces
(results/ablation_profile.csv, e4_ablation_profile.py): each category's
seconds summed over the queries, over the summed windows, so a share is
time-weighted, not a geomean.  UVM was not profiled and has no breakdown.

FAILURES ARE MARKED.  A query a setting did not complete is left out of every
bar and named in red above that setting's bar ("OOM on ...", "T/O on ..." for a
run stopped or timed out).

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page two columns wide).
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
from sigmod_style import FAIL, SYSTEMS, TEXT, bar_style, save

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

# Setting key -> (name on the figure, bar style), in bar order.  Cumulative
# ablation naming: the baseline by its name, then "+<mechanism>" for each bar
# that adds one to the bar on its left; the last bar is the full system.
SETTINGS = {
    "uvm":          ("UVM", bar_style("UVM")),
    "tqp_raw":      ("+GPU-Native\nBuffer", bar_style("PageFlow", "#c6dcf5", "////")),
    "tqp_ans":      ("+Compression", bar_style("PageFlow", "#86b4ea", "\\\\\\\\")),
    "tqp_adaptive": ("+Adaptivity", bar_style("PageFlow", SYSTEMS["PageFlow"][0], "")),
}
OFF_SCALE = 2.5     # a bar this many times the next tallest gets its own axis
# The breakdown's categories, bottom up: (name, the profile CSV's columns,
# face, hatch, label ink).  Greys, so they read apart from the settings' blues.
PARTS = [
    ("Compute", ("compute_only_s", "d2d_only_s"), "#4a4a4a", "", "white"),
    ("Overlap", ("overlap_s",), "#9b9b9b", "", "black"),
    ("PCIe", ("pcie_only_s",), "#dcdcdc", "", "black"),
    ("Host Work", ("idle_s",), "white", "////", "black"),
]


def load(path, sf):
    """{setting: {"time": {q: s or None}, "error": {q: str}, "partial": {q}}}
    from E2's CSV at scale factor ``sf`` (the CSV also holds the scaling
    figure's rows at other ones), settings in SETTINGS order."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["sf"]) != sf:
                continue
            d = data.setdefault(r["setting"], dict(time={}, error={}, partial=set()))
            q = int(r["query"])
            if q in d["time"]:
                raise SystemExit(f"{path}: repeated ({r['setting']}, SF{sf}, Q{q}): rows "
                                 "at several chunk sizes or partition budgets")
            d["time"][q] = float(r["mean_s"]) if r["mean_s"] else None
            if d["time"][q] is None:
                d["error"][q] = r["error"]
            elif "NA" in r["rep_seconds"].split(";"):
                d["partial"].add(q)
    missing = [k for k in SETTINGS if k not in data]
    if missing:
        raise SystemExit(f"{path}: no rows at SF{sf} for {missing} (--sf picks the scale factor)")
    return {k: data[k] for k in SETTINGS}


def load_profile(path, queries):
    """{setting: {part: share 0..1}} over ``queries`` from E4's CSV; a setting
    missing any of them is left out."""
    rows = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "OK" and r["window_s"]:
                rows.setdefault(r["setting"], {})[int(r["query"])] = r
    out = {}
    for k, by_q in rows.items():
        if not all(q in by_q for q in queries):
            continue
        win = sum(float(by_q[q]["window_s"]) for q in queries)
        out[k] = {name: sum(float(by_q[q][c]) for q in queries for c in cols) / win
                  for name, cols, *_ in PARTS}
    return out


def failure_word(errors):
    text = " ".join(errors).lower()
    if "out of memory" in text or "oom" in text:
        return "OOM"
    if any(w in text for w in ("stopped", "timeout", "time limit", "no progress")):
        return "T/O"
    return "Fails"


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def qlist(qs):
    return ", ".join(f"Q{q}" for q in sorted(qs))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=os.path.join(RESULTS, "ablation_results.csv"))
    ap.add_argument("--sf", type=int, default=1000, help="the ladder's scale factor "
                    "(default SF1000, E2's)")
    ap.add_argument("--profile", default=os.path.join(RESULTS, "ablation_profile.csv"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_ablation.pdf"))
    args = ap.parse_args()

    data = load(args.csv, args.sf)
    keys = list(data)
    queries = sorted(set.union(*(set(d["time"]) for d in data.values())))
    common = [q for q in queries
              if all(d["time"].get(q) is not None for d in data.values())]
    gm = {k: geomean([data[k]["time"][q] for q in common]) for k in keys}

    print(f"geomean over the {len(common)} queries every setting completed "
          f"(left out: {qlist(set(queries) - set(common)) or 'none'})")
    total = math.log(gm[keys[0]] / gm[keys[-1]])
    for i, k in enumerate(keys):
        d, name = data[k], SETTINGS[k][0]
        step = (f"{gm[keys[i - 1]] / gm[k]:6.2f}x over previous "
                f"({math.log(gm[keys[i - 1]] / gm[k]) / total:4.0%} of the log speedup)"
                if i else " " * 50)
        fails = "; ".join(f"Q{q}: {e[:60]}" for q, e in sorted(d["error"].items()))
        print(f"  {name:<14} {gm[k]:8.2f} s  {step}  failed: {fails or '-'}"
              f"  partial: {qlist(d['partial']) or '-'}")
    print(f"  {SETTINGS[keys[-1]][0]} over {SETTINGS[keys[0]][0]}: "
          f"{gm[keys[0]] / gm[keys[-1]]:.1f}x")
    prof = load_profile(args.profile, common)
    print(f"breakdown (E4, time-weighted over the same {len(common)} queries):")
    for k in keys:
        if k in prof:
            print(f"  {SETTINGS[k][0].replace(chr(10), ' '):<20} " + "  ".join(
                f"{n} {100 * v:4.1f}%" for n, v in prof[k].items()))

    ranked = sorted(gm.values(), reverse=True)
    off = {k for k in keys if gm[k] == ranked[0] and ranked[0] > OFF_SCALE * ranked[1]}
    top = (ranked[1] if off else ranked[0]) * 1.35
    step = 10 ** math.floor(math.log10(top)) / 2      # end the axis on a tick
    top = math.ceil(top / step) * step
    w = 0.56

    # (a) on the left, (b) right of it, in one row.  An off-scale bar in (a)
    # takes a short upper axis of its own above a break in the y axis, one
    # tick (the round value just under it) on it.
    fig = plt.figure(figsize=(TEXT, 1.75), layout="constrained")
    top_fig, bot_fig = fig.subfigures(1, 2, width_ratios=[1.1, 1], wspace=0.04)
    if off:
        hi, ax = top_fig.subplots(2, 1, sharex=True, height_ratios=[1, 5],
                                  gridspec_kw=dict(hspace=0))
        big = ranked[0]
        step_hi = 10 ** math.floor(math.log10(big)) / 10
        tick = math.floor(big / step_hi) * step_hi
        hi.set_ylim(tick - 0.6 * step_hi, big + 0.4 * step_hi)
        hi.set_yticks([tick])
        hi.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        hi.spines["bottom"].set_visible(False)
        hi.tick_params(axis="x", length=0, labelbottom=False)
        # Break marks: a short slash on the y axis either side of the gap.
        for a, y in ((hi, 0), (ax, 1)):
            a.plot([0], [y], transform=a.transAxes, marker=[(-1, -0.6), (1, 0.6)],
                   ms=6, mew=0.6, color="black", ls="none", clip_on=False)
    else:
        ax, hi = top_fig.subplots(), None
    bx = bot_fig.subplots()

    def at(y):
        """The axes the height ``y`` is drawn in."""
        return hi if hi is not None and y > top else ax

    def label(a, x, y, text, lines=0, **kw):
        """Text centred above (x, y), ``lines`` text lines up."""
        a.annotate(text, (x, y), xytext=(0, 2 + 11 * lines), textcoords="offset points",
                   ha="center", va="bottom", annotation_clip=False, **kw)

    # (a) geomean time per setting.
    for x, k in enumerate(keys):
        name, st = SETTINGS[k]
        for a in {ax, at(gm[k])}:
            a.bar(x, gm[k], width=w, **st)
        lines = 0
        if k in off:
            label(hi, x, gm[k], f"{gm[k]:.0f} s")
            lines = 1
        if data[k]["error"]:
            word = failure_word(data[k]["error"].values())
            label(at(gm[k]), x, gm[k], f"{word} on {qlist(data[k]['error'])}",
                  lines, color=FAIL)
        if x == 0:
            continue
        prev = keys[x - 1]
        # The left bar's top carried over this one: the gap is the gain.
        a = at(gm[prev])
        a.plot([x - 1 + w / 2, x + w / 2], [gm[prev]] * 2, color="black",
               lw=0.6, ls=(0, (2, 1.5)), zorder=4)
        r = gm[prev] / gm[k]
        label(a, x, gm[prev], f"{r:.2f}×" if r < 10 else f"{r:.1f}×")

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([SETTINGS[k][0] for k in keys])
    ax.tick_params(axis="x", length=0)
    # The second name is long and the gap left of it wide: shift it 4 pt left,
    # so it clears the third.
    lab = ax.get_xticklabels()[1]
    lab.set_transform(lab.get_transform() + ScaledTranslation(-4 / 72, 0, fig.dpi_scale_trans))
    ax.set_xlim(-0.4, len(keys) - 0.66)
    ax.set_ylim(0, top)
    ax.set_ylabel("Geomean time (s)")

    # (b) the breakdown: one 100% bar per profiled setting, in (a)'s order top
    # down.  Host Work (the hatched end) is not numbered: it is what is left.
    pk = [k for k in keys if k in prof]
    for y, k in enumerate(pk):
        left = 0.0
        for pname, _, face, hatch, ink in PARTS:
            v = 100 * prof[k][pname]
            bx.barh(y, v, left=left, height=0.62, facecolor=face, hatch=hatch,
                    edgecolor="black", linewidth=0.5)
            if v >= 9 and pname != "Host Work":
                bx.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", color=ink)
            left += v
    bx.set_yticks(range(len(pk)))
    bx.set_yticklabels([SETTINGS[k][0].replace("\n", " ") for k in pk])
    bx.tick_params(axis="y", length=0)
    bx.set_ylim(len(pk) - 0.5, -0.5)
    bx.set_xlim(0, 100)
    bx.set_xticks([0, 25, 50, 75, 100])
    bx.set_xlabel("Share of GPU time (%)")
    bx.grid(axis="y", visible=False)
    bx.grid(axis="x", visible=True)
    # The legend right-aligned above (b), leaving its row's left end to "(b)".
    bot_fig.legend([Patch(facecolor=face, hatch=hatch, edgecolor="black", linewidth=0.5)
                    for _, _, face, hatch, _ in PARTS], [p[0] for p in PARTS],
                   loc="outside upper right", ncol=len(PARTS), handlelength=1.0,
                   handletextpad=0.4, columnspacing=0.8)
    # Panel labels: each panel's top-left corner, outside its axes.
    for sf, t in ((top_fig, "(a)"), (bot_fig, "(b)")):
        sf.text(0.0, 1.0, t, ha="left", va="top")
    save(fig, args.out)


if __name__ == "__main__":
    main()
