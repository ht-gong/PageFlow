#!/usr/bin/env python3
"""fig_allocation_ablation -- E3's query time per allocation setting on a few
queries, and its geomean over the suite; one ACM column wide.

    python3 make_allocation_fig.py [--csv CSV] [--out PDF] [--queries 8,13,21]
    -> experiments/paper_scripts/results/fig_allocation_ablation.pdf

One bar per setting of e3_allocation_ablation.py (free list over the whole
arena, free list with a reserve, PageFlow), read from
results/allocation_ablation_results.csv; none is carried here.  Panel (a), on
the left, is each of ``--queries`` (default Q8, Q13, Q21) on its own: a bar is that
query's mean time (``mean_s``).  Panel (b), GeoMean, is the geometric
mean of the per-query mean time over the queries every setting completed at
least once, so all three bars average the same queries.  Both panels share one
log time axis.  Above every free-list bar is its time over PageFlow's, as a ratio.

FAILURES ARE MARKED.  In (a), a query a setting never completed (every try
failed) is a white hatched bar to the top of the axis, labelled OOM in red when
the error is an out-of-memory.  Such a query is left out of every GeoMean bar,
and (b) names them in red at its top, with a line down to that setting's bar.  The
summary printed to stdout also lists the queries that failed in some tries but
not all, and those with fewer successful runs than ``need``.

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page one column wide).
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from sigmod_style import COLUMN, FAIL, bar_style, save

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

# Setting key -> name on the figure, in bar order.
SETTINGS = {
    "freelist_full": "Free list",
    "freelist_reserve": "Free list reserved",
    "tqp": "PageFlow",
}
OURS = "tqp"
# The queries drawn one by one on the left (--queries).
QUERIES = (8, 13, 21)


def load(path):
    """{setting: {"label", "time": {q: s or None}, "fail": {q}, "flaky": {q},
    "short": {q}, "oom": bool}} from E3's CSV, settings in SETTINGS order."""
    data = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = data.setdefault(r["setting"], dict(label=r["label"], time={}, fail=set(),
                                                   flaky=set(), short=set(), oom=True))
            q = int(r["query"])
            tries = r["try_seconds"].split(";") if r["try_seconds"] else []
            n_bad = sum(t == "NA" for t in tries)
            d["time"][q] = float(r["mean_s"]) if r["mean_s"] else None
            if d["time"][q] is None:
                d["fail"].add(q)
                d["oom"] &= "out of memory" in r["error"].lower()
            elif n_bad:
                d["flaky"].add(q)
            elif int(r["n_ok"]) < int(r["need"]):
                d["short"].add(q)
    missing = [k for k in SETTINGS if k not in data]
    if missing:
        raise SystemExit(f"{path}: no rows for {missing}")
    return {k: data[k] for k in SETTINGS}


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def qlist(qs):
    return ", ".join(f"Q{q}" for q in sorted(qs))


def ladder(lo, hi):
    """The 1-2-5 values from the largest one <= lo to the smallest one >= hi."""
    vals = [m * 10.0 ** k for k in range(-3, 5) for m in (1, 2, 5)]
    a = max(v for v in vals if v <= lo)
    b = min(v for v in vals if v >= hi)
    return [v for v in vals if a <= v <= b]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default=os.path.join(RESULTS, "allocation_ablation_results.csv"))
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_allocation_ablation.pdf"))
    ap.add_argument("--queries", default=",".join(map(str, QUERIES)),
                    help="the queries drawn one by one on the left (default %(default)s)")
    args = ap.parse_args()
    shown = [int(q.strip().removeprefix("Q").removeprefix("q"))
             for q in args.queries.split(",") if q.strip()]

    data = load(args.csv)
    queries = sorted(set.union(*(set(d["time"]) for d in data.values())))
    unknown = [q for q in shown if q not in queries]
    if unknown:
        raise SystemExit(f"{args.csv}: no rows for {qlist(unknown)}")
    common = [q for q in queries
              if all(d["time"].get(q) is not None for d in data.values())]
    gm = {k: geomean([d["time"][q] for q in common]) for k, d in data.items()}

    print(f"geomean over the {len(common)} queries every setting completed "
          f"(left out: {qlist(set(queries) - set(common)) or 'none'})")
    for k, d in data.items():
        print(f"  {SETTINGS[k]:<20} {gm[k]:7.2f} s  {gm[k] / gm[OURS]:5.2f}x {SETTINGS[OURS]}"
              f"  never completed: {qlist(d['fail']) or '-'}"
              f"  failed some tries: {qlist(d['flaky']) or '-'}"
              f"  fewer than need runs: {qlist(d['short']) or '-'}")
    print("per query (s): " + "  ".join(
        f"Q{q} " + "/".join("FAIL" if data[k]["time"][q] is None
                            else f"{data[k]['time'][q]:.1f}" for k in data)
        for q in shown) + f"  ({' / '.join(SETTINGS.values())})")

    keys = list(data)
    n, w = len(keys), 0.8 / len(keys)
    live = [t for k in keys for q in shown if (t := data[k]["time"][q]) is not None]
    ticks = ladder(min(live + list(gm.values())), max(live + list(gm.values())) * 1.6)
    lo, top = ticks[0], ticks[-1]

    fig, (ax, gx) = plt.subplots(
        1, 2, figsize=(COLUMN, 1.9), sharey=True, layout="constrained",
        gridspec_kw=dict(width_ratios=[len(shown), 1.6]))
    # A wide gap sets (b) apart from (a).
    fig.get_layout_engine().set(wspace=0.25)
    for i, k in enumerate(keys):
        st, off = bar_style(SETTINGS[k]), (i - (n - 1) / 2) * w
        d = data[k]
        for j, q in enumerate(shown):
            t = d["time"][q]
            if t is not None:
                ax.bar(j + off, t, width=w, label=SETTINGS[k] if j == 0 else None, **st)
                if k != OURS and d["time"][q] and data[OURS]["time"][q]:
                    ax.text(j + off, t * 1.08, f"{t / data[OURS]['time'][q]:.1f}×",
                            rotation=90, ha="center", va="bottom")
                continue
            # Never completed: a white hatched bar to the top, named in red.
            ax.bar(j + off, top, width=w, facecolor="white", hatch="////",
                   edgecolor="black", linewidth=0.5)
            ax.text(j + off, lo * (top / lo) ** 0.6, "OOM" if d["oom"] else "FAIL",
                    rotation=90, ha="center", va="center", color=FAIL,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        bar = gx.bar(off, gm[k], width=w, **st)[0]
        above = bar
        if k != OURS:
            above = gx.text(off, gm[k] * 1.08, f"{gm[k] / gm[OURS]:.1f}×", rotation=90,
                            ha="center", va="bottom")
        # The queries a setting failed (left out of every bar): a note at the
        # top of (b), from the bar's left edge, with a line down to the bar.
        if d["fail"]:
            gx.annotate(f"{SETTINGS[k]} {'OOM' if d['oom'] else 'fails'}:\n"
                        f"{qlist(d['fail'])}", (0.5, 1.0), xycoords=above,
                        xytext=(off - w / 2, top / 1.2), textcoords="data",
                        ha="left", va="top", color=FAIL, linespacing=1.1,
                        arrowprops=dict(arrowstyle="-", color=FAIL, lw=0.6,
                                        shrinkA=1, shrinkB=1, relpos=(0.15, 0.0)))

    ax.set_yscale("log")
    ax.set_ylim(lo, top)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("Query time (s)")
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([f"Q{q}" for q in shown])
    ax.set_xlim(-0.5, len(shown) - 0.5)
    gx.set_xticks([0])
    gx.set_xticklabels(["GeoMean"])
    gx.set_xlim(-0.6, 0.6)
    gx.tick_params(axis="y", left=False)
    gx.spines["left"].set_visible(False)
    for a in (ax, gx):
        a.tick_params(axis="x", length=0)
    # Panel captions under each panel.
    ax.set_xlabel("(a) Selected queries")
    gx.set_xlabel(f"(b) {len(common)} queries")
    fig.legend(*ax.get_legend_handles_labels(), ncol=n, loc="outside upper center")
    save(fig, args.out)

if __name__ == "__main__":
    main()
