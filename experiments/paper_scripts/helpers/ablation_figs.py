#!/usr/bin/env python3
"""Regenerate the ablation figures from the CSVs of an ablation.py run.

    python3 ablation_figs.py [ablation_dir]     # -> <ablation_dir>/plot/*.{pdf,png}

`ablation_dir` is the directory holding the ablation_*.csv files and defaults to the
2026-07-29 SF700 study.  This script lives under paper_scripts/ rather than
beside its output because experiments/results/ is gitignored (`result*`), so a
generator kept next to the figures would never be committed.

The runtime figures (1, 2 and 5) read ablation_raw.csv, which ablation.py
writes: each cell is the median of the arm's reps, A0 is the run's `baseline`
arm, and a query an arm did not complete is left out of that arm's bars and of
the totals.  A directory without it is read as the 2026-07-29 study:

  Run of 2026-07-29, engine @ cf4d1bd, RTX 5090, --cmp_pool_gb 29, --slot_pool_gb
  4.0, per-query B, single repetition, in ablation_runtimes_round1.csv and
  ablation_runtimes_baseline_c{512,1024}.csv.  Baseline A0 is the static-1024 MB
  configuration measured the same night on the same HEAD (--slot_pool_gb 0.5); it
  is NOT the arm labelled `baseline` in ablation_runtimes_round1.csv, which ran at
  the engine-default 256 MB because the harness had no --chunk_size_mb flag yet
  (added in 57e76cd).  Ten of the 88 runtime cells are IMPUTED: those queries
  aborted with `Pool out of memory` under that arm.  See `impute()` for the model
  and its validation.

The buffer study's CSV (figure 4) comes from the movement ledger since
ablation.py moved onto it; see `fig_buffer` for the one column whose
meaning changed.
"""
import csv
import math
import os
import statistics
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_DATA = os.path.join(REPO, "experiments", "results", "2026-07-29",
                            "ablation_study_sf700")
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
HERE = os.path.join(DATA, "plot")

# ---------------------------------------------------------------- style
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#78776f"
GRID, AXIS = "#e2e1dc", "#c9c8c2"
BASE = "#77766f"                                    # baseline / "none" role
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"        # validated categorical slots
CRIT = "#e34948"

ARMS = ["a0", "sizing", "dataflow", "adaptive"]
COL = {"a0": BASE, "sizing": S1, "dataflow": S2, "adaptive": S3}
LBL = {"a0": "A$_0$ baseline", "sizing": "A$_1$ +sizing",
       "dataflow": "A$_2$ +dataflow", "adaptive": "A$_3$ +adaptive"}
# The 2026-07-29 runtime CSVs: A0 is the static-1024 configuration.
ARCHIVED_RUNTIMES = ("ablation_runtimes_round1.csv",
                     "ablation_runtimes_baseline_c512.csv",
                     "ablation_runtimes_baseline_c1024.csv")

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
    "hatch.linewidth": 0.75,
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


# ---------------------------------------------------------------- data
def load():
    """Runtimes ``{arm: {query: seconds}}`` with ``a0`` set, and whether they
    are the archived 2026-07-29 set (see the module docstring)."""
    a = defaultdict(dict)
    raw = os.path.join(DATA, "ablation_raw.csv")
    if os.path.isfile(raw):
        reps = defaultdict(list)
        with open(raw) as f:
            for r in csv.DictReader(f):
                if r["seconds"]:
                    reps[(r["arm"], int(r["query"]))].append(float(r["seconds"]))
        for (arm, q), secs in reps.items():
            a[arm][q] = statistics.median(secs)
        a["a0"] = a["baseline"]
        return a, False
    with open(os.path.join(DATA, "ablation_runtimes_round1.csv")) as f:
        for r in csv.DictReader(f):
            a[r["arm"]][int(r["query"])] = float(r["seconds"])
    for fn in ("ablation_runtimes_baseline_c512.csv",
               "ablation_runtimes_baseline_c1024.csv"):
        with open(os.path.join(DATA, fn)) as f:
            for row in csv.reader(f):
                a[row[0]][int(row[1])] = float(row[2])
    a["a0"] = a["baseline_c1024"]
    return a, True


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


def gmean(vals):
    return math.exp(sum(map(math.log, vals)) / len(vals))


def pct(x):
    """Signed percentage with a typographic minus, not an ASCII hyphen."""
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.0%}"


def impute(a, qs):
    """Estimate the 10 OOM'd cells.

    Arm benefit tracks each query's static chunk-size sensitivity g = t256/t1024
    (log-log r = -0.64/-0.78/-0.70) and NOT query size (-0.13/-0.27/-0.17), which
    matches the mechanism: the sizing search's job is to find a bigger window.
    So fit log(t_arm/t256) = p + q*log(g) per arm on that arm's completed queries.

    adaptive/Q7 instead uses the sibling arm: A3 differs from A2 only on the
    compression axis, whose ratio is 0.944 (geomean, 18 shared queries), and A2
    measured Q7 directly.  adaptive/Q10,17,18 take the geometric mean of both
    routes (they agree to 2.5-12%).

    Validation: this predicts Q10/+sizing = 13.72 s; a later run at HEAD 99c9a5d
    with the sizing clamp fixed measured 13.78 s (-0.4%).  Returns
    {arm: {q: (estimate_s, log_sigma)}}.
    """
    c256, c1024 = a["baseline"], a["baseline_c1024"]
    g = {q: c256[q] / c1024[q] for q in qs}
    fit, loo = {}, {}
    for arm in ("sizing", "dataflow", "adaptive"):
        done = [q for q in qs if q in a[arm]]
        xs = [math.log(g[q]) for q in done]
        ys = [math.log(a[arm][q] / c256[q]) for q in done]
        fit[arm] = _ols(xs, ys)
        res = []
        for i in range(len(done)):
            pi, qi = _ols(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
            res.append(ys[i] - (pi + qi * xs[i]))
        loo[arm] = math.sqrt(sum(r * r for r in res) / len(res))

    def gpred(arm, q):
        p, s = fit[arm]
        return math.exp(p + s * math.log(g[q])) * c256[q]

    shared = [q for q in qs if q in a["adaptive"] and q in a["dataflow"]]
    ratio = gmean([a["adaptive"][q] / a["dataflow"][q] for q in shared])
    imp = {
        "sizing": {q: (gpred("sizing", q), loo["sizing"]) for q in (10, 17, 18)},
        "dataflow": {q: (gpred("dataflow", q), loo["dataflow"]) for q in (10, 17, 18)},
        "adaptive": {7: (a["dataflow"][7] * ratio, 0.1302)},
    }
    for q in (10, 17, 18):
        imp["adaptive"][q] = (math.sqrt(gpred("adaptive", q) * gpred("dataflow", q) * ratio),
                              loo["adaptive"])
    return imp


# ---------------------------------------------------------------- figures
def fig_runtime(a, imp, qs, archived, logy=False):
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    n, w = len(ARMS), 0.20
    for i, arm in enumerate(ARMS):
        src = a[arm]
        xs, ys, hx, hy, lo, hi = [], [], [], [], [], []
        for j, q in enumerate(qs):
            x = j + (i - (n - 1) / 2) * w
            if q in src:
                xs.append(x)
                ys.append(src[q])
            elif q in imp[arm]:
                t, s = imp[arm][q]
                hx.append(x)
                hy.append(t)
                lo.append(t - t / math.exp(s))
                hi.append(t * math.exp(s) - t)
        ax.bar(xs, ys, w * 0.9, color=COL[arm], label=LBL[arm], linewidth=0)
        if hx:
            ax.bar(hx, hy, w * 0.9, facecolor=COL[arm], alpha=0.38,
                   edgecolor=COL[arm], hatch="////", linewidth=0.6)
            ax.errorbar(hx, hy, yerr=[lo, hi], fmt="none", ecolor=COL[arm],
                        elinewidth=0.8, capsize=1.6, capthick=0.8)
    vals = [v for arm in ARMS for v in a[arm].values()]
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(*((0.9, 60) if archived
                      else (min(vals) * 0.8, max(vals) * 1.5)))
    else:
        ax.set_ylim(0, 44 if archived else max(vals) * 1.15)
    ax.set_xticks(range(len(qs)))
    ax.set_xticklabels([f"Q{q}" for q in qs])
    ax.set_xlim(-0.65, len(qs) - 0.35)
    ax.set_ylabel("runtime (s)")
    bare(ax)
    handles, labels = ax.get_legend_handles_labels()
    if any(imp[arm] for arm in ARMS):
        handles.append(Patch(facecolor="white", edgecolor=INK3, hatch="////",
                             linewidth=0.6))
        labels.append("imputed (OOM), 1$\\sigma$")
    ax.legend(handles, labels, ncol=5, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.17), handlelength=1.3, columnspacing=1.4)
    save(fig, "fig1_runtime_log" if logy else "fig1_runtime")


def fig_totals(a, imp, qs, archived):
    """Totals over ``qs``, which every arm completed or has imputed."""
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    tots, los, his, geos = [], [], [], []
    for arm in ARMS:
        src = dict(a[arm])
        src.update({q: v[0] for q, v in imp[arm].items()})
        t = sum(src[q] for q in qs)
        tots.append(t)
        geos.append(gmean([a["a0"][q] / src[q] for q in qs]))
        if arm == "a0" or not imp[arm]:
            los.append(0.0)
            his.append(0.0)
        else:
            fixed = sum(src[q] for q in qs if q not in imp[arm])
            lo = fixed + sum(v[0] / math.exp(v[1]) for v in imp[arm].values())
            hi = fixed + sum(v[0] * math.exp(v[1]) for v in imp[arm].values())
            los.append(t - lo)
            his.append(hi - t)
    x = range(len(ARMS))
    ax.bar(x, tots, 0.56, color=[COL[k] for k in ARMS], linewidth=0)
    ax.errorbar(x, tots, yerr=[los, his], fmt="none", ecolor=INK2,
                elinewidth=0.9, capsize=2.2, capthick=0.9)
    top = max(t + h for t, h in zip(tots, his))
    pad = 4 if archived else top * 0.02
    for i, (t, h) in enumerate(zip(tots, his)):
        ax.text(i, t + h + pad, f"{t:.1f}" if top >= 10 else f"{t:.3f}",
                ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([
        f"A$_{i}$\n" + ("baseline" if ARMS[i] == "a0" else f"{geos[i]:.3f}$\\times$")
        for i in range(len(ARMS))])
    ax.set_ylabel(f"total runtime, {len(qs)} queries (s)")
    ax.set_ylim(0, 240 if archived else top * 1.2)
    bare(ax)
    save(fig, "fig2_totals")


BAND = 1.3      # accuracy band, as a factor of the parity line: within 30%


def fig_memory(qs):
    """Predicted vs observed transient, one point per query.

    Single series on purpose: tick-pipe and gather-window sizing are two planes
    of the SAME envelope model, and the question this figure answers -- does
    the envelope track the transient it is supposed to bound? -- is asked of
    the model as a whole.  A second marker/colour would encode a split the
    reader is not being asked to make anything of.

    x is the largest transient OBSERVED, and y is the prediction paired with that
    same sample -- NOT the largest prediction, which is a different sample and
    lives in the CSV's own `max_predicted_mb` column (it differs for 22 of the 35
    plane-rows here).  Selecting on the observation keeps the x-axis a physical
    measurement and leaves y free to be right or wrong about it; selecting on the
    error, or on the prediction, would build the answer into the axis.

    The shaded band is the accuracy band: predictions within 30% of the observed
    transient.  It replaces the per-point breach circles, which marked a
    different quantity (any breach anywhere in the run) at the coordinates of the
    peak sample, and so could not be read off the axes.  Everything else is
    deliberately unlabelled -- the band factor and the sample count belong in the
    caption, not on the canvas.
    """
    # One point per QUERY.  The CSV carries a row per (query, plane) and each row
    # is already the plane's largest-OBSERVED sample -- so keeping the larger-
    # observed of a query's two rows extends that same reduction by one level:
    # the biggest single transient the query ever demanded, paired with what the
    # envelope predicted for that very sample.  (Not a sum: the two planes' peaks
    # need not co-occur, so adding them would plot a joint peak that never
    # happened.  In this run the tick-pipe plane dominates in all 14 two-plane
    # queries, so the merge keeps its row every time.)
    best = {}
    with open(os.path.join(DATA, "ablation_memory_model.csv")) as f:
        for r in csv.DictReader(f):
            if not r["peak_observed_mb"]:
                continue        # a query whose sizing plane took no sample
            q, obs = int(r["query"]), float(r["peak_observed_mb"])
            if q not in best or obs > best[q]["obs"]:
                best[q] = {"q": q, "obs": obs,
                           "pred": float(r["predicted_at_peak_mb"])}
    rows = [best[q] for q in sorted(best)]
    if not rows:
        print("  skipping fig3 (no sizing sample in ablation_memory_model.csv)")
        return

    fig, ax = plt.subplots(figsize=(4.3, 3.7))
    vals = [v for d in rows for v in (d["obs"], d["pred"]) if v > 0]
    lo, hi = min(vals) * 0.7, max(vals) * 1.45

    # Band and parity line, both recessive: they are the reference the points are
    # read against, not marks competing with them.
    ax.fill_between([lo, hi], [lo / BAND, hi / BAND], [lo * BAND, hi * BAND],
                    color=INK3, alpha=0.075, linewidth=0, zorder=0)
    for e in (BAND, 1 / BAND):  # a wash alone has no readable boundary
        ax.plot([lo, hi], [lo * e, hi * e], ls=":", lw=0.8, color=AXIS, zorder=1)
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.9, color=INK3, zorder=1)
    ax.scatter([d["obs"] for d in rows], [d["pred"] for d in rows], s=28, marker="o",
               facecolor=S1, edgecolor="white", linewidth=0.7, zorder=3)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    # Per-query merging leaves ~0.5 decade of range, where matplotlib's default
    # log ticker falls back to minor ticks typeset as "4 x 10^3".  Place plain
    # 1-2-5(-7.5) ticks instead, thinning the mantissa set until <= 6 survive.
    for mant in ([1, 1.5, 2, 3, 5, 7.5], [1, 2, 5], [1]):
        ticks = [m * 10 ** k for k in range(1, 9) for m in mant
                 if lo <= m * 10 ** k <= hi]
        if len(ticks) <= 6:
            break
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("observed peak transient (MB)")
    ax.set_ylabel("predicted transient at that sample (MB)")

    # The only text on the canvas: which side of the parity line means what.
    # Each sits in the empty half of its own region, clear of the bands.
    ax.text(0.055, 0.62, "over-predicted", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.5, color=INK2)
    ax.text(0.945, 0.34, "under-predicted", transform=ax.transAxes,
            ha="right", va="center", fontsize=7.5, color=INK2)

    bare(ax, ygrid=False)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    save(fig, "fig3_memory_model")


def fig_memory_outliers():
    """One bar: where the envelope lands in the converged state.

    Reads the AMORT row of ablation_memory_converged.csv -- the harness owns the
    convergence rule (`_converged_ratios` in ablation.py) so it is defined
    once, and this script stays a consumer of CSVs like every other figure here.
    AMORT, not POOLED: each query is one unit of evidence regardless of how many
    ticks it ran, since the highest-count queries are also the most over-padded.
    """
    path = os.path.join(DATA, "ablation_memory_converged.csv")
    summary = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["query"] in ("AMORT", "POOLED"):
                summary[r["query"]] = r
    row = summary.get("AMORT")
    pctg = int(round((BAND - 1) * 100))
    if row is None or f"within_pct_{pctg}" not in row:
        print(f"  skipping fig3b (no AMORT row at the {pctg}% band in {path})")
        return
    under = float(row[f"under_pct_{pctg}"])
    over = float(row[f"over_pct_{pctg}"])
    within = 100 - under - over

    fig, ax = plt.subplots(figsize=(5.4, 1.15))
    segs = [(under, S2, "under-predicted"), (within, BASE, "within band"),
            (over, S1, "over-predicted")]
    left = 0.0
    for v, c, lab in segs:
        ax.barh(0, v, left=left, height=0.5, color=c, linewidth=0)
        if v > 6:       # a sliver holds neither its value nor its name
            ax.text(left + v / 2, 0, f"{v:.0f}%", ha="center", va="center",
                    color="white", fontsize=8)
            ax.text(left + v / 2, -0.46, lab, ha="center", va="top",
                    fontsize=7.5, color=INK2)
        else:
            # A sliver's label goes on its own row below, value folded in --
            # centred under a 0.6% segment it would collide with its neighbour.
            ax.text(left, -0.95, f"{lab} {v:.1f}%", ha="left", va="top",
                    fontsize=7.5, color=INK2)
        left += v
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.6, 0.4)
    ax.axis("off")
    save(fig, "fig3b_memory_model_outliers")


# arm -> colour, keyed by the CSV's own arm names (the ablation ladder rungs).
# Colour follows the entity across every figure: the +sizing arm is S1 wherever
# it appears, the +dataflow arm S2 -- never reassigned by rank within a chart.
BC = {"baseline": BASE, "sizing": S1, "dataflow": S2, "adaptive": S3}


GB = float(1 << 30)


def fig_buffer():
    """The buffer-strategy leg in bytes: A1 (+sizing, LRU) -> A2 (+dataflow).

    Both arms hold sizing_search=mid_stream, fixed BITCOMP, both compression
    searches off, and the 25% write-through mirror on; the eviction policy is
    the ONLY difference, so this pair isolates dataflow on top of LRU.

    (a) `pcie_h2d_avoided_bytes` (the movement ledger's `cache_hit` wire
        bytes) -- bytes a consumer read out of the GPU-side buffer instead of
        re-fetching over PCIe.  Supersedes the earlier hit-rate view: a rate
        weights a 4 KB page and a 2 GB column equally, so a strategy can raise
        the rate while moving fewer bytes.  Q1 is exactly that case (rate
        0.05 -> 0.12, but 2.4 -> 10.2 GB).
    (b) `mirror_critical_spill_avoided_bytes` (the ledger's `avoided_clean`
        wire bytes) -- bytes an eviction was able to DROP instead of spilling
        D2H, because cleaning (the 25% eviction frontier) had already copied
        it home.  Attribution is unambiguous here: the share is fixed at 25%
        (the 2026-07-29 CSV's wt_frontier_percent column, since dropped, read
        25 on all 78 rows), and the engine has no other source of clean
        copies.

    Both columns keep their values across the move to the ledger.  The pooled
    byte hit rate in (a)'s note does not: its `h2d_miss_bytes` is the ledger's
    `restore_h2d` + `restore_evicted_h2d`, every intermediate read back over
    PCIe whether or not it was offered to the buffer, where the 2026-07-29 CSV
    counted only pages offered
    to it.  On a CSV written since, the rate is lower for the same run, and the
    caption has to say what the misses count.

    Shared y (GB) by request -- the panels are the same unit, and the point of
    putting them on one scale is that under LRU the mirror rescues MORE bytes
    than the buffer serves (144 vs 106 GB); under dataflow it rescues far
    fewer, because dataflow evicts less and evicts better.
    """
    d = defaultdict(dict)
    with open(os.path.join(DATA, "ablation_buffer_manager.csv")) as f:
        for r in csv.DictReader(f):
            d[r["arm"]][int(r["query"])] = r
    qs = sorted(set(d["sizing"]) & set(d["dataflow"]))
    leg = [("sizing", "A$_1$  LRU"), ("dataflow", "A$_2$  +dataflow")]
    # Two standalone figures, no prose: the explanation belongs in the paper caption.
    # They keep IDENTICAL y limits and ticks so the pair still reads on one scale --
    # that comparability was the point of the shared axis before the split.  Note the
    # good direction is opposite in each (above 1.0 in (a), below 1.0 in (b)); with
    # the words gone, the caption is the only place that can say so.
    panels = [
        ("fig4a_buffer_served", "pcie_h2d_avoided_bytes",
         "buffer bytes served $\\div$ A$_1$ LRU",
         [(1, "left", -7, 3.5), (21, "right", 9, 3.5), (7, "right", 10, 15)]),
        ("fig4b_mirror_avoided", "mirror_critical_spill_avoided_bytes",
         "mirror-avoided bytes $\\div$ A$_1$ LRU",
         [(21, "right", 9, 3.5)]),
    ]
    # Byte-weighted hit rate, pooled over the 19 queries: hit bytes / (hit + miss
    # bytes).  NOT the CSV's own `hit_rate`, which counts requests and so weights a
    # 4 KB page like a 2 GB column -- the two disagree on the SIGN of Q7 and rate Q1
    # at 0.12 where the byte view says dataflow served literally all of it.  The
    # misses are every intermediate read back over PCIe (see the docstring).
    hr = {}
    for arm, _ in leg:
        hb = sum(float(d[arm][q]["pcie_h2d_avoided_bytes"]) for q in qs)
        mb = sum(float(d[arm][q]["h2d_miss_bytes"]) for q in qs)
        hr[arm] = hb / (hb + mb) if hb + mb else 0.0

    for name, col, ylab, marks in panels:
        fig, ax = plt.subplots(figsize=(4.1, 2.55))
        v = {arm: [float(d[arm][q][col]) / GB for q in qs] for arm, _ in leg}
        # ratio to LRU; None where LRU itself moved nothing (0/0 is not a ratio)
        rel = [(b / a if a > 0 else None) for a, b in zip(v["sizing"], v["dataflow"])]
        ax.axhline(1.0, color=BASE, ls="--", lw=0.9, zorder=1)
        ax.bar([j for j, r in enumerate(rel) if r is not None],
               [r for r in rel if r is not None], 0.62,
               color=BC["dataflow"], linewidth=0, zorder=2)
        # A zero-height bar is invisible, and "exactly zero" must not read the same as
        # "no measurement".  Both markers sit ON the axis: a square stub above it just
        # reads as a short bar, i.e. as a small nonzero value.
        ax.scatter([j for j, r in enumerate(rel) if r == 0], [0.03] * sum(r == 0 for r in rel),
                   s=6, marker="o", color=BC["dataflow"], zorder=3, clip_on=False)
        ax.scatter([j for j, r in enumerate(rel) if r is None], [0.03] * rel.count(None),
                   s=9, marker="x", color=INK3, linewidth=0.7, zorder=3, clip_on=False)
        ta, tb = sum(v["sizing"]), sum(v["dataflow"])
        both = [r for r in rel if r]
        rate = (f"\nhit rate {hr['sizing']:.0%} $\\rightarrow$ {hr['dataflow']:.0%}"
                if col == "pcie_h2d_avoided_bytes" else "")
        geo = f"{gmean(both):.2f}$\\times$" if both else "no"
        ax.text(0.985, 0.975,
                f"{ta:.0f} $\\rightarrow$ {tb:.0f} GB\n"
                f"{geo} geo. mean" + rate,
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                color=INK2, linespacing=1.5)
        for q, ha, dx, dy in marks:      # selective labels: only what carries the point
            if q not in qs or rel[qs.index(q)] is None:
                continue                 # a run without that query's ratio
            j = qs.index(q)
            ax.annotate(f"Q{q}  {rel[j]:.2f}$\\times$", (j, rel[j]),
                        textcoords="offset points", xytext=(dx, dy), ha=ha,
                        fontsize=6.4, color=BC["dataflow"])
        # bare query numbers, not "Q<n>": 19 groups leaves ~0.18 in per label, and the
        # "Q" prefix is exactly what makes them collide
        ax.set_xticks(range(len(qs)))
        ax.set_xticklabels([str(q) for q in qs], fontsize=6.6)
        ax.set_xlim(-0.7, len(qs) - 0.3)
        ax.set_ylim(0, 4.6)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["0", "1.0", "2.0", "3.0", "4.0"])
        ax.set_ylabel(ylab)
        bare(ax)
        save(fig, name)


CODECS = ["BITCOMP", "ANS", "CASCADED", "RAW"]
CC = {"BITCOMP": S1, "ANS": S2, "CASCADED": S3, "RAW": BASE}


def fig_compression(a, archived):
    """Read-leg codec picks per query (top) beside A3's runtime change
    against A2 (bottom).

    On a directory ablation.py writes, the panels come from different
    invocations: the runtimes from the timed runs, the picks from
    ablation_compression.csv, which it extracts from the untimed run with the
    movement ledger on.  The compression search decides from measured timing
    and the ledger adds host time, so the timed run may have picked
    differently (identical SF1 runs already disagree on 5-11 of 50 searched
    pipes).  The caption has to say so, e.g. "Codec picks are from a separate
    run of A3 with the movement ledger on; the search decides from measured
    timing, so the timed runs may have picked differently."
    """
    rows = list(csv.DictReader(open(os.path.join(DATA, "ablation_compression.csv"))))
    qs = sorted({int(r["query"]) for r in rows}
                & set(a["adaptive"]) & set(a["dataflow"]))
    if not qs:
        print("  skipping fig5 (no searched pipe in ablation_compression.csv)")
        return
    read, ratio = {}, {}
    for q in qs:
        rs = [r for r in rows if int(r["query"]) == q]
        c = Counter()
        for r in rs:
            for alg in (r["read_final_algs"] or "").split("|"):
                if alg:
                    c[alg] += 1
        read[q] = c
        ratio[q] = a["adaptive"][q] / a["dataflow"][q]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.3), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1], "hspace": 0.12})
    x = range(len(qs))
    bottom = [0.0] * len(qs)
    for k in CODECS:
        vals = [read[q].get(k, 0) for q in qs]
        ax1.bar(x, vals, 0.6, bottom=bottom, color=CC[k], edgecolor="white", linewidth=0.9,
                label=k + (" (uncompressed)" if k == "RAW" else ""))
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax1.set_ylabel("read-leg columns")
    ax1.set_ylim(0, max(bottom) + 1.4)
    bare(ax1)
    ax1.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22),
               handlelength=1.3, columnspacing=1.5)

    pct = [(ratio[q] - 1) * 100 for q in qs]
    ax2.bar(x, pct, 0.6, color=[S1 if v < 0 else CRIT for v in pct], linewidth=0)
    ax2.axhline(0, color=AXIS, lw=0.9)
    ax2.set_ylabel("A$_3$ vs A$_2$ runtime")
    if archived:
        ax2.set_yticks([-40, -20, 0])
        ax2.set_yticklabels(["$-$40%", "$-$20%", "0"])
        ax2.set_ylim(min(pct) * 1.2, max(max(pct) * 2.4, 12))
    else:
        ax2.set_ylim(min(min(pct), 0) * 1.2 - 2, max(max(pct) * 1.2, 12))
        ax2.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: "0" if v == 0 else f"{'$-$' if v < 0 else ''}{abs(v):.0f}%"))
    ax2.text(len(qs) - 0.4, 4, "slower", ha="right", fontsize=6.5, color=INK3)
    ax2.text(len(qs) - 0.4, -9, "faster", ha="right", fontsize=6.5, color=INK3)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"Q{q}" for q in qs])
    ax2.set_xlim(-0.7, len(qs) - 0.3)
    bare(ax2)
    save(fig, "fig5_compression")


def have(*names):
    """True when every named CSV is present in DATA.

    Some ablation directories carry only a subset of the CSVs (the 2026-07-31
    sizefix/pin500 run holds the memory model alone), so each figure is gated on
    its own inputs rather than the whole script failing on the first missing one.
    """
    return all(os.path.isfile(os.path.join(DATA, n)) for n in names)


def main():
    if not os.path.isdir(DATA):
        sys.exit(f"no such ablation directory: {DATA}")
    os.makedirs(HERE, exist_ok=True)
    qs = list(range(1, 23))
    print("generating figures in", HERE)
    a = archived = None
    if have("ablation_raw.csv") or have(*ARCHIVED_RUNTIMES):
        a, archived = load()
        imp = {arm: {} for arm in ARMS}
        if archived:
            LBL["a0"] = "A$_0$ static 1024"
            imp.update(impute(a, qs))
        else:
            # The run's own queries; totals compare like with like.
            qs = sorted(set().union(*(a[arm] for arm in ARMS)))
        done = [q for q in qs if all(q in a[arm] or q in imp[arm] for arm in ARMS)]
        if done != qs:
            print(f"  totals leave out {sorted(set(qs) - set(done))}: "
                  f"not completed by every arm")
        fig_runtime(a, imp, qs, archived, logy=False)
        fig_runtime(a, imp, qs, archived, logy=True)
        fig_totals(a, imp, done, archived)
    else:
        print("  skipping fig1/fig2 (ablation_raw.csv absent)")
    if have("ablation_memory_model.csv"):
        fig_memory(qs)
    else:
        print("  skipping fig3 (ablation_memory_model.csv absent)")
    if have("ablation_memory_converged.csv"):
        fig_memory_outliers()
    else:
        print("  skipping fig3b (ablation_memory_converged.csv absent)")
    if have("ablation_buffer_manager.csv"):
        fig_buffer()
    else:
        print("  skipping fig4 (ablation_buffer_manager.csv absent)")
    if a is not None and have("ablation_compression.csv"):
        fig_compression(a, archived)
    else:
        print("  skipping fig5 (ablation_compression.csv or runtimes absent)")


if __name__ == "__main__":
    main()
