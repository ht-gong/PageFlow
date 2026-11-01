#!/usr/bin/env python3
"""fig_baselines -- PageFlow's per-query speedup over each baseline at one
scale factor (``--sf``, default SF1000, E2's), one ACM text width wide.

    python3 make_baselines_fig.py [--e2 CSV] [--baselines CSV] [--sf N]
    -> experiments/paper_scripts/results/fig_baselines.pdf

PageFlow's times are E2's full setting at that scale factor (``tqp_adaptive``
in results/ablation_results.csv, e2_end_to_end_ablation.py); the baselines'
are E1's rows at the same scale factor in baselines/results/all_results.csv
(e1_motivation.py: cudf-Polars', Spark-RAPIDS' and DuckDB's parquet rows, and
Sirius's ``sirius_host`` rows, its second build with the query's columns
pinned in its host tier, as in fig_motivation_geomean; ``TQP_BASELINES_CSV``
or ``--baselines`` overrides the path).  None is carried here.

Each bar is a baseline's time over PageFlow's on one query, on a log axis, and
grows from 1 (PageFlow's own time): up where PageFlow is faster, down where it
is slower.  The last group, GeoMean, is each baseline's geometric mean over the
queries both it and PageFlow completed, so a baseline's failures leave only its
own bar's query set smaller (the bar's label then gives its count, 20 queries).

FAILURES ARE MARKED.  A query a baseline failed is a white hatched bar to the
top of the axis in its colour, labelled OOM, T/O or FAIL; it is left out of
that baseline's GeoMean bar only.  A query PageFlow failed has no bars, and a red x
on the 1 line.

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page one text width wide).
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from sigmod_style import FAIL, SYSTEMS, TEXT, bar_style, save

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
BASELINES_CSV = os.environ.get("TQP_BASELINES_CSV") or os.path.join(
    ROOT, "baselines", "results", "all_results.csv")

OURS = "PageFlow"
E2_SETTING = "tqp_adaptive"
SF = 1000   # E2's default scale factor
# all_results.csv engine key -> the paper's name, in bar order.  Sirius is the
# second build, host-pinned (``sirius_host``), as in fig_motivation_geomean.
ENGINES = {"sirius_host": "Sirius", "polars_gpu": "cudf-Polars",
           "rapids": "Spark-RAPIDS", "duckdb_cpu": "DuckDB"}
QUERIES = list(range(1, 23))


def load_e2(path, sfs):
    """{sf: {query: seconds or None}} of E2_SETTING in E2's CSV, at the
    scale factors of ``sfs`` it has rows for: ``mean_s`` when the status is
    OK (every rep OK), else None."""
    cells = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sf = int(r["sf"])
            if r["setting"] != E2_SETTING or sf not in sfs:
                continue
            cell = cells.setdefault(sf, {})
            q = int(r["query"])
            if q in cell:
                raise SystemExit(f"{path}: repeated ({E2_SETTING}, SF{sf}, Q{q}): rows "
                                 "at several chunk sizes or partition budgets")
            cell[q] = float(r["mean_s"]) if r["status"] == "OK" else None
    return cells


def failure_word(status, error):
    e = error.lower()
    if status == "TIMEOUT" or "timeout" in e:
        return "T/O"
    if "oom" in e or "out of memory" in e or "pinnedpoolexhausted" in e:
        return "OOM"
    return "FAIL"


def load_baselines(path, sf):
    """{name: {query: (seconds or None, failure word or None)}} at ``sf``."""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f)
        need = {"engine", "scale_factor", "query", "status", "seconds", "rows_or_error"}
        if not need <= set(rows.fieldnames or ()):
            raise SystemExit(f"{path}: header {rows.fieldnames} lacks {sorted(need)}")
        for r in rows:
            if r["engine"] not in ENGINES or int(r["scale_factor"]) != sf:
                continue
            q = int(r["query"].removeprefix("query"))
            cell = out.setdefault(ENGINES[r["engine"]], {})
            if q in cell:
                raise SystemExit(f"{path}: repeated ({r['engine']}, {sf}, {r['query']})")
            cell[q] = ((float(r["seconds"]), None) if r["status"] == "OK"
                       else (None, failure_word(r["status"], r["rows_or_error"])))
    missing = [n for n in ENGINES.values() if n not in out]
    if missing:
        raise SystemExit(f"{path}: no rows at SF{sf} for {missing} (run "
                         f"`e1_motivation.py --engine <engine> --sf {sf}`, or point "
                         "--baselines / TQP_BASELINES_CSV at the all_results.csv "
                         "that holds them)")
    return {n: out[n] for n in ENGINES.values()}


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def ladder(lo, hi):
    """The 1-2-5 values from the largest one <= lo to the smallest one >= hi."""
    vals = [m * 10.0 ** k for k in range(-3, 5) for m in (1, 2, 5)]
    a = max(v for v in vals if v <= lo)
    b = min(v for v in vals if v >= hi)
    return [v for v in vals if a <= v <= b]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--e2", default=os.path.join(RESULTS, "ablation_results.csv"))
    ap.add_argument("--baselines", default=BASELINES_CSV)
    ap.add_argument("--sf", type=int, default=SF,
                    help=f"the scale factor drawn (default SF{SF}, E2's)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_baselines.pdf"))
    args = ap.parse_args()

    sf = args.sf
    ours = load_e2(args.e2, {sf}).get(sf)
    if ours is None:
        raise SystemExit(f"{args.e2}: no {E2_SETTING} rows at SF{sf}")
    base = load_baselines(args.baselines, sf)
    print(f"SF{sf}: {OURS} = {E2_SETTING} in {args.e2}; baselines from {args.baselines}")

    ratio = {n: {q: c[q][0] / ours[q] for q in QUERIES
                 if c.get(q, (None,))[0] is not None and ours.get(q) is not None}
             for n, c in base.items()}
    failed = {n: {q: c[q][1] for q in QUERIES if q in c and c[q][1]} for n, c in base.items()}
    ours_failed = [q for q in QUERIES if ours.get(q) is None]
    gm = {n: geomean(list(ratio[n].values())) for n in base}
    common = [q for q in QUERIES if all(q in r for r in ratio.values())]

    print(f"geomean speedup of {OURS} over the queries both it and the baseline completed")
    for n in base:
        pair = list(ratio[n].values())
        fails = ", ".join(f"Q{q} {w}" for q, w in failed[n].items())
        print(f"  {n:<13} {gm[n]:6.2f}x over {len(pair)} queries   (over the "
              f"{len(common)} every system completed "
              f"{geomean([ratio[n][q] for q in common]):6.2f}x; {OURS} faster on "
              f"{sum(v > 1 for v in pair)})  failed: {fails or '-'}")
    if ours_failed:
        print(f"  {OURS} failed: {', '.join(f'Q{q}' for q in ours_failed)}")

    live = [v for r in ratio.values() for v in r.values()] + list(gm.values())
    ticks = ladder(min(live + [1.0]), max(live + [1.0]) * 1.15)
    lo, top = ticks[0], ticks[-1]

    names = list(base)
    n, group = len(names), 0.8
    w = group / n
    # The GeoMean group is spread wider, so each bar can carry its value.
    gm_step, gm_w = 0.42, 0.3
    gm_x = len(QUERIES) - 1 + 0.6 + (n - 1) / 2 * gm_step + gm_w
    fig, ax = plt.subplots(figsize=(TEXT, 2.3), layout="constrained")
    for i, name in enumerate(names):
        st = bar_style(name)
        off = (i - (n - 1) / 2) * w
        xs = [j + off for j, q in enumerate(QUERIES) if q in ratio[name]]
        hs = [ratio[name][q] for q in QUERIES if q in ratio[name]]
        ax.bar(xs, [h - 1 for h in hs], bottom=1, width=w, label=name, **st)
        x = gm_x + (i - (n - 1) / 2) * gm_step
        ax.bar(x, gm[name] - 1, bottom=1, width=gm_w, **st)
        nq = len(ratio[name])
        ax.text(x, gm[name] * 1.1, f"{gm[name]:.1f}×" + (
                f" ({nq} queries)" if nq < len(QUERIES) else ""), rotation=90, ha="center",
                va="bottom")
        for q, word in failed[name].items():
            x = QUERIES.index(q) + off
            ax.bar(x, top - 1, bottom=1, width=w, facecolor="white", hatch="////",
                   edgecolor=st["facecolor"], linewidth=0.5)
            ax.text(x, top ** 0.72, word, rotation=90, ha="center", va="center",
                    color=st["facecolor"],
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
    for q in ours_failed:
        ax.plot([QUERIES.index(q)], [1], ls="none", marker="x", ms=5, mew=1.2,
                color=FAIL, zorder=5)
    ax.axhline(1, color=SYSTEMS[OURS][0], lw=1.0, zorder=4)
    ax.axvline(len(QUERIES) - 0.4, color="#d0d0d0", lw=0.6)

    ax.set_yscale("log")
    ax.set_ylim(lo, top)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks(list(range(len(QUERIES))) + [gm_x])
    ax.set_xticklabels([f"Q{q}" for q in QUERIES] + ["GeoMean"])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, gm_x + (n - 1) / 2 * gm_step + gm_w)
    ax.set_ylabel(f"{OURS} speedup (×)")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=SYSTEMS[OURS][0], lw=1.0))
    labels.append(f"{OURS} = 1")
    if any(failed.values()):
        handles.append(Patch(facecolor="white", edgecolor="black", hatch="////",
                             linewidth=0.5))
        labels.append("Failed")
    ax.legend(handles, labels, ncol=len(labels), loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    save(fig, args.out)


if __name__ == "__main__":
    main()
