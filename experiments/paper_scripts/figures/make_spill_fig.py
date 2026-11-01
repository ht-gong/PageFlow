#!/usr/bin/env python3
"""fig_spill -- the bytes each GPU engine moves off the GPU on every TPC-H
query at one scale factor, one ACM text width wide.

    python3 make_spill_fig.py [--spill CSV] [--tqp JSONL ...]
        [--sf 1000] -> experiments/paper_scripts/results/fig_spill.pdf
                       (+ results/spill_comparison_sf<SF>.csv, the plotted bytes)

The bytes are the spill pass's (baselines/AGENTS.md, "Spill pass"): the
baselines' records in baselines/results/spill_profile.csv (``TQP_SPILL_CSV``
or ``--spill`` overrides the path; the worktree has no copy), PageFlow's from
``--tqp`` records.jsonl files of bench/tqp_spill_stats.py, or, without
``--tqp``, its ``tqp_vortex`` rows in the same CSV.  A later ``--tqp`` file
replaces an earlier one's queries (a rerun of one query).

Each bar is a system's logical (decoded) bytes off the GPU on one query,
spill_results.headline()'s ``off_gpu``, split in two: the solid part is
``pressure_spill`` (data evicted because the GPU was full), the light part on
top the rest (moved off by design: RAPIDS' shuffle, PageFlow's pipe outputs
written home).  Panel (b) sums both over the queries every system completed.

``--sirius-split`` (sirius_split_stats.py's per-query JSON, from the quent
traces) keeps only Sirius' spilled intermediate batches: its scan batches'
spill is left out, as every other engine's scan is.

FAILURES ARE MARKED.  A query a system did not complete in its spill pass is a
white hatched bar to the top of the axis in its colour, labelled OOM, T/O,
Disk (killed for its disk use) or FAIL, and is left out of (b) for every
system.

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page one text width wide).
"""
import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sigmod_style import SYSTEMS, TEXT, bar_style, save

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "helpers"))
import spill_profile  # noqa: E402
from spill_results import headline  # noqa: E402

SPILL_CSV = os.environ.get("TQP_SPILL_CSV") or os.path.normpath(os.path.join(
    HERE, os.pardir, os.pardir, os.pardir, "baselines", "results", "spill_profile.csv"))
OURS = "PageFlow"
# spill_profile.csv engine key -> the paper's name, in bar order.
ENGINES = {"tqp_vortex": OURS, "sirius": "Sirius", "polars_gpu": "cudf-Polars",
           "rapids": "Spark-RAPIDS"}
QUERIES = list(range(1, 23))
GB = 1e9


def failure_word(status, error=""):
    e = (error or "").lower()
    if status == "TIMEOUT":
        return "T/O"
    if status == "KILLED_DISK":
        return "Disk"
    if "out of memory" in e or "oom" in e:
        return "OOM"
    return "FAIL"


def cell(h, status, error=""):
    """(pressure GB, design GB, None) of an OK query, else (None, None, word)."""
    if status != "OK":
        return None, None, failure_word(status, error)
    off = h["off_gpu"][0] or 0.0
    pressure = (h["pressure_spill"] or (0.0,))[0] or 0.0
    return pressure / GB, (off - pressure) / GB, None


def load(spill_csv, tqp_files, sf):
    """{name: {query: (pressure GB, design GB, failure word)}} at ``sf``."""
    cells = spill_profile.load(spill_csv)
    heads = spill_profile.headlines(cells)
    out = {}
    for key, name in ENGINES.items():
        if key == "tqp_vortex" and tqp_files:
            continue
        if (key, sf) not in cells:
            raise SystemExit(f"{spill_csv}: no {key} records at SF{sf}"
                             + (" (pass --tqp records.jsonl)" if key == "tqp_vortex" else ""))
        out[name] = {q: cell(heads[(key, sf)][q], c["run_status"])
                     for q, c in cells[(key, sf)].items()}
    if tqp_files:
        ours = {}
        for path in tqp_files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["scale_factor"] != sf:
                        raise SystemExit(f"{path}: a record at SF{r['scale_factor']}, not SF{sf}")
                    q = int(r["query"].removeprefix("query"))
                    ours[q] = cell(headline("tqp_vortex", r["rows"]), r["run_status"],
                                   r.get("error"))
        out[OURS] = ours
    for name, c in out.items():
        if sorted(c) != QUERIES:
            raise SystemExit(f"{name}: records for {sorted(c)}, not Q1-Q22")
    return {n: out[n] for n in ENGINES.values()}


def lighten(hex_color, f=0.55):
    """``hex_color`` mixed with white by ``f``: the by-design part of a bar."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * f):02x}" for c in rgb)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spill", default=SPILL_CSV)
    ap.add_argument("--tqp", nargs="*", default=[],
                    help="PageFlow's records.jsonl files; a later one replaces an earlier one's queries")
    ap.add_argument("--sirius-split", default=None,
                    help="per-query JSON with spill_gpu_host_intermediate_b (Sirius' intermediates only)")
    ap.add_argument("--common-only", action="store_true",
                    help="plot only the queries every system completed (no failure bars)")
    ap.add_argument("--sf", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_spill.pdf"))
    args = ap.parse_args()

    data = load(args.spill, args.tqp, args.sf)
    if args.sirius_split:
        with open(args.sirius_split, encoding="utf-8") as f:
            split = json.load(f)
        sirius = data["Sirius"]
        for q in QUERIES:
            if sirius[q][2] is None:
                r = split[str(q)]
                sirius[q] = ((r.get("spill_gpu_host_intermediate_b", 0)
                              + r.get("spill_gpu_disk_intermediate_b", 0)) / GB, 0.0, None)
    names = list(data)
    common = [q for q in QUERIES if all(data[n][q][2] is None for n in names)]
    totals = {n: (sum(data[n][q][0] for q in common), sum(data[n][q][1] for q in common))
              for n in names}

    table = os.path.join(RESULTS, f"spill_comparison_sf{args.sf}.csv")
    os.makedirs(RESULTS, exist_ok=True)
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "query", "status", "pressure_spill_gb", "by_design_gb"])
        for n in names:
            for q in QUERIES:
                p, d, word = data[n][q]
                w.writerow([n, q, word or "OK", "" if p is None else f"{p:.3f}",
                            "" if d is None else f"{d:.3f}"])
    print(f"SF{args.sf}: baselines from {args.spill}; {OURS} from "
          f"{', '.join(args.tqp) if args.tqp else 'the same CSV'}  -> {table}")
    print(f"totals over the {len(common)} queries every system completed "
          f"(left out: {', '.join(f'Q{q}' for q in QUERIES if q not in common)})")
    for n in names:
        fails = ", ".join(f"Q{q} {data[n][q][2]}" for q in QUERIES if data[n][q][2])
        print(f"  {n:<13} pressure {totals[n][0]:7.1f} GB  by design {totals[n][1]:7.1f} GB"
              f"  failed: {fails or '-'}")

    shown = common if args.common_only else QUERIES
    peak = max(data[n][q][0] + data[n][q][1] for n in names for q in shown
               if data[n][q][2] is None)
    top = (int(peak / 100) + 1) * 100
    tot_peak = max(p + d for p, d in totals.values())
    tot_top = (int(tot_peak * 1.15 / 200) + 1) * 200

    n, group = len(names), 0.8
    w = group / n
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(TEXT, 2.3), layout="constrained",
                                  gridspec_kw=dict(width_ratios=[22, 3.2]))
    for i, name in enumerate(names):
        st = bar_style(name)
        light = dict(st, facecolor=lighten(st["facecolor"]))
        light.pop("label", None)
        off = (i - (n - 1) / 2) * w
        ok = [q for q in shown if data[name][q][2] is None]
        xs = [shown.index(q) + off for q in ok]
        ps = [data[name][q][0] for q in ok]
        ds = [data[name][q][1] for q in ok]
        ax.bar(xs, ps, width=w, label=name, **st)
        ax.bar(xs, ds, bottom=ps, width=w, **light)
        for q in shown:
            word = data[name][q][2]
            if word is None:
                continue
            x = shown.index(q) + off
            ax.bar(x, top, width=w, facecolor="white", hatch="////",
                   edgecolor=st["facecolor"], linewidth=0.5)
            # Staggered by system, so adjacent failures' labels do not collide.
            ax.text(x, top * (0.22 + 0.2 * i), word, rotation=90, ha="center", va="center",
                    color=st["facecolor"],
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        # (b): the totals, each labelled with its sum.
        tw = 0.8 / n
        x = (i - (n - 1) / 2) * tw
        p, d = totals[name]
        axt.bar(x, p, width=tw, **{k: v for k, v in st.items() if k != "label"})
        axt.bar(x, d, bottom=p, width=tw, **light)
        axt.text(x, p + d + tot_top * 0.02, f"{p + d:.0f}", rotation=90, ha="center",
                 va="bottom")

    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([f"Q{q}" for q in shown])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Data off GPU (GB)")
    ax.set_xlabel("(a) Per query\n")

    axt.set_ylim(0, tot_top)
    axt.set_xlim(-0.6, 0.6)
    axt.set_xticks([])
    axt.set_xlabel(f"(b) Total,\n{len(common)} queries")

    handles, labels = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="#7f7f7f", edgecolor="black", linewidth=0.5),
                Patch(facecolor=lighten("#7f7f7f"), edgecolor="black", linewidth=0.5),
                Patch(facecolor="white", edgecolor="black", hatch="////", linewidth=0.5)]
    labels += ["Under pressure", "By design", "Failed"]
    if args.common_only:
        handles, labels = handles[:-1], labels[:-1]
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")
    save(fig, args.out)


if __name__ == "__main__":
    main()
