#!/usr/bin/env python3
"""fig_intermediates -- the intermediate bytes each GPU engine produces on
every TPC-H query at one scale factor, one ACM text width wide.

    python3 make_intermediates_fig.py --tqp JSONL --polars CSV
        [--spill CSV] [--sf 1000] -> experiments/paper_scripts/results/fig_intermediates.pdf
                       (+ results/intermediates_sf<SF>.csv, the plotted bytes)

An intermediate is what one pipeline hands to the next: the data that is
materialized, on the GPU or wherever the engine puts it, whether or not it is
ever spilled.  Logical (decoded) bytes, per engine:

  PageFlow      each pipe's output once (``born`` in bench/tqp_spill_stats.py's
                records, ``--tqp``)
  Sirius        the batches its non-scan pipelines produce (``materialized``
                in spill_profile.csv; scan batches are its scan)
  cudf-Polars   the chunks a non-scan plan node hands to a join, group-by,
                sort, distinct, shuffle, repartition or the result.  ESTIMATED:
                that share of the node outputs at SF100 (the polars-boundary
                meter), times the SF1000 spill pass's node outputs (``--polars``)
  Spark-RAPIDS  its exchanges' data (``shuffle_d2h`` + ``broadcast``): Spark's
                stage outputs

The baselines' records are baselines/results/spill_profile.csv's
(``TQP_SPILL_CSV`` or ``--spill``).  Panel (b) sums the queries every system
completed.  A query a system did not complete in its spill pass is a white
hatched bar to the top of the axis, labelled OOM, T/O, Disk or FAIL.

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page one text width wide).
"""
import argparse
import csv
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sigmod_style import TEXT, bar_style, save
from make_spill_fig import ENGINES, GB, OURS, QUERIES, RESULTS, SPILL_CSV, failure_word

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "helpers"))
import spill_profile  # noqa: E402

ESTIMATED = "cudf-Polars"
# Each baseline's intermediates: the sum of these kinds' logical bytes.
KINDS = {"sirius": ("materialized",), "rapids": ("shuffle_d2h", "broadcast")}


def load(spill_csv, tqp_file, polars_csv, sf):
    """{name: {query: (GB or None, failure word or None)}} at ``sf``."""
    cells = spill_profile.load(spill_csv)
    out = {}
    for key, kinds in KINDS.items():
        out[ENGINES[key]] = {
            q: ((sum(c["rows"][k]["logical_b"] for k in kinds) / GB, None)
                if c["run_status"] == "OK" else (None, failure_word(c["run_status"])))
            for q, c in cells[(key, sf)].items()}
    est = {}
    with open(polars_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            est[int(r["query"])] = r["sf1000_boundary_est_b"]
    out[ENGINES["polars_gpu"]] = {
        q: ((float(est[q]) / GB, None) if c["run_status"] == "OK"
            else (None, failure_word(c["run_status"])))
        for q, c in cells[("polars_gpu", sf)].items()}
    ours = {}
    with open(tqp_file, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["scale_factor"] != sf:
                raise SystemExit(f"{tqp_file}: a record at SF{r['scale_factor']}, not SF{sf}")
            q = int(r["query"].removeprefix("query"))
            born = {x["kind"]: x["logical_b"] for x in r["rows"]}["born"]
            ours[q] = ((born / GB, None) if r["run_status"] == "OK"
                       else (None, failure_word(r["run_status"], r.get("error"))))
    out[OURS] = ours
    for name, c in out.items():
        if sorted(c) != QUERIES:
            raise SystemExit(f"{name}: records for {sorted(c)}, not Q1-Q22")
    return {n: out[n] for n in ENGINES.values()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spill", default=SPILL_CSV)
    ap.add_argument("--tqp", required=True, help="PageFlow's spill-pass records.jsonl")
    ap.add_argument("--polars", required=True,
                    help="polars_boundary_estimate_sf1000.csv (sf1000_boundary_est_b per query)")
    ap.add_argument("--sf", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_intermediates.pdf"))
    args = ap.parse_args()

    data = load(args.spill, args.tqp, args.polars, args.sf)
    names = list(data)
    common = [q for q in QUERIES if all(data[n][q][1] is None for n in names)]
    totals = {n: sum(data[n][q][0] for q in common) for n in names}

    table = os.path.join(RESULTS, f"intermediates_sf{args.sf}.csv")
    os.makedirs(RESULTS, exist_ok=True)
    with open(table, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "query", "status", "intermediates_gb", "estimated"])
        for n in names:
            for q in QUERIES:
                v, word = data[n][q]
                w.writerow([n, q, word or "OK", "" if v is None else f"{v:.3f}",
                            int(n == ESTIMATED)])
    print(f"SF{args.sf}: -> {table}")
    print(f"totals over the {len(common)} queries every system completed "
          f"(left out: {', '.join(f'Q{q}' for q in QUERIES if q not in common)})")
    for n in names:
        fails = ", ".join(f"Q{q} {data[n][q][1]}" for q in QUERIES if data[n][q][1])
        print(f"  {n:<13} {totals[n]:7.1f} GB  failed: {fails or '-'}")

    peak = max(v for n in names for v, word in data[n].values() if word is None)
    top = (int(peak / 100) + 1) * 100
    tot_top = (int(max(totals.values()) * 1.15 / 500) + 1) * 500

    n, group = len(names), 0.8
    w = group / n
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(TEXT, 2.3), layout="constrained",
                                  gridspec_kw=dict(width_ratios=[22, 3.2]))
    for i, name in enumerate(names):
        st = bar_style(name)
        label = f"{name} (est.)" if name == ESTIMATED else name
        off = (i - (n - 1) / 2) * w
        ok = [q for q in QUERIES if data[name][q][1] is None]
        ax.bar([QUERIES.index(q) + off for q in ok], [data[name][q][0] for q in ok],
               width=w, label=label, **st)
        for q in QUERIES:
            word = data[name][q][1]
            if word is None:
                continue
            x = QUERIES.index(q) + off
            ax.bar(x, top, width=w, facecolor="white", hatch="////",
                   edgecolor=st["facecolor"], linewidth=0.5)
            # Staggered by system, so adjacent failures' labels do not collide.
            ax.text(x, top * (0.22 + 0.2 * i), word, rotation=90, ha="center", va="center",
                    color=st["facecolor"],
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
        tw = 0.8 / n
        x = (i - (n - 1) / 2) * tw
        axt.bar(x, totals[name], width=tw, **{k: v for k, v in st.items() if k != "label"})
        axt.text(x, totals[name] + tot_top * 0.02, f"{totals[name]:.0f}", rotation=90,
                 ha="center", va="bottom")

    ax.set_ylim(0, top)
    ax.set_xlim(-0.6, len(QUERIES) - 0.4)
    ax.set_xticks(range(len(QUERIES)))
    ax.set_xticklabels([f"Q{q}" for q in QUERIES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Intermediates (GB)")
    ax.set_xlabel("(a) Per query\n")

    axt.set_ylim(0, tot_top)
    axt.set_xlim(-0.6, 0.6)
    axt.set_xticks([])
    axt.set_xlabel(f"(b) Total,\n{len(common)} queries")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="black", hatch="////", linewidth=0.5))
    labels.append("Failed")
    fig.legend(handles, labels, ncol=len(labels), loc="outside upper center")
    save(fig, args.out)


if __name__ == "__main__":
    main()
