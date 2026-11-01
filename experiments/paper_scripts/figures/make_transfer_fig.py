#!/usr/bin/env python3
"""fig_transfer -- PageFlow's adaptive setting against RAW, as the geomean query
time over the TPC-H suite, one group per GPU configuration; one ACM column wide.

    python3 make_transfer_fig.py [--panel CSV LABEL ...] [--out PDF]
    -> experiments/paper_scripts/results/fig_transfer.pdf

The claim: PageFlow's gain over RAW holds on other hardware and memory sizes.
Each configuration is one CSV in all_results.csv format
(engine,scale_factor,query,status,seconds,rows_or_error) with an engine
``tqp_raw`` row and an engine ``tqp_adaptive`` row per query (either name may
carry a config suffix, e.g. tqp_raw_cmp30_lru20), each with E2's settings of
that name (e2_end_to_end_ablation.py).  No driver on main writes this format:
the CSVs are made per GPU configuration and are never committed (AGENTS.md,
"Results are never committed").  The defaults are the three paths

    results/h100_30gb_raw_adaptive.csv     H100 limited to 30 GB
    results/h100_full_raw_adaptive.csv     H100, whole card
    results/e2_raw_adaptive_all_results.csv  RTX 5090, 32 GB

``--panel CSV LABEL`` (repeatable) replaces the defaults.

Each group has a RAW bar and an adaptive bar: the geometric mean of the query
time over the queries both settings completed in that configuration.  Above
the adaptive bar is RAW's geomean over adaptive's.

FAILURES ARE COUNTED.  A query either setting failed (status not OK) or has no
row for is left out of that configuration's geomean; the group's label then
says how many queries the geomean is over, and stdout names the left-out ones.

Style: sigmod_style.py (upright Linux Biolinum, 9 pt, the page one column wide).
"""
import argparse
import csv
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from sigmod_style import COLUMN, SYSTEMS, bar_style, save

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")

# Engine in the CSV -> (name on the figure, bar style), in bar order.  The
# styles are fig_ablation's for the same two settings.
SETTINGS = {
    "tqp_raw":      ("RAW", bar_style("PageFlow", "#c6dcf5", "////")),
    "tqp_adaptive": ("Adaptive", bar_style("PageFlow", SYSTEMS["PageFlow"][0], "")),
}
BASE, OURS = "tqp_raw", "tqp_adaptive"
PANELS = [
    (os.path.join(RESULTS, "h100_30gb_raw_adaptive.csv"), "H100 limited\nto 30 GB"),
    (os.path.join(RESULTS, "h100_full_raw_adaptive.csv"), "H100\nfull card"),
    (os.path.join(RESULTS, "e2_raw_adaptive_all_results.csv"), "RTX 5090\n32 GB"),
]
QUERIES = range(1, 23)


def setting(engine):
    """The SETTINGS key of a CSV engine: the key itself or the key plus a
    config suffix (tqp_raw_cmp30_lru20 -> tqp_raw); None for other engines."""
    return next((k for k in SETTINGS if engine == k or engine.startswith(k + "_")), None)


def load(path):
    """{setting: {q: (status, seconds or None)}} for the SETTINGS engines."""
    data = {k: {} for k in SETTINGS}
    engines = {k: set() for k in SETTINGS}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k = setting(r["engine"])
            if k is None:
                continue
            engines[k].add(r["engine"])
            q = int(r["query"].removeprefix("query"))
            ok = r["status"] == "OK" and r["seconds"]
            data[k][q] = (r["status"], float(r["seconds"]) if ok else None)
    mixed = {k: sorted(v) for k, v in engines.items() if len(v) > 1}
    if mixed:
        raise SystemExit(f"{path}: more than one engine per setting: {mixed}")
    if not any(data.values()):
        raise SystemExit(f"{path}: no {'/'.join(SETTINGS)} rows yet; paste the timings in")
    return data


def geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def summary(label, data):
    """Print the configuration's geomeans; return (common OK queries, {setting: geomean})."""
    ok = [q for q in QUERIES if all(data[k].get(q, ("", None))[1] for k in SETTINGS)]
    if not ok:
        raise SystemExit(f"{label}: no query both settings completed")
    gm = {k: geomean([data[k][q][1] for q in ok]) for k in SETTINGS}
    print(f"{label.replace(chr(10), ' ')}: geomean over {len(ok)} queries "
          f"RAW {gm[BASE]:.2f} s, adaptive {gm[OURS]:.2f} s ({gm[BASE] / gm[OURS]:.2f}x)")
    for k in SETTINGS:
        out = [f"Q{q} {data[k][q][0] if q in data[k] else 'no row'}"
               for q in QUERIES if not data[k].get(q, ("", None))[1]]
        if out:
            print(f"  {SETTINGS[k][0]} left out: {', '.join(out)}")
    return ok, gm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", nargs=2, action="append", metavar=("CSV", "LABEL"),
                    help="a configuration's CSV and name (repeatable; replaces the two H100 ones)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "fig_transfer.pdf"))
    args = ap.parse_args()
    groups = [(lbl.replace("\\n", "\n"), load(p)) for p, lbl in (args.panel or PANELS)]
    stats = [summary(lbl, d) for lbl, d in groups]

    fig, ax = plt.subplots(figsize=(COLUMN, 1.9), layout="constrained")
    n = len(SETTINGS)
    w = 0.7 / n
    for x, (ok, gm) in enumerate(stats):
        for j, (k, (_, st)) in enumerate(SETTINGS.items()):
            ax.bar(x + (j - (n - 1) / 2) * w, gm[k], width=w, **st)
        ax.text(x + (n - 1) / 2 * w, gm[OURS], f"{gm[BASE] / gm[OURS]:.2f}×",
                ha="center", va="bottom")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([lbl if len(ok) == len(QUERIES) else f"{lbl}\n({len(ok)} queries)"
                        for (lbl, _), (ok, _) in zip(groups, stats)])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, max(g for _, gm in stats for g in gm.values()) * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("Geomean time (s)")
    ax.legend([Patch(**st) for _, st in SETTINGS.values()],
              [name for name, _ in SETTINGS.values()],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n)
    save(fig, args.out)


if __name__ == "__main__":
    main()
