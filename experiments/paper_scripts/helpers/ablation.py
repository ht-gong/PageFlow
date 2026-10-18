#!/usr/bin/env python3
"""
Cumulative ablation study for the paper evaluation (E6).  Runs inside the
``dev`` container; ``e6_ablation.py`` launches it and draws the figures.

Four arms, each adding one mechanism to the one before it:

  A0  baseline    LRU cache      + fixed intermediate sizing + fixed BITCOMP
  A1  +sizing     LRU cache      + ADAPTIVE sizing search    + fixed BITCOMP
  A2  +dataflow   DATAFLOW cache + ADAPTIVE sizing search    + fixed BITCOMP
  A3  +adaptive   DATAFLOW cache + ADAPTIVE sizing search    + adaptive codec
                                                               + read-and-write search

``--slot_pool_gb`` and ``--chunk_size_mb`` are held fixed across the arms;
the chunk defaults to 512, the sizing searches' start, so the sizing axis
measures adaptivity, not a chunk change.

TIMING (AGENTS.md).  Runtimes are ``query_times.csv`` rows from each arm's
timed invocation (``perf/timed/<arm>/``), run with every telemetry switch
and debug validator stripped from the environment.  Each arm is ONE main.py
invocation whose query list is the suite repeated ``--reps`` times; the
median over the reps is reported.  Cross-arm deltas inside the roughly 3%
invocation-to-invocation band are noise.

Results 2-4 come from a separate UNTIMED pass per arm with
``TQP_EXPERIMENT_TELEMETRY=1`` (``perf/arms/<arm>/``): ``sizing_model``,
``compression``, ``peak_gpu_bytes`` and the movement ledger.  Repeated
occurrences overwrite ``SF<sf>_<qid>.json``, so these describe the perf
pass's final rep, whose search choices can differ from the timed run's.

Products (``--out``):
    ablation_raw.csv             every (rep, arm, query) sample      [result 1]
    ablation_medians.csv         median seconds per arm + speedups
    ablation_memory_model.csv    sizing model, predicted vs observed [result 2]
    ablation_memory_converged.csv  the same, converged state only
    ablation_compression.csv     codec picks per searched pipe       [result 3]
    ablation_buffer_manager.csv  per (arm, query) cache and spill
                                 traffic from the movement ledger     [result 4]
    ablation_table.tex, plot/ablation.{png,pdf}, results.json, logs/
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass, asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import perf_json

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)


# ── The ablation ladder ──────────────────────────────────────────────────

@dataclass(frozen=True)
class Arm:
    key: str                 # filename / csv-column safe
    label: str               # short label for plots and tables
    buffer_strategy: str     # --buffer_strategy
    sizing_search: str       # --sizing_search
    compression: str         # --compression
    compression_search: str  # --compression_search

    def flags(self) -> list[str]:
        return [
            "--buffer_strategy", self.buffer_strategy,
            "--sizing_search", self.sizing_search,
            "--compression", self.compression,
            "--compression_search", self.compression_search,
        ]

    def describe(self) -> str:
        return (f"buffer={self.buffer_strategy} sizing={self.sizing_search} "
                f"compression={self.compression} "
                f"search={self.compression_search}")


ARMS: list[Arm] = [
    Arm("baseline", "Baseline",
        buffer_strategy="lru", sizing_search="off",
        compression="BITCOMP", compression_search="off"),
    Arm("sizing", "+ Adaptive sizing",
        buffer_strategy="lru", sizing_search="mid_stream",
        compression="BITCOMP", compression_search="off"),
    Arm("dataflow", "+ Dataflow cache",
        buffer_strategy="dataflow", sizing_search="mid_stream",
        compression="BITCOMP", compression_search="off"),
    Arm("adaptive", "+ Adaptive compression",
        buffer_strategy="dataflow", sizing_search="mid_stream",
        compression="adaptive", compression_search="read-and-write"),
]

# Perf extraction reads the UNSUFFIXED SF<sf>_<qid>.json (the final rep's
# report — repeated occurrences overwrite it).  An ambient
# TQP_PERF_PER_OCCURRENCE would suffix every report and leave nothing at
# the extracted path, so strip it.  The telemetry switch and the debug
# validators are stripped too, so the timed runs run with all of them off.
# The untimed perf passes then set the experiment switch explicitly.
SWITCH_ENV_KEYS = ("TQP_EXPERIMENT_TELEMETRY", "TQP_PICK_DEBUG", "TQP_VARCHK",
                   "TQP_TIMED_GC_VALIDATE")
HARNESS_ENV_STRIP = ("TQP_PERF_PER_OCCURRENCE",) + SWITCH_ENV_KEYS


def harness_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment of an arm's main.py: the ambient one minus
    ``HARNESS_ENV_STRIP``, plus ``extra``."""
    env = {k: v for k, v in os.environ.items() if k not in HARNESS_ENV_STRIP}
    env.update(extra or {})
    return env


# ── Running one arm ──────────────────────────────────────────────────────

def _query_times_rows(perf_dir: str) -> list[tuple[int, float]]:
    """Every (qid, seconds) row of the query_times.csv main.py wrote into
    ``perf_dir``; ``run_arm`` clears the file before each launch, so these
    are this invocation's rows only."""
    path = os.path.join(perf_dir, "query_times.csv")
    if not os.path.exists(path):
        return []
    rows: list[tuple[int, float]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            qid_s, _, secs_s = line.partition(",")
            try:
                rows.append((int(qid_s), float(secs_s)))
            except ValueError:
                continue
    return rows


def run_arm(
    arm: Arm,
    *,
    sf: int,
    queries: list[int],
    reps: int,
    slot_pool_gb: float,
    chunk_size_mb: int,
    extra_flags: list[str],
    perf_dir: str,
    log_path: str,
    dry_run: bool,
    telemetry: bool,
    env: dict[str, str] | None = None,
) -> tuple[dict[int, list[float]], list[int]]:
    """Run one arm as a SINGLE main.py invocation (one host pin) whose query
    list is `queries` repeated `reps` times back to back, so occurrence k of
    a qid is its rep-k sample.  It writes its perf reports and its own
    query_times.csv into `perf_dir`, after removing the ones an earlier run
    left there (a failed launch therefore yields no rows); `telemetry` sets
    TQP_EXPERIMENT_TELEMETRY=1 (the movement ledger), every other telemetry
    and debug switch stays off.

    Returns (times_by_qid -> per-rep list in occurrence order, failed_qids).
    Timing is `perf_dir`'s query_times.csv -- never wall clock (AGENTS.md
    timing rule).
    """
    qids = [q for _ in range(reps) for q in queries]
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "main.py"),
        "--SF", str(sf),
        "--q", *[str(q) for q in qids],
        "--slot_pool_gb", str(slot_pool_gb),
        "--chunk_size_mb", str(chunk_size_mb),
        "--no_debug",
        *arm.flags(),
        *extra_flags,
        "--perf_report", perf_dir,
    ]
    run_env = {"TQP_EXPERIMENT_TELEMETRY": "1"} if telemetry else {}
    run_env.update(env or {})

    if dry_run:
        env_s = " ".join(f"{k}={v}" for k, v in run_env.items())
        print("    " + (env_s + " " if env_s else "") + " ".join(cmd))
        return {}, []

    os.makedirs(perf_dir, exist_ok=True)
    perf_json.clear_run_outputs(perf_dir)
    # Arms run pure engine defaults for the mirror machinery and no ambient
    # telemetry or debug switch: strip them, then overlay this run's env.
    sub_env = harness_env(run_env)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, stdout=log,
            stderr=subprocess.STDOUT, text=True, env=sub_env,
        )
    if proc.returncode != 0:
        print(f"    WARNING: main.py exited {proc.returncode} "
              f"(log: {log_path})", file=sys.stderr)

    # Queries the engine reported as failed; it prints [QFAIL] and continues.
    failed: list[int] = []
    with open(log_path, encoding="utf-8") as log:
        for line in log:
            if line.startswith("[QFAIL]"):
                token = line.split()[1].lstrip("Q").rstrip(":")
                try:
                    failed.append(int(token))
                except ValueError:
                    pass

    # Rows land in execution order, so a qid's k-th row is its rep-k sample
    # (a [QFAIL]ed occurrence writes no row and shortens that qid's list;
    # which rep failed is not recoverable, but the median doesn't need it).
    times: dict[int, list[float]] = {}
    for qid, secs in _query_times_rows(perf_dir):
        times.setdefault(qid, []).append(secs)

    short = [q for q in queries if len(times.get(q, [])) < reps]
    if short:
        print(f"    note: fewer than {reps} timing rows for "
              f"{['Q%d' % q for q in short]}"
              f"{' (reported [QFAIL])' if failed else ''}")
    return times, failed


# ── Aggregation ──────────────────────────────────────────────────────────

def median_of(samples: list[float]) -> float | None:
    vals = [s for s in samples if s is not None and s > 0]
    return statistics.median(vals) if vals else None


def geomean_speedup(base: list[float | None],
                    arm: list[float | None]) -> float | None:
    """Geometric mean of base/arm over the queries where both are present."""
    ratios = [b / a for b, a in zip(base, arm)
              if b is not None and a is not None and b > 0 and a > 0]
    if not ratios:
        return None
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios))


# ── Reporting ────────────────────────────────────────────────────────────

def print_variance_report(queries, arms, samples):
    """Per (arm, query): the raw reps, their median, and relative spread.

    This is the verification surface the run is judged on -- a cell whose
    spread is comparable to the effect being measured cannot support a claim.
    Reps share one invocation, so this spread does NOT include the +-3%
    cross-invocation offset (see PROTOCOL in the module docstring).
    """
    print(f"\n{'=' * 78}")
    print("  Per-query variance across in-invocation reps (verification)")
    print(f"{'=' * 78}")
    high = []
    for arm in arms:
        print(f"\n  {arm.label}  [{arm.describe()}]")
        print(f"    {'Query':<8}{'median':>10}{'min':>10}{'max':>10}"
              f"{'spread':>10}   runs")
        print("    " + "-" * 66)
        for q in queries:
            vals = [v for v in samples[arm.key][q] if v is not None]
            if not vals:
                print(f"    Q{q:<7}{'--':>10}{'--':>10}{'--':>10}"
                      f"{'--':>10}   (no samples)")
                continue
            med = statistics.median(vals)
            spread = (max(vals) - min(vals)) / med if med > 0 else 0.0
            if spread > 0.05 and len(vals) > 1:
                high.append((arm.key, q, spread))
            runs = ", ".join(f"{v:.4f}" for v in vals)
            print(f"    Q{q:<7}{med:>10.4f}{min(vals):>10.4f}{max(vals):>10.4f}"
                  f"{spread * 100:>9.1f}%   {runs}")

    if high:
        print(f"\n  ⚠  {len(high)} (arm, query) cells vary >5% across reps:")
        for key, q, spread in sorted(high, key=lambda t: -t[2])[:15]:
            print(f"       {key:<12} Q{q:<4} {spread * 100:.1f}%")
        print("     Treat differences smaller than these spreads as noise.")
    else:
        print("\n  All cells vary <5% across reps.")


def print_summary(queries, arms, medians, totals):
    print(f"\n{'=' * 78}")
    print("  Ablation summary (median over reps, seconds)")
    print(f"{'=' * 78}\n")
    w = max(16, max(len(a.label) for a in arms) + 2)
    header = f"{'Query':<8}" + "".join(f"{a.label:>{w}}" for a in arms)
    print(header)
    print("-" * len(header))
    for i, q in enumerate(queries):
        row = f"Q{q:<7}"
        for a in arms:
            v = medians[a.key][i]
            row += f"{v:>{w}.4f}" if v is not None else f"{'--':>{w}}"
        print(row)
    print("-" * len(header))
    row = f"{'Total':<8}"
    for a in arms:
        row += f"{totals[a.key]:>{w}.4f}"
    print(row)

    base = arms[0]
    print(f"\n  Speedup vs {base.label} (geometric mean over queries):")
    for a in arms:
        gm = geomean_speedup(medians[base.key], medians[a.key])
        tot = totals[base.key] / totals[a.key] if totals[a.key] > 0 else 0.0
        marker = "  (baseline)" if a.key == base.key else ""
        gm_s = f"{gm:.4f}x" if gm else "--"
        print(f"    {a.label:<26} geomean {gm_s:>9}   "
              f"total {tot:.4f}x{marker}")


# ── Products ─────────────────────────────────────────────────────────────

def write_raw_csv(path, arms, samples, queries, reps):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rep", "arm", "arm_label", "query", "seconds"])
        for r in range(reps):
            for a in arms:
                for q in queries:
                    v = samples[a.key][q][r]
                    w.writerow([r + 1, a.key, a.label, q,
                                f"{v:.6f}" if v is not None else ""])
    print(f"  raw samples  -> {path}")


def write_medians_csv(path, arms, medians, totals, queries):
    base = arms[0]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query"] + [a.key for a in arms]
                   + [f"speedup_{a.key}_vs_{base.key}" for a in arms[1:]])
        for i, q in enumerate(queries):
            row = [q]
            for a in arms:
                v = medians[a.key][i]
                row.append(f"{v:.6f}" if v is not None else "")
            b = medians[base.key][i]
            for a in arms[1:]:
                v = medians[a.key][i]
                row.append(f"{b / v:.4f}"
                           if (b and v and v > 0) else "")
            w.writerow(row)
        # Totals + geomean rows
        w.writerow(["TOTAL"] + [f"{totals[a.key]:.6f}" for a in arms]
                   + [f"{totals[base.key] / totals[a.key]:.4f}"
                      if totals[a.key] > 0 else "" for a in arms[1:]])
        w.writerow(["GEOMEAN_SPEEDUP"] + [""] * len(arms)
                   + [f"{geomean_speedup(medians[base.key], medians[a.key]):.4f}"
                      if geomean_speedup(medians[base.key], medians[a.key])
                      else "" for a in arms[1:]])
    print(f"  medians      -> {path}")


# ── Perf-JSON extraction: results 2-4 ────────────────────────────────────
#
# Sources are the per-query perf reports the untimed passes wrote with
# TQP_EXPERIMENT_TELEMETRY=1, perf/arms/<arm>/SF<sf>_<qid>.json (the final
# rep: each occurrence overwrote the file), read through perf_json.

MB = 1024 ** 2


def load_perf(perf_dir: str, sf: int, qid: int):
    """One query's report, read through perf_json.  A missing report (the
    query failed) skips its row with a warning; a report written without
    TQP_EXPERIMENT_TELEMETRY=1 raises."""
    try:
        return perf_json.load(perf_dir, sf, qid, movement=True)
    except FileNotFoundError:
        print(f"    WARNING: missing perf json "
              f"{perf_json.report_path(perf_dir, sf, qid)}", file=sys.stderr)
        return None


# Columns of the buffer study (result 4), all wire bytes or pages of the
# movement ledger.  A hit is a read served from a cache slot (cache_hit), a
# miss an intermediate read back H2D (MISS_KINDS: an uncached region's runs,
# restore_h2d, and a cached region's home pieces, restore_evicted_h2d), a
# compulsory read a base-table page (scan_h2d).
BUFFER_COLUMNS = [
    "hit_rate", "pcache_hits", "h2d_misses", "compulsory",
    "pcie_h2d_avoided_bytes", "h2d_miss_bytes", "compulsory_bytes",
    # spill written D2H: eviction write-backs, the overflow, every spill kind
    "spill_evict_d2h_bytes", "spill_overflow_d2h_bytes", "spill_total_bytes",
    # spill avoided: clean evictions (mirrored) and dead drops
    "mirror_critical_spill_avoided_bytes", "deaddrop_spill_avoided_bytes",
    "wt_mirror_bytes", "peak_gpu_bytes",
]


MISS_KINDS = ("restore_h2d", "restore_evicted_h2d")


def _hit_rate(hits, misses):
    denom = hits + misses
    return f"{hits / denom:.4f}" if denom else ""


def buffer_values(perf: dict) -> list:
    """One buffer-study row (``BUFFER_COLUMNS``) from a report."""
    mv = perf_json.mv
    hits = mv(perf, "cache_hit", "pages")
    misses = perf_json.mv_sum(perf, MISS_KINDS, "pages")
    return [
        _hit_rate(hits, misses),
        hits, misses, mv(perf, "scan_h2d", "pages"),
        mv(perf, "cache_hit", "wire_b"),
        perf_json.mv_sum(perf, MISS_KINDS, "wire_b"),
        mv(perf, "scan_h2d", "wire_b"),
        mv(perf, "spill_evict", "wire_b"),
        perf_json.mv_sum(perf, ("spill_bypass", "spill_rejected"), "wire_b"),
        perf_json.spill_total(perf),
        mv(perf, "avoided_clean", "wire_b"),
        mv(perf, "avoided_dead", "wire_b"),
        mv(perf, "spill_mirror", "wire_b"),
        perf_json.peak_gpu_bytes(perf),
    ]


def write_memory_model_csv(path, perf_dir, sf, queries):
    """Result 2: sizing-model accuracy — predicted vs actual transient
    memory, per query (one row per (query, plane); the envelope is per-
    decision, so 'predicted vs actual' is summarised at the peak-observed
    sample plus tightness statistics over all samples)."""
    rows_written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "query", "plane", "n_samples", "n_breaches", "n_seeded",
            "peak_observed_mb", "predicted_at_peak_mb",
            "max_predicted_mb", "mean_pred_over_obs", "min_pred_over_obs",
            "skipped_mixed", "skipped_partial", "skipped_no_pending",
            "samples_dropped",
            "arena_peak_gpu_mb",
        ])
        for q in queries:
            perf = load_perf(perf_dir, sf, q)
            if perf is None:
                continue
            sm = perf.get("sizing_model") or {}
            samples = sm.get("samples") or []
            ctx = [f"{perf_json.peak_gpu_bytes(perf) / MB:.1f}"]
            by_plane: dict[str, list[dict]] = {}
            for s in samples:
                by_plane.setdefault(s["plane"], []).append(s)
            if not by_plane:
                w.writerow([q, "", 0, 0, 0, "", "", "", "", "",
                            sm.get("skipped_mixed", 0),
                            sm.get("skipped_partial", 0),
                            sm.get("skipped_no_pending", 0),
                            sm.get("dropped", 0), *ctx])
                rows_written += 1
                continue
            for plane in sorted(by_plane):
                ss = by_plane[plane]
                peak = max(ss, key=lambda s: s["observed_b"])
                ratios = [s["predicted_b"] / s["observed_b"]
                          for s in ss if s["observed_b"] > 0]
                w.writerow([
                    q, plane, len(ss),
                    sum(1 for s in ss if s["breach"]),
                    sum(1 for s in ss if s["seeded"]),
                    f"{peak['observed_b'] / MB:.1f}",
                    f"{peak['predicted_b'] / MB:.1f}",
                    f"{max(s['predicted_b'] for s in ss) / MB:.1f}",
                    f"{statistics.mean(ratios):.3f}" if ratios else "",
                    f"{min(ratios):.3f}" if ratios else "",
                    sm.get("skipped_mixed", 0),
                    sm.get("skipped_partial", 0),
                    sm.get("skipped_no_pending", 0),
                    sm.get("dropped", 0), *ctx,
                ])
                rows_written += 1
    print(f"  memory model -> {path} ({rows_written} rows)")
    return rows_written


# Accuracy bands reported by result 2b, as tolerance fractions: a prediction is
# "within" band f when 1/(1+f) <= predicted/observed <= 1+f.  Column suffixes are
# the percentages (10, 20, ...).
SIZING_BANDS = (0.1, 0.2, 0.3, 0.5, 1.0)


def _converged_ratios(samples):
    """predicted/observed over the CONVERGED samples, grouped by (plane, pipe).

    A sizing session's converged state is the trailing run of samples it spent at
    its FINAL size -- everything after the controller's last size change.  The
    search phase is excluded on purpose: while the controller is still moving c
    it probes sizes the envelope has never observed, so its misses there measure
    the search, not the model.

    Seeded predictions are KEPT.  A session that converges early may never see
    two distinct sizes, so it never leaves the seed prior; dropping seeded
    samples would discard most of the converged set and retain only the sessions
    that kept moving, which is the opposite of converged.

    Returns (ratios, n_sessions, n_seeded, n_breach).
    """
    by_pipe = {}
    for s in samples:
        by_pipe.setdefault((s["plane"], s["pipe"]), []).append(s)
    ratios, seeded, breach = [], 0, 0
    for v in by_pipe.values():
        i, final = len(v), v[-1]["c_b"]
        while i > 0 and v[i - 1]["c_b"] == final:
            i -= 1
        for s in v[i:]:
            if s["observed_b"] > 0:
                ratios.append(s["predicted_b"] / s["observed_b"])
                seeded += bool(s["seeded"])
                breach += bool(s["breach"])
    return ratios, len(by_pipe), seeded, breach


def write_memory_converged_csv(path, perf_dir, sf, queries):
    """Result 2b: envelope accuracy in the converged state, per query.

    Result 2 summarises the sizing model at each query's single worst-observed
    sample; this one reports the whole steady-state distribution behind it, which
    is what the envelope is actually judged on.

    Two trailing rows carry the study-level summary.  AMORT averages each share
    over QUERIES, so a query contributing 221 converged decisions and one
    contributing four count equally; POOLED pools the decisions themselves and
    therefore reports mostly on whichever queries ran the most ticks.  Prefer
    AMORT when quoting a single number -- the two differ by 6-12 points, always
    in the same direction, because the highest-count queries are also the most
    over-padded.
    """
    def stat(rs, f, default=""):
        return f"{f(rs):.3f}" if rs else default

    bands = [(int(round(f * 100)), 1.0 + f) for f in SIZING_BANDS]
    head = ["query", "n_sessions", "n_samples", "n_converged", "n_seeded",
            "n_breach", "median", "p25", "p75", "p95", "min", "max"]
    for pctg, _ in bands:
        head += [f"within_pct_{pctg}", f"over_pct_{pctg}", f"under_pct_{pctg}"]

    per_q, rows_written = {}, 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        for q in queries:
            perf = load_perf(perf_dir, sf, q)
            if perf is None:
                continue
            samples = (perf.get("sizing_model") or {}).get("samples") or []
            rs, n_sess, seeded, breach = _converged_ratios(samples)
            if not rs:
                continue
            per_q[q] = rs
            row = [q, n_sess, len(samples), len(rs), seeded, breach,
                   stat(rs, statistics.median),
                   stat(rs, lambda v: float(np.percentile(v, 25))),
                   stat(rs, lambda v: float(np.percentile(v, 75))),
                   stat(rs, lambda v: float(np.percentile(v, 95))),
                   stat(rs, min), stat(rs, max)]
            for _, b in bands:
                row += [f"{100 * sum(1 / b <= r <= b for r in rs) / len(rs):.1f}",
                        f"{100 * sum(r > b for r in rs) / len(rs):.1f}",
                        f"{100 * sum(r < 1 / b for r in rs) / len(rs):.1f}"]
            w.writerow(row)
            rows_written += 1

        if per_q:
            pool = [r for rs in per_q.values() for r in rs]
            amort = lambda fn: sum(100 * sum(fn(r) for r in rs) / len(rs)
                                   for rs in per_q.values()) / len(per_q)
            arow = ["AMORT", len(per_q), "", len(pool), "", "",
                    f"{statistics.median([statistics.median(rs) for rs in per_q.values()]):.3f}",
                    "", "", "", "", ""]
            prow = ["POOLED", len(per_q), "", len(pool), "", "",
                    stat(pool, statistics.median),
                    stat(pool, lambda v: float(np.percentile(v, 25))),
                    stat(pool, lambda v: float(np.percentile(v, 75))),
                    stat(pool, lambda v: float(np.percentile(v, 95))),
                    stat(pool, min), stat(pool, max)]
            for _, b in bands:
                arow += [f"{amort(lambda r, b=b: 1 / b <= r <= b):.1f}",
                         f"{amort(lambda r, b=b: r > b):.1f}",
                         f"{amort(lambda r, b=b: r < 1 / b):.1f}"]
                prow += [f"{100 * sum(1 / b <= r <= b for r in pool) / len(pool):.1f}",
                         f"{100 * sum(r > b for r in pool) / len(pool):.1f}",
                         f"{100 * sum(r < 1 / b for r in pool) / len(pool):.1f}"]
            w.writerow(arow)
            w.writerow(prow)
    print(f"  mem converged-> {path} ({rows_written} query rows)")
    return rows_written


def write_compression_csv(path, perf_dir, sf, queries):
    """Result 3: what the compression search chose, per searched pipe.
    Static (non-searched) pipes keep the policy default."""
    rows_written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "query", "pipe", "total_ticks",
            "read_n_cols", "read_k_star", "read_converged", "read_pregated",
            "read_probes", "read_final_algs",
            "write_searched", "write_n_cols", "write_k_star",
            "write_flips_to_raw", "write_mode", "write_policy",
            "write_pregated", "write_benefit_gated", "write_encode_gated",
            "write_probes", "write_final_algs",
        ])
        for q in queries:
            perf = load_perf(perf_dir, sf, q)
            if perf is None:
                continue
            for entry in (perf.get("compression") or {}).get("pipes", []):
                if "error" in entry:
                    w.writerow([q, entry.get("pipe", "?")] + [""] * 17
                               + [f"ERROR {entry['error']}"])
                    rows_written += 1
                    continue
                rd = entry.get("read") or {}
                wr = entry.get("write")
                flips = ""
                if wr and wr.get("k_star") is not None:
                    flips = wr["n_decision_cols"] - wr["k_star"]
                w.writerow([
                    q, entry["pipe"], entry["total_ticks"],
                    rd.get("n_decision_cols"), rd.get("k_star"),
                    rd.get("converged"), rd.get("pregate_skipped"),
                    rd.get("probes_spent"),
                    "|".join(rd["final_input_algs"])
                    if rd.get("final_input_algs") else "",
                    bool(wr and wr.get("searched")),
                    wr.get("n_decision_cols") if wr else "",
                    wr.get("k_star") if wr else "",
                    flips,
                    wr.get("mode") if wr else "",
                    wr.get("write_policy") if wr else "",
                    wr.get("pregate_skipped") if wr else "",
                    wr.get("benefit_gated") if wr else "",
                    wr.get("encode_gated") if wr else "",
                    wr.get("probes_spent") if wr else "",
                    "|".join(wr["final_output_algs"])
                    if wr and wr.get("final_output_algs") else "",
                ])
                rows_written += 1
    print(f"  compression  -> {path} ({rows_written} rows)")
    return rows_written


def write_buffer_manager_csv(path, sf, queries,
                             arm_sources: list[tuple[Arm, str]]):
    """Result 4: the buffer-manager study.  One row per (arm, query);
    ``arm_sources`` is [(arm, perf_dir)].  The buffer_strategy column
    carries the LRU-vs-dataflow contrast."""
    rows_written = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "arm", "buffer_strategy", "sizing_search", "compression",
            "compression_search", "query",
            *BUFFER_COLUMNS,
        ])
        for arm, perf_dir in arm_sources:
            for q in queries:
                perf = load_perf(perf_dir, sf, q)
                if perf is None:
                    continue
                w.writerow([
                    arm.key, arm.buffer_strategy, arm.sizing_search,
                    arm.compression, arm.compression_search, q,
                    *buffer_values(perf),
                ])
                rows_written += 1
    print(f"  buffer study -> {path} ({rows_written} rows)")
    return rows_written


def perf_sources(perf_home, arms):
    """The buffer study's [(arm, perf_dir)] of a run's perf tree."""
    return [(a, os.path.join(perf_home, "perf", "arms", a.key)) for a in arms]


def adaptive_perf_dir(perf_home, arms):
    """The adaptive arm's perf dir (results 2-3), or None without one."""
    adaptive = next((a for a in arms if a.compression == "adaptive"), None)
    if adaptive is None:
        return None
    return os.path.join(perf_home, "perf", "arms", adaptive.key)


def emit_perf_csvs(out_dir, sf, queries, arm_sources, adaptive_dir):
    """Write results 2-4 from the captured perf JSONs."""
    if adaptive_dir is None:
        print("  note: no adaptive arm ran — skipping memory-model and "
              "compression CSVs")
    else:
        write_memory_model_csv(
            os.path.join(out_dir, "ablation_memory_model.csv"),
            adaptive_dir, sf, queries)
        write_memory_converged_csv(
            os.path.join(out_dir, "ablation_memory_converged.csv"),
            adaptive_dir, sf, queries)
        write_compression_csv(
            os.path.join(out_dir, "ablation_compression.csv"),
            adaptive_dir, sf, queries)
    write_buffer_manager_csv(
        os.path.join(out_dir, "ablation_buffer_manager.csv"),
        sf, queries, arm_sources)


def write_latex(path, arms, medians, totals, queries, sf, reps):
    base = arms[0]
    # Pick a precision that stays informative at any scale factor: SF100/700
    # queries run for seconds, an SF1 shakeout for milliseconds.
    seen = [v for a in arms for v in medians[a.key] if v]
    hi = max(seen) if seen else 1.0
    prec = 2 if hi >= 10 else (3 if hi >= 1 else 4)
    lines = [
        "% Generated by experiments/paper_scripts/helpers/ablation.py",
        "% Timing source: query_times.csv (per-query approx_time), "
        f"median of {reps} back-to-back reps in one invocation per arm.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\begin{tabular}{l" + "r" * len(arms) + "}",
        "    \\toprule",
        "    Query & " + " & ".join(a.label.replace("+", "$+$")
                                    for a in arms) + " \\\\",
        "    \\midrule",
    ]
    for i, q in enumerate(queries):
        cells = []
        for a in arms:
            v = medians[a.key][i]
            cells.append(f"{v:.{prec}f}" if v is not None else "--")
        lines.append(f"    Q{q} & " + " & ".join(cells) + " \\\\")
    lines += [
        "    \\midrule",
        "    Total & " + " & ".join(f"{totals[a.key]:.{prec}f}" for a in arms)
        + " \\\\",
    ]
    gm_cells = ["--"]
    for a in arms[1:]:
        gm = geomean_speedup(medians[base.key], medians[a.key])
        gm_cells.append(f"{gm:.2f}$\\times$" if gm else "--")
    lines += [
        "    Geo-mean speedup & " + " & ".join(gm_cells) + " \\\\",
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{Cumulative ablation at SF{sf}. Each column adds one "
        f"mechanism to the column on its left. Times are per-query "
        f"\\texttt{{approx\\_time}} in seconds, median of {reps} back-to-back "
        f"repetitions within a single engine invocation per configuration; "
        f"speedup is the geometric mean over queries relative to the "
        f"baseline.}}",
        f"  \\label{{tab:ablation-sf{sf}}}",
        "\\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  latex table  -> {path}")


def plot_ablation(path_png, arms, medians, samples, totals, queries, sf, reps):
    """Two panels: per-query grouped bars (with min-max whiskers) on top,
    the cumulative total-runtime ladder underneath."""
    colors = ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"]
    fig, (ax_q, ax_t) = plt.subplots(
        2, 1,
        figsize=(max(11.0, len(queries) * 0.62), 8.6),
        gridspec_kw={"height_ratios": [2.5, 1.0]},
    )

    # ── Panel 1: per query ────────────────────────────────────────────
    x = np.arange(len(queries))
    bar_w = 0.8 / len(arms)
    for i, a in enumerate(arms):
        vals = [medians[a.key][j] if medians[a.key][j] else 0.0
                for j in range(len(queries))]
        # Whiskers span the observed rep-to-rep range.
        lo, hi = [], []
        for q, med in zip(queries, vals):
            obs = [v for v in samples[a.key][q] if v is not None]
            if obs and med > 0:
                lo.append(max(0.0, med - min(obs)))
                hi.append(max(0.0, max(obs) - med))
            else:
                lo.append(0.0)
                hi.append(0.0)
        offset = bar_w * (i - (len(arms) - 1) / 2)
        ax_q.bar(x + offset, vals, bar_w, label=a.label,
                 color=colors[i % len(colors)], edgecolor="black",
                 linewidth=0.3,
                 yerr=[lo, hi] if reps > 1 else None,
                 error_kw={"elinewidth": 0.7, "capsize": 1.5,
                           "ecolor": "#333333"})
    ax_q.set_xticks(x)
    ax_q.set_xticklabels([f"Q{q}" for q in queries], rotation=45)
    ax_q.set_ylabel("Query time (s)")
    ax_q.set_title(
        f"Cumulative ablation — SF{sf} "
        f"(median of {reps} back-to-back rep{'s' if reps > 1 else ''} "
        f"per arm invocation; whiskers = observed range)")
    ax_q.legend(fontsize=9, ncol=len(arms), loc="upper left")
    ax_q.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_q.set_axisbelow(True)

    # ── Panel 2: the ladder ───────────────────────────────────────────
    labels = [a.label for a in arms]
    tot = [totals[a.key] for a in arms]
    bars = ax_t.bar(labels, tot,
                    color=[colors[i % len(colors)] for i in range(len(arms))],
                    edgecolor="black", linewidth=0.4, width=0.55)
    base_total = tot[0]
    hi = max(tot) if tot else 1.0
    prec = 1 if hi >= 10 else (2 if hi >= 1 else 4)
    for i, (b, v) in enumerate(zip(bars, tot)):
        if i == 0:
            tag = "baseline"
        elif v > 0:
            ratio = base_total / v
            tag = (f"{ratio:.2f}× faster" if ratio >= 1.0
                   else f"{1 / ratio:.2f}× slower")
        else:
            tag = "--"
        ax_t.text(b.get_x() + b.get_width() / 2,
                  v + hi * 0.02, f"{v:.{prec}f}s\n{tag}",
                  ha="center", va="bottom", fontsize=9)
    ax_t.set_ylabel("Total time (s)")
    ax_t.set_ylim(0, max(tot) * 1.25 if max(tot) > 0 else 1.0)
    ax_t.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax_t.set_axisbelow(True)
    ax_t.set_title("Cumulative effect on total suite runtime")

    plt.tight_layout()
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    plt.savefig(path_png, dpi=300)
    path_pdf = os.path.splitext(path_png)[0] + ".pdf"
    plt.savefig(path_pdf)
    plt.close()
    print(f"  figure       -> {path_png}")
    print(f"  figure (pdf) -> {path_pdf}")


def emit_products(out_dir, arms, samples, medians, totals, queries, sf, reps):
    """Write every product from already-aggregated numbers."""
    print(f"\n{'=' * 78}")
    print("  Products")
    print(f"{'=' * 78}")
    write_raw_csv(os.path.join(out_dir, "ablation_raw.csv"),
                  arms, samples, queries, reps)
    write_medians_csv(os.path.join(out_dir, "ablation_medians.csv"),
                      arms, medians, totals, queries)
    write_latex(os.path.join(out_dir, "ablation_table.tex"),
                arms, medians, totals, queries, sf, reps)
    plot_ablation(os.path.join(out_dir, "plot", "ablation.png"),
                  arms, medians, samples, totals, queries, sf, reps)


def replot(results_json: str, out_dir: str | None):
    """Regenerate csv/latex/figure from a previous run's results.json.

    Lets the presentation be iterated on without paying for the engine again.
    """
    with open(results_json, encoding="utf-8") as f:
        data = json.load(f)

    arms = [Arm(**a) for a in data["arms"]]
    queries = data["queries"]
    # "rounds" is the pre-back-to-back name for the same per-(arm, query)
    # sample dimension; accept old results.json files.
    reps = data.get("reps", data.get("rounds"))
    samples = {k: {int(q): v for q, v in d.items()}
               for k, d in data["samples"].items()}
    medians = {a.key: [median_of(samples[a.key][q]) for q in queries]
               for a in arms}
    totals = {a.key: sum(v for v in medians[a.key] if v) for a in arms}

    out_dir = out_dir or os.path.dirname(os.path.abspath(results_json))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Replotting {results_json}\n  -> {out_dir}")
    print_summary(queries, arms, medians, totals)
    emit_products(out_dir, arms, samples, medians, totals,
                  queries, data["sf"], reps)
    # Regenerate results 2-4 when the run's perf tree is present (beside the
    # original results.json, not --out).
    perf_home = os.path.dirname(os.path.abspath(results_json))
    if os.path.isdir(os.path.join(perf_home, "perf", "arms")):
        emit_perf_csvs(out_dir, data["sf"], queries,
                       perf_sources(perf_home, arms),
                       adaptive_perf_dir(perf_home, arms))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Cumulative ablation study (compression / buffer "
                    "strategy / intermediate sizing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--SF", type=int, default=100,
                    help="Scale factor to evaluate.")
    ap.add_argument("--q", type=int, nargs="+", default=list(range(1, 23)),
                    help="Query ids.")
    ap.add_argument("--reps", "--rounds", type=int, default=3, dest="reps",
                    help="Back-to-back repetitions of the query list inside "
                         "each arm's single invocation; median is reported. "
                         "(--rounds is accepted as a legacy alias.)")
    ap.add_argument("--slot_pool_gb", type=float, default=4.0,
                    help="Slot pool size, held fixed across all arms.")
    ap.add_argument("--chunk_size_mb", type=int, default=512,
                    help="Static chunk budget, held fixed across all arms; "
                         "512 matches the sizing searches' c0 baseline so "
                         "the sizing axis isolates adaptivity.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output directory (default: "
                         "experiments/results/<date>/ablation_study).")
    ap.add_argument("--arms", type=str, nargs="+", default=None,
                    help="Subset of arm keys to run "
                         f"(default all: {[a.key for a in ARMS]}).")
    ap.add_argument("--no-perf-report", action="store_true",
                    help="Skip the untimed passes that write the perf "
                         "reports with the movement ledger (timing is "
                         "unaffected; results 2-4 CSVs need them and are "
                         "skipped).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the main.py command lines and exit.")
    ap.add_argument("--replot", type=str, default=None,
                    help="Path to a previous results.json: regenerate the csv "
                         "/ latex / figure from it WITHOUT re-running the "
                         "engine. For iterating on presentation.")
    ap.add_argument("--extra", type=str, nargs=argparse.REMAINDER, default=[],
                    help="Everything after --extra is forwarded to main.py.")
    args = ap.parse_args()

    if args.replot:
        replot(args.replot, args.out)
        return

    arms = ARMS if not args.arms else [a for a in ARMS if a.key in args.arms]
    if not arms:
        ap.error(f"no arms matched {args.arms}; "
                 f"valid keys: {[a.key for a in ARMS]}")
    queries, reps = args.q, max(1, args.reps)

    out_dir = args.out or os.path.join(
        PROJECT_ROOT, "experiments", "results",
        datetime.date.today().isoformat(), "ablation_study")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'=' * 78}")
    print(f"  Ablation study — SF{args.SF}, {len(queries)} queries x "
          f"{reps} back-to-back rep{'s' if reps > 1 else ''}, "
          f"{len(arms)} arms (one invocation / host pin each)")
    print(f"  Timing source: query_times.csv (approx_time) — per AGENTS.md; "
          f"timed runs with every telemetry switch off")
    print(f"  chunk_size_mb={args.chunk_size_mb} (all arms), "
          f"slot_pool_gb={args.slot_pool_gb} (all arms)")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 78}")
    for a in arms:
        print(f"    {a.key:<10} {a.label:<24} {a.describe()}")

    # samples[arm.key][qid] -> list indexed by rep; all of an arm's reps
    # come out of its single back-to-back invocation.
    samples = {a.key: {q: [None] * reps for q in queries} for a in arms}
    failures: dict[str, list[int]] = {a.key: [] for a in arms}

    for i, a in enumerate(arms):
        print(f"\n[arm {i + 1}/{len(arms)}] {a.label} — "
              f"{len(queries)} queries x {reps} reps, one invocation")
        times, failed = run_arm(
            a, sf=args.SF, queries=queries, reps=reps,
            slot_pool_gb=args.slot_pool_gb,
            chunk_size_mb=args.chunk_size_mb,
            extra_flags=list(args.extra),
            perf_dir=os.path.join(out_dir, "perf", "timed", a.key),
            log_path=os.path.join(out_dir, "logs", f"{a.key}.log"),
            dry_run=args.dry_run,
            telemetry=False,
        )
        for q, secs_list in times.items():
            if q in samples[a.key]:
                for r, secs in enumerate(secs_list[:reps]):
                    samples[a.key][q][r] = secs
        for q in failed:
            if q not in failures[a.key]:
                failures[a.key].append(q)
        if not args.dry_run:
            done = sum(1 for q in queries
                       for v in samples[a.key][q] if v is not None)
            print(f"    {done}/{len(queries) * reps} query reps timed")

    # ── Perf passes (untimed, movement ledger on) ──────────────────────
    # The timed runs above had every telemetry switch off.  Each arm's perf
    # reports come from a second invocation that replays the same query
    # list with TQP_EXPERIMENT_TELEMETRY=1; its timing rows are not used.
    perf_failures: dict[str, list[int]] = {}
    if not args.no_perf_report:
        for i, a in enumerate(arms):
            print(f"\n[perf {i + 1}/{len(arms)}] {a.label} — untimed, "
                  f"TQP_EXPERIMENT_TELEMETRY=1")
            _, perf_failures[a.key] = run_arm(
                a, sf=args.SF, queries=queries, reps=reps,
                slot_pool_gb=args.slot_pool_gb,
                chunk_size_mb=args.chunk_size_mb,
                extra_flags=list(args.extra),
                perf_dir=os.path.join(out_dir, "perf", "arms", a.key),
                log_path=os.path.join(out_dir, "logs", f"{a.key}_perf.log"),
                dry_run=args.dry_run,
                telemetry=True,
            )

    if args.dry_run:
        return

    # ── Aggregate ──────────────────────────────────────────────────────
    medians = {a.key: [median_of(samples[a.key][q]) for q in queries]
               for a in arms}
    totals = {a.key: sum(v for v in medians[a.key] if v) for a in arms}

    print_variance_report(queries, arms, samples)
    print_summary(queries, arms, medians, totals)

    any_fail = {k: v for k, v in failures.items() if v}
    if any_fail:
        print("\n  ⚠  queries the engine reported as failed ([QFAIL]):")
        for k, qs in any_fail.items():
            print(f"       {k:<12} {['Q%d' % q for q in sorted(qs)]}")
        print("     These are excluded from totals and the geometric mean,")
        print("     so the Total row is NOT comparable across arms unless the")
        print("     same queries succeeded everywhere.")

    # ── Products ───────────────────────────────────────────────────────
    emit_products(out_dir, arms, samples, medians, totals,
                  queries, args.SF, reps)

    # Results 2-4 come from perf/arms/<arm>/SF<sf>_<qid>.json, written by
    # the untimed perf passes; repeated occurrences overwrote each file, so
    # these are the FINAL rep's metrics.
    if args.no_perf_report:
        print("  note: --no-perf-report — results 2-4 CSVs skipped")
    else:
        emit_perf_csvs(out_dir, args.SF, queries, perf_sources(out_dir, arms),
                       adaptive_perf_dir(out_dir, arms))

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "sf": args.SF,
            "queries": queries,
            "reps": reps,
            "protocol": "back-to-back reps, one invocation per arm",
            "slot_pool_gb": args.slot_pool_gb,
            "chunk_size_mb": args.chunk_size_mb,
            "timing_source": "query_times.csv approx_time "
                             "(perf/timed/<arm>/, every telemetry switch off)",
            "arms": [asdict(a) for a in arms],
            "samples": {k: {str(q): v for q, v in d.items()}
                        for k, d in samples.items()},
            "medians": {k: {str(q): m for q, m in zip(queries, v)}
                        for k, v in medians.items()},
            "totals": totals,
            "geomean_speedup_vs_baseline": {
                a.key: geomean_speedup(medians[arms[0].key], medians[a.key])
                for a in arms},
            "failures": failures,
            "perf_pass_failures": perf_failures,
        }, f, indent=2)
    print(f"  raw json     -> {results_path}")


if __name__ == "__main__":
    main()
