#!/usr/bin/env python
"""tqp_vortex's spill-pass record builder (baselines/AGENTS.md, "Spill pass";
docs/engine_spill_plan.md §3.4).

  tqp_spill_stats.py record <q_dir> --sf SF --query N --evidence-dir DIR
  tqp_spill_stats.py settings <evidence_dir> --sf SF [--git REV]

The timed run of one main.py process (bench/run_tqp.sh with SPILL=1: the
bench protocol, TQP_EXPERIMENT_TELEMETRY=1), from <q_dir>:
  perf/SF<SF>_<N>.json  the perf report: the movement ledger
                        (IO/telemetry/movement.py; `kinds` the query, `output`
                        the output step), the residency peaks
                        (IO/telemetry/residency.py) and each pipe's arena peak;
                        a failed query's is partial (wall.status FAIL)
  perf/cmdline.json     the settings as main.py resolved them
Bytes are the ledger's: logical_b decoded, wire_b as stored (nvCOMP codecs; a
granule's decoded bytes are its records', pro-rated by stored bytes); peaks
are whole blocks of the arena (5 MiB) and of the pinned pool (0.5 MiB). count
is the ledger's pages of one kind (a run, a region, a granule, a piece, a
column); none for the zero-copy reads, which have no pages, and for sums of
kinds.

`record` appends the query's record to DIR/records.jsonl (spill_results.py),
an all-NA stub with the reason when there is no usable report (a timeout, a
kill, a warm-pass failure). Its checks are the identities that hold by
construction: under --buffer_strategy cpu no cache or pressure traffic and
every pipe output written home; in the cache modes, clean reclaims mirrored
first, every filled granule home once at most and out of its slot once at
most, and home reads only of what a pressure kind wrote (not a bound on
their bytes: each consumer reads); in every mode, peaks within their pools
and the arena's peak above every pipe's.

stdlib only: it runs with the baselines' Python natively and with the host's
python3 in Docker mode.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "baselines"))
import spill_results as sr  # noqa: E402

ENGINE = "tqp_vortex"
# The ledger kinds a record carries as they are (spill_results.KINDS).
LEDGER = ("scan_h2d", "scan_uva", "spill_bypass", "spill_mirror", "spill_evict",
          "spill_rejected", "restore_h2d", "restore_evicted_h2d", "restore_uva",
          "cache_hit", "resident_d2d", "cache_fill", "avoided_dead", "avoided_clean")
READS = ("scan_h2d", "scan_uva", "restore_h2d", "restore_evicted_h2d", "restore_uva",
         "cache_hit", "resident_d2d")
UVA = ("scan_uva", "restore_uva")                            # zero-copy reads: no pages
CACHE = ("cache_hit", "cache_fill", "spill_mirror", "restore_evicted_h2d",
         "avoided_dead", "avoided_clean")


def load(q_dir, sf, num):
    """(perf report, cmdline.json) of the query's process."""
    with open(os.path.join(q_dir, "perf", f"SF{sf}_{num}.json")) as f:
        perf = json.load(f)
    with open(os.path.join(q_dir, "perf", "cmdline.json")) as f:
        cmd = json.load(f)
    for key in ("movement", "residency"):
        if key not in perf:
            raise LookupError(f"perf report has no {key!r}: not written under "
                              "TQP_EXPERIMENT_TELEMETRY=1")
    if "effective" not in cmd:
        raise LookupError("cmdline.json has no resolved settings")
    return perf, cmd


def pipe_peak(perf):
    """The largest per-pipe arena peak (perf_json.peak_gpu_bytes), or None."""
    peaks = [sec["totals"]["peak_gpu_bytes"] for sec in perf.values()
             if isinstance(sec, dict) and isinstance(sec.get("totals"), dict)
             and "peak_gpu_bytes" in sec["totals"]]
    return max(peaks) if peaks else None


def rows(perf, cmd, status, sf, num):
    """The contract rows and the identity checks of one query's report."""
    partial = status != "OK" or perf["wall"].get("status") == "FAIL"
    q = "PARTIAL" if partial else "OK"
    kinds, out = perf["movement"]["kinds"], perf["movement"]["output"]
    res = perf["residency"]
    arena, host, slots = res["gpu_arena"], res["host_pool"], res["slot_pool"]
    mode = cmd["effective"]["buffer_strategy"]
    lg = lambda ks, f, at=kinds: sum(at[k][f] for k in ks)

    def ledger(name, src_kinds, at, src):
        pages = lg(src_kinds, "pages", at) if len(src_kinds) == 1 and name not in UVA else None
        return sr.row(name, lg(src_kinds, "logical_b", at), lg(src_kinds, "wire_b", at),
                      pages, quality=q, source=src)

    def peak(name, b, src):
        return sr.row(name, b, b, quality=q, source=src) if b is not None else sr.na(name, src)
    born = ("cache_fill", "spill_bypass", "spill_rejected")
    pp = pipe_peak(perf)
    peak_inter = host["peak_b"].get("intermediate", 0)
    no_slots = f"no slot pool (--buffer_strategy {mode})"
    out_rows = [ledger(k, (k,), kinds, "ledger:" + k) for k in LEDGER] + [
        ledger("born", born, kinds, "ledger:" + "+".join(born)),
        ledger("result_d2h", ("result_d2h",), out, "ledger output step:result_d2h"),
        ledger("output_read", READS, out, "ledger output step: reads of the result columns"),
        peak("peak_gpu_arena", arena["peak_b"], "residency:gpu_arena.peak_b (arena get_peak_bytes)"),
        peak("peak_host_intermediate", peak_inter, "residency:host_pool.peak_b.intermediate"),
        peak("peak_pipe_gpu", pp, "largest pipe peak_gpu_bytes"),
        peak("peak_host_pool", host["peak_total_b"], "residency:host_pool.peak_total_b"),
        peak("host_scan", (host["at_start_b"] or {}).get("scan", 0) if res["timed"] else None,
             "residency:host_pool.at_start_b.scan"),
        peak("slot_slab", slots["slab_b"], "residency:slot_pool.slab_b") if slots
        else sr.row("slot_slab", 0, 0, quality=q, source=no_slots),
        peak("peak_slot", slots["peak_used_b"], "residency:slot_pool.peak_used_b") if slots
        else sr.row("peak_slot", 0, 0, quality=q, source=no_slots),
    ]
    w = lambda ks, at=kinds: lg(ks, "wire_b", at)
    checks = [
        sr.check("arena peak <= arena", arena["peak_b"], arena["capacity_b"], "<="),
        sr.check("pinned pool peak <= pool", host["peak_total_b"], host["capacity_b"], "<="),
        sr.check("intermediates' peak <= pinned pool peak", peak_inter, host["peak_total_b"], "<="),
    ]
    if pp is not None:
        checks.append(sr.check("arena peak >= every pipe's peak", arena["peak_b"], pp, ">="))
    if not partial and not (num == 16 and sf >= 100):     # main.py skips Q16's output there
        checks.append(sr.check("the result copy is counted", out["result_d2h"]["pages"], 1, ">="))
    if mode == "cpu":
        checks += [
            sr.check("cpu: spill_evict = 0", w(("spill_evict",)), 0),
            sr.check("cpu: spill_rejected = 0", w(("spill_rejected",)), 0),
            sr.check("cpu: no cache traffic", w(CACHE), 0),
            sr.check("cpu: born = spill_bypass", lg(born, "logical_b"), lg(("spill_bypass",), "logical_b")),
        ]
    else:
        # A filled granule goes home once at most, by a reclaim or by
        # cleaning; its decoded bytes round apart from the fill's by a byte.
        # It leaves its slot once at most too (a reclaim or a dead release;
        # admission is the only way in): its bytes, exactly as filled.
        home, fill = ("spill_evict", "spill_mirror"), kinds["cache_fill"]
        both = lambda ks: w(ks) + w(ks, out)
        checks += [
            sr.check("cache: clean reclaims were mirrored first", w(("avoided_clean",)),
                     w(("spill_mirror",)), "<="),
            sr.check("cache: granules home <= granules filled", lg(home, "pages"), fill["pages"], "<="),
            sr.check("cache: decoded bytes home <= decoded bytes filled", lg(home, "logical_b"),
                     fill["logical_b"] + fill["pages"], "<="),
            sr.check("cache: bytes out of slots <= bytes filled",
                     both(("spill_evict", "avoided_clean", "avoided_dead")), both(("cache_fill",)), "<="),
            # Home bytes are read only once a pressure kind wrote them. Not
            # how many: a region is read once per consumer (expected_reads),
            # and a home read leaves it home, so the read-back can exceed
            # the spill (plan §10).
            sr.check("cache: evicted reads only with copies home",
                     both(("spill_evict", "spill_rejected", "spill_mirror")),
                     min(1, both(("restore_evicted_h2d",))), ">="),
        ]
        if slots:
            checks.append(sr.check("cache: slots in use <= slot pool", slots["peak_used_b"],
                                   slots["capacity_b"], "<="))
            if slots["slab_b"] is not None:
                checks.append(sr.check("cache: arena peak >= slot slab", arena["peak_b"], slots["slab_b"], ">="))
    return out_rows, checks


def cmd_record(a):
    status, pt = sr.pass_time(os.path.join(a.evidence_dir, "pass.csv"), a.query)
    try:
        perf, cmd = load(a.q_dir, a.sf, a.query)
        rs, checks = rows(perf, cmd, status, a.sf, a.query)
        rec = sr.record(ENGINE, a.sf, a.query, status, rs + [pt], checks,
                        error=perf["wall"].get("error"))
    except (OSError, LookupError, ValueError, KeyError, TypeError) as e:
        rec = sr.stub(ENGINE, a.sf, a.query, a.evidence_dir, f"{type(e).__name__}: {e}")
    sr.append_record(a.evidence_dir, rec)


def _git(rev):
    """The checkout's commit, with "+dirty" when it has uncommitted changes."""
    if rev:
        return rev
    try:
        sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        return sha + ("+dirty" if dirty else "") if sha else None
    except (OSError, subprocess.SubprocessError):
        return None


def cmd_settings(a):
    """The pools as main.py sized them (a query's residency and cmdline.json),
    the cache and codec settings, the versions, the GPU."""
    perf = cmd = None
    for p in sorted(glob.glob(os.path.join(a.evidence_dir, "q*", "perf", "SF*_*.json"))):
        try:
            perf, cmd = load(os.path.dirname(os.path.dirname(p)), a.sf,
                             os.path.basename(p).rsplit("_", 1)[1][:-5])
            break
        except (OSError, LookupError, ValueError):
            continue
    # Where each query's threads were pinned (main.py's cpu_pin; with
    # TQP_NUMA_BIND off also where its pinned pool landed): node None when
    # the pin failed.
    pins = {}
    for c in sorted(glob.glob(os.path.join(a.evidence_dir, "q*", "perf", "cmdline.json"))):
        try:
            with open(c) as f:
                pin = json.load(f).get("cpu_pin")
        except (OSError, ValueError):
            continue
        if pin is not None:
            pins[os.path.basename(os.path.dirname(os.path.dirname(c)))] = pin
    res = (perf or {}).get("residency", {})
    cmd = cmd or {}
    budgets = {"gpu_b": res.get("gpu_arena", {}).get("capacity_b"),
               "host_b": res.get("host_pool", {}).get("capacity_b"), "disk_b": None}
    codec_map = os.path.join(ROOT, "profiling", "results", "codec_map.json")
    sr.write_settings(a.evidence_dir, ENGINE, a.sf, budgets, [],
                      [codec_map] if os.path.exists(codec_map) else [],
                      {**cmd.get("versions", {}), "tqp_vortex": _git(a.git)},
                      effective=cmd.get("effective"), env=cmd.get("env"),
                      codec_map=cmd.get("codec_map"), engine_gpu=cmd.get("gpu"),
                      slot_pool_b=(res.get("slot_pool") or {}).get("capacity_b"),
                      argv=cmd.get("argv"), native=os.environ.get("TQP_NATIVE") == "1",
                      cpu_pin_nodes=sorted({p["node"] for p in pins.values()
                                            if p["node"] is not None}),
                      cpu_pin_failed=sorted(q for q, p in pins.items() if p["node"] is None),
                      smoke=os.environ.get("SPILL_SMOKE") == "1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("q_dir")
    r.add_argument("--sf", type=int, required=True)
    r.add_argument("--query", type=int, required=True)
    r.add_argument("--evidence-dir", required=True)
    s = sub.add_parser("settings")
    s.add_argument("evidence_dir")
    s.add_argument("--sf", type=int, required=True)
    s.add_argument("--git", default=None, help="the engine's commit (default: git rev-parse here)")
    a = ap.parse_args()
    {"record": cmd_record, "settings": cmd_settings}[a.cmd](a)


if __name__ == "__main__":
    main()
