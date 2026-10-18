#!/usr/bin/env python3
"""Where each hot query's timed window goes, from one Nsight trace taken with
``nvtx_ranges.py`` (E5; docs/sf100-bottleneck-study.md §1.4).  Runs on the
host (sqlite3 only); export the trace first with ``nsys export --type sqlite``.

  [EXCLUDE_Q=1,..] [ONLY_Q=7,..] [ONLY_ARM=<arm>] [HOT_FROM=1] [BEFORE=<json>] \\
      trace_analyze.py <trace.sqlite> <out.json> [<perf dir>]

HOT_FROM: the first occurrence index analysed per qid (default 1: skip the
cold one; 0 for a single cold pass).  ONLY_ARM: keep the windows inside one
"ARM <arm>" range (``switch_ab.py --nvtx`` marks every occurrence with its
arm).  <perf dir>: a ``--perf_report`` directory whose ``query_times.csv``
gives each query's untraced time (``T_canon``: the median of its occurrences
after the first).  BEFORE: the out.json of an earlier analysis; the
output-step and exposed-D2H tables then print it beside this one, with the
ratio.

Window W per query occurrence = its "Query N" range up to the start of its last
output operator (what approx_time covers).  Every instant of W gets two labels:
  pipe  the innermost "P <tag>" range (the pipeline, else the operator's own
        code outside its pipelines = "glue", else "between operators");
  step  the innermost "S <step>" range (a tick phase), else "(no step)".
For each (pipe class, step), the GPU state over that time is split into
  K     some kernel running;
  C     copies only (no kernel);
  idle  nothing on the GPU; idle is split by what the main thread was doing:
        inside a blocking CUDA call ("sync"), inside another CUDA call
        (launch, copy submit, event, alloc), or outside CUDA ("host").
Kernel and copy time are also attributed to pipe classes by kernel class,
and to the host step that launched them (through the CUDA correlation id).
Hot = every occurrence but the first of each qid.

The output path's host steps (OUTPUT_STEPS) get their own tables: GPU idle in
them per pipe class (study §5), and the D2H copies of each pipe class by the
step that issued them, with their exposed time (study §4).  CUDA calls and
step ranges are also counted per step (e.g. synchronize calls per gather
chunk: ``n.sync`` of ``rg.gather`` ÷ its ``ranges``).
"""
import bisect
import csv
import json
import os
import re
import sqlite3
import statistics as st
import sys
from collections import defaultdict

trace, out_path = sys.argv[1], sys.argv[2]
canon_dir = sys.argv[3] if len(sys.argv) > 3 else None

STEPS = {"pipe.init", "tick.pre_op", "tick.post", "pipe.flush", "tick.rebuild",
         "tick.mask_counts", "x.launch", "x.sync", "out.plan", "out.transfer",
         "tick.push_out", "tick.clean", "tick.decompress", "tick.zero_copy",
         "tick.apply_mask", "compress", "ob.feed", "ob.flush", "rg.gather", "col_to_gpu",
         "rp.pack", "rp.ship", "rp.drain", "tick.wait", "tick.collect",
         "x.bind"}
# The output path's host steps: direct compress, the output buffer's feed and
# flush, output planning, the post-compute dispatch and the end-of-pipe drain,
# and the region producer's pack, copies home and drain.
OUTPUT_STEPS = ("ob.feed", "ob.flush", "pipe.flush", "out.plan", "tick.post", "compress",
                "rp.pack", "rp.ship", "rp.drain")


def pipe_class(tag):
    rules = [
        (r"^join \d+ \d+ sort pipeline$", "join: partition sort"),
        (r"^join \d+ core pipeline$", "join: core (match)"),
        (r"^join \d+ rearrange (left|right)$", "join: payload gather"),
        (r"^join \d+ lineage (left|right) g\d+$", "join: lineage gather"),
        (r"^deferred gather \d+ g\d+$", "deferred gather"),
        (r"^aggregation \d+ two-pass aggregation pass1$", "agg: pass1"),
        (r"^aggregation \d+ two-pass aggregation pass2$", "agg: pass2"),
        (r"^aggregation \d+ aggregation pipe$", "agg: single pass"),
        (r"^filter \d+ filter pipe$", "filter/scan pipe"),
        (r"^project \d+ project pipeline$", "project pipe"),
        (r"^sort( lim)? \d+$", "final sort"),
        (r"^scan \d+$", "scan (metadata)"),
        (r"^(join|aggregation|filter|project) \d+$", None),
    ]
    for pat, c in rules:
        if re.match(pat, tag):
            return c if c else "glue: " + tag.split()[0]
    return "other: " + re.sub(r"\d+", "N", tag)


def kernel_class(n):
    if n.startswith("zero_copy_rearrange"):
        return "UVA gather (zero-copy)"
    if n.startswith("zero_copy_mask"):
        return "UVA masked read (zero-copy)"
    if n in ("batch_decoder_kernel", "decompress_kernel", "setup_batched_decomp_llif_buffers",
             "construct_decoding_table_kernel"):
        return "codec decode (nvCOMP)"
    if n in ("batch_encoder_kernel", "compress_kernel",
             "batched_compact_comp_buffers_and_header_output", "setup_batched_comp_llif_buffers",
             "compute_histogram", "normalize_counts", "construct_encoding_table_kernel"):
        return "codec encode (nvCOMP)"
    if n in ("coalesce_sub_chunks", "max_device_status_kernel_batch",
             "batched_device_status_writeout_kernel", "round_up_alignment_kernel") \
            or "nvcomp" in n.lower() or "bitcomp" in n.lower() or "cascaded" in n.lower() \
            or "lz4" in n.lower() or "snappy" in n.lower() or "deflate" in n.lower():
        return "codec other (nvCOMP)"
    if n == "batched_memcpy_kernel":
        return "batched memcpy (pages)"
    if "RadixSort" in n or "MergeSort" in n or n.startswith("radixSort") or "bitonic" in n.lower():
        return "sort"
    if n.startswith(("k_expand", "k_unique_", "k_take_partner", "k_set_build")):
        return "join core (CUDA)"
    if n.startswith("searchsorted"):
        return "searchsorted"
    if n in ("index_elementwise_kernel", "_scatter_gather_elementwise_kernel", "scatter_kernel") \
            or n.startswith("indexSelect") or n.startswith("index") or "gather" in n.lower():
        return "index/gather (HBM)"
    if "elementwise" in n:
        return "elementwise"
    if "reduce" in n.lower() or "Reduce" in n or "Scan" in n or "Select" in n or "Histogram" in n \
            or "histogram" in n or n.startswith("compute_cuda"):
        return "reduce/scan/select"
    if n.startswith("CatArray"):
        return "cat"
    return "other: " + n[:40]


def api_class(n):
    if "Synchronize" in n or n in ("cudaMemcpy", "cuMemcpy", "cuCtxSynchronize"):
        return "sync"
    if "Launch" in n:
        return "launch"
    if "Memcpy" in n or "Memset" in n:
        return "copy submit"
    if "Event" in n:
        return "event"
    if "Malloc" in n or "Free" in n or "Host" in n or "Register" in n:
        return "alloc"
    return "other api"


class IvSet:
    """Merged intervals with prefix sums: covered(a, b) in O(log n)."""

    def __init__(self, iv, merged=False):
        if not merged:
            iv = sorted(iv)
            m = []
            for s, e in iv:
                if m and s <= m[-1][1]:
                    if e > m[-1][1]:
                        m[-1][1] = e
                else:
                    m.append([s, e])
            iv = m
        self.s = [x[0] for x in iv]
        self.e = [x[1] for x in iv]
        self.cum = [0]
        for s, e in iv:
            self.cum.append(self.cum[-1] + e - s)

    def covered(self, a, b):
        if b <= a or not self.s:
            return 0
        i = bisect.bisect_right(self.e, a)       # first interval ending after a
        j = bisect.bisect_left(self.s, b)        # intervals [i, j) start before b
        if i >= j:
            return 0
        tot = self.cum[j] - self.cum[i]
        if self.s[i] < a:
            tot -= a - self.s[i]
        if self.e[j - 1] > b:
            tot -= self.e[j - 1] - b
        return tot

    def minus(self, iv):
        """Pieces of the (unmerged, sorted) intervals ``iv`` not covered by self."""
        out = []
        for s, e in iv:
            i = bisect.bisect_right(self.e, s)
            cur = s
            while i < len(self.s) and self.s[i] < e:
                if self.s[i] > cur:
                    out.append((cur, self.s[i]))
                cur = max(cur, self.e[i])
                i += 1
            if cur < e:
                out.append((cur, e))
        return out


c = sqlite3.connect(trace)
S = dict(c.execute("select id, value from StringIds"))
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
kern = sorted(c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"))
copies = sorted(c.execute("select start, end, bytes, copyKind, srcKind, dstKind, correlationId "
                          "from CUPTI_ACTIVITY_KIND_MEMCPY"))
msets = (sorted(c.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET"))
         if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else [])
nv = sorted((s, e, t if t is not None else S.get(tid, ""), g) for s, e, t, tid, g in c.execute(
    "select start, end, text, textId, globalTid from NVTX_EVENTS where end is not null"))
qranges = [x for x in nv if re.fullmatch(r"Query \d+", x[2])]
main_tid = qranges[0][3]
nv = [x for x in nv if x[3] == main_tid]
api_rows = list(c.execute(
    "select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid = ?",
    (main_tid,)))
api = sorted((s, e, re.sub(r"_v\d+$", "", S.get(n, "?"))) for s, e, n, _ in api_rows)
launch_at = {corr: s for s, e, n, corr in api_rows}   # correlationId -> host launch time
del api_rows
print(f"kernels {len(kern)}, copies {len(copies)}, memsets {len(msets)}, nvtx {len(nv)}, "
      f"main-thread CUDA calls {len(api)}", flush=True)

K = IvSet([(s, e) for s, e, *_ in kern])
BUSY = IvSet([(s, e) for s, e, *_ in kern] + [(s, e) for s, e, *_ in copies] + list(msets))
H2D = IvSet([(s, e) for s, e, b, k, *_ in copies if k == 1])
D2H = IvSet([(s, e) for s, e, b, k, *_ in copies if k == 2])
busy_iv = list(zip(BUSY.s, BUSY.e))
# Copy-only time split by direction: D2H with no kernel and no H2D running,
# H2D with no kernel and no D2H, and both directions at once.
_kh = IvSet([(s, e) for s, e, *_ in kern] + list(zip(H2D.s, H2D.e)))
_kd = IvSet([(s, e) for s, e, *_ in kern] + list(zip(D2H.s, D2H.e)))
EXP_D2H = IvSet(_kh.minus(list(zip(D2H.s, D2H.e))))
EXP_H2D = IvSet(_kd.minus(list(zip(H2D.s, H2D.e))))
api_by_cls = defaultdict(list)
for s, e, n in api:
    api_by_cls[api_class(n)].append((s, e))
API_IDLE = {k: IvSet(BUSY.minus(v)) for k, v in api_by_cls.items()}
API_ALL = {k: IvSet(v) for k, v in api_by_cls.items()}
api_starts = [x[0] for x in api]

occ = defaultdict(int)
windows = []
for s, e, name, _ in qranges:
    q = int(name.split()[1])
    k_ = occ[q]
    occ[q] += 1
    outs = [x for x in nv if s <= x[0] <= e and x[2].startswith("AdaptiveSparkPlan")]
    w1 = outs[-1][0] if outs else e
    windows.append((q, k_, s, w1))

# chunk_ab.py wraps every occurrence in an "ARM <name>" range; ONLY_ARM keeps
# the windows of one arm (the first, warm occurrence of each qid is still
# dropped as cold; with HOT_FROM=0 it is kept, for its own arm only).
_arm_ranges = [(s, e, t[4:]) for s, e, t, _ in nv if t.startswith("ARM ")]


def arm_of(t):
    for s, e, a in _arm_ranges:
        if s <= t <= e:
            return a
    return None

nv_starts = [x[0] for x in nv]
res = {"queries": {}, "pipe": {}, "step": {}, "pipe_step": {}, "kernels": {}, "copies": {}}
agg_pipe = defaultdict(lambda: defaultdict(float))
agg_step = defaultdict(lambda: defaultdict(float))
agg_ps = defaultdict(lambda: defaultdict(float))
agg_kern = defaultdict(lambda: defaultdict(float))
agg_kstep = defaultdict(lambda: defaultdict(float))    # launching step -> kernel class -> s
agg_cstep = defaultdict(lambda: defaultdict(float))    # launching step -> copy kind -> s / GB
agg_copy = defaultdict(lambda: defaultdict(float))
agg_q = defaultdict(lambda: defaultdict(list))
cold = defaultdict(float)

kern_starts = [x[0] for x in kern]
copy_starts = [x[0] for x in copies]
max_kd = max((e - s for s, e, *_ in kern), default=0)
max_cd = max((e - s for s, e, *_ in copies), default=0)

_excl = {int(x) for x in os.environ.get("EXCLUDE_Q", "").split(",") if x}
_only = {int(x) for x in os.environ.get("ONLY_Q", "").split(",") if x}
_only_arm = os.environ.get("ONLY_ARM")
for q, k_, w0, w1 in windows:
    if q in _excl or (_only and q not in _only):
        continue
    if (_only_arm and (k_ > 0 or os.environ.get("HOT_FROM") == "0")
            and arm_of(w0) != _only_arm):
        continue
    if k_ < int(os.environ.get("HOT_FROM", "1")):
        cold[q] = (w1 - w0) / 1e9
        continue
    lo = bisect.bisect_left(nv_starts, w0)
    rng = [x for x in nv[lo:bisect.bisect_right(nv_starts, w1)] if x[1] > w0 and x[0] < w1]
    P = [(max(s, w0), min(e, w1), t[2:]) for s, e, t, _ in rng
         if t.startswith("P ") and not re.fullmatch(r"P Query \d+", t)]
    ST = [(max(s, w0), min(e, w1), t) for s, e, t, _ in rng if t in STEPS]
    ev = []
    for i, (s, e, _) in enumerate(P):
        ev += [(s, 1, "P", i), (e, 0, "P", i)]
    for i, (s, e, _) in enumerate(ST):
        ev += [(s, 1, "S", i), (e, 0, "S", i)]
    ev.sort()
    actP, actS = {}, {}
    segs = []
    cur = w0
    for t, typ, kind, i in ev + [(w1, 0, None, None)]:
        if t > cur:
            p = max(actP.items(), key=lambda x: (x[1], -P[x[0]][1]))[0] if actP else None
            sidx = max(actS.items(), key=lambda x: (x[1], -ST[x[0]][1]))[0] if actS else None
            pl = pipe_class(P[p][2]) if p is not None else "between operators"
            sl = ST[sidx][2] if sidx is not None else "(no step)"
            if segs and segs[-1][2] == pl and segs[-1][3] == sl and segs[-1][1] == cur:
                segs[-1][1] = t
            else:
                segs.append([cur, t, pl, sl])
            cur = t
        if kind == "P":
            (actP.__setitem__(i, P[i][0]) if typ else actP.pop(i, None))
        elif kind == "S":
            (actS.__setitem__(i, ST[i][0]) if typ else actS.pop(i, None))
    qa = defaultdict(float)
    for a, b, pl, sl in segs:
        d = b - a
        kk = K.covered(a, b)
        bb = BUSY.covered(a, b)
        row = {"W": d, "K": kk, "C": bb - kk, "idle": d - bb,
               "h2d": H2D.covered(a, b), "d2h": D2H.covered(a, b),
               "C.d2h_only": EXP_D2H.covered(a, b), "C.h2d_only": EXP_H2D.covered(a, b)}
        for ac, ivs in API_IDLE.items():
            row["idle." + ac] = ivs.covered(a, b)
        for ac, ivs in API_ALL.items():
            row["in." + ac] = ivs.covered(a, b)
        row["idle.host"] = row["idle"] - sum(row["idle." + ac] for ac in API_IDLE)
        for key, v in row.items():
            agg_pipe[pl][key] += v / 1e9
            agg_step[sl][key] += v / 1e9
            agg_ps[(pl, sl)][key] += v / 1e9
            qa[key] += v / 1e9
    # counts: API calls and ticks per pipe class
    seg_starts = [s[0] for s in segs]

    def label_at(t):
        i = bisect.bisect_right(seg_starts, t) - 1
        return segs[i][2] if 0 <= i < len(segs) else "between operators"

    def step_at(t):
        i = bisect.bisect_right(seg_starts, t) - 1
        return segs[i][3] if 0 <= i < len(segs) else "(outside window)"

    for s, e, t in ST:
        agg_step[t]["ranges"] += 1
        if t == "x.sync":
            agg_pipe[label_at(s)]["ticks"] += 1
        if t == "pipe.init":
            agg_pipe[label_at(s)]["pipes_built"] += 1
    i0 = bisect.bisect_left(api_starts, w0)
    for s, e, n in api[i0:bisect.bisect_left(api_starts, w1)]:
        agg_pipe[label_at(s)]["n." + api_class(n)] += 1
        agg_step[step_at(s)]["n." + api_class(n)] += 1
        qa["n." + api_class(n)] += 1
    for s, e, n, corr in kern[bisect.bisect_left(kern_starts, w0 - max_kd):bisect.bisect_left(kern_starts, w1)]:
        a, b = max(s, w0), min(e, w1)
        if b <= a:
            continue
        pl = label_at((a + b) // 2)
        kc = kernel_class(S.get(n, "?"))
        lt = launch_at.get(corr)
        stp = step_at(lt) if lt is not None else "(other thread)"
        agg_kstep[stp][kc] += (b - a) / 1e9
        agg_kstep[pl + " | " + stp][kc] += (b - a) / 1e9
        agg_kern[pl][kc] += (b - a) / 1e9
        agg_kern["ALL"][kc] += (b - a) / 1e9
        agg_kern["ALL:names"][S.get(n, "?")] += (b - a) / 1e9
    for s, e, nb, ck, sk, dk, corr in copies[bisect.bisect_left(copy_starts, w0 - max_cd):
                                      bisect.bisect_left(copy_starts, w1)]:
        a, b = max(s, w0), min(e, w1)
        if b <= a:
            continue
        pl = label_at((a + b) // 2)
        kind = {1: "H2D", 2: "D2H", 8: "D2D"}.get(ck, f"kind{ck}")
        if ck in (1, 2) and nb < 4096:
            kind += " tiny"
        if ck == 1 and sk == 0 or ck == 2 and dk == 0:
            kind += " pageable"
        lt = launch_at.get(corr)
        stp = step_at(lt) if lt is not None else "(other thread)"
        # exposed: no kernel and no copy in the other direction running
        exp = (EXP_D2H.covered(a, b) if ck == 2 else EXP_H2D.covered(a, b) if ck == 1 else 0)
        for key in (stp, pl + " | " + stp):
            agg_cstep[key][kind + " s"] += (b - a) / 1e9
            agg_cstep[key][kind + " GB"] += nb / 1e9
            agg_cstep[key][kind + " exp"] += exp / 1e9
        for key in (pl, "ALL"):
            agg_copy[key][kind + " s"] += (b - a) / 1e9
            agg_copy[key][kind + " GB"] += nb / 1e9
            agg_copy[key][kind + " n"] += 1
    for key, v in qa.items():
        agg_q[q][key].append(v)

canon = {}
if canon_dir:
    runs = defaultdict(list)
    with open(os.path.join(canon_dir, "query_times.csv")) as f:
        for row in csv.reader(f):
            if row and row[0].strip().isdigit():
                runs[int(row[0])].append(float(row[1]))
    canon = {q: st.median(v[1:]) if len(v) > 1 else v[0] for q, v in runs.items()}

nq = len(agg_q)
n_hot = {q: len(v["W"]) for q, v in agg_q.items()}
# Totals are sums over hot occurrences; divide by the hot count per query
# (uniform here) so the tables are "one hot pass".
reps_hot = max(n_hot.values())
scale = 1.0 / reps_hot
for q, v in agg_q.items():
    res["queries"][q] = {k: st.median(x) for k, x in v.items()}
    res["queries"][q]["T_canon"] = canon.get(q)
    res["queries"][q]["W_cold"] = cold.get(q)
res["pipe"] = {k: {kk: vv * scale for kk, vv in v.items()} for k, v in agg_pipe.items()}
res["step"] = {k: {kk: vv * scale for kk, vv in v.items()} for k, v in agg_step.items()}
res["pipe_step"] = {f"{k[0]} | {k[1]}": {kk: vv * scale for kk, vv in v.items()} for k, v in agg_ps.items()}
res["kernels"] = {k: {kk: vv * scale for kk, vv in v.items()} for k, v in agg_kern.items()}
res["copies"] = {k: {kk: vv * scale for kk, vv in v.items()} for k, v in agg_copy.items()}
res["kernels_by_launch_step"] = {k: {kk: vv * scale for kk, vv in v.items()} for k, v in agg_kstep.items()}
res["copies_by_launch_step"] = {k: {kk: vv * scale for kk, vv in v.items()} for k, v in agg_cstep.items()}


def output_tables(r):
    """Per pipe class: GPU idle in each output step, and the pinned D2H copies
    of 4 KiB or more by the step that issued them, as (GB, copy s, exposed s).
    Reads only ``pipe_step`` and ``copies_by_launch_step``, so it also reads
    an out.json written before these tables existed."""
    idle = defaultdict(dict)
    for key, v in r["pipe_step"].items():
        pl, stp = key.split(" | ", 1)
        if stp in OUTPUT_STEPS:
            idle[pl][stp] = v["idle"]
    d2h = defaultdict(dict)
    for key, v in r["copies_by_launch_step"].items():
        if " | " in key and "D2H GB" in v:
            pl, stp = key.split(" | ", 1)
            d2h[pl][stp] = (v["D2H GB"], v["D2H s"], v["D2H exp"])
    return idle, d2h


out_idle, out_d2h = output_tables(res)
res["output_steps"] = {pl: {**d, "total": sum(d.values())} for pl, d in out_idle.items()}
res["exposed_d2h"] = {pl: {stp: {"GB": g, "s": s, "exposed": x} for stp, (g, s, x) in d.items()}
                      for pl, d in out_d2h.items()}
json.dump(res, open(out_path, "w"), indent=1)
before = output_tables(json.load(open(os.environ["BEFORE"]))) if os.environ.get("BEFORE") else None

TW = sum(r["W"] for r in res["queries"].values())
TC = sum((r["T_canon"] or 0) for r in res["queries"].values())
tot = defaultdict(float)
for r in res["queries"].values():
    for k, v in r.items():
        if isinstance(v, (int, float)):
            tot[k] += v
print(f"\n{nq} queries, hot occurrences per query {reps_hot}; traced W {TW:.3f} s (untraced T {TC:.3f} s)")
print(f"  GPU kernel {tot['K']:.3f} s ({100*tot['K']/TW:.0f}%) | copy-only {tot['C']:.3f} s "
      f"({100*tot['C']/TW:.0f}%) | idle {tot['idle']:.3f} s ({100*tot['idle']/TW:.0f}%)")
print(f"  idle split: host (outside CUDA) {tot['idle.host']:.3f} s, " + ", ".join(
    f"{k[5:]} {tot[k]:.3f}" for k in sorted(tot) if k.startswith("idle.") and k != "idle.host"))
print(f"  main thread in blocking syncs (any GPU state): {tot['in.sync']:.3f} s; CUDA calls: " + ", ".join(
    f"{k[2:]} {tot[k]:.0f}" for k in sorted(tot) if k.startswith("n.")))


def table(title, d, keyw=34):
    print(f"\n{title}")
    print(f"  {'':{keyw}s} {'W':>7s} {'%':>5s} {'K':>7s} {'C':>6s} {'idle':>7s} | {'host':>6s} {'sync':>6s} "
          f"{'launch':>6s} {'copy':>6s} {'event':>6s} {'alloc':>6s} | {'ticks':>6s}")
    for k, v in sorted(d.items(), key=lambda x: -x[1].get("W", 0)):
        if v.get("W", 0) < 0.002:
            continue
        print(f"  {k[:keyw]:{keyw}s} {v['W']:7.3f} {100*v['W']/TW:4.1f}% {v['K']:7.3f} {v['C']:6.3f} "
              f"{v['idle']:7.3f} | {v.get('idle.host',0):6.3f} {v.get('idle.sync',0):6.3f} "
              f"{v.get('idle.launch',0):6.3f} {v.get('idle.copy submit',0):6.3f} {v.get('idle.event',0):6.3f} "
              f"{v.get('idle.alloc',0):6.3f} | {v.get('ticks',0):6.0f}")


table("by pipe class (s, one hot pass; idle columns split GPU-idle time by main-thread state)", res["pipe"])
table("by host step", res["step"])
table("by pipe class | step (top 40)", dict(sorted(res["pipe_step"].items(),
                                                   key=lambda x: -x[1]["W"])[:40]), 50)
print("\nkernel time by class (s, one hot pass):")
for kc, v in sorted(res["kernels"]["ALL"].items(), key=lambda x: -x[1]):
    print(f"  {kc:34s} {v:7.3f}")
print("\nkernel time by pipe class x kernel class (s):")
for pl, d in sorted(res["kernels"].items(), key=lambda x: -sum(x[1].values())):
    if pl.startswith("ALL"):
        continue
    top = sorted(d.items(), key=lambda x: -x[1])[:5]
    print(f"  {pl:28s} {sum(d.values()):6.3f} | " + ", ".join(f"{k} {v:.3f}" for k, v in top))
print("\nkernel time by launching host step (s):")
for stp, d in sorted(res["kernels_by_launch_step"].items(), key=lambda x: -sum(x[1].values())):
    if " | " in stp:
        continue
    top = sorted(d.items(), key=lambda x: -x[1])[:5]
    print(f"  {stp:20s} {sum(d.values()):6.3f} | " + ", ".join(f"{k} {v:.3f}" for k, v in top))
print("\ncopies by launching host step:")
for stp, d in sorted(res["copies_by_launch_step"].items(), key=lambda x: -sum(v for k, v in x[1].items() if k.endswith(" s"))):
    if " | " in stp and not stp.endswith(("pipe.flush", "out.transfer", "x.launch",
                                          "rp.ship", "rp.drain")):
        continue
    parts = sorted({k.rsplit(' ', 1)[0] for k in d if not k.startswith(("D2H tiny", "H2D tiny", "D2D"))})
    if not parts:
        continue
    print(f"  {stp:20s} " + "; ".join(f"{p}: {d[p+' GB']:.2f} GB {d[p+' s']:.3f} s (exposed {d.get(p+' exp', 0):.3f})"
                                       for p in parts))
print("\ncopies (one hot pass):")
for pl, d in sorted(res["copies"].items(), key=lambda x: -sum(v for k, v in x[1].items() if k.endswith(" s"))):
    parts = sorted({k.rsplit(' ', 1)[0] for k in d})
    print(f"  {pl:28s} " + "; ".join(f"{p}: {d[p+' GB']:.2f} GB {d[p+' s']:.3f} s n={d[p+' n']:.0f}"
                                     for p in parts))
b_idle, b_d2h = before if before is not None else ({}, {})


def vs_before(now, then):
    if before is None:
        return ""
    return f"  | before {then:7.3f}  ratio " + (f"{now / then:5.2f}" if then else "  n/a")


print("\nGPU idle in output steps by pipe class (s, one hot pass)")
print(f"  {'':28s} " + " ".join(f"{s:>10s}" for s in OUTPUT_STEPS) + f" {'total':>8s}")
for pl in sorted(set(out_idle) | set(b_idle), key=lambda p: -sum(out_idle.get(p, {}).values())):
    d = out_idle.get(pl, {})
    tot, b_tot = sum(d.values()), sum(b_idle.get(pl, {}).values())
    if max(tot, b_tot) < 0.002:
        continue
    print(f"  {pl[:28]:28s} " + " ".join(f"{d.get(s, 0):10.3f}" for s in OUTPUT_STEPS)
          + f" {tot:8.3f}" + vs_before(tot, b_tot))
print("\nD2H by pipe class and issuing step (pinned copies of 4 KiB or more; one hot pass)")
for pl in sorted(set(out_d2h) | set(b_d2h), key=lambda p: -sum(x[2] for x in out_d2h.get(p, {}).values())):
    d, bd = out_d2h.get(pl, {}), b_d2h.get(pl, {})
    gb, s, x = (sum(v[i] for v in d.values()) for i in range(3))
    b_x = sum(v[2] for v in bd.values())
    if max(gb, sum(v[0] for v in bd.values())) < 0.01:
        continue
    print(f"  {pl[:28]:28s} {gb:7.2f} GB  copy {s:6.3f} s  exposed {x:6.3f} s" + vs_before(x, b_x))
    for stp in sorted(set(d) | set(bd), key=lambda t: -d.get(t, (0, 0, 0))[2]):
        g1, s1, x1 = d.get(stp, (0.0, 0.0, 0.0))
        print(f"    {stp:26s} {g1:7.2f} GB  copy {s1:6.3f} s  exposed {x1:6.3f} s"
              + vs_before(x1, bd.get(stp, (0.0, 0.0, 0.0))[2]))
print("\nCUDA calls and ranges by host step (one hot pass)")
print(f"  {'':20s} {'ranges':>8s} {'sync':>8s} {'launch':>8s} {'copy':>8s} {'event':>9s} {'other':>7s}")
for stp, v in sorted(res["step"].items(), key=lambda x: -x[1].get("ranges", 0)):
    if not any(k == "ranges" or k.startswith("n.") for k in v):
        continue
    print(f"  {stp:20s} {v.get('ranges', 0):8.0f} {v.get('n.sync', 0):8.0f} {v.get('n.launch', 0):8.0f} "
          f"{v.get('n.copy submit', 0):8.0f} {v.get('n.event', 0):9.0f} "
          f"{v.get('n.alloc', 0) + v.get('n.other api', 0):7.0f}")
print("\nper query (ms): T_canon W_hot W_cold | K C idle (host sync) | calls")
for q, r in sorted(res["queries"].items()):
    print(f"  Q{q:<3d} {1e3*(r['T_canon'] or 0):6.0f} {1e3*r['W']:6.0f} {1e3*(r['W_cold'] or 0):6.0f} | "
          f"{1e3*r['K']:6.0f} {1e3*r['C']:5.0f} {1e3*r['idle']:6.0f} ({1e3*r['idle.host']:5.0f} "
          f"{1e3*r.get('idle.sync',0):4.0f}) | {sum(v for k, v in r.items() if k.startswith('n.')):7.0f}")
