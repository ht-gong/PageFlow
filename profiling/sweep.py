"""The offline codec sweep: measure every codec x chunk on SF5 TPC-H columns
through the engine's own compress path and write the codec map.

    python3 -m profiling.sweep [--reps 3]          # make profiling

What it does (``profiling/README.md`` has the same, with the outputs):

* **Inputs, all through engine code.**  The scan set and byte weights come
  from the 22 cached plans (``queries/pickles/planobj-q*.pkl`` ->
  ``disk_read_cols``; weight = scan frequency x in-memory bytes).  Columns
  are read from the SF5 parquet under ``load_config()["tables path"]`` (SF5
  is required) with ``IO.parquet_reader.ParquetTable`` and cast as the scan
  casts them (``IO.data_scanner.cast`` / ``ensure_2d``).
  Compress is the region producer's (``IO.codec.region_producer.
  RegionProducer`` with an explicit ``chunk_b``: records of whole rows,
  one batched nvCOMP call, packed into a region whose home is in a pinned
  pool on GPU 0's NUMA node), the path every pipe's outputs and the scan's
  compressed representations take; the read is the record reader's
  (``IO.transfer.in_stage.DirectIn`` over the region's record table: the
  runs copied H2D into the staging buffer, then one batched decompress).
  RAW takes the same paths (a RAW region, read back by a ``DirectIn``).
  Threads are pinned to GPU 0's NUMA node with ``utility.numa``, as a timed
  ``main.py`` run does.
* **Grid.**  codec in the seven engine codecs, nvCOMP chunk in
  {16, 32, 64, 128, 256, 512} KiB, run in {unsorted (seeded permutation),
  sorted (ascending / row-wise lexicographic)}, every scanned column, minimum
  over ``--reps``.  Codec options are fixed (nvcomp_manager.cu).
* **Measured per cell** with CUDA events: compress (the producer's compress
  and pack), D2H of the region to its pinned home (the home writer's copy
  span), H2D back (the reader's run copies), decompress of the bytes that
  made the round trip (verified equal to the input).  Cost = the sum of the four
  (``profiling/objective.py``).  RAW costs the D2H + H2D of a RAW region,
  measured the same way, once per column and run.
* **Selection.**  Per class ``(type, run)`` the byte-weighted cost decides
  the codec; each codec then gets ONE chunk (the best over the classes it
  won, or over all classes if it won none), and the class winners are
  re-picked with every codec at its final chunk so the map and the chunks
  are consistent.  ``stable`` records whether four robustness schemes agree
  with the winner.
* **Outputs** in ``profiling/results/``: ``codec_map.json`` (read by
  ``compression/format_policy.py``), ``sweep_per_column.csv``,
  ``sweep_by_class.csv``, ``scan_frequency.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from compression.format_policy import CODECS, MAPPED_TYPES, RAW  # noqa: E402
from IO.data_scanner import TYPE_MAPPING, cast, ensure_2d     # noqa: E402
from IO.parquet_reader import ParquetTable, table_dir            # noqa: E402
from profiling.objective import LEGS, cost_s, raw_cost_s         # noqa: E402
from utility.config import load_config                           # noqa: E402

RESULTS_DIR = REPO / "profiling" / "results"
SF = 5
CHUNKS_B = [k << 10 for k in (16, 32, 64, 128, 256, 512)]
RUNS = ("unsorted", "sorted")
PINNED_BLOCK_MB = 0.5                 # main.py's pinned-pool block
DEVICE = "cuda:0"
GB = 1024 ** 3

# ---------------------------------------------------------------------------
# Inputs: schema, scan sets, tensors
# ---------------------------------------------------------------------------

def load_schema() -> dict:
    with open(REPO / "queries" / "schema.json", encoding="utf-8") as f:
        return json.load(f)


def col_type(schema, table, col) -> str:
    return TYPE_MAPPING[schema[table][col].split(",")[0]]


def is_pk(schema, table, col) -> bool:
    # A label in the CSVs only: the scan marks no column sorted, key or not.
    return schema[table].get("PRIMARY_KEY") == col


def scanned_cols_per_query() -> Dict[int, set]:
    """{qid: {(TABLE, COL)}} from the cached plan pickles' disk_read_cols."""
    out = {}
    for qid in range(1, 23):
        with open(REPO / "queries" / "pickles" / f"planobj-q{qid}.pkl", "rb") as f:
            plans = pickle.load(f)
        out[qid] = {(st.table.upper(), st.name.upper())
                    for plan in plans for st in plan.disk_read_cols}
    return out


def read_column(pt: ParquetTable, col: str, ttype: str) -> torch.Tensor:
    """The whole column on the host, cast as the scan casts it (1-D
    numeric, 2-D string)."""
    parts = [chunk[col].cpu() for _, chunk in pt.batches([col])]
    return cast(parts[0] if len(parts) == 1 else torch.cat(parts), ttype)


def make_runs(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(unsorted, sorted) 2-D contiguous CPU tensors of the column.

    unsorted = seeded random permutation; sorted = ascending (numeric/date)
    or row-wise lexicographic (multi-byte string).
    """
    g = torch.Generator().manual_seed(0)
    if t.dim() == 1:
        n = t.shape[0]
        uns = t[torch.randperm(n, generator=g)].unsqueeze(1).contiguous()
        srt = torch.sort(t)[0].unsqueeze(1).contiguous()
        return uns, srt
    n, w = t.shape
    uns = t[torch.randperm(n, generator=g)].contiguous()
    a = np.ascontiguousarray(t.cpu().numpy())
    order = np.argsort(a.view(f"S{w}").reshape(-1), kind="stable")
    srt = torch.from_numpy(a[order]).contiguous()
    return uns, srt


# ---------------------------------------------------------------------------
# Measurement (CUDA events, engine library code)
# ---------------------------------------------------------------------------

def _timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    res = fn()
    end.record()
    torch.cuda.synchronize()
    return res, start.elapsed_time(end) / 1000.0


def _round_trip(gpu: torch.Tensor, codec: str, chunk_b, pool):
    """One round trip of ``gpu`` through the engine's write and read paths:
    a region producer writes it as one tick, one region in ``pool`` (the
    pinned host pool; RAW ships the tensor's bytes at its compress), and a
    ``DirectIn`` reads the region back and decodes it.  Returns the four
    legs in seconds (compress and pack; the home writer's D2H span; the
    reader's H2D; its decode, ~0 for RAW), the region's stored bytes and
    the decoded column."""
    from IO.codec.region_producer import RegionProducer
    from IO.transfer.in_stage import DirectIn
    from variable import TableVariable

    producer = RegionProducer(pool, "sweep", chunk_b=chunk_b, spills=False)

    def _compress():
        producer.compress([(0, gpu, [gpu.shape[0]], codec)])
        producer.ticks_left = 1
        producer.pack()

    try:
        _, c_s = _timed(_compress)
        producer.wait_copies()
        o_ms, _ = producer.collect_lane()
        producer.drain()
    except BaseException:
        producer.abort()
        raise
    table = producer.tables()[0]
    try:
        stage = DirectIn(TableVariable(table, "int"))
        DirectIn.install_compiled_tick_layout([stage])
        agg_buf, plan = DirectIn.plan_input_data_movement([stage])
        _, i_s = _timed(lambda: DirectIn.launch_h2d(agg_buf, plan))
        stage.phase_finalize(None)
        _, d_s = _timed(stage.decompress)
        out = stage.gpu_col.data().clone()
        plan.release()
        stored = sum(r.nbytes for r in table.regions)
        del stage, agg_buf, plan
    finally:
        table.free(pool, None)
    return (c_s, o_ms / 1000.0, i_s, d_s), stored, out


def measure_raw(gpu: torch.Tensor, reps: int, pool) -> Dict[str, float]:
    """RAW = the D2H of a RAW region to its pinned home and the reader's
    H2D back, through the same paths as a codec's cell; min over reps."""
    best_o = best_i = float("inf")
    for rep in range(reps):
        (_, o, i, _), _, out = _round_trip(gpu, "RAW", None, pool)
        if rep == 0 and not torch.equal(out.reshape(-1), gpu.reshape(-1)):
            raise RuntimeError("RAW: round trip mismatch")
        best_o, best_i = min(best_o, o), min(best_i, i)
    return {"raw_d2h_s": best_o, "raw_h2d_s": best_i}


def measure_cell(gpu: torch.Tensor, codec: str, chunk_b: int, reps: int,
                 pool) -> Dict[str, float]:
    """One (column run, codec, chunk): the four legs, min over reps, plus
    the compression ratio, through ``_round_trip``; the decoded column is
    verified against the input."""
    best = {k: float("inf") for k in LEGS}
    comp_bytes = None
    for rep in range(reps):
        legs, comp_bytes, out = _round_trip(gpu, codec, chunk_b, pool)
        if rep == 0 and not torch.equal(out.reshape(-1), gpu.reshape(-1)):
            raise RuntimeError(f"{codec} @ {chunk_b} B: round trip mismatch")
        for k, v in zip(LEGS, legs):
            best[k] = min(best[k], v)
    best["ratio"] = gpu.nbytes / comp_bytes
    return best


# ---------------------------------------------------------------------------
# Aggregation and selection
# ---------------------------------------------------------------------------

def weighted_median(pairs: List[Tuple[float, float]]) -> float:
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return v
    return pairs[-1][0]


def winsorize(vals: List[float], frac: float = 0.10) -> List[float]:
    s = sorted(vals)
    k = int(len(s) * frac)
    lo, hi = s[k], s[-(k + 1)]
    return [min(max(v, lo), hi) for v in vals]


def class_cost(cols: list, per_col: dict, run: str, codec: str, chunk_b: int
               ) -> float:
    """Byte-weighted round-trip seconds per byte of ``codec``@``chunk_b``
    over ``cols`` (weight = scan_freq x raw bytes).  ``inf`` if the codec
    failed on any column of the class, so it can never be picked."""
    num = den = 0.0
    for tc in cols:
        e = per_col[tc]
        cell = e["runs"][run].get((codec, chunk_b))
        if cell is None:
            return float("inf")
        num += e["freq"] * cost_s(cell)
        den += e["freq"] * e["raw_bytes"]
    return num / den


def class_raw_cost(cols: list, per_col: dict, run: str) -> float:
    num = den = 0.0
    for tc in cols:
        e = per_col[tc]
        num += e["freq"] * raw_cost_s(e["raw"][run])
        den += e["freq"] * e["raw_bytes"]
    return num / den


def col_cost_per_byte(e: dict, run: str, codec: str, chunk_b: int) -> float:
    if codec == RAW:
        return raw_cost_s(e["raw"][run]) / e["raw_bytes"]
    cell = e["runs"][run].get((codec, chunk_b))
    return float("inf") if cell is None else cost_s(cell) / e["raw_bytes"]


def robustness_winners(cols: list, per_col: dict, run: str,
                       chunk_of: Dict[str, int]) -> Dict[str, str]:
    """Class winner under four outlier-robust schemes, every codec at its
    final chunk (RAW included as a candidate)."""
    cands = [RAW] + list(CODECS)
    v = {c: [col_cost_per_byte(per_col[tc], run, c, chunk_of.get(c, 0))
             for tc in cols] for c in cands}
    w = [per_col[tc]["freq"] * per_col[tc]["raw_bytes"] for tc in cols]
    out = {}
    out["median"] = min(cands, key=lambda c: sorted(v[c])[len(cols) // 2])
    out["wmedian"] = min(cands, key=lambda c: weighted_median(list(zip(v[c], w))))
    out["winsor_bw"] = min(
        cands, key=lambda c: (sum(x * y for x, y in zip(winsorize(v[c]), w)) / sum(w)
                              if all(np.isfinite(v[c])) else float("inf")))
    votes = defaultdict(int)
    for i in range(len(cols)):
        votes[min(cands, key=lambda c: v[c][i])] += 1
    out["vote"] = max(cands, key=lambda c: votes[c])
    return out


def select(per_col: dict, classes: Dict[Tuple[str, str], list]):
    """Pick the codec map and one chunk per codec.

    1. Per class ``(type, run)``, the byte-weighted cost of every
       (codec, chunk) cell and of RAW.
    2. Per class, the cheapest codec at its best chunk for that class (RAW
       wins if nothing beats it).
    3. Per codec, one chunk: the cheapest over the classes it won, or over
       all classes if it won none.
    4. Re-pick each class with every codec at its step-3 chunk, so the map
       and the chunks agree.

    Returns (best_map, chunk_of, step2_map, class_bytes)."""
    cost = {}        # (type, run) -> {(codec, chunk): s/B}
    rawc = {}        # (type, run) -> s/B
    class_bytes = {}
    for key, cols in classes.items():
        t, run = key
        cost[key] = {(c, ch): class_cost(cols, per_col, run, c, ch)
                     for c in CODECS for ch in CHUNKS_B}
        rawc[key] = class_raw_cost(cols, per_col, run)
        class_bytes[key] = sum(per_col[tc]["freq"] * per_col[tc]["raw_bytes"]
                               for tc in cols)

    # Step 2: per class, cheapest codec at its own best chunk; RAW if nothing beats it.
    step2 = {}
    for key in classes:
        best_c, best_v = RAW, rawc[key]
        for c in CODECS:
            v = min(cost[key][(c, ch)] for ch in CHUNKS_B)
            if v < best_v:
                best_c, best_v = c, v
        step2[key] = best_c

    # Step 3: one chunk per codec, byte-weighted over the classes it won
    # (all classes for a codec that won none).
    chunk_of = {}
    for c in CODECS:
        won = [k for k in classes if step2[k] == c] or list(classes)
        chunk_of[c] = min(
            CHUNKS_B, key=lambda ch: sum(cost[k][(c, ch)] * class_bytes[k] for k in won))

    # Step 4: re-pick every class with each codec at its final chunk.
    final = {}
    for key in classes:
        best_c, best_v = RAW, rawc[key]
        for c in CODECS:
            v = cost[key][(c, chunk_of[c])]
            if v < best_v:
                best_c, best_v = c, v
        final[key] = best_c
    return final, chunk_of, step2, cost, rawc, class_bytes


def pareto(pts: Dict[str, Tuple[float, float]]) -> set:
    """Names not dominated in (cost lower, ratio higher)."""
    keep = set()
    for a, (ta, ra) in pts.items():
        if any(b != a and tb <= ta and rb >= ra and (tb < ta or rb > ra)
               for b, (tb, rb) in pts.items()):
            continue
        keep.add(a)
    return keep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline codec sweep -> profiling/results/codec_map.json")
    ap.add_argument("--reps", type=int, default=3,
                    help="repetitions per cell; each leg keeps its minimum")
    ap.add_argument("--out", default=str(RESULTS_DIR),
                    help="results directory (default profiling/results)")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps must be >= 1")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from TQPlib.tqpcompress import NVCOMP_MGR_VERSION, mgr_release_scratch
    from IO.pages.pinned_mem import PinnedMemory
    from utility.numa import pin_to_gpu_node

    torch.cuda.init()
    torch.zeros(1, device=DEVICE)          # CUDA context before the NUMA pin
    t_start = time.time()
    print(f"[sweep] GPU {torch.cuda.get_device_name(0)} | nvCOMP {NVCOMP_MGR_VERSION} "
          f"| torch {torch.__version__} | SF{SF} | reps={args.reps}",
          flush=True)
    print(f"[sweep] grid: codecs {list(CODECS)} x chunk_b "
          f"{[c >> 10 for c in CHUNKS_B]} KiB x runs {list(RUNS)}", flush=True)
    node, cpu0, ncpu = pin_to_gpu_node(0)
    if node is not None:
        print(f"[variance] pinned all threads to NUMA node{node} ({ncpu} CPUs, "
              f"from cpu{cpu0}) local to GPU0; PyTorch intra-op threads="
              f"{torch.get_num_threads()}", flush=True)
    else:
        print(f"[variance] NUMA pin failed ({cpu0}); continuing unpinned", flush=True)

    schema = load_schema()
    freq: Dict[Tuple[str, str], int] = defaultdict(int)
    for cols in scanned_cols_per_query().values():
        for tc in cols:
            freq[tc] += 1
    unique_cols = sorted(freq)
    root = load_config()["tables path"]
    tables = sorted({t.lower() for t, _ in unique_cols})
    missing = [table_dir(root, SF, t) for t in tables
               if not os.path.isdir(table_dir(root, SF, t))]
    if missing:
        print(f"[sweep] ERROR: {len(missing)} SF{SF} table(s) missing, e.g. "
              f"{missing[0]}\n[sweep] run: make data SF={SF}",
              file=sys.stderr)
        return 1
    readers = {t: ParquetTable(root, SF, t) for t in tables}
    print(f"[sweep] {len(unique_cols)} unique scanned columns across 22 queries",
          flush=True)
    # The regions' homes: a pinned pool on GPU 0's NUMA node, as main.py's,
    # sized for one compressed copy of the largest column (a region is at
    # most its worst case) and freed after every cell.
    largest = 0
    for t, c in unique_cols:
        pt = readers[t.lower()]
        dtype, width = pt.encoding(c)
        largest = max(largest, pt.rows * width * dtype.itemsize)
    pool_gb = max(1, -(-3 * largest // GB))
    pool = PinnedMemory(capacity_gb=pool_gb, block_size_mb=PINNED_BLOCK_MB)
    print(f"[sweep] pinned pool {pool_gb} GiB on NUMA node {pool.numa_node}",
          flush=True)

    # per_col[(table, col)] = {type, is_pk, freq, raw_bytes,
    #                          raw: {run: {raw_d2h_s, raw_h2d_s}},
    #                          runs: {run: {(codec, chunk): legs+ratio}}}
    per_col: dict = {}
    for ci, (table, col) in enumerate(unique_cols, 1):
        t0 = time.time()
        ttype = col_type(schema, table, col)
        raw = read_column(readers[table.lower()], col, ttype)
        # make_runs takes the column as loaded (1-D numeric, 2-D string) and
        # returns 2-D runs, the shape ensure_2d gives the scan path.
        uns, srt = make_runs(raw)
        assert uns.shape == ensure_2d(raw).shape
        del raw
        e = {"type": ttype, "is_pk": is_pk(schema, table, col),
             "freq": freq[(table, col)], "raw_bytes": uns.nbytes,
             "raw": {}, "runs": {}}
        for run, cpu_t in (("unsorted", uns), ("sorted", srt)):
            gpu = cpu_t.to(DEVICE)
            e["raw"][run] = measure_raw(gpu, args.reps, pool)
            cells = {}
            for codec in CODECS:
                for chunk_b in CHUNKS_B:
                    try:
                        cells[(codec, chunk_b)] = measure_cell(gpu, codec, chunk_b,
                                                               args.reps, pool)
                    except Exception as ex:       # recorded as a hole, printed
                        print(f"  ! {col} {run} {codec}@{chunk_b >> 10}K failed: {ex}",
                              flush=True)
                    mgr_release_scratch()
                    torch.cuda.empty_cache()
            e["runs"][run] = cells
            del gpu
            torch.cuda.empty_cache()
        per_col[(table, col)] = e

        def _best(run):
            ok = [(k, v) for k, v in e["runs"][run].items()]
            if not ok:
                return "none"
            k, v = min(ok, key=lambda kv: cost_s(kv[1]))
            return f"{k[0]}@{k[1] >> 10}K {v['ratio']:.1f}x"
        print(f"[col {ci:>2}/{len(unique_cols)}] {col:<16} {ttype:<7}"
              f"{'PK' if e['is_pk'] else '  '} x{e['freq']:<2} "
              f"{e['raw_bytes'] / 1e6:8.1f} MB | unsorted {_best('unsorted'):<22} "
              f"| sorted {_best('sorted'):<22} ({time.time() - t0:.0f}s)", flush=True)
        del uns, srt

    # ---- classes and selection ----
    classes: Dict[Tuple[str, str], list] = {}
    for t in MAPPED_TYPES:
        cols_t = [tc for tc in unique_cols if per_col[tc]["type"] == t]
        if not cols_t:
            continue
        for run in RUNS:
            classes[(t, run)] = cols_t
    final, chunk_of, step2, cost, rawc, class_bytes = select(per_col, classes)

    stable = {}
    for key, cols in classes.items():
        wins = robustness_winners(cols, per_col, key[1], chunk_of)
        stable[key] = all(w == final[key] for w in wins.values())
        flag = "stable" if stable[key] else "UNSTABLE"
        print(f"[stable] {key[0]:<7}{key[1]:<9} -> {final[key]:<9} {flag}  "
              + " ".join(f"{s}={w}" for s, w in wins.items()), flush=True)

    # ---- codec_map.json ----
    codec_json = {t: {"unsorted": final.get((t, "unsorted"), RAW),
                      "sorted": final.get((t, "sorted"), RAW)} for t in MAPPED_TYPES}
    codec_map = {"codec": codec_json,
                 "settings": {c: {"chunk_b": chunk_of[c]} for c in CODECS}}
    from compression.format_policy import parse_codec_map
    parse_codec_map(codec_map, "codec_map (in memory)")     # never write a bad file
    with open(out_dir / "codec_map.json", "w", encoding="utf-8") as f:
        json.dump(codec_map, f, indent=2)
        f.write("\n")

    print("\n[codec_map] codec per (type, run):", flush=True)
    for t in MAPPED_TYPES:
        for run in RUNS:
            key = (t, run)
            if key not in classes:
                continue
            note = "" if step2[key] == final[key] else f"  (step 2 picked {step2[key]})"
            print(f"  {t:<7} {run:<9} -> {final[key]:<9} "
                  f"{cost[key].get((final[key], chunk_of.get(final[key], 0)), rawc[key]) * GB * 1e3:8.2f} ms/GB"
                  f"  (RAW {rawc[key] * GB * 1e3:8.2f} ms/GB){note}", flush=True)
    print("[codec_map] chunk_b per codec: "
          + " ".join(f"{c}={chunk_of[c]}" for c in CODECS), flush=True)

    # ---- sweep_per_column.csv ----
    with open(out_dir / "sweep_per_column.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "column", "type", "run", "is_pk", "scan_freq",
                    "raw_bytes", "codec", "chunk_b", "selected", "ratio",
                    "compress_s", "d2h_s", "h2d_s", "decompress_s",
                    "raw_d2h_s", "raw_h2d_s"])
        for (table, col), e in per_col.items():
            for run in RUNS:
                r = e["raw"][run]
                for (codec, chunk_b), cell in sorted(e["runs"][run].items()):
                    w.writerow([table, col, e["type"], run, int(e["is_pk"]), e["freq"],
                                e["raw_bytes"], codec, chunk_b,
                                int(chunk_of[codec] == chunk_b), f"{cell['ratio']:.4f}",
                                *(f"{cell[k]:.6f}" for k in LEGS),
                                f"{r['raw_d2h_s']:.6f}", f"{r['raw_h2d_s']:.6f}"])

    # ---- sweep_by_class.csv ----
    with open(out_dir / "sweep_by_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "run", "scanned_bytes", "codec", "chunk_b", "ratio",
                    "compress_s_per_gb", "d2h_s_per_gb", "h2d_s_per_gb",
                    "decompress_s_per_gb", "cost_s_per_gb", "selected",
                    "on_pareto", "is_best", "stable"])
        for key, cols in classes.items():
            t, run = key
            den = class_bytes[key]
            rows = {}
            for codec in CODECS:
                for chunk_b in CHUNKS_B:
                    legs = {k: 0.0 for k in LEGS}
                    comp = 0.0
                    ok = True
                    for tc in cols:
                        e = per_col[tc]
                        cell = e["runs"][run].get((codec, chunk_b))
                        if cell is None:
                            ok = False
                            break
                        for k in LEGS:
                            legs[k] += e["freq"] * cell[k]
                        comp += e["freq"] * e["raw_bytes"] / cell["ratio"]
                    if not ok:
                        continue
                    rows[(codec, chunk_b)] = {
                        "ratio": den / comp,
                        **{k: legs[k] / den * GB for k in LEGS},
                        "cost": cost[key][(codec, chunk_b)] * GB,
                        "selected": int(chunk_of[codec] == chunk_b)}
            raw_o = sum(per_col[tc]["freq"] * per_col[tc]["raw"][run]["raw_d2h_s"]
                        for tc in cols) / den * GB
            raw_i = sum(per_col[tc]["freq"] * per_col[tc]["raw"][run]["raw_h2d_s"]
                        for tc in cols) / den * GB
            rows[(RAW, 0)] = {"ratio": 1.0, "compress_s": 0.0, "d2h_s": raw_o,
                              "h2d_s": raw_i, "decompress_s": 0.0,
                              "cost": rawc[key] * GB, "selected": 1}
            front = pareto({k: (v["cost"], v["ratio"])
                            for k, v in rows.items() if v["selected"]})
            for (codec, chunk_b), v in sorted(rows.items()):
                w.writerow([t, run, den, codec, chunk_b, f"{v['ratio']:.4f}",
                            *(f"{v[k]:.6f}" for k in LEGS), f"{v['cost']:.6f}",
                            v["selected"], int((codec, chunk_b) in front),
                            int(v["selected"] and codec == final[key]),
                            int(stable[key])])

    # ---- scan_frequency.csv ----
    scanned = {tc: per_col[tc]["freq"] * per_col[tc]["raw_bytes"] for tc in unique_cols}
    with open(out_dir / "scan_frequency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "column", "type", "is_pk", "scan_freq",
                    "raw_bytes", "scanned_bytes_workload"])
        for tc in sorted(unique_cols, key=lambda x: -scanned[x]):
            e = per_col[tc]
            w.writerow([tc[0], tc[1], e["type"], int(e["is_pk"]), e["freq"],
                        e["raw_bytes"], scanned[tc]])

    print(f"\n[sweep] wrote codec_map.json, sweep_per_column.csv, sweep_by_class.csv, "
          f"scan_frequency.csv to {out_dir} ({time.time() - t_start:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
