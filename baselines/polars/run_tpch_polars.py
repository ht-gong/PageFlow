#!/usr/bin/env python
"""Run native-Polars TPC-H queries 1-22 on the GPU (cudf-polars) over a parquet
dataset (normally the ramdisk copy).

Protocol (as every engine, see AGENTS.md): an untimed warm pass first runs the
same queries on the SF1 dataset in $WARM_PARQUET (GPU init, JIT, reader setup);
then each query runs once on <input_dir>, timed. A query that fails in the warm
pass is recorded FAIL and not timed. Results are written incrementally to a CSV
so a watchdog kill never loses prior rows. q11's FRACTION is for $TPCH_SF, the
target scale factor, in both passes.

Spill pass (AGENTS.md; run_polars_gpu.sh sets SPILL_STATS_DIR and
RAPIDSMPF_STATISTICS=1): each pass is metered (spill_stats.Meter) and the meters
go to $SPILL_STATS_DIR/q<N>/meter.json.

LOAD_MODE=resident (engine key polars_gpu_resident, see AGENTS.md): the tables
are decoded into host DataFrames, untimed, and the in-memory executor runs over
them with device_cache.py's patch, which keeps each uploaded column on the GPU.
The SF1 warm pass records which columns the query scans; before the timed run
exactly those target columns are uploaded (untimed), and a timed run that still
copies a column host->device is FAIL. No CPU fallback (raise_on_fail).

  run_tpch_polars.py <input_dir> <out_csv> [SUBSET] [append]
    SUBSET  comma-separated query numbers, e.g. "9" or "1,2,3"  (default: 1..22)
"""
import json
import os
import sys
import time
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tpch_queries import queries
import spill_stats

SPILL_STATS_DIR = os.environ.get("SPILL_STATS_DIR", "")

RESIDENT = os.environ.get("LOAD_MODE", "parquet") == "resident"
if RESIDENT:
    import device_cache


def build_engine():
    """The streaming executor with modest partitions, spilling device->host via
    rapidsmpf (so SF500 fits one 32 GB GPU), on the explicit SPMDEngine of
    spill_stats.build_engine (why: its docstring). GPU_MR: "async" (rapidsmpf
    spill) or "managed" (UVM: oversubscribes into host RAM, robust but slow).
    Resident: the in-memory executor, the only one that scans a host frame
    through the patched DataFrameScan (device_cache.py), and no CPU fallback.
    Its memory resource is given explicitly: left unset, the in-memory executor
    allocates from a managed (UVM) pool, which pages the "resident" columns out
    to host memory instead of failing once the working set exceeds the GPU."""
    if RESIDENT:
        import rmm
        return pl.GPUEngine(device=0, executor="in-memory", raise_on_fail=True,
                            memory_resource=rmm.mr.CudaAsyncMemoryResource())
    part_mb = int(os.environ.get("GPU_PART_MB", "256"))
    return spill_stats.build_engine(part_mb, os.environ.get("GPU_MR", "async"))


class PassMeters:
    """Spill pass: each pass's meter record, kept in q<N>/meter.json as soon as
    the pass ends (a killed process keeps the passes it finished). A meter
    that fails is reported and left out (the query's record is then a stub),
    never the query's failure."""

    def __init__(self, engine):
        self.meter = spill_stats.Meter(engine) if SPILL_STATS_DIR else None
        self.done = {}

    def begin(self):
        if self.meter:
            self.meter.begin()

    def end(self, label, n):
        if self.meter:
            try:
                self.done.setdefault(n, {})[label] = self.meter.end()
            except Exception as e:
                print(f"[spill] q{n} {label} pass: meter failed: {e}", flush=True)
                return
            with open(os.path.join(SPILL_STATS_DIR, f"q{n}", "meter.json"), "w") as f:
                json.dump(self.done[n], f, indent=1)

TABLES = ["customer", "lineitem", "nation", "orders",
          "part", "partsupp", "region", "supplier"]
# Monetary columns stored as Decimal -> read as Float64 for robust large sums.
DECIMAL_COLS = {"l_quantity", "l_extendedprice", "l_discount", "l_tax",
                "o_totalprice", "ps_supplycost", "c_acctbal",
                "s_acctbal", "p_retailprice"}


def scan(input_dir, table):
    lf = pl.scan_parquet(f"{input_dir}/{table}/*.parquet")
    cols = lf.collect_schema().names()
    exprs = []
    for c in cols:
        if c == "ignore":
            continue
        exprs.append(pl.col(c).cast(pl.Float64) if c in DECIMAL_COLS else pl.col(c))
    return lf.select(exprs)


def load_host(input_dir):
    """Resident: every table decoded into a host DataFrame (untimed), as lazy
    frames over it."""
    t0 = time.time()
    lf = {t: scan(input_dir, t).collect().lazy() for t in TABLES}
    print(f"decoded {input_dir} into host memory in {time.time()-t0:.1f}s (excluded from query timings)")
    return lf


def upload(lf, cols, engine):
    """Resident: put the given columns of the tables in lf on the GPU, untimed. A
    min() per column keeps every column through projection pushdown, so the scan
    uploads each one (into device_cache) while only one row comes back."""
    t0 = time.time()
    device_cache.clear()
    device_cache.fill()
    for t, frame in lf.items():
        mine = [c for c in cols if c in frame.collect_schema().names()]
        if mine:
            frame.select([pl.col(c).min() for c in mine]).collect(engine=engine)
    device_cache.freeze()
    print(f"uploaded {len(cols)} columns ({device_cache.resident_bytes()/2**30:.2f} GiB) to the GPU "
          f"in {time.time()-t0:.1f}s (excluded from query timings)")


def main():
    INPUT = sys.argv[1]
    OUT_CSV = sys.argv[2]
    SUBSET = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else list(range(1, 23))
    APPEND = len(sys.argv) > 4 and sys.argv[4] == "append"

    warm_dir = os.environ.get("WARM_PARQUET") or sys.exit(
        "WARM_PARQUET is not set (the SF1 dataset of the warm pass)")
    sf = os.environ.get("TPCH_SF") or sys.exit("TPCH_SF is not set (the target scale factor)")
    query = queries(sf)

    print(f"Polars {pl.__version__} | engine=gpu | threads={pl.thread_pool_size()} | "
          f"load={'resident (in-memory executor, device_cache)' if RESIDENT else 'parquet'}")
    engine = build_engine()
    try:
        run_passes(engine, INPUT, OUT_CSV, SUBSET, APPEND, warm_dir, query)
    finally:
        # the SPMDEngine, on the thread that built it, or rapidsmpf calls
        # std::terminate (a plain pl.GPUEngine has nothing to shut down)
        if hasattr(engine, "shutdown"):
            engine.shutdown()


def run_passes(engine, INPUT, OUT_CSV, SUBSET, APPEND, warm_dir, query):
    meters = PassMeters(engine)

    # Warm pass: the same queries on SF1, untimed (q11 keeps the target SF's
    # FRACTION, so the warm pass runs the timed run's query). Resident: over
    # host frames, keeping what it uploads; its uploads are the columns the
    # queries scan.
    if RESIDENT:
        warm = load_host(warm_dir)
        device_cache.clear()
        device_cache.fill()
    else:
        warm = {t: scan(warm_dir, t) for t in TABLES}
    warm_failed = {}
    t0 = time.time()
    for n in SUBSET:
        tq = time.time()
        meters.begin()
        try:
            query[n](warm).collect(engine=engine)
        except Exception as e:
            warm_failed[n] = (time.time() - tq, str(e).splitlines()[0][:90].replace(",", ";"))
        meters.end("warm", n)
    print(f"SF1 warm pass done in {time.time()-t0:.1f}s (excluded from query timings)")

    if RESIDENT:
        scanned = list(dict.fromkeys(device_cache.misses))
        del warm
        device_cache.clear()
        lf = load_host(INPUT)
        tq = time.time()
        try:
            upload(lf, scanned, engine)
        except Exception as e:
            # The query's columns do not fit on the GPU: every query fails.
            device_cache.clear()
            msg = "resident upload: " + str(e).splitlines()[0].replace(",", ";")
            for n in SUBSET:
                warm_failed.setdefault(n, (time.time() - tq, msg))
    else:
        lf = {t: scan(INPUT, t) for t in TABLES}

    write_header = not (APPEND and os.path.exists(OUT_CSV))
    f = open(OUT_CSV, "a" if APPEND else "w")
    if write_header:
        f.write("query,status,seconds,result_rows_or_error\n")
        f.flush()

    print(f"\n{'query':10} {'status':8} {'secs':>9} {'rows':>10}")
    ok = 0
    for n in SUBSET:
        name = f"query{n}"
        if n in warm_failed:
            dt, msg = warm_failed[n]
            msg = (msg if msg.startswith("resident upload") else f"SF1 warm pass: {msg}")[:90]
            print(f"{name:10} {'FAIL':8} {dt:9.2f}   {msg}")
            f.write(f"{name},FAIL,{dt:.3f},{msg}\n")
            f.flush()
            continue
        meters.begin()
        if RESIDENT:
            device_cache.reset_counters()
        t0 = time.time()
        try:
            res, err = query[n](lf).collect(engine=engine), None
        except Exception as e:
            res, err = None, e
        dt = time.time() - t0
        meters.end("timed", n)
        if err is None:
            nrows = res.height
            if RESIDENT and device_cache.misses:
                # A column copied host->device inside the timed run.
                msg = "device_cache_miss:" + "+".join(sorted(set(device_cache.misses)))
                print(f"{name:10} {'FAIL':8} {dt:9.2f}   {msg}")
                f.write(f"{name},FAIL,{dt:.3f},{msg[:90]}\n")
                f.flush()
                continue
            print(f"{name:10} {'OK':8} {dt:9.2f} {nrows:10d}"
                  + (f"   (scans served from the GPU: {device_cache.hits})" if RESIDENT else ""))
            f.write(f"{name},OK,{dt:.3f},{nrows}\n")
            ok += 1
        else:
            msg = str(err).splitlines()[0][:90].replace(",", ";")
            print(f"{name:10} {'FAIL':8} {dt:9.2f}   {msg}")
            f.write(f"{name},FAIL,{dt:.3f},{msg}\n")
        f.flush()
    f.close()

    print(f"\n{ok}/{len(SUBSET)} queries OK | csv -> {OUT_CSV}")
    sys.exit(0 if ok == len(SUBSET) else 1)


if __name__ == "__main__":
    main()
