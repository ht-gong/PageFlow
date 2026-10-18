#!/usr/bin/env python
"""Run NDS-H (TPC-H) queries 1-22 on Spark with the RAPIDS GPU accelerator.

Registers each parquet table (one sub-dir per table) as a temp view, then
executes the marker-delimited query stream produced by qgen, read through
results/queries/stream.py (incl. multi-statement Q15) with q11's FRACTION for
$TPCH_SF.

Protocol (as every engine, see AGENTS.md): an untimed warm pass first runs the
same queries on the SF1 dataset in $WARM_PARQUET (GPU init, kernel JIT, reader
setup); then the views point at <input_dir> and each query runs once, timed.
A query that fails in the warm pass is recorded FAIL and not timed. Per-query
wall time + GPU-operator count are reported.

Spill pass (AGENTS.md; run_tpch_safe.sh sets RAPIDS_STATS_DIR to the query's
evidence dir and turns the event log on): jobs and SQL executions are tagged
with their pass for eventlog_stats.py, and PassStats records what the event log
lacks.

LOAD_MODE=resident (engine key rapids_resident, see AGENTS.md): before each
pass, untimed, the query's columns (../sirius/pin_columns.py) of each table it
reads are persisted MEMORY_ONLY in Spark's cache and materialized, and the
views point at the cached frames. Spark-RAPIDS has no GPU-resident cache, so
this is the closest it gets: the cache is in host memory, in the serializer
run_tpch_safe.sh configures (ParquetCachedBatchSerializer: parquet-encoded
batches the GPU decodes). The frames are read through hard links that are
deleted once the cache is built, so a timed run that falls back to the parquet
(an evicted or never-cached block) fails instead of reading it: FAIL,
cache_miss:<tables>.

  spark-submit ... run_tpch_queries.py <input_dir> <stream.sql> <out_csv> [SUBSET] [append]
"""
import glob, json, os, re, shutil, sys, threading, time
from pyspark.sql import SparkSession

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "results", "queries"))
from stream import queries as stream_queries

RESIDENT = os.environ.get("LOAD_MODE", "parquet") == "resident"
if RESIDENT:
    from pyspark import StorageLevel
    sys.path.insert(0, os.path.join(HERE, "..", "sirius"))
    from pin_columns import QUERY_COLUMNS
# Resident: the per-process directory of hard links the cache is built from.
LINK_DIR = f".rapids_resident_{os.getpid()}"

INPUT, STREAM, OUT_CSV = sys.argv[1], sys.argv[2], sys.argv[3]
# optional 4th arg: comma-separated template numbers to run (e.g. "9" or "1,2,3")
SUBSET = {int(q) for q in sys.argv[4].split(",")} if len(sys.argv) > 4 and sys.argv[4] else None
# optional 5th arg: append to OUT_CSV instead of overwriting (for per-query runs)
APPEND = len(sys.argv) > 5 and sys.argv[5] == "append"
TABLES = ["customer", "lineitem", "nation", "orders",
          "part", "partsupp", "region", "supplier"]
# Movement stats (eventlog_stats.py): run_tpch_safe.sh sets it to the query's stats dir.
STATS_DIR = os.environ.get("RAPIDS_STATS_DIR")


def tag(spark, pass_name, num):
    """Label the jobs and SQL executions that follow: eventlog_stats.py splits the event
    log into passes on tpch.pass / tpch.query (stages) and the description (executions).
    Local properties, not only setJobGroup: a broadcast exchange's job replaces the job
    group with its own run id, but inherits the local properties. Spill pass only."""
    if not STATS_DIR:
        return
    sc = spark.sparkContext
    sc.setLocalProperty("tpch.pass", pass_name)
    sc.setLocalProperty("tpch.query", str(num))
    sc.setJobDescription(f"tpch:{pass_name}:q{num}")


class PassStats:
    """Per-pass residency the event log lacks, written to $RAPIDS_STATS_DIR:
    - runner.json: per pass the process-wide GPU high-water mark of live RMM allocations
      (cudf's tracking adaptor, always installed by the plugin; reset at the pass start),
      start/end times;
    - monitor.csv: 1 s samples (appended, so a killed process keeps them) of the bytes
      under spark.local.dir by kind (shuffle_* files, temp_local_* spill files), the
      RAPIDS host spill store's size, RMM's current allocation and /proc/<jvm>/io."""

    def __init__(self, spark):
        self.jvm, self.passes, self.cur = spark.sparkContext._jvm, {}, "init"
        self.local = spark.conf.get("spark.local.dir")
        self.pid = os.getppid()          # spark-submit's JVM runs this python as its child
        threading.Thread(target=self._sample, daemon=True).start()

    def _j(self, fn):
        try:
            return fn()
        except Exception:
            return ""

    def begin(self, pass_name, num):
        self.cur = f"{pass_name}:q{num}"
        # scoped max = max of (initial value + net allocations since the reset): seeding it
        # with the live total makes it the absolute high-water mark of this pass
        rmm = self.jvm.ai.rapids.cudf.Rmm
        self._j(lambda: rmm.resetScopedMaximumBytesAllocated(rmm.getTotalBytesAllocated()))
        self.passes[self.cur] = {"t0": time.time()}

    def end(self):
        p = self.passes[self.cur]
        p["t1"] = time.time()
        p["gpu_peak_process_bytes"] = self._j(
            lambda: self.jvm.ai.rapids.cudf.Rmm.getScopedMaximumBytesAllocated())
        self.cur = "between"
        self.write()

    def write(self):
        with open(os.path.join(STATS_DIR, "runner.json"), "w") as f:
            json.dump({"passes": self.passes}, f)

    def _sample(self):
        j = self.jvm
        with open(os.path.join(STATS_DIR, "monitor.csv"), "a") as f:
            f.write("t,pass,shuffle_bytes,spill_bytes,host_store_bytes,rmm_bytes,rchar,wchar\n")
            while True:
                kinds = {"shuffle_": 0, "temp_": 0}
                for root, _, files in os.walk(self.local):
                    for n in files:
                        k = "shuffle_" if n.startswith("shuffle_") else "temp_" if n.startswith("temp_") else None
                        if k:
                            try:
                                kinds[k] += os.lstat(os.path.join(root, n)).st_blocks * 512
                            except OSError:
                                pass
                io = {}
                try:
                    with open(f"/proc/{self.pid}/io") as g:
                        io = dict((a.strip(), int(b)) for a, b in (ln.split(":") for ln in g))
                except OSError:
                    pass
                hs = self._j(lambda: j.com.nvidia.spark.rapids.spill.SpillFramework.stores().hostStore().totalSize())
                rmm = self._j(lambda: j.ai.rapids.cudf.Rmm.getTotalBytesAllocated())
                f.write(f"{time.time():.1f},{self.cur},{kinds['shuffle_']},{kinds['temp_']},{hs},{rmm},"
                        f"{io.get('rchar', '')},{io.get('wchar', '')}\n")
                f.flush()
                time.sleep(1)


def _code_only(stmt):
    """statement text with leading '--' comment lines removed, lowercased."""
    return "\n".join(l for l in stmt.splitlines()
                     if not l.strip().startswith("--")).strip().lower()


def register(spark, root):
    """One temp view per table over the parquet under root (replacing earlier ones)."""
    for t in TABLES:
        spark.read.parquet(f"{root}/{t}").createOrReplaceTempView(t)


def cache(spark, root, nums):
    """Resident: drop the earlier cache, then one temp view per table the queries
    in nums read, over its columns (union over nums) cached MEMORY_ONLY and
    materialized. The cache is built from hard links to root's files (copies
    across file systems), deleted afterwards. Raises if a cached frame is not
    wholly in memory."""
    spark.catalog.clearCache()
    cols = {}
    for n in nums:
        for t, cs in QUERY_COLUMNS[n].items():
            have = cols.setdefault(t, [])
            have += [c for c in cs if c not in have]
    links = os.path.join(os.path.dirname(root.rstrip("/")), LINK_DIR)
    shutil.rmtree(links, ignore_errors=True)
    t0 = time.time()
    try:
        for t, cs in cols.items():
            os.makedirs(f"{links}/{t}")
            for f in glob.glob(f"{root}/{t}/*.parquet"):
                dst = f"{links}/{t}/{os.path.basename(f)}"
                try:
                    os.link(f, dst)
                except OSError:
                    shutil.copy2(f, dst)
            df = spark.read.parquet(f"{links}/{t}").select(*cs).persist(StorageLevel.MEMORY_ONLY)
            df.createOrReplaceTempView(t)
            df.count()
    finally:
        shutil.rmtree(links, ignore_errors=True)
    infos = spark.sparkContext._jsc.sc().getRDDStorageInfo()
    short = [f"{i.numCachedPartitions()}/{i.numPartitions()}" for i in infos
             if i.numCachedPartitions() < i.numPartitions() or i.diskSize() > 0]
    if len(infos) < len(cols) or short:
        raise RuntimeError(f"cache incomplete: {len(infos)} of {len(cols)} tables; "
                           f"partitions cached {' '.join(short) or 'all'}")
    gib = sum(i.memSize() for i in infos) / 2**30
    print(f"cached {sum(map(len, cols.values()))} columns of {len(cols)} tables from {root} "
          f"({gib:.2f} GiB in memory) in {time.time()-t0:.1f}s (excluded from query timings)", flush=True)


def load(spark, root, nums):
    """A pass's views: over the parquet under root, or (resident) over the cache
    of the queries in nums. Resident: returns the error that fails every query
    when the cache cannot be built (else None)."""
    if not RESIDENT:
        register(spark, root)
        return None
    try:
        cache(spark, root, nums)
    except Exception as e:
        return short(getattr(e, "java_exception", None) or e)  # Java's message, not Py4J's


def cache_miss(e):
    """Resident: the tables a failed run tried to read from the deleted links."""
    return sorted(set(re.findall(re.escape(LINK_DIR) + r"/(\w+)/", str(e))))


def run(spark, stmts):
    """Run one query's statements; return (result rows, GPU operators in the plan)."""
    gpu_ops, nrows = 0, 0
    for stmt in stmts:
        df = spark.sql(stmt)
        code = _code_only(stmt)
        if code.startswith("select") or code.startswith("with"):
            res = df.collect()
            nrows = len(res)
            # read the plan AFTER execution so AQE's finalized (GPU) plan is reflected
            plan = df._jdf.queryExecution().executedPlan().toString()
            gpu_ops = sum(1 for ln in plan.splitlines() if "Gpu" in ln)
        else:
            df.collect()  # DDL (create/drop view)
    return nrows, gpu_ops


def short(e):
    """An exception's first line, fit for the CSV."""
    return str(e).splitlines()[0][:80].replace(",", ";")


def collect_garbage(spark):
    """Force a JVM GC so Spark's ContextCleaner deletes finished shuffle files now,
    instead of letting scratch accumulate across the session."""
    try:
        spark.sparkContext._jvm.System.gc()
        time.sleep(2)
    except Exception:
        pass


def main():
    warm = os.environ.get("WARM_PARQUET") or sys.exit(
        "WARM_PARQUET is not set (the SF1 dataset of the warm pass)")
    sf = os.environ.get("TPCH_SF") or sys.exit("TPCH_SF is not set (the target scale factor)")
    spark = SparkSession.builder.appName("NDS-H Power Run (RAPIDS GPU)").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("Spark", spark.version, "| rapids.sql.enabled =",
          spark.conf.get("spark.rapids.sql.enabled", "false"),
          "| plugins =", spark.conf.get("spark.plugins", "<none>"))

    queries = stream_queries(sf, STREAM)
    if SUBSET:
        queries = {k: v for k, v in queries.items() if k in SUBSET}

    # Warm pass: the same queries on SF1, untimed (q11 keeps the target SF's
    # FRACTION, so the warm pass runs the timed run's text). Resident: over
    # SF1's cache, built the same way.
    stats = PassStats(spark) if STATS_DIR else None
    tag(spark, "warm", 0)
    t0 = time.time()
    err = load(spark, warm, queries)
    warm_failed = {num: (time.time() - t0, err) for num in queries} if err else {}
    for num, stmts in queries.items():
        if num in warm_failed:
            continue
        tag(spark, "warm", num)
        stats and stats.begin("warm", num)
        tq = time.time()
        try:
            run(spark, stmts)
        except Exception as e:
            warm_failed[num] = (time.time() - tq, short(e))
        stats and stats.end()
    tag(spark, "gc", 0)
    collect_garbage(spark)
    print(f"SF1 warm pass done in {time.time()-t0:.1f}s (excluded from query timings)")

    tag(spark, "setup", 0)
    t0 = time.time()
    err = load(spark, INPUT, queries)
    load_failed = {num: (time.time() - t0, "resident cache: " + err) for num in queries} if err else {}
    print("Registered temp views:", ", ".join(TABLES))
    rows = []
    print(f"\n{'query':10} {'status':8} {'secs':>8} {'gpu_ops':>8} {'result_rows':>12}")
    for num, stmts in queries.items():
        name = f"query{num}"
        if num in warm_failed or num in load_failed:
            dt, msg = warm_failed[num] if num in warm_failed else load_failed[num]
            msg = (f"SF1 warm pass: {msg}" if num in warm_failed else msg)[:80]
            print(f"{name:10} {'FAIL':8} {dt:8.2f}   {msg}")
            rows.append((name, "FAIL", f"{dt:.3f}", 0, msg))
            continue
        tag(spark, "timed", num)
        stats and stats.begin("timed", num)
        t0 = time.time()
        try:
            (nrows, gpu_ops), exc = run(spark, stmts), None
        except Exception as e:
            exc = e
        dt = time.time() - t0
        stats and stats.end()
        if exc is None:
            tag(spark, "gc", num)
            print(f"{name:10} {'OK':8} {dt:8.2f} {gpu_ops:8d} {nrows:12d}")
            rows.append((name, "OK", f"{dt:.3f}", gpu_ops, nrows))
            collect_garbage(spark)
        else:
            missed = cache_miss(exc) if RESIDENT else []
            # Resident: the run fell back to the deleted parquet links.
            msg = ("cache_miss:" + "+".join(missed)) if missed else short(exc)
            print(f"{name:10} {'FAIL':8} {dt:8.2f}   {msg}")
            rows.append((name, "FAIL", f"{dt:.3f}", 0, msg))

    write_header = not (APPEND and os.path.exists(OUT_CSV))
    with open(OUT_CSV, "a" if APPEND else "w") as f:
        if write_header:
            f.write("query,status,seconds,gpu_ops,result_rows_or_error\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")

    ok = [r for r in rows if r[1] == "OK"]
    total = sum(float(r[2]) for r in ok)
    print(f"\n{len(ok)}/{len(rows)} queries OK | total GPU wall time {total:.1f}s | csv -> {OUT_CSV}")
    spark.stop()
    sys.exit(0 if len(ok) == len(rows) else 1)


if __name__ == "__main__":
    main()
