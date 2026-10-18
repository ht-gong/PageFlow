#!/usr/bin/env python
"""rapids column probes (see probe_common.py). Launched with the TPC-H runner's
spark-submit flags (rapids_run_args in rapids/activate.sh); the same temp views
as rapids/run_tpch_queries.py, then a GPU warm-up query. A probe whose executed
plan has any operator outside the GPU (a CPU scan or aggregate) is recorded as
FALLBACK."""
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import probe_common as pc  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

# Plan nodes that are not operators or are GPU-neutral AQE wrappers.
NEUTRAL = {"AdaptiveSparkPlan", "ShuffleQueryStage", "BroadcastQueryStage",
           "AQEShuffleRead", "ResultQueryStage", "InputAdapter"}


def cpu_operators(plan):
    """Operator names in the executed (AQE final) plan that are not Gpu*."""
    if "== Final Plan ==" in plan:
        plan = plan.split("== Final Plan ==", 1)[1].split("== Initial Plan ==", 1)[0]
    ops = []
    for line in plan.splitlines():
        m = re.match(r"[\s:+|-]*(?:\*\(\d+\)\s*)?([A-Za-z]+)", line)
        if m and not m.group(1).startswith("Gpu") and m.group(1) not in NEUTRAL:
            ops.append(m.group(1))
    return ops


def main():
    parquet, cols, out_csv = pc.args()
    spark = SparkSession.builder.appName("format probe (RAPIDS GPU)").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("Spark", spark.version, "| rapids.sql.enabled =",
          spark.conf.get("spark.rapids.sql.enabled", "false"), flush=True)
    for t in pc.TABLES:
        spark.read.parquet(f"{parquet}/{t}").createOrReplaceTempView(t)
    t0 = time.time()
    spark.sql("select l_returnflag, count(*) c, sum(l_quantity) q "
              "from lineitem where l_orderkey < 100000 group by l_returnflag").collect()
    print(f"GPU warm-up done in {time.time() - t0:.1f}s", flush=True)

    def probe(table, column):
        t0 = time.time()
        df = spark.sql(pc.probe_sql(table, column))
        df.collect()
        dt = time.time() - t0
        cpu = cpu_operators(df._jdf.queryExecution().executedPlan().toString())
        if cpu:
            return "FALLBACK", dt, "cpu:" + "|".join(dict.fromkeys(cpu))
        return "OK", dt, ""

    pc.run(cols, out_csv, probe)
    spark.stop()


if __name__ == "__main__":
    main()
