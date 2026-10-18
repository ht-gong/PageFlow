#!/usr/bin/env python
"""Smoke test: confirm the RAPIDS Accelerator actually runs SQL on the GPU.

Run via the helper:   source activate.sh && rapids-submit test_gpu.py
Or standalone:        source activate.sh && python test_gpu.py
"""
import os

# Vast.ai sets CONTAINER_ID, which makes Spark think it runs inside a YARN
# container ("Yarn Local dirs can't be empty"). Drop it so local mode works.
# The JVM pyspark launches inherits this environment.
os.environ.pop("CONTAINER_ID", None)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

RAPIDS_JAR = os.environ.get(
    "RAPIDS_JAR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "jars", "rapids-4-spark_2.12-26.04.2-cuda12.jar"),
)

spark = (
    SparkSession.builder.appName("spark-rapids-smoke-test")
    .master("local[*]")
    # If launched with spark-submit --jars these are already set; setting them
    # here too makes `python test_gpu.py` work standalone.
    .config("spark.jars", RAPIDS_JAR)
    .config("spark.plugins", "com.nvidia.spark.SQLPlugin")
    .config("spark.rapids.sql.enabled", "true")
    .config("spark.rapids.sql.explain", "ALL")
    .config("spark.rapids.sql.concurrentGpuTasks", "2")
    .config("spark.sql.adaptive.enabled", "false")  # keep the plan simple to inspect
    .getOrCreate()
)

print("\n=== Spark / plugin info ===")
print("Spark version :", spark.version)
print("Plugins       :", spark.conf.get("spark.plugins"))
print("RAPIDS enabled:", spark.conf.get("spark.rapids.sql.enabled"))

# Build some data and run an aggregation — the kind of thing the GPU accelerates.
df = spark.range(0, 20_000_000).withColumn("k", (F.col("id") % 100))
agg = df.groupBy("k").agg(F.sum("id").alias("s"), F.count("*").alias("c"))

print("\n=== Physical plan (look for 'Gpu*' operators) ===")
plan = agg._jdf.queryExecution().executedPlan().toString()
print(plan)

result = agg.orderBy("k").limit(5).collect()
print("\n=== Sample result ===")
for row in result:
    print(row)

gpu_ops = [ln.strip() for ln in plan.splitlines() if "Gpu" in ln]
on_gpu = any("GpuHashAggregate" in ln for ln in gpu_ops)
print("\n=== Verdict ===")
print(f"GPU operators in plan: {len(gpu_ops)}")
print("✅ Aggregation ran on the GPU" if on_gpu
      else "⚠️  No GpuHashAggregate found — check spark.rapids.sql.explain output above")

spark.stop()
raise SystemExit(0 if on_gpu else 1)
