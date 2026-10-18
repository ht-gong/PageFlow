# spark-rapids (RAPIDS Accelerator for Apache Spark)

The `rapids` engine: Spark 3.5.8 (local mode, one GPU) with the
[NVIDIA RAPIDS Accelerator](https://github.com/NVIDIA/spark-rapids). Its
environment is part of the image (`docker/Dockerfile`, pins in `versions.env`):
the RAPIDS jar at `$RAPIDS_JAR`, Spark from the `py` env (`$SPARK_HOME`), and a
Temurin 17 JDK (`$JAVA_HOME`).

## What's here

| path | purpose |
|------|---------|
| `run_tpch_safe.sh` | the runner: one `spark-submit` per query, a disk watchdog and a per-query time limit (`RAPIDS_TIMEOUT`, 600 s by default; a smaller `QUERY_TIMEOUT` wins); merges into `results/all_results.csv` |
| `run_tpch_queries.py` | runs the qgen stream's queries in one Spark session: an untimed warm pass on SF1 (`WARM_PARQUET`), then each query once on the target dataset, timed; wall time and GPU-operator count. `LOAD_MODE=resident` (key `rapids_resident`) runs them over Spark's cache in host memory, built before each pass (`../AGENTS.md`, "Resident mode") |
| `activate.sh` | `source` it for the Spark toolchain, `$RAPIDS_JAR`/`$RAPIDS_CONF`, a `rapids-submit` helper and `rapids_run_args` (the benchmark's spark-submit flags, shared with `ablation/probe.sh`) |
| `conf/spark-rapids.conf` | defaults for interactive `rapids-submit` use |
| `test_gpu.py` | smoke test that proves a query runs on the GPU |

## Versions / hardware

- **spark-rapids 26.04.2**, Scala 2.12 → Spark 3.3.x–3.5.x (3.5.8 here).
- **cuda12** classifier: the jar bundles its own cuDF native library (sm_70
  through sm_120), so it needs only an NVIDIA driver; it runs on the H100s' CUDA
  13 driver through backward compatibility.

## Usage

```bash
make bench SF=100 ENGINES=rapids        # TPC-H q1-22 -> results/all_results.csv
make shell                              # then, inside the container:
source rapids/activate.sh && rapids-submit rapids/test_gpu.py
```

Confirm GPU execution with `df.explain()` — operators should be prefixed `Gpu*`
(e.g. `GpuHashAggregate`). Set `spark.rapids.sql.explain=ALL` to see *why* any
operator falls back to CPU.

## Notes / gotchas

- `CONTAINER_ID` in the environment makes Spark think it is in a YARN container
  ("Yarn Local dirs can't be empty"); the runner and `rapids-submit` use
  `env -u CONTAINER_ID`, `test_gpu.py` pops it.
- Spark's local dirs go to `$SCRATCH/rapids` and are emptied between queries.
- Each query's JVM fetches the 880 MB jar into its local dir at start-up. On a
  heavily loaded box (load ~100, two Spark apps starting at once) this once took
  over 15 s, a heartbeat arrived before the executor's BlockManager registered,
  and that query failed with `BlockManagerId ... is null`; a rerun passed.
- spark-rapids 26.04.2 hangs on some reads of delta-encoded + zstd parquet
  (cuDF decoder); the format profiling pass times those probes out.
