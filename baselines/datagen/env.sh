#!/bin/bash
# Source this to get the Spark toolchain (RAPIDS runner) on
# PATH from the variables the image sets (docker/Dockerfile):
#   PY          python of the py env (pyspark 3.5.8, pyarrow, duckdb)
#   JAVA_HOME   Temurin JDK 17 (Spark 3.5 does not support Java 21)
#   SPARK_HOME  the pyspark package dir (its bin/ has spark-submit)
#   source datagen/env.sh
: "${PY:?PY is not set (run through make / docker/run.sh)}"
: "${JAVA_HOME:?JAVA_HOME is not set}" "${SPARK_HOME:?SPARK_HOME is not set}"
export PATH="$(dirname "${PY}"):${JAVA_HOME}/bin:${SPARK_HOME}/bin:${PATH}"
export PYSPARK_PYTHON="${PY}" PYSPARK_DRIVER_PYTHON="${PY}"
# Spark mistakes a Vast-style CONTAINER_ID for a YARN container and dies with
# "Yarn Local dirs can't be empty"; the runners already `env -u CONTAINER_ID`.
unset CONTAINER_ID
# Local-mode Spark only; in containers whose hostname does not resolve to a
# bindable address the driver fails with "Cannot assign requested address".
export SPARK_LOCAL_IP="${SPARK_LOCAL_IP:-127.0.0.1}"
