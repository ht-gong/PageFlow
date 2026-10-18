# syntax=docker/dockerfile:1.7
#
# TQP-Vortex container image.
#
#   Target
#     engine  (default) — the built engine plus every side entry point:
#                         pytest suites, the codec sweep,
#                         experiment scripts, and Spark plan parsing.
#
#   The TPC-H data is made outside it: `make data` runs baselines/ in its
#   own image.
#
#   Nothing here needs a GPU at build time: cmake/CrossTorchGPU.cmake selects
#   the CUDA backend from the CUDA *toolkit* plus the torch wheel flavor, both
#   of which are present in a plain `docker build`.
#
#   See docker-compose.yml and the Makefile for how these are invoked, and
#   README.md "Running with Docker" for the host-side requirements.

ARG CUDA_IMAGE=nvidia/cuda:12.6.2-devel-ubuntu22.04

# ──────────────────────────────────────────────────────────────────────────
# base — CUDA toolchain, Python 3.11, Torch, JDK, Nsight Systems
# ──────────────────────────────────────────────────────────────────────────
FROM ${CUDA_IMAGE} AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
# PINNED, and it has to stay pinned.  torch >= 2.9 headers hard-error with
# "C++20 or later compatible compiler is required to use PyTorch", while this
# repo builds its extensions at C++17 (CMakeLists.txt, memory/CMakeLists.txt).
ARG TORCH_SPEC="torch==2.6.0+cu126 torchvision==0.21.0+cu126"
ARG WITH_NSYS=1
# The CUDA apt repo carries only version-suffixed packages (there is no
# unversioned nsight-systems-cli); `apt-cache pkgnames nsight-systems` lists
# what a given base image offers.
ARG NSYS_PACKAGE=nsight-systems-2026.1.3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Python 3.11 from the deadsnakes PPA: Ubuntu 22.04 ships 3.10, and the cuDF
# that requirements.txt pins needs 3.11 or later.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential git curl wget ca-certificates xz-utils \
      software-properties-common gpg-agent \
      numactl pciutils procps \
      openjdk-17-jdk-headless \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv \
 && rm -rf /var/lib/apt/lists/*

# Every python3 / pip3 of the image is this 3.11 environment, first on PATH;
# the system's 3.10 is left to apt's own tools.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV CUDA_HOME=/usr/local/cuda \
    CUDAToolkit_ROOT=/usr/local/cuda
ENV PATH=${JAVA_HOME}/bin:${CUDA_HOME}/bin:${PATH}

# Ubuntu 22.04 ships CMake 3.22; CPM.cmake and nvCOMP's package config want
# newer.  Pinned to a range so image builds stay reproducible without chasing
# a CMake 4 migration.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip "setuptools>=61" wheel && \
    pip3 install "cmake>=3.28,<4" ninja

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --index-url ${TORCH_INDEX} ${TORCH_SPEC}

ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Nsight Systems for `main.py --cuda_profile`.  Not fatal if the CUDA apt
# repo has no matching package — docker/preflight.py reports it, and only the
# profiling paths need it.
RUN if [ "${WITH_NSYS}" = "1" ]; then \
      apt-get update && \
      { apt-get install -y --no-install-recommends "${NSYS_PACKAGE}" \
        && ln -sf "$(find /opt/nvidia/nsight-systems -name nsys -type f -perm -u+x -print -quit)" /usr/local/bin/nsys \
        || echo ">>> WARNING: ${NSYS_PACKAGE} unavailable; --cuda_profile will not work in this image"; } && \
      rm -rf /var/lib/apt/lists/* ; \
    fi

# ──────────────────────────────────────────────────────────────────────────
# deps — Python requirements only, so source edits do not re-resolve pip.
#        They bring cuDF (pylibcudf, RMM) and with it the one nvCOMP of the
#        image: the nvidia-libnvcomp-cu12 wheel that libcudf pins.
# ──────────────────────────────────────────────────────────────────────────
FROM base AS deps

WORKDIR /opt/TQP-Vortex
COPY requirements.txt requirements-dev.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt -r requirements-dev.txt

# ──────────────────────────────────────────────────────────────────────────
# builder — compile the custom modules and install the package
# ──────────────────────────────────────────────────────────────────────────
FROM deps AS builder

# The build runs without a GPU, so the arch list cannot be probed.  Default
# covers A100 (80), Ampere consumer (86), Ada (89) and Hopper (90).
ARG CUDA_ARCHITECTURES="80;86;89;90"

WORKDIR /opt/TQP-Vortex
COPY . .

# compression/ builds against the nvCOMP wheel cuDF pins and links the
# libnvcomp.so.5 libcudf loads (compression/CMakeLists.txt): one nvCOMP per
# process.  CMake does not search lib64/ on Ubuntu, so nvcomp_DIR names the
# wheel's package config.
RUN NVCOMP_ROOT="$(python3 -c 'import os, nvidia.libnvcomp as m; print(os.path.dirname(m.__file__))')" && \
    cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
      -DCUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}" \
      -Dnvcomp_DIR="${NVCOMP_ROOT}/lib64/cmake/nvcomp" \
 && cmake --build build -j"$(nproc)"

# setup.py's build_py stages build/<module>/*.so into TQPlib/<pkg>/.  Because
# `python main.py` runs with the repo root on sys.path[0], that in-tree copy —
# not site-packages — is what gets imported.
RUN pip3 install --no-cache-dir . && \
    for p in tqpmemory zero_copy_reader tqpcompress tqpkernels tqpcache; do \
      ls TQPlib/$p/*.so >/dev/null || { echo "missing TQPlib/$p/*.so"; exit 1; }; \
    done

# ──────────────────────────────────────────────────────────────────────────
# engine — the image every service runs
# ──────────────────────────────────────────────────────────────────────────
FROM builder AS engine

# Pristine copy of the compiled extensions.  The `dev` service bind-mounts the
# host source over /opt/TQP-Vortex, which shadows TQPlib/*/*.so (gitignored, so
# absent on the host); the entrypoint restores them from here.
RUN mkdir -p /opt/tqp-artifacts && cp -a TQPlib /opt/tqp-artifacts/

ENV TQP_TABLES_PATH=/data/tables \
    TQP_OUT_DIR=/opt/TQP-Vortex/out \
    OMP_NUM_THREADS=32 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/TQP-Vortex
# Invoked through bash so a bind-mounted source tree that lost the exec bit
# still starts.
ENTRYPOINT ["/bin/bash", "/opt/TQP-Vortex/docker/entrypoint.sh"]
CMD ["python3", "main.py", "--help"]
