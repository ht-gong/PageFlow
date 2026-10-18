# The one environment every workflow runs in: the datagen toolchain and all four
# engines (rapids, polars_gpu, duckdb_cpu, sirius). Every version comes from
# versions.env (passed as build args by docker/run.sh) and docker/*.lock.
# The image holds environments only, under /opt/envs; the repo is bind-mounted
# at runtime, so code changes need no rebuild. Build with `make image`.
#
# No CUDA base image: each engine brings its own CUDA user space (the RAPIDS jar
# bundles cuDF, the cudf-polars wheels bundle libcudf, the Sirius pixi env brings
# CUDA 13.3); the NVIDIA container runtime injects the driver.
ARG UBUNTU_IMAGE

# ------------------------------------------------------------------ base
# apt packages come from the UBUNTU_SNAPSHOT archive snapshot, which is served
# over https only. The bare image has no CA bundle, so `certs` fetches one from
# the live archive; it is mounted only while apt downloads (package integrity
# is checked by apt's signatures either way), and the pinned ca-certificates
# package is what the image keeps.
FROM ${UBUNTU_IMAGE} AS certs
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates

FROM ${UBUNTU_IMAGE} AS base
ARG UBUNTU_SNAPSHOT
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN --mount=type=bind,from=certs,source=/etc/ssl/certs/ca-certificates.crt,target=/tmp/boot-ca.crt \
    apt-get -o Acquire::https::CaInfo=/tmp/boot-ca.crt update --snapshot "${UBUNTU_SNAPSHOT}" \
 && apt-get -o Acquire::https::CaInfo=/tmp/boot-ca.crt install -y --no-install-recommends \
      --snapshot "${UBUNTU_SNAPSHOT}" ca-certificates curl python3 python3-venv procps \
 && rm -rf /var/lib/apt/lists/*

# build-only tools
FROM base AS build
ARG UBUNTU_SNAPSHOT
RUN apt-get update --snapshot "${UBUNTU_SNAPSHOT}" \
 && apt-get install -y --no-install-recommends --snapshot "${UBUNTU_SNAPSHOT}" git build-essential \
 && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------ jdk
FROM base AS jdk
ARG JDK_URL
ARG JDK_SHA256
RUN curl -fsSL -o /tmp/jdk.tgz "${JDK_URL}" \
 && echo "${JDK_SHA256}  /tmp/jdk.tgz" | sha256sum -c - \
 && mkdir -p /opt/envs/jdk && tar xzf /tmp/jdk.tgz -C /opt/envs/jdk --strip-components=1 \
 && rm /tmp/jdk.tgz

# ------------------------------------------------------------------ py
# pyspark (RAPIDS driver), pyarrow, duckdb (duckdb_cpu).
FROM base AS py
COPY docker/py.lock /tmp/py.lock
RUN python3 -m venv /opt/envs/py \
 && /opt/envs/py/bin/pip install --no-cache-dir --disable-pip-version-check --require-hashes --no-deps \
      -r /tmp/py.lock

# ------------------------------------------------------------------ ndsgen
# datagen/ndsgen, the TPC-H generator, built with the pinned Rust toolchain from
# its Cargo.lock (--locked). Unlike the rest of the repo it is compiled into the
# image, so its sources are part of the image tag: an edit needs `make image`.
FROM build AS ndsgen
ARG RUSTUP_URL
ARG RUSTUP_SHA256
ARG RUST_VERSION
ENV RUSTUP_HOME=/opt/rust/rustup CARGO_HOME=/opt/rust/cargo
RUN curl -fsSL -o /tmp/rustup-init "${RUSTUP_URL}" \
 && echo "${RUSTUP_SHA256}  /tmp/rustup-init" | sha256sum -c - \
 && chmod +x /tmp/rustup-init \
 && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain "${RUST_VERSION}" \
 && rm /tmp/rustup-init
COPY datagen/ndsgen/Cargo.toml datagen/ndsgen/Cargo.lock /src/ndsgen/
COPY datagen/ndsgen/src /src/ndsgen/src
RUN /opt/rust/cargo/bin/cargo build --release --locked --manifest-path /src/ndsgen/Cargo.toml \
      --target-dir /tmp/ndsgen-target \
 && mkdir -p /opt/envs/ndsgen && cp /tmp/ndsgen-target/release/ndsgen /opt/envs/ndsgen/ \
 && rm -rf /tmp/ndsgen-target

# ------------------------------------------------------------------ rapids-jar
FROM base AS rapids-jar
ARG RAPIDS_JAR_NAME
ARG RAPIDS_JAR_URL
ARG RAPIDS_JAR_SHA1
RUN mkdir -p /opt/envs/jars \
 && curl -fsSL -o "/opt/envs/jars/${RAPIDS_JAR_NAME}" "${RAPIDS_JAR_URL}" \
 && echo "${RAPIDS_JAR_SHA1}  /opt/envs/jars/${RAPIDS_JAR_NAME}" | sha1sum -c -

# ------------------------------------------------------------------ polars
FROM base AS polars
COPY docker/polars.lock /tmp/polars.lock
RUN python3 -m venv /opt/envs/polars \
 && /opt/envs/polars/bin/pip install --no-cache-dir --disable-pip-version-check --require-hashes --no-deps \
      --extra-index-url https://pypi.nvidia.com -r /tmp/polars.lock

# ------------------------------------------------------------------ sirius-build
# Sirius at the pinned commit with its submodules (minus the unused
# experimental/starrocks ones), its toolchain from its own pixi.lock (--frozen),
# `make` as upstream documents, for H100 only. The binary's RPATH points at the
# env's lib dir, so the final stage keeps that dir at the same path.
FROM build AS sirius-build
ARG PIXI_URL
ARG PIXI_SHA256
ARG SIRIUS_URL
ARG SIRIUS_REF
ARG SIRIUS_CUDAARCHS
RUN curl -fsSL -o /tmp/pixi.tgz "${PIXI_URL}" \
 && echo "${PIXI_SHA256}  /tmp/pixi.tgz" | sha256sum -c - \
 && tar xzf /tmp/pixi.tgz -C /usr/local/bin pixi && rm /tmp/pixi.tgz
RUN git clone -q "${SIRIUS_URL}" /opt/envs/sirius \
 && cd /opt/envs/sirius && git checkout -q "${SIRIUS_REF}" \
 && git submodule update --init --recursive -- \
      $(git config -f .gitmodules --get-regexp '\.path$' | awk '{print $2}' | grep -v '^experimental/')
WORKDIR /opt/envs/sirius
RUN --mount=type=cache,target=/root/.cache pixi install --frozen
RUN --mount=type=cache,target=/root/.cache \
    pixi run --frozen bash -c "export CUDAARCHS='${SIRIUS_CUDAARCHS}'; CMAKE_BUILD_PARALLEL_LEVEL=\$(nproc) make" \
 && test -x build/release/duckdb
# Runtime subset of the env: shared libs (some are symlinks into targets/).
RUN E=/opt/envs/sirius/.pixi/envs/default \
 && mkdir -p /opt/sirius-rt/targets/x86_64-linux \
 && cp -a "$E/lib" /opt/sirius-rt/lib \
 && rm -rf /opt/sirius-rt/lib/rustlib /opt/sirius-rt/lib/python3* \
 && find /opt/sirius-rt/lib -name '*.a' -delete \
 && cp -a "$E/targets/x86_64-linux/lib" /opt/sirius-rt/targets/x86_64-linux/lib

# ------------------------------------------------------------------ final
FROM base AS final
ARG RAPIDS_JAR_NAME
COPY --from=jdk /opt/envs/jdk /opt/envs/jdk
COPY --from=py /opt/envs/py /opt/envs/py
COPY --from=polars /opt/envs/polars /opt/envs/polars
COPY --from=ndsgen /opt/envs/ndsgen /opt/envs/ndsgen
COPY --from=rapids-jar /opt/envs/jars /opt/envs/jars
COPY --from=sirius-build /opt/sirius-rt/lib /opt/envs/sirius/.pixi/envs/default/lib
COPY --from=sirius-build /opt/sirius-rt/targets /opt/envs/sirius/.pixi/envs/default/targets
COPY --from=sirius-build /opt/envs/sirius/build/release/duckdb /opt/envs/sirius/build/release/duckdb
COPY versions.env docker/py.lock docker/polars.lock datagen/ndsgen/Cargo.lock /opt/envs/pins/
ENV PY=/opt/envs/py/bin/python \
    POLARS_PY=/opt/envs/polars/bin/python \
    JAVA_HOME=/opt/envs/jdk \
    SPARK_HOME=/opt/envs/py/lib/python3.12/site-packages/pyspark \
    RAPIDS_JAR=/opt/envs/jars/${RAPIDS_JAR_NAME} \
    SIRIUS_DUCKDB=/opt/envs/sirius/build/release/duckdb \
    SIRIUS_ENVLIB=/opt/envs/sirius/.pixi/envs/default/lib \
    NDSGEN=/opt/envs/ndsgen/ndsgen \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility
