FROM nvidia/cuda:13.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl pkg-config python3 python3-pip \
    libaio-dev liblz4-dev libnuma-dev libsnappy-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir cmake==3.31.6 matplotlib pandas

WORKDIR /workspace
COPY DPFProto /workspace/DPFProto
COPY scripts /workspace/scripts

RUN cd /workspace/DPFProto && scripts/setup/install_nvidia_libs.sh \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
    && cmake --build build -j"$(nproc)" --target tpchdb tpchloader

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    DPF_DEPS=/root/.local/dpfproto-deps \
    PATH=/usr/local/cuda/bin:/usr/local/cuda/gds/tools:${PATH} \
    LD_LIBRARY_PATH=/root/libs/nvcomp/lib:/usr/local/cuda/lib64:/usr/local/cuda/gds/lib64

CMD ["scripts/run_golap_pruning.sh"]
