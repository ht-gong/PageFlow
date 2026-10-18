# GOLAP Docker

This runs the ramdisk GOLAP pruning experiment. Build data first with the
instructions below; Docker mounts that data and does not generate it.

## Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA driver 580.82.07 or newer
- This repository with `dpfproto/DPFProto` checked out locally
- SF300 input at `../data/tpch/sideways/sf300`
- 512 GB free host RAM in `/dev/shm`

### Get DPFProto

Clone upstream DPFProto at the commit the patch is based on, then apply
`dpfproto/patches/golap-baseline.patch`. From the TQP-Vortex root:

```bash
git clone --recursive https://github.com/dbc-utokyoiis/DPFProto.git baselines/dpfproto/DPFProto
cd baselines/dpfproto/DPFProto
git checkout 697fe24fc500db52c60c17bb021d3dcfd38289e5
git apply ../patches/golap-baseline.patch
```

The patch adds the ramdisk runner, the data scripts, a configurable GPU-memory
budget and page count, and the Q3/Q5 memory tuning. Docker copies this local
checkout; it does not clone DPFProto itself.

### Prepare SF300 data

Run this once before Docker. TPC-H provides `dbgen`, which generates the data.

1. Download [TPC-H Tools 3.0.1](https://www.tpc.org/TPC_Documents_Current_Versions/download_programs/tools-download-request5.asp), accept the license, and unzip it.
2. Build `dbgen`:

```bash
cd /path/to/TPC-H V3.0.1/dbgen
cp makefile.suite Makefile
sed -i 's/^CC[[:space:]]*=.*/CC      = gcc/' Makefile
sed -i 's/^DATABASE[[:space:]]*=.*/DATABASE= INFORMIX/' Makefile
sed -i 's/^MACHINE[[:space:]]*=.*/MACHINE = LINUX/' Makefile
sed -i 's/^WORKLOAD[[:space:]]*=.*/WORKLOAD= TPCH/' Makefile
make
```

3. Install DuckDB if needed, then generate the GOLAP input:

```bash
cd ~/work/TQP-Vortex/baselines
command -v duckdb || PREFIX="$HOME/.local" dpfproto/DPFProto/scripts/setup/install_duckdb.sh
export PATH="$HOME/.local/bin:$PATH"
export TPCH_DBGEN_DIR="/path/to/TPC-H V3.0.1/dbgen"
dpfproto/scripts/prepare_golap_data.sh
```

This creates `dpfproto/data/tpch/sideways/sf300`, which Docker mounts read-only.

## Run

```bash
cd dpfproto/docker

# build DPFProto and its dependencies
docker compose build


# paper experiment
SF=300 docker compose run --rm golap


cd ~/work/TQP-Vortex/baselines/dpfproto/docker
docker compose run --rm golap scripts/call_q3_q5_tuned_sf300.sh
```

The SF300 run writes one timestamped directory under `../logs/golap_ramdisk/`.
It contains `pruning_off/`, `pruning_on/`, `pruning_runtime.png`, and the plot CSV.
Q5 is expected to OOM at SF300, so the command ends non-zero after saving results.

## Notes

`ipc: host` shares the host's `/dev/shm` with the container. The runner creates
four 128 GB ramdisk device files there. CUDA 13.2 requires
`NVIDIA_DISABLE_REQUIRE=1` with the evaluation host's driver. It enables cuFile
compatibility mode because tmpfs files are not GDS NVMe devices. This is a
ramdisk compatibility baseline, not a direct NVMe GDS measurement.
