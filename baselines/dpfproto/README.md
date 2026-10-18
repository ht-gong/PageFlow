# GOLAP and DPFProto Baseline

More of a general note file; instructions here are not clear

## Paper notes
GOLAP: 
- uses GPU direct storage to stream data from SSD to GPU
- decompress on GPU during scan
- prune and query operators on GPU
- result: inflate effective SSD bandwith beyond raw SSD bandwidth
    - We should expect 4×-23× speedup over CPU SSD-based systems

DPFProto: 
- Data path fusion (DPF): exeutes a sequence of data path ops in 1 kernel (OP, decomp, database ops)
- Other GPU optimization techniques
    - type-specific compression/depression
    - variable length attribute support
    - BaM-based IO
    - works togther with DPF
Reported results:
- reimplements GOLAP baseline
- reports better TPC-H performance than GOLAP
- Main gains come from reducing GPU memory traffic and intermediate materialization

```bash
cd ~/work/TQP-Vortex/baselines/dpfproto

python3 -m venv .venv
source .venv/bin/activate
pip install matplotlib pandas
```

## Code running notes

### DPFProto
DPFProto reports results on a partial TPC-H suite: Q1, Q3, Q5, Q6, Q13, and Q16

This note tracks the minimal path for setting up DPFProto's TPC-H evaluation on the evaluation host

The BaM-backed modes require bare-metal access to NVMe devices

```bash
# 1st clone the official dpfproto repo and apply the baseline patch
cd dpfproto
git clone --recursive https://github.com/dbc-utokyoiis/DPFProto.git
cd DPFProto
git checkout 697fe24fc500db52c60c17bb021d3dcfd38289e5
git apply ../patches/golap-baseline.patch

# MAKE COPY FOR RAM DISK
cd ~/work/TQP-Vortex/baselines/dpfproto

cp -a DPFProto DPFProto-ramdisk

# setup checks
nvcc --version
cmake --version
gdscheck.py -p
lsmod | grep nvidia_fs

# provided by DPFProto to install nvidia compression libs
scripts/setup/install_nvidia_libs.sh

# shared local build env
export DPF_DEPS=$HOME/.local/dpfproto-deps
export CUDA_HOME=/usr/local/cuda-13.2
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$CUDA_HOME/gds/tools:$DPF_DEPS/bin:$PATH
export PKG_CONFIG_PATH=$DPF_DEPS/lib/pkgconfig:$DPF_DEPS/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=$DPF_DEPS/lib:$DPF_DEPS/lib64:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$DPF_DEPS/lib:$DPF_DEPS/lib64:$LIBRARY_PATH
export CMAKE_PREFIX_PATH=$DPF_DEPS:$CMAKE_PREFIX_PATH

# dependency fixups for the evaluation host
mkdir -p "$DPF_DEPS/src"
cd "$DPF_DEPS/src"
rm -f zlib-1.3.1.tar.gz
rm -rf zlib-1.3.1
curl -L -o zlib-1.3.1.tar.gz https://zlib.net/fossils/zlib-1.3.1.tar.gz
tar xf zlib-1.3.1.tar.gz
cd zlib-1.3.1
./configure --prefix="$DPF_DEPS"
make -j$(nproc)
make install
cd ..

# ====== config issue fixing ==========
# add missing snappy pkg-config file
mkdir -p "$DPF_DEPS/lib/pkgconfig"
cat > "$DPF_DEPS/lib/pkgconfig/snappy.pc" <<EOF
prefix=$DPF_DEPS
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: snappy
Description: Fast compression/decompression library
Version: 1.2.2
Libs: -L\${libdir} -lsnappy
Cflags: -I\${includedir}
EOF

# check deps
pkg-config --modversion liblz4 zlib snappy
# ================

# build DPFProto
cd ~/work/TQP-Vortex/baselines/dpfproto/DPFProto
git submodule update --init --recursive
mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=86
make -j$(nproc) tpchdb tpchloader
ls -lh tpchdb tpchloader

# prepare data -> data is put into DATA_BASE
# alter the following as needed
cd ~/work/TQP-Vortex/baselines/dpfproto/DPFProto
export DATA_BASE=$HOME/work/TQP-Vortex/baselines/dpfproto/data/tpch
mkdir -p "$DATA_BASE"


# build official TPC-H dbgen
# this was downloaded from web and placed into the dir below
cd "/path/to/TPC-H V3.0.1/dbgen"
cp makefile.suite Makefile
sed -i 's/^CC[[:space:]]*=.*/CC      = gcc/' Makefile
sed -i 's/^DATABASE[[:space:]]*=.*/DATABASE= INFORMIX/' Makefile
sed -i 's/^MACHINE[[:space:]]*=.*/MACHINE = LINUX/' Makefile
sed -i 's/^WORKLOAD[[:space:]]*=.*/WORKLOAD= TPCH/' Makefile
make clean 2>/dev/null || true
make -j$(nproc)
./dbgen -h

# generate raw TPC-H data
export TPCH_DBGEN_DIR="/path/to/TPC-H V3.0.1/dbgen"
cd ~/work/TQP-Vortex/baselines/dpfproto/DPFProto

# DPFProto's dbgen wrapper does not quote dbgen paths, so patch a local copy.
cp scripts/tpch/dbgen.sh scripts/tpch/dbgen.local.sh
sed -i 's|$dbgendir/dbgen|"$dbgendir/dbgen"|g' scripts/tpch/dbgen.local.sh

cp scripts/tpch/run_dbgen.sh scripts/tpch/run_dbgen.local.sh
sed -i 's|bash "${SCRIPT_DIR}/dbgen.sh"|bash "${SCRIPT_DIR}/dbgen.local.sh"|g' \
  scripts/tpch/run_dbgen.local.sh

rm -rf "$DATA_BASE/input1"
scripts/tpch/run_dbgen.local.sh 1 $(nproc) "$DATA_BASE"
find "$DATA_BASE/input1" -maxdepth 2 -type f | head -20

# prepare sideways/GOLAP-style data
DATA_BASE="$DATA_BASE" bash scripts/golap/01_sideways_pruning_data_base.sh -s 1 -n $(nproc)

# run full non-BaM suite
scripts/tpch_run_all.sh \
  -s 1 \
  -q q1,q3,q5,q6,q13,q16 \
  -n 3 \
  --skip gidp+bam \
  --skip gidp+bam+fusion \
  --skip datapathfusion \
  --no-revenue

# run BaM suite (requires sudo / bare-metal NVMe access)
sudo scripts/tpch_run_all.sh \
  -s 1 \
  -q q1,q3,q5,q6,q13,q16 \
  -n 3 \
  --skip gidp \
  --no-revenue
  
# outputs:
# logs/tpch_run_all/<timestamp>/sf<SF>/<mode>/<query>.txt
```

### GOLAP (baseline in DPFProto codebase)

Running GOLAP end-to-end

re-exporting data for different SF
```bash
cd ~/work/TQP-Vortex/baselines

export DATA_BASE=$PWD/dpfproto/data/tpch
export TPCH_DBGEN_DIR=$HOME/work/tpch_official_dbgen

cd dpfproto/DPFProto

# raw TPC-H SF100 tables
scripts/tpch/run_dbgen.sh 100 "$(nproc)" "$DATA_BASE"

# GOLAP sideway layout SF100
DATA_BASE="$DATA_BASE" bash scripts/golap/01_sideways_pruning_data_base.sh -s 100 -n "$(nproc)"

# sanity check
find "$DATA_BASE/input100" -maxdepth 2 -type f | head
find "$DATA_BASE/sideways/sf100" -maxdepth 2 -type f | head
```

```bash
# From DPFProto root
SF=100
THREADS=$(nproc)
TRIALS=1
QUERIES="q1 q3 q5 q6 q13 q16"

# Device list from scripts/common.sh.
# Confirm these are safe on your machine before using them.
source scripts/common.sh
export DATA_BASE=$HOME/work/TQP-Vortex/baselines/dpfproto/data/tpch

# 1. Prepare GOLAP-style sideways data.
# No sudo needed: this only reads/writes files under DATA_BASE.
DATA_BASE="$DATA_BASE" bash scripts/golap/01_sideways_pruning_data_base.sh -s ${SF} -n ${THREADS}

# 2. Load GOLAP-style compressed layout.
# Sudo needed: this writes DPFProto pages to DEVICES_NVME.
sudo env CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON}" \
  ./build/tpchloader \
  -i "${DATA_BASE}/sideways/sf${SF}" \
  -d "${DEVICES_NVME}" \
  -x gidp \
  -A \
  -c

# 3. Run the filesystem/GDS GOLAP-style baseline.
# Sudo likely needed: this reads DEVICES_NVME through the GDS path.
mkdir -p logs/golap_gidp_sf${SF}
for q in ${QUERIES}; do
  scripts/bench.sh -n ${TRIALS} -w -t 30 \
    -o logs/golap_gidp_sf${SF}/${q}.txt \
    -- sudo env CUFILE_ENV_PATH_JSON="${CUFILE_ENV_PATH_JSON}" \
      ./build/tpchdb \
      -q "${q}" \
      -x gidp \
      -w ${THREADS} \
      -Z \
      "${DEVICES_NVME}"
done

# 4. Optional BaM baseline.
# Sudo definitely needed: this can unmount/rebind NVMe devices.
mkdir -p logs/golap_bam_sf${SF}
sudo scripts/tpch_run_all.sh \
  -s ${SF} \
  -q q1,q3,q5,q6,q13,q16 \
  -n ${TRIALS} \
  --skip gidp \
  --skip gidp+bam+fusion \
  --skip datapathfusion \
  --no-load \
  --no-revenue
```

### GOLAP ramdisk variant

The ramdisk runner uses `/dev/shm` files as DPFProto device images. Because
tmpfs rejects `O_DIRECT`, the script applies the local fallback patch and
rebuilds the touched binaries before running:

```bash
cd ~/work/TQP-Vortex/baselines
SF=100 DEVICE_SIZE=64G dpfproto/scripts/run_golap_ramdisk.sh

# same run documented as a wrapper
dpfproto/scripts/call_all_ramdisk_sf100.sh
```

Defaults are SF100, 64 GB per tmpfs device image, 3 trials, and q1 q3 q5 q6 q13
q16. Logs land under `dpfproto/logs/golap_ramdisk/<timestamp>/`.

## Further Investigation


### Q1: Are GOLAP OOMs caused when intermediates grow large?

- Intermediates: temporary data created during query execution before final answer

- Method: 
  - If memory rises with SF and crosses the GPU limit at the failing SF, q5 OOM is explained by scale GPU memory pressure
    - secondary question: why does q5 still fail if they claim to spill seamlessly to disk?
    - status
    - runtime
    - peak GPU memory
    - last printed query metadata
    - OOM allocation sites (important)
  
  - plot
    - x axis = SF
    - y axis = peak GPU memory GB
    - RTX 3090 usable VRAM as a line 


#### Investigation Results

GOLAP can avoid keeping all input data in GPU memory because it streams/pages data from storage.
But Q5 fails on runtime GPU allocation of intermediates shown by the following line:

`[cuda_alloc_failed] request_mb=2049 free_mb=165 error=out of memory`

These intermediates are not spillable and results in OOM when the VRAM is full on our RTX 3090. 
  


### Q2: Is GOLAP bottlenecked on GPU compute when run on RTX3090?

- Whether GOLAP is limited by the GPU's compute/decompression/query kernels instead of storage

- Method:
  - Pick q1 and q6
  - Measure ramdisk runtime and throughput, 1st check that its not already compute bottlenecked
  - Minimal evidence to proof:
    - Ramdisk IO throughput 
    - Effective throughput
    - GPU util
    - CPI IOwait near zero


  - dd if=/dev/shm/dpfproto_golap_filedev/sf100/dev0.img of=/dev/null bs=1M count=8192

Signs of compute bottleneck:
CPU iowait near 0
GPU util high
ramdisk io_throughput_gbs comfortably below possible ramdisk bandwidth
runtime stable even though storage is much faster



It doesn't look like that GPU is the primary bottle neck. Evidence points to mixed pipeline cost in decompression, transfer, and syncs.

Q1:
runtime: 800 ms
actual data read: 6173 MB
logical data processed: 16128 MB
GPU memory allocated: 17.9 GB

CPU iowait very low:
cpu_usage[
	user: 2464
	nice: 0
	system: 3656
	idle 635743,
	iowait: 42
]

Q6:
runtime: 758 ms
actual data read: 4636 MB
logical data processed: 9216 MB
GPU memory allocated: 10.2 GB
cpu_usage[
  user: 2306
  nice: 0
  system: 3548
  idle: 625518
  iowait: 55
]

- This shows that Q1 is not limited by saturated GPU compute 
- Nsight shows bursty GPU activity with GPUs
  - especially looking at the compute in flight section
- GOLAP has GPU-compute-saturated phases, but end-to-end Q1/Q6 runtime is not proven to be purely GPU-compute-bound 
  - The remaining delay is likely pipeline synchronization and data movement



### Q3: How large does storage pruning affect the runtime for GOLAP?

- Storage pruning: skipping chunks via metadata before reading them

- Method:
  - Compare end-to-end runtime with pruning disabled versus enabled for the full standard GOLAP query set
  - collect:
    - runtime
    - read_mb
    - uncompressed_read_mb
    - io_reduction_ratio
    - effective_throughput_gbs

  - expecting q6 to be affected more than q1

- At SF100, storage pruning substantially improves selective queries
  - Q3 improves by 4.19x and Q6 by 2.76x. Q1, Q13, and Q16 show no meaningful improvement
  - This shows that pruning only helps when query predicates allow GOLAP to skip many storage pages
  - One more note is Q5 still OOMs because its failure occurs during GPU buffer allocation before pruning is applied


| Query | Pruning off | Pruning on | Result |
|---|---:|---:|---:|
| Q1 | 586 ms | 594 ms | no benefit |
| Q3 | 1903 ms | 454 ms | 4.19x faster |
| Q6 | 536 ms | 194 ms | 2.76x faster |
| Q13 | 1385 ms | 1378 ms | no meaningful change |
| Q16 | 471 ms | 468 ms | no meaningful change |
| Q5 | OOM | OOM | unchanged |


Q5 is scale-limited by its GPU buffer allocation. It fits at SF1, but at SF100 its required GPU-resident buffers exceed the RTX 3090’s 24 GB VRAM. This is not a fixed bug that happens at every scale, and pruning does not help because the OOM occurs before query execution and pruning.


## Final 

The benchmark reads `dpfproto/data/tpch/sideways/sf300`. Keep this directory on
host disk and mount it read-only into Docker. It is input data, not a result.

0. Construct docker reproducibility -> DONE
  1. documented command for: build binaries, load SF300 data, run both pruning modes, plotting
  2. mount data results
  3. document ramdisk and cuFile-compat requirements

1. Create SF300 sideways data; do all necessary checking -> DONE
2. Run pruning/no-pruning on data
  1. Run SF300 ZONEMAP=0
  2. Run ZONEMAP=1

3. Produce the paper table
  1. look up table format for paper?

4. Paragraph length analysis on GOLAP scaling issues seen previously
  1. Describe pruning effects with pruning on/off (Q3, Q6)
  2. Describe existing bottlenecks and how our paper resolves these

dig a bit deeper into the reason
- why do we have cuFile calls?
- why does the GPU gets called intermittently? 
- Are there any forced sync points? 
- How does CPU/GPU co-process the data
- how does everything fit into the larger picture in both the GOLAP paper and GPU database execution in general

This paragraph describes the scaling issues scene on GOLAP on TCP-H queries at high scale factors. 

when intermediate results exceed GPU memory, GOLAP falls back to GPU-CPU co-execution rather than keeping the entire pipeline GPU-resident.
GOLAP avoids loading the entire input relation into GPU memory by streaming compressed table chunks from SSD and processing them incrementally.

However, certain situatios can still result OOM on intermediates when they are not spillable and are required to be present on the GPU at once.

- GOLAP mainly attacks storage -> GPU ingress
  - storage

- cuFileRead is the GPU Direct Storage API used to move data from an NVMe/file-backed device directly into GPU memory
  - It avoids the traditional bounce-buffer path (storage → CPU buffer → cudaMemcpy → GPU)
  - The important qualification is that GDS is direct in the data path, but still CPU-controlled. CPU threads submit the cuFileRead requests and coordinate the buffers. GOLAP deliberately uses synchronous GDS calls issued by multiple CPU threads because the paper found that its asynchronous GDS path did not saturate storage bandwidth reliably.

- Query runs are done as a seuqnece of dependent stage
  - rage compressed chunks -> wait until. required columns arrive -> decompress -> query kernel -> materialize/transfer output -> next chunk

- Pruning is effective only when predicates can rule out many chunks
  - It does not solve Q5 because Q5’s failure happens when allocating large GPU-resident intermediate buffers, before pruning can reduce the problematic state

ROTAS deals with these:
B. PCIe transfer and pipeline overlap
C. GPU-resident intermediate capacity
page-based CPU/GPU buffer management;
spillable intermediate streams;
external radix-partitioned joins and group-by;
dataflow-aware eviction;
adaptive division between ephemeral memory and the swap buffer;
compression chosen according to compute-versus-PCIe balance;
UVA gather for sparse late materialization.



## Docker
1. Build fixed image with CUDA 13.2, compiler, CMake, and DPFProto dependencies
2. Find a way to deal with data -> regenerate or mount
3. Mount dpfproto/logs/ back to host -> persist results
4. Given container the GPU and a large /dev/shm -> maybe not necessary? Bc this was more a logical solution
5. Run existing GOLAP runner twice
6. Generate comparison plot in proper paper format








