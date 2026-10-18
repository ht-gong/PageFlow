#!/usr/bin/env bash
set -euo pipefail

TAG=q1_sf100
SF=100
export QUERIES=q5
TRIALS=1
THREADS=1
DEVICE_SIZE=64G 
./dpfproto/scripts/run_golap_ramdisk.sh

# grep cuda_alloc dpfproto/logs/golap_ramdisk/<latest>/q5.txt
