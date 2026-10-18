#!/usr/bin/env bash
set -euo pipefail

# document sf100 ramdisk run
export QUERIES="${QUERIES:-q1 q3 q5 q6 q13 q16}"
SF=100 
DEVICE_SIZE=64G 
"$(dirname "$0")/run_golap_ramdisk.sh"
