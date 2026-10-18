#!/bin/bash
# Ensure the dataset of one scale factor (`make data`, run.sh): generate
# $DATA_DIR/sf<SF> with ndsgen if it has no parquet/ yet, then validate it.
#
#   datagen/dataset.sh <SF>
#
# Env: DATA_DIR, PY, NDSGEN (the image's datagen/ndsgen binary); GEN_PARALLEL
# (dbgen chunks, default 2*SF but at least 20), GEN_BATCH (chunks per batch,
# default 25), GEN_OPTS (extra ndsgen options, e.g. "--compression zstd";
# default: snappy, dictionary, v1 pages).
set -euo pipefail
SF="${1:?scale factor}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
OUT="${DATA_DIR:?DATA_DIR is not set}/sf${SF}"

if [ -d "${OUT}/parquet" ]; then
  exec "${PY:?PY is not set}" "${ROOT}/results/validate_dataset.py" "${OUT}/parquet" "${SF}"
fi
par="${GEN_PARALLEL:-$(( SF * 2 < 20 ? 20 : SF * 2 ))}"
batch="${GEN_BATCH:-25}"
# Built in _gen/ and moved in only once it validates, so a half-built or invalid
# dataset never looks finished.
rm -rf "${OUT}/_gen"
mkdir -p "${OUT}"
echo "[$(date +%T)] generating SF${SF} into ${OUT} (PARALLEL=${par} BATCH=${batch}${GEN_OPTS:+ ${GEN_OPTS}}; log ${OUT}/ndsgen.log)"
# shellcheck disable=SC2086
"${NDSGEN:?NDSGEN is not set (run through make / docker/run.sh)}" --sf "${SF}" \
  --parallel "${par}" --batch "${batch}" --out "${OUT}/_gen" ${GEN_OPTS:-} 2>&1 \
  | tee "${OUT}/ndsgen.log" | tail -1
"${PY:?PY is not set}" "${ROOT}/results/validate_dataset.py" "${OUT}/_gen/parquet" "${SF}"
mv "${OUT}/_gen/NDSGEN.json" "${OUT}/NDSGEN.json"
mv "${OUT}/_gen/parquet" "${OUT}/parquet"
rmdir "${OUT}/_gen"
