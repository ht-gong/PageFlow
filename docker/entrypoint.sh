#!/usr/bin/env bash
# Entrypoint for the `engine` image.
#
# Three jobs before handing over to the real command:
#   1. restore the compiled extensions when a bind-mounted source tree has
#      shadowed them (the `dev` service),
#   2. redirect main.py's CWD-relative query_times.csv into the mounted out/
#      directory so a benchmark survives the container,
#   3. run the preflight checks that catch the host settings this engine
#      silently depends on.
#
# TQP_SKIP_PREFLIGHT=1 skips (3).
set -euo pipefail

REPO=/opt/TQP-Vortex
ARTIFACTS=/opt/tqp-artifacts
OUT_DIR="${TQP_OUT_DIR:-$REPO/out}"

cd "$REPO"

# 1. TQPlib/*/*.so is gitignored, so a bind-mounted host tree has none.  Copy
#    the image's build back in rather than forcing a rebuild on every start.
for pkg in tqpmemory zero_copy_reader tqpcompress tqpkernels tqpcache; do
    if compgen -G "$REPO/TQPlib/$pkg/"'*.so' > /dev/null; then
        continue
    fi
    if compgen -G "$ARTIFACTS/TQPlib/$pkg/"'*.so' > /dev/null; then
        echo "[tqp] restoring TQPlib/$pkg from the image build" >&2
        mkdir -p "$REPO/TQPlib/$pkg"
        cp -a "$ARTIFACTS/TQPlib/$pkg/." "$REPO/TQPlib/$pkg/"
    fi
done

# 2. main.py appends query_times.csv (the canonical benchmark number, see
#    AGENTS.md) relative to the CWD, which is the image's own filesystem.
#    Symlink it into the mounted out/ dir; appending through a dangling
#    symlink creates the target.
mkdir -p "$OUT_DIR"
if [ ! -e "$REPO/query_times.csv" ] || [ -L "$REPO/query_times.csv" ]; then
    ln -sfn "$OUT_DIR/query_times.csv" "$REPO/query_times.csv"
fi

# 3.
if [ "${TQP_SKIP_PREFLIGHT:-0}" != "1" ]; then
    python3 "$REPO/docker/preflight.py"
fi

exec "$@"
