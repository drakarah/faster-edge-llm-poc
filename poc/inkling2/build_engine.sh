#!/bin/bash
# Builds the Inkling 2 Mini engine with the settings of poc/config/inkling2-mini.sh and the INT8 drafter file.
# usage: build_engine.sh CHECKPOINT_DIR ENGINE_DIR   (run inside the builder image; CHECKPOINT_DIR = converted Inkling 2 checkpoint, also holding the original hf files for the drafter)
set -euo pipefail
CK=${1:?checkpoint dir}; OUT=${2:?engine dir}
HF=${HF_DIR:-$CK}                                    # directory with mtp.safetensors and the target safetensors (embedding norm, unembedding)
POC=$(cd "$(dirname "$0")/.." && pwd)
set +e; die() { echo "$*"; }; . "$POC/config/inkling2-mini.sh"; set -e   # the config's die line only warns here
for kv in $BUILD_ENV; do export "$kv"; done
python3 - "$CK" "$OUT" "$MAXIN" "$CAP" <<'PY'
import sys
ck, out, maxin, cap = sys.argv[1:5]
sys.argv = ['build', '--model-dir', ck, '--engine-dir', out, '--max-input-len', maxin, '--max-kv-cache-capacity', cap, '--max-batch-size', '1']
from experimental.builder.cli import main
main()
PY
python3 "$POC/tools/inkling2_drafter_int8.py" "$HF" "$OUT/drafter_int8.safetensors"
echo "engine and drafter written to $OUT"
