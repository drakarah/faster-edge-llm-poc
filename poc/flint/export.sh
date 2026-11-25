#!/bin/bash
# Exports the Flint checkpoint (RecursiveSI/flint-int4, MLX affine INT4) to the ONNX directory llm_build reads: model.onnx, the INT8 embedding, external_flint_moe_weights.safetensors, tokenizer files (about 22 GB).
# usage: export.sh CHECKPOINT_DIR ONNX_DIR   (run inside the builder image, cwd = the fork checkout /opt/edge-llm; needs about 27 GB of RAM and swap for it)
set -euo pipefail
CK=${1:?checkpoint dir}; OUT=${2:?onnx dir}
POC=$(cd "$(dirname "$0")/.." && pwd)
set +e; die() { :; }; . "$POC/config/flint.sh"; set -e
cd "${STACK_ROOT:-/opt/edge-llm}"
env $EXPORT_ENV python3 -m tensorrt_edgellm.scripts.export "$CK" "$OUT"
echo "exported to $OUT"
