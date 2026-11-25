#!/bin/bash
# Builds the Flint engine with the settings of poc/config/flint.sh.
# usage: build_engine.sh ONNX_DIR ENGINE_DIR   (run inside the builder image; ONNX_DIR from export.sh)
set -euo pipefail
ONNX=${1:?onnx dir}; OUT=${2:?engine dir}
POC=$(cd "$(dirname "$0")/.." && pwd)
set +e; die() { :; }; . "$POC/config/flint.sh"; set -e
STACK=${STACK_ROOT:-/opt/edge-llm}
"$STACK/build/examples/llm/llm_build" --onnxDir "$ONNX" --engineDir "$OUT" $BUILD_ARGS
echo "engine written to $OUT"
