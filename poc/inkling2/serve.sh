#!/bin/bash
# Runs a prompt file through the engine with the drafter (llm_inference); EDGELLM_INKLING_SPEC=lookup|drafter|hybrid, unset = plain decode.
# usage: serve.sh ENGINE_DIR INPUT_JSON [OUTPUT_JSON]
set -euo pipefail
ENG=${1:?engine dir}; IN=${2:?input json}; OUT=${3:-/tmp/inkling2_out.json}
POC=$(cd "$(dirname "$0")/.." && pwd)
DRAFTER_FILE=$ENG/drafter_int8.safetensors
set +e; die() { :; }; . "$POC/config/inkling2-mini.sh"; set -e
for kv in $(eval echo "$SERVE_ENV"); do export "$kv"; done
exec llm_inference --engineDir="$ENG" --inputFile="$IN" --outputFile="$OUT" --warmup=0 --dumpProfile
