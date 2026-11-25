#!/bin/bash
# Serves the Flint engine over HTTP (OpenAI-compatible) with the settings of poc/config/flint.sh. ASCII prompts only (see the limits in the config); run nothing next to it (26-27 GB resident).
# usage: serve.sh ENGINE_DIR [PORT=18080]   (EDGELLM_PLUGIN_PATH and PYTHONPATH as in the image)
set -euo pipefail
ENG=${1:?engine dir}; PORT=${2:-18080}
POC=$(cd "$(dirname "$0")/.." && pwd)
set +e; die() { :; }; . "$POC/config/flint.sh"; set -e
cd "${STACK_ROOT:-/opt/edge-llm}"
exec python3 -m experimental.server "$ENG" --host 0.0.0.0 --port "$PORT" --served-model-name local $SERVE_ARGS
