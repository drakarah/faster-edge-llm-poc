#!/bin/bash
# HTTP server (OpenAI-compatible, experimental.server) on the Inkling 2 Mini engine with the profile's speculative settings: chain verification, n-gram lookup and the INT8 DSpark drafter.
# Greedy and sampled requests both use it (sampled: every verify row is sampled from the target, a draft is accepted when the sample equals it; same-seed output equals plain decoding's for short generations).
# Requests with logprobs, penalties, logit bias, guided decoding, a reasoning budget or a batch of more than one run the plain step.
# Context reuse is on (CONTEXT_REUSE=0 turns it off): a request that reuses a cached prefix resumes the drafter from the rows of the previous request when the reused tokens match (otherwise it runs without the drafter).
# usage: serve_http.sh ENGINE_DIR [PORT=18080]   (ENGINE_DIR holds llm.engine and drafter_int8.safetensors; EDGELLM_PLUGIN_PATH and PYTHONPATH as in the image)
set -euo pipefail
ENG=${1:?engine dir}; PORT=${2:-18080}
POC=$(cd "$(dirname "$0")/.." && pwd)
DRAFTER_FILE=$ENG/drafter_int8.safetensors
set +e; die() { :; }; . "$POC/config/inkling2-mini.sh"; set -e
for kv in $(eval echo "$SERVE_ENV"); do export "$kv"; done
cd "${STACK_ROOT:-/opt/edge-llm}"
REUSE=""
[ "${CONTEXT_REUSE:-1}" = 1 ] && REUSE="--enable-context-reuse --context-cache-snapshots-in-host-memory --context-cache-recurrent-snapshot-pool-bytes 536870912 --context-cache-partial-kv-snapshot-pool-bytes 67108864"
exec python3 -m experimental.server "$ENG" --host 0.0.0.0 --port "$PORT" --served-model-name local $REUSE
