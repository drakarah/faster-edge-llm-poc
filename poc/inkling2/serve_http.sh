#!/bin/bash
# HTTP server (OpenAI-compatible, experimental.server) on the Inkling 2 Mini engine with the profile's speculative settings: chain verification, n-gram lookup and the INT8 DSpark drafter.
# Greedy and sampled requests both use it (sampled: every verify row is sampled from the target, a draft is accepted when the sample equals it; same-seed output equals plain decoding's for short generations).
# Requests with logprobs, penalties, logit bias, guided decoding, a reasoning budget or a batch of more than one run the plain step.
# A request that reuses a cached prefix resumes the drafter from the rows of the previous request when the reused tokens match (otherwise it runs without the drafter).
# Prefix reuse is always on. CONTEXT_REUSE=auto (default) picks the mechanism from the engine's KV capacity: up to 65,536 the generic reuse flag; above it (the 262,144-token engine, CAP=262144 KV_INT8=1)
# only the host prefix cache, because the generic flag keeps the sliding-window layers at full length (about 22 GB, the Orin's memory is exhausted); the host prefix cache below
# gives append-only chats near-full reuse instead (METRICS 246, 247: 20k-token chat, turns 2-3 in 1.1-1.7 s against 77.8 s cold; 3,072 MiB budget, a snapshot every 1,024 prompt tokens).
# PREFIX_MB and CAPTURE_INTERVAL override the host prefix cache budget and snapshot spacing. The sampling defaults, parsers and thinking default are the ones used for the pi harness.
# usage: serve_http.sh ENGINE_DIR [PORT=18080]   (ENGINE_DIR holds llm.engine and drafter_int8.safetensors; EDGELLM_PLUGIN_PATH and PYTHONPATH as in the image)
set -euo pipefail
ENG=${1:?engine dir}; PORT=${2:-18080}
POC=$(cd "$(dirname "$0")/.." && pwd)
DRAFTER_FILE=$ENG/drafter_int8.safetensors
set +e; die() { :; }; . "$POC/config/inkling2-mini.sh"; set -e
for kv in $(eval echo "$SERVE_ENV"); do export "$kv"; done
cd "${STACK_ROOT:-/opt/edge-llm}"
REUSE=""
CAPKV=$(grep -o '"max_kv_cache_capacity": *[0-9]*' "$ENG/config.json" 2>/dev/null | grep -o '[0-9]*$' | head -1); CAPKV=${CAPKV:-0}
MODE=${CONTEXT_REUSE:-auto}; [ "$MODE" = auto ] && { [ "$CAPKV" -gt 65536 ] && MODE=0 || MODE=1; }
echo "serve_http: KV capacity $CAPKV, generic context reuse flag: $MODE, host prefix cache ${PREFIX_MB:-3072} MiB" >&2
[ "$MODE" = 1 ] && REUSE="--enable-context-reuse --context-cache-snapshots-in-host-memory --context-cache-recurrent-snapshot-pool-bytes 536870912 --context-cache-partial-kv-snapshot-pool-bytes 67108864"
export EDGELLM_SWA_PREFIX_CACHE_MB=${PREFIX_MB:-3072} EDGELLM_SWA_PREFIX_CAPTURE_INTERVAL=${CAPTURE_INTERVAL:-1024}
export EDGELLM_DEFAULT_ENABLE_THINKING=${EDGELLM_DEFAULT_ENABLE_THINKING:-1} EDGELLM_DEFAULT_TEMPERATURE=${EDGELLM_DEFAULT_TEMPERATURE:-1.0} EDGELLM_DEFAULT_TOP_P=${EDGELLM_DEFAULT_TOP_P:-0.95} EDGELLM_DEFAULT_TOP_K=${EDGELLM_DEFAULT_TOP_K:-20}
exec python3 -m experimental.server "$ENG" --host 0.0.0.0 --port "$PORT" --served-model-name local --enable-auto-tool-choice --tool-call-parser inkling --reasoning-parser inkling $REUSE
