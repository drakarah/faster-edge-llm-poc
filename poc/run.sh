#!/bin/bash
# Entry point of the container. Modes (all steps resume from /data, so a failed or interrupted run continues where it stopped):
#   up       prepare (fetch or quantise the weights, build the engine; skipped when already built) and serve the OpenAI-compatible API on $PORT  (docker compose up)
#   bench    time a running server (BENCH_URL, default the compose `serve` service) on the recorded replay, HumanEval and long prompts; print the table  (docker compose run --rm bench)
#   builder  the full comparison pipeline: download, quantise, build the optimised engine, let the stock stack build its own
#   compare  time stock and optimised stacks alternately on the same prompts (needs `builder` first); write the comparison table
#   serve    keep the optimised server up on $PORT (engine must exist)
#   all      builder, compare, serve
set -euo pipefail
cd "$(dirname "$0")"
source config/common.sh
CONFIG=config/$MODEL.sh; [ -f "config/$MODEL.$PROFILE.sh" ] && CONFIG=config/$MODEL.$PROFILE.sh   # a model can have a config of its own per profile
[ -f "$CONFIG" ] || die "unknown MODEL '$MODEL' (gemma4-e4b | gemma4-12b | gemma4-26b | qwen35-4b | qwen36-35b | cybertiel-35b)"
source "$CONFIG"
[ "${BENCH_LONGCTX-__default__}" = __default__ ] && BENCH_LONGCTX=$LONGCTX_DEFAULT
BENCH_LONGCTX=${BENCH_LONGCTX}   # long-context test prompt lengths in tokens, comma-separated; empty skips it (default per model in config/)
source lib.sh

mode=${1:-up}
case "$mode" in
  up)      prepare; serve ;;
  prepare) prepare ;;
  bench)   bench_api ;;
  builder) builder ;;
  compare) bench ;;
  serve)   serve ;;
  all)     builder; bench; serve ;;
  shell)   exec bash ;;
  *) die "mode must be up | prepare | bench | builder | compare | serve | all | shell" ;;
esac
