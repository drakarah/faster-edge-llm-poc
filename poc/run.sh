#!/bin/bash
# Entry point of the container. Modes (all steps resume from /data, so a failed or interrupted run continues where it stopped):
#   builder  download the model, quantise it, build the optimised engine, let the stock stack build its own
#   bench    time stock and optimised stacks alternately on the same prompts; write the comparison table
#   serve    keep the optimised OpenAI-compatible server up on $PORT
#   all      builder, bench, serve
set -euo pipefail
cd "$(dirname "$0")"
source config/common.sh
[ -f "config/$MODEL.sh" ] || die "unknown MODEL '$MODEL' (gemma4-e4b | gemma4-12b | gemma4-26b | qwen35-4b)"
source "config/$MODEL.sh"
[ "${BENCH_LONGCTX-__default__}" = __default__ ] && BENCH_LONGCTX=$LONGCTX_DEFAULT
BENCH_LONGCTX=${BENCH_LONGCTX}   # long-context test prompt lengths in tokens, comma-separated; empty skips it (default per model in config/)
source lib.sh

mode=${1:-all}
case "$mode" in
  builder) builder ;;
  bench)   bench ;;
  serve)   serve ;;
  all)     builder; bench; serve ;;
  shell)   exec bash ;;
  *) die "mode must be builder | bench | serve | all | shell" ;;
esac
