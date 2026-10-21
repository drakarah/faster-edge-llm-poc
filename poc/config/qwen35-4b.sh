# Qwen3.5-4B (Qwen/Qwen3.5-4B): hybrid model (gated delta-net + attention) with its built-in MTP layer.
KIND=qwen
BASE_REPO=Qwen/Qwen3.5-4B
ASSIST_REPO=
MAXIN=10240; CAP=10240; POOL=320; TREE=12
EXPORT_ENV=""
EXPORT_ARGS="--mtp --draft-reduced-vocab-dir $POC/data/vocab/qwen35"
INT8_EMBEDDINGS=0
if [ "$PROFILE" = orin ]; then SNAP=1610612736; else SNAP=536870912; fi        # recurrent-state snapshot pool for prefix reuse (bytes)
SERVE_ARGS="--draft-top-k 1 --draft-step 3 --verify-tree-size 4 --enable-context-reuse --context-cache-recurrent-snapshot-pool-bytes $SNAP --context-cache-partial-kv-snapshot-pool-bytes 134217728 --tool-call-parser generic"
SERVE_ENV="EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_HYBRID_DIVERGENCE_CHECKPOINT=256 EDGELLM_GDN_CHUNK_PREFILL=1"
[ "$PROFILE" = orin ] && SERVE_ENV="$SERVE_ENV EDGELLM_INT4_GEMV_MAX_M=1"
# Stock gets exactly the optimised arm's settings where it has the same knob: drafting tree (top-k 1, step 3, verify 4), the engine's build maximum for the
# tree ($TREE), context length, the prefix-reuse flag. Fork-only features (recurrent-state snapshot pool for hybrid reuse, suffix draft, GDN chunked prefill,
# fused projections, plugins) are what is being measured, so stock does not get them.
# If stock rejects --enable-context-reuse for this hybrid model, it runs without: that is an upstream feature gap (the fork's hybrid prefix reuse exists for
# that reason), not a setting withheld from it; the README must say so next to the result.
STOCK_TREE="--draft-top-k 1 --draft-step 3 --verify-tree-size 4"
STOCK_ARGS="--max-input-len $MAXIN --max-kv-cache-capacity $CAP --enable-context-reuse --enable-auto-tool-choice $STOCK_TREE --max-verify-tree-size $TREE --max-draft-tree-size $TREE"
STOCK_SPEC='{"method":"mtp"}'
