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
STOCK_ARGS="--max-input-len $MAXIN --max-kv-cache-capacity $CAP"
STOCK_SPEC='{"method":"mtp"}'
