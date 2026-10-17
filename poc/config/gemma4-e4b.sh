# Gemma 4 E4B (google/gemma-4-E4B-it) with its MTP assistant.
KIND=gemma4
BASE_REPO=google/gemma-4-E4B-it-qat-w4a16-ct          # Google's QAT checkpoint (group 32): the quantiser's starting point
ASSIST_REPO=google/gemma-4-E4B-it-assistant
GPTQ_SAMPLES=256
KV_SCALES=$POC/data/gemma4-e4b-int8-kv-scales.json
# --- optimised stack ---
MAXIN=2048; CAP=10240; POOL=320; TREE=12              # engine: prompt tokens per pass, KV capacity, KV pool pages (prefix reuse), max verify tree
EXPORT_ENV="EDGELLM_FUSE_GEMMA4_INT4=1 EDGELLM_INT8_KV_SCALES=$KV_SCALES EDGELLM_GEMMA4_SPARSE_DRAFT_LOGITS=1"
EXPORT_ARGS="--externalize-weights int4_ffn --mtp"
INT8_EMBEDDINGS=1
SERVE_ARGS="--draft-top-k 2 --draft-step 5 --verify-tree-size 12 --enable-context-reuse --tool-call-parser gemma4"
SERVE_ENV="EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MTP_SUFFIX_DRAFT=6 EDGELLM_MTP_SUFFIX_MIN=2"
[ "$PROFILE" = orin ] && SERVE_ENV="$SERVE_ENV EDGELLM_INT4_GEMV_MAX_M=1"
[ "$PROFILE" = dgpu ] && SERVE_ENV="$SERVE_ENV EDGELLM_GRAMMAR_CACHE_MB=4096"
# --- stock stack: upstream's documented path (the server builds its own engine from the checkpoint) ---
STOCK_ARGS="--max-input-len $MAXIN --max-kv-cache-capacity $CAP --enable-context-reuse"
STOCK_SPEC='{"method":"mtp","model":"@ASSIST@"}'
