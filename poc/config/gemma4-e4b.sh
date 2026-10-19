# Gemma 4 E4B (google/gemma-4-E4B-it) with its MTP assistant.
KIND=gemma4
BASE_REPO=google/gemma-4-E4B-it-qat-w4a16-ct          # Google's QAT checkpoint (group 32): the quantiser's starting point
ASSIST_REPO=google/gemma-4-E4B-it-assistant
GPTQ_SAMPLES=256
KV_SCALES=$POC/data/gemma4-e4b-int8-kv-scales.json
# --- optimised stack ---
MAXIN=2048; CAP=10240; POOL=320; TREE=12              # engine: prompt tokens per pass, KV capacity, KV pool pages (prefix reuse), max verify tree
EXPORT_ENV="EDGELLM_GEMMA4_GELU_PLUGIN=1 EDGELLM_FUSE_GEMMA4_INT4=1 EDGELLM_INT8_KV_SCALES=$KV_SCALES EDGELLM_GEMMA4_SPARSE_DRAFT_LOGITS=1"
EXPORT_ARGS="--externalize-weights int4_ffn --mtp"
INT8_EMBEDDINGS=1
SERVE_ARGS="--draft-top-k 2 --draft-step 5 --verify-tree-size 12 --enable-context-reuse"
SERVE_ENV="EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MTP_SUFFIX_DRAFT=6 EDGELLM_MTP_SUFFIX_MIN=2"
[ "$PROFILE" = orin ] && SERVE_ENV="$SERVE_ENV EDGELLM_INT4_GEMV_MAX_M=1"
[ "$PROFILE" = dgpu ] && SERVE_ENV="$SERVE_ENV EDGELLM_GRAMMAR_CACHE_MB=4096"
# INT8 tensor-core prefill GEMM (W4A8) for prompts of 64+ tokens. On the Jetson the two ops with outlier inputs (per-layer-embedding gate,
# down projection) stay on the FP16 path, which restored HumanEval there; on the discrete GPU all ops use it (not yet checked on that profile).
[ "$PROFILE" = orin ] && SERVE_ENV="$SERVE_ENV EDGELLM_INT4_W4A8_MIN_M=64 EDGELLM_INT4_W4A8_MAX_K=8192 EDGELLM_INT4_W4A8_MIN_N=512"
[ "$PROFILE" = dgpu ] && SERVE_ENV="$SERVE_ENV EDGELLM_INT4_W4A8_MIN_M=64"
# --- stock stack: upstream's documented path (the server builds its own engine from the checkpoint) ---
# upstream has no chunked prompt pass: the engine must take a whole prompt, so its input limit is the full context
# stock's own default drafting (one chain of 6, verify size 7) decodes at 19 tok/s on the Orin, slower than no speculation (29); stock's tree flags
# with the same values the optimised stack uses give 32 tok/s (measured on engines built for the default maximum tree of 60; 32.5 on the engines built for TREE, README), so the stock arm gets those: its best setting, not a strawman.
STOCK_ARGS="--max-input-len $CAP --max-kv-cache-capacity $CAP --enable-context-reuse --enable-auto-tool-choice --draft-top-k 2 --draft-step 5 --verify-tree-size 12 --max-verify-tree-size $TREE --max-draft-tree-size $TREE"
STOCK_SPEC='{"method":"mtp","model":"@ASSIST@"}'
