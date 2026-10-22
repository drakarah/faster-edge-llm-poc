# Gemma 4 E4B (google/gemma-4-E4B-it) with its MTP assistant.
KIND=gemma4
LONGCTX_DEFAULT=2048,4096,8192,16384,24576,30000   # the optimised engine holds 32,768 tokens; stock holds 10,240 (it has no chunked prefill) and is recorded as not reached above that
BASE_REPO=google/gemma-4-E4B-it-qat-w4a16-ct          # Google's QAT checkpoint (group 32): the quantiser's starting point
ASSIST_REPO=google/gemma-4-E4B-it-assistant
GPTQ_SAMPLES=256
KV_SCALES=$POC/data/gemma4-e4b-int8-kv-scales.json
# --- optimised stack ---
MAXIN=2048; CAP=32768; STOCK_CAP=10240; POOL=320; TREE=12              # engine: prompt tokens per pass, KV capacity, KV pool pages (prefix reuse), max verify tree
# Export switches (README rows have the measurements): GELU_PLUGIN = fused GELU(gate)*up kernel (prompt passes -11%); FUSE_GEMMA4_INT4 = fused q+k+v and gate+up INT4 GEMMs (decode +7%);
# INT8_KV_SCALES = INT8 KV cache with per-layer scales (calibrated, $KV_SCALES); SPARSE_DRAFT_LOGITS = the drafter returns 4,096 candidates instead of a 262,144-entry row (about -1% per token).
EXPORT_ENV="EDGELLM_GEMMA4_GELU_PLUGIN=1 EDGELLM_FUSE_GEMMA4_INT4=1 EDGELLM_INT8_KV_SCALES=$KV_SCALES EDGELLM_GEMMA4_SPARSE_DRAFT_LOGITS=1"
EXPORT_ARGS="--externalize-weights int4_ffn --mtp"
INT8_EMBEDDINGS=1           # token and per-layer embedding tables stored as INT8 rows: GPU 11.1 -> 7.78 GB, no speed or quality change by design
# Tree 2 x 5 with 12 verify nodes: swept on replayed requests (README row "Tree shape and size", -7% to -10% against 3 x 2 / 8; neighbouring shapes: 2x6/12 -0.7%, 3x4/12 +0.4%, 4x3/12 +1.8%, chains 1x5/6 +12% and 1x4/5 +16%, all single runs on 60 calls). --enable-context-reuse + POOL=320: prefix reuse for the replay's shared prompts.
SERVE_ARGS="--draft-top-k 2 --draft-step 5 --verify-tree-size 12 --enable-context-reuse"
# FORCE_GREEDY: the MTP tree verifies greedily only. JSON_MAX_WHITESPACE: bounds whitespace runs in constrained JSON. SUFFIX_DRAFT=6 / MIN=2: suffix draft merged into the verify tree (replay-396 -11.9%).
# INT4_GEMV_MAX_M=1 (Orin): on the Orin the tensor-core GEMM beats the GEMV from 2 tokens, upstream switches at 4 (+11% on a 1-draft-token decode probe; on the E4B tree replay it is neutral, 68.0 vs 68.3 s). GRAMMAR_CACHE_MB=4096 (dgpu): a compiled JSON-schema grammar is ~250 MB for the 262k vocabulary and the default cache is 64 MB, so every request recompiled its schema (~60 ms): the app's 396-call replay was +3.7% slower with the default (measured with the grammar cache). Host RAM, not GPU memory.
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
STOCK_ARGS="--max-input-len $STOCK_CAP --max-kv-cache-capacity $STOCK_CAP --enable-context-reuse --enable-auto-tool-choice --draft-top-k 2 --draft-step 5 --verify-tree-size 12 --max-verify-tree-size $TREE --max-draft-tree-size $TREE"
STOCK_SPEC='{"method":"mtp","model":"@ASSIST@"}'
