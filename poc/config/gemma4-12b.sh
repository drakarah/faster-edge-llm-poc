# Gemma 4 12B (google/gemma-4-12B-it) with its MTP assistant. Long context on small memory: bounded sliding-window KV, INT8 KV.
KIND=gemma4
BASE_REPO=google/gemma-4-12B-it-qat-w4a16-ct
ASSIST_REPO=google/gemma-4-12B-it-assistant
GPTQ_SAMPLES=128
KV_SCALES=$POC/data/gemma4-12b-int8-kv-scales.json
MAXIN=1024; CAP=32768; POOL=0; TREE=20
EXPORT_ENV="EDGELLM_FUSE_GEMMA4_INT4=1 EDGELLM_INT8_KV_SCALES=$KV_SCALES EDGELLM_GEMMA4_DRAFT_VOCAB_MAP=$POC/data/vocab/gemma4 EDGELLM_GEMMA4_BOUNDED_KV_SPEC=1 EDGELLM_GEMMA4_BOUNDED_KV_QUANT=1"
EXPORT_ARGS="--externalize-weights int4_ffn --mtp"
INT8_EMBEDDINGS=0
SERVE_ARGS="--draft-top-k 6 --draft-step 3 --verify-tree-size 20 --tool-call-parser gemma4"
# Jetson Orin: the engine's max verify size is also the size its INT4 GEMM tiles are tuned at. The GEMM runs in 16-row tiles (16, 32, 64, 128): a
# verify tree of 20 is padded to a 32-row tile (37.5% idle rows, the tensor cores are the limit), a tree of 16 fits one 16-row tile. Measured on the
# Orin (same weights, replay of 60 recorded calls, HumanEval): decode round 72.8 -> 60.7 ms, -16% per output token for 3.5% fewer tokens per round,
# replay 327 -> 295 s, HumanEval unchanged. Not measured on the RTX 4060 (its tree sweep, tree 20, predates this), so the dgpu profile keeps 20.
if [ "$PROFILE" = orin ]; then
  TREE=16
  SERVE_ARGS="--draft-top-k 5 --draft-step 3 --verify-tree-size 16 --tool-call-parser gemma4"
fi
SERVE_ENV="EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MTP_SUFFIX_DRAFT=19 EDGELLM_MTP_SUFFIX_MIN=3 EDGELLM_MTP_SUFFIX_PROMPT=16384"
if [ "$PROFILE" = dgpu ]; then
  SERVE_ENV="$SERVE_ENV EDGELLM_EMBEDDING_ON_HOST=1 EDGELLM_SWA_PREFIX_CACHE_MB=16384"   # an 8 GB card: tables and prefix snapshots in pinned host RAM
else
  SERVE_ENV="$SERVE_ENV EDGELLM_INT4_GEMV_MAX_M=1 EDGELLM_SWA_PREFIX_CACHE_MB=4096"        # unified memory: no host-side tricks
  # INT8 tensor-core prompt GEMM (W4A8), the same per-op filter as the E4B config: prompt speed +15%, replay of 60 calls -8.5%, HumanEval 93.9% (off 90.9%,
  # a single greedy run each: no visible loss). The down projection (K = 15 360) stays on the FP16 path because of the K limit.
  SERVE_ENV="$SERVE_ENV EDGELLM_INT4_W4A8_MIN_M=64 EDGELLM_INT4_W4A8_MAX_K=8192 EDGELLM_INT4_W4A8_MIN_N=512"
fi
STOCK_ARGS="--max-input-len $MAXIN --max-kv-cache-capacity $CAP"
STOCK_SPEC='{"method":"mtp","model":"@ASSIST@"}'
