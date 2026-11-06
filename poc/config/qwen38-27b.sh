# Qwen3.8-27B (Qwen/Qwen3.8-27B, RedHatAI/Qwen3.8-27B-INT4 compressed-tensors checkpoint): dense hybrid, 64 layers = 48 gated delta-net + 16 full attention (4 KV heads of 256), one MTP layer, on the Jetson AGX Orin 32 GB.
# Orin profile only: the 27B does not fit an 8 GB card.
[ "$PROFILE" = orin ] || die "MODEL=qwen38-27b needs PROFILE=orin (not measured on a discrete GPU)"
KIND=qwen38
NO_STOCK=1              # stock has no INT8-KV hybrid path, no INT4-head engine and no MTP chain sampling
BASE_REPO=RedHatAI/Qwen3.8-27B-INT4
ASSIST_REPO=
LONGCTX_DEFAULT=2048,8192,24576,49000
# Engine: passes of 2,048 tokens (longer prompts run in several passes), KV capacity 131,072 with the INT8 KV cache (32 KiB per token = 4 GiB; the model's own FP8 scales converted, data/qwen38-27b-int8-kv-scales.json).
# The 32 GB board then has about 3 GB free: 262,144 does not fit as built. Draft verify window 8 rows (room for the lookup windows); the served chain is 3 draft steps.
MAXIN=2048; CAP=131072; POOL=0; TREE=8
BUILD_ENV=""
INT8_EMBEDDINGS=0
DRAFTER_INT4_MLP=0      # the MTP layer stays bf16: an INT4 MLP of the drafter gave 1.01 tokens per round
EXPORT_ENV="EDGELLM_INT8_KV_SCALES=$POC/data/qwen38-27b-int8-kv-scales.json"
EXPORT_ARGS="--mtp --draft-reduced-vocab-dir $POC/data/vocab/qwen35 --externalize-weights all --skip-visual --skip-audio"
# Weights: INT4 GPTQ g128 (repacked bit-exact), INT4 lm_head for the base model, bf16 MTP layer and bf16 reduced drafter head (build_fast_qwen38 in lib.sh; the drafter must not inherit a head sliced from INT4 weights).
# Context reuse with host-memory snapshots: the recurrent state of this model is about 150 MB per slot, so the pools are 1 GiB and 512 MiB.
REUSE="--enable-context-reuse --context-cache-snapshots-in-host-memory --context-cache-recurrent-snapshot-pool-bytes 1073741824 --context-cache-partial-kv-snapshot-pool-bytes 536870912"
# --draft-step 3: best under sampling (greedy probes are equal for steps 3-5; sampled acceptance falls with the window). --reasoning-parser qwen3: the template ends the generation prompt with <think>.
SERVE_ARGS="--tool-call-parser generic --reasoning-parser qwen3 --draft-top-k 1 --draft-step 3 $REUSE"
# EDGELLM_INT4_GEMV_MAX_M=1: the INT4 GEMV is slower than the GEMM from 2 verify rows on the Orin (round 156 ms at 3 rows, 117 ms at 5); with the switch at 1 every window runs on the GEMM (+15 to +46% decode).
# EDGELLM_INT4_W4A8_MIN_M=64: INT8 tensor-core prefill GEMMs from 64 rows (+20% prefill, 347 -> 418 tok/s on 2k-10k prompts; HumanEval 152 of 164 as without, needles 3 of 3 at 15.7k tokens, no longer answers).
# Lookup 7 with prompt indexing: +52% on code edits, neutral on prose. EDGELLM_IGNORE_UNSUPPORTED_PENALTIES=1: agent harnesses send frequency_penalty, which the runtime has no kernel for; it is dropped instead of a 400.
SERVE_ENV="EDGELLM_INT4_GEMV_MAX_M=1 EDGELLM_INT4_W4A8_MIN_M=64 EDGELLM_MTP_LOOKUP=7 EDGELLM_MTP_LOOKUP_PROMPT=65536 EDGELLM_HYBRID_PLAIN_TAIL_CHECKPOINT=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_IGNORE_UNSUPPORTED_PENALTIES=1"
# Chat defaults for a served endpoint (the benchmarks send their own fields): thinking on and preserved, the model card's sampling. The MTP chain samples (sample-then-compare verification) and takes presence_penalty.
# REASONING_BUDGETS turns a client's reasoning level into a hard budget (the level alone only changes an instruction and does not reliably shorten thinking): the fork's budget for MTP engines.
CHAT_ENV="EDGELLM_DEFAULT_ENABLE_THINKING=${THINKING:-1} EDGELLM_DEFAULT_TEMPERATURE=${TEMPERATURE:-1.0} EDGELLM_DEFAULT_TOP_P=${TOP_P:-0.95} EDGELLM_DEFAULT_TOP_K=${TOP_K:-20} EDGELLM_PRESERVE_THINKING=${PRESERVE_THINKING:-1} EDGELLM_REQUEST_STATS=${REQUEST_STATS:-1}"
REASONING_BUDGETS=${REASONING_BUDGETS-minimal=512,low=1024,medium=4096,high=8192}   # xhigh and a request without a level keep thinking unlimited; REASONING_BUDGETS= (empty) turns the mapping off
[ -n "$REASONING_BUDGETS" ] && CHAT_ENV="$CHAT_ENV EDGELLM_REASONING_BUDGETS=$REASONING_BUDGETS"
export BENCH_SERVER_DEFAULTS=1 BENCH_NEEDLE_MAX_TOKENS=4096 BENCH_HUMANEVAL_MAX_TOKENS=4096
