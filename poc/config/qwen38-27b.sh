# Qwen3.8-27B (Qwen/Qwen3.8-27B, RedHatAI/Qwen3.8-27B-INT4 compressed-tensors checkpoint): dense hybrid, 64 layers = 48 gated delta-net + 16 full attention (4 KV heads of 256), on the Jetson AGX Orin 32 GB,
# with the DFlash2 block drafter (z-lab/Qwen3.8-27B-DFlash2: 5 layers, proposes a block of 8 tokens per round from five tapped target layers) and a 16-row draft tree (DDTree: a best-first tree over the drafter's top-8 candidates per position, verified in one pass).
# Against the MTP profile (MODEL=qwen38-27b-mtp) on this board: code +60% and thinking-on +45% with the DFlash2 chain (prose level), and the tree lifts DFlash2 prose from 22.8 to 27.5 tok/s greedy (code about 47 tok/s);
# about 3 GB more fixed memory, about 211k tokens of context tested instead of 224k.
# Orin profile only: the 27B does not fit an 8 GB card.
[ "$PROFILE" = orin ] || die "MODEL=qwen38-27b needs PROFILE=orin (not measured on a discrete GPU)"
KIND=qwen38df2
NO_STOCK=1              # stock has no INT8-KV hybrid path, no INT4-head engine, no block drafter over a hybrid base and no draft tree for it
BASE_REPO=RedHatAI/Qwen3.8-27B-INT4
ASSIST_REPO=z-lab/Qwen3.8-27B-DFlash2
LONGCTX_DEFAULT=2048,8192,24576,49000
# Engine (checkpoint-direct builder, `python -m experimental.builder.cli --spec-type dflash`): passes of 1,024 tokens (longer prompts run in several passes; 2,048 costs 0.6 GB for 2% prefill),
# KV capacity 262,144 with the INT8 KV cache (32 KiB per token, mapped on demand: EDGELLM_KV_INITIAL_TOKENS below; the model's own FP8 scales converted, data/qwen38-27b-int8-kv-scales.json).
# Verify and draft profile 16 rows: the tree. 24 and 32 rows accept more per round (3.73 and 3.83 tokens against 3.58) but the round grows from 136 to 212 ms, so 16 it is; do not build a bigger profile "for room", the build tunes the GEMM for its maximum.
MAXIN=1024; CAP=262144; POOL=0; TREE=16
# Draft KV as a 32-page ring (the draft attends a 2,048-token window; 80 MiB instead of 20 KiB per token of context, exact with context reuse). GDN tree scratch 2 rows (843 MiB less than the default 8).
# --int8-embedding (token embedding as INT8, about -1.27 GB) exists and is not in the measured engine; add it to BUILD_ARGS for more context, at the cost of re-checking the outputs.
BUILD_ARGS="--draft-kv-ring-pages 32 --gdn-tree-scratch-rows 2 --externalize-weights all"
# EDGELLM_LINEAR_F32_VIA_FP16=1: the draft's fp32-output projections run as fp16 GEMMs (a TensorRT cast of the 5,120 x 25,600 projection cost 44.7 ms of a 196.5 ms round).
# EDGELLM_BUILDER_FUSED_ADD_RMSNORM=1: each residual add and the RMSNorm after it run as one plugin kernel (-2% prefill, decode level).
# EDGELLM_BUILDER_FUSED_SILU_DOWN=1 exists and is off: SiLU(gate) * up inside the down projection's GEMM plugin gives another -2% prefill, prose decode level, code decode -1.2%.
BUILD_ENV="EDGELLM_XQA_MULTI_BLOCK=auto EDGELLM_INT8_KV_SCALES=$POC/data/qwen38-27b-int8-kv-scales.json EDGELLM_LINEAR_F32_VIA_FP16=1 EDGELLM_BUILDER_FUSED_ADD_RMSNORM=1"
# Weights: INT4 GPTQ g128 (RedHatAI's compressed tensors repacked bit-exact), INT4 lm_head (the draft shares it: -656 MB), DFlash2 draft quantised to INT4 g128 including its fc projection (tools/dflash2_quantize.py:
# draft GEMMs 23 -> about 6 ms per round, 3.85 -> about 1.1 GB; acceptance on prose level with the bf16 draft).
DRAFT_CKPT=$CKPT_DIR/dflash2_q4
# Context reuse with host-memory snapshots: each checkpoint carries the recurrent state plus a copy of the 80 MiB draft ring (two 236 MiB-class slots in 512 MiB).
REUSE="--enable-context-reuse --context-cache-snapshots-in-host-memory --context-cache-recurrent-snapshot-pool-bytes 536870912 --context-cache-partial-kv-snapshot-pool-bytes 67108864"
# --draft-top-k 8 --verify-tree-size 16: the tree. --reasoning-parser qwen3: the template ends the generation prompt with <think>.
SERVE_ARGS="--tool-call-parser generic --reasoning-parser qwen3 --speculative-config {\"method\":\"dflash\",\"model\":\"$DRAFT_CKPT\"} --draft-top-k 8 --verify-tree-size 16 $REUSE"
# EDGELLM_INT4_GEMV_MAX_M=1: the INT4 GEMV is slower than the GEMM from 2 verify rows on the Orin.
# EDGELLM_INT4_W4A8_MIN_M=64 with EDGELLM_INT4_W4A8_MAX_K=20480: INT8 tensor-core prefill GEMMs from 64 rows, including the MLP down projections (K = 17,408; the default cap of 16,384 left them on fp16: -6 to -8% prefill time).
# EDGELLM_FMHA_INT8_QK=1: INT8 Q.K^T in the paged prefill attention (-1% prefill time at 8k tokens, -5% at 39k, more at depth). The three prefill settings together: 39,420 tokens cold 116.1 -> about 101 s;
# the next generated token is the same on 24 of 24 probe prompts (1k-40k tokens) with and without them, mean KL at most 0.008; outputs are not bit-identical to the fp16 prefill.
# EDGELLM_INT4_FORCE_TACTIC pins the 16-row GEMM variant of three shapes (N,K:variant): the builder chooses by timing and one of three builds picked a variant for the down projection that cost 5% of decode. EDGELLM_LOG_INT4_TACTIC=1 prints what an engine runs.
# EDGELLM_KV_INITIAL_TOKENS: KV mapped on demand from 16,384 tokens; EDGELLM_KV_MEM_RESERVE_MB: the runtime refuses to map KV that would leave less than this free (integrated memory: an oversized request otherwise freezes the board).
# EDGELLM_IGNORE_UNSUPPORTED_PENALTIES=1: agent harnesses send frequency_penalty, which the runtime has no kernel for; it is dropped instead of a 400.
SERVE_ENV="EDGELLM_CHECKPOINT_DIR=$CKPT_DIR/int4head EDGELLM_DRAFT_CHECKPOINT_DIR=$DRAFT_CKPT EDGELLM_INT4_GEMV_MAX_M=1 EDGELLM_INT4_W4A8_MIN_M=64 EDGELLM_INT4_W4A8_MAX_K=20480 EDGELLM_FMHA_INT8_QK=1 EDGELLM_INT4_FORCE_TACTIC=248320,5120:1;5120,17408:3;6144,5120:17 EDGELLM_KV_INITIAL_TOKENS=16384 EDGELLM_KV_CACHE_SLACK_PAGES=32 EDGELLM_KV_MEM_RESERVE_MB=1800 EDGELLM_HYBRID_PLAIN_TAIL_CHECKPOINT=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_IGNORE_UNSUPPORTED_PENALTIES=1"
# Chat defaults for a served endpoint (the benchmarks send their own fields): thinking on and preserved, the model card's sampling. Under the tree: sampling (lossless sample-and-match acceptance; without top-k it is bounded to the 128 most likely tokens),
# presence_penalty, JSON schema / grammar and the hard reasoning budget all work for one request at a time.
CHAT_ENV="EDGELLM_DEFAULT_ENABLE_THINKING=${THINKING:-1} EDGELLM_DEFAULT_TEMPERATURE=${TEMPERATURE:-1.0} EDGELLM_DEFAULT_TOP_P=${TOP_P:-0.95} EDGELLM_DEFAULT_TOP_K=${TOP_K:-20} EDGELLM_PRESERVE_THINKING=${PRESERVE_THINKING:-1} EDGELLM_REQUEST_STATS=${REQUEST_STATS:-1}"
REASONING_BUDGETS=${REASONING_BUDGETS-minimal=512,low=1024,medium=4096,high=8192}   # xhigh and a request without a level keep thinking unlimited; REASONING_BUDGETS= (empty) turns the mapping off
[ -n "$REASONING_BUDGETS" ] && CHAT_ENV="$CHAT_ENV EDGELLM_REASONING_BUDGETS=$REASONING_BUDGETS"
export BENCH_SERVER_DEFAULTS=1 BENCH_NEEDLE_MAX_TOKENS=4096 BENCH_HUMANEVAL_MAX_TOKENS=4096
