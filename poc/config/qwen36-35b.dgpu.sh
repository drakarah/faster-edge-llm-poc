# Qwen3.6-35B-A3B (Qwen/Qwen3.6-35B-A3B), hybrid mixture of experts (40 layers: 30 gated delta-net + 10 full attention; 256 experts, top-8, 3B active parameters) on a discrete 8 GB GPU
# (measured on an RTX 4060 laptop-class card, PCIe 4.0 x8, 31 GB RAM). Loaded instead of qwen36-35b.sh when PROFILE=dgpu. Every value says why.
# The experts (about 16 GB as INT4) do not fit next to the rest in 8 GB: they stay in pinned host memory and the plugin streams what a step needs over PCIe, with a per-layer expert cache on the GPU.
# NO MTP (decode 17.5 tok/s with the model's MTP head against 45 without: verify steps touch more experts per step). No stock arm: stock keeps every expert on the GPU and cannot load this model in 8 GB.
# DISK: about 70 GB for the BF16 download plus 30 GB for the quantised checkpoint; the build needs 28 GB of RAM + swap for the export (the builder says how to add swap).
KIND=qwen36moe          # model family handled by the builder: layer-wise expert GPTQ, then layer-wise dense GPTQ, INT8 KV scales shipped in poc/data
NO_STOCK=1              # no stock arm (see above): the table has the optimised arm only
NO_SPEC=1               # plain autoregressive engine, no drafter
LONGCTX_DEFAULT=2048,6000,8500   # the engine holds 12,288 tokens (MAXIN = CAP: a hybrid engine cannot chunk its prompt, so the input limit is the longest prompt); 16,384 does not load next to the expert cache
BASE_REPO=Qwen/Qwen3.6-35B-A3B
ASSIST_REPO=            # empty: no MTP assistant (see above)
# Quantisation, all with generic calibration text (Wikipedia nl/en/fr + source code), never the benchmark prompts:
#  experts  INT4 g128 symmetric, layer-wise GPTQ on the routed tokens of each expert (tools/gptq_experts_layerwise_qwen36.py, 256 samples x 1,024 tokens)
#  dense    INT4 g128 layer-wise GPTQ of ALL dense linears: shared expert, lm_head, gated-delta-net projections, attention q/k/v/o (tools/gptq_dense_layerwise_qwen36.py, --parts shared,head,gdn,attn):
#           replay 682 -> 397 s against FP16 dense, HumanEval 155/164 against 150-156 for every variant, 80/80 retrieval needles at 2k and 4k tokens. Router, norms, embeddings and the 32-row GDN a/b projections stay FP16.
GPTQ_SAMPLES=256        # the setting used; not swept
DENSE_PARTS=shared,head,gdn,attn
# INT8 KV cache (only the 10 full-attention layers own a KV cache) with per-layer scales calibrated on this model (tools/calibrate_int8_kv_layerwise_qwen36.py, 64 x 1,024 tokens; the file ships in poc/data):
# 89/90 needles, replay -3% at 12k. XQA_MULTI_BLOCK = split-KV decode attention.
KV_SCALES=$POC/data/qwen36-35b-int8-kv-scales.json
MAXIN=12288; CAP=12288; POOL=0; TREE=1   # POOL=0: no extra KV pages; TREE unused
EXPORT_ENV="EDGELLM_INT8_KV_SCALES=$KV_SCALES"
# --externalize-weights int4_moe keeps the expert weights (and the embedding table) as separate files next to the engine, which the plugin streams from host memory.
EXPORT_ARGS="--externalize-weights int4_moe --skip-visual --skip-audio"
INT8_EMBEDDINGS=0
# EDGELLM_MOE_STAGE_EXPERTS=2 at engine build: the MoE plugin stages routed experts into private buffers so the prefill copy overlaps the previous layer's GEMMs.
# EDGELLM_MOE_TOKEN_TILE=4096 at engine build: the expert GEMM pipeline of a prefill pass runs over tiles of 4,096 tokens, so the 12k engine's workspace shrinks and 9 more cache slots fit (55 -> 64 per layer, decode +5%).
BUILD_ENV="EDGELLM_MOE_STAGE_EXPERTS=2 EDGELLM_MOE_TOKEN_TILE=4096"
SERVE_ARGS="--tool-call-parser generic"   # no speculation flags; prefix reuse stays off (each checkpoint split costs an extra expert-streaming pass: replay-20 77.5 -> 100 s)
# Serve environment, one reason each:
#  EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe  experts live in pinned host memory (the point of this profile)       EDGELLM_EMBEDDING_ON_HOST=1  the embedding table too
#  EDGELLM_JSON_MAX_WHITESPACE=20   as in every profile (README). EDGELLM_FORCE_GREEDY is not set: this profile has no speculation, so a request keeps its own temperature (forced greedy decoding can make a thinking model loop in long reasoning)
#  EDGELLM_MOE_CACHE_SLOTS=auto   per-layer GPU expert cache sized from the free VRAM at the first decode step; EDGELLM_MOE_CACHE_RESERVE_MB=250  VRAM kept free after sizing
#  EDGELLM_MOE_PREFILL_OVERLAP=1  the experts of layer L+1 are copied while layer L computes (prefill floor 1.9 s per pass; overlap reaches 92% of it)
#  EDGELLM_MOE_CACHE_BIAS=0.5     LOSSY (decode only): the router adds 0.5 to the logits of experts already in the GPU cache before the top-8 choice: 65% fewer cache misses for +0.09% perplexity,
#                                 needles 40/40 at 2k and 6k tokens; replay-60 363 -> 270 s, decode 49 -> 62 tok/s. 0 = exact routing, 1 = faster (245 s) and +1.5% perplexity.
#  EDGELLM_MOE_TAIL_SKIP=0        LOSSY, opt-in (TAIL_SKIP in .env): keep the top N experts always and do not fetch a cache-missing expert ranked N+1 or lower, dropping its weight. A LOWER N skips more.
#                                 With CACHE_BIAS=0 (exact routing otherwise): TAIL_SKIP=6 gives decode 47.5 -> 55.9 tok/s (+18%) for +0.04% perplexity (inside the noise), TAIL_SKIP=5 61.5 tok/s (+29%) for +0.26%.
#                                 With CACHE_BIAS=0.5: TAIL_SKIP=6 65.6 tok/s (+9% over bias alone) for +0.83% perplexity in total, TAIL_SKIP=5 68.6 tok/s. Needles 27/27 at 2k, 4k and 6k tokens for both combinations tested (CACHE_BIAS=0 + 6, 0.5 + 6).
#  EDGELLM_MOE_CPU_MISS=1         CPU miss path (x86 only; off if the CPU lacks AVX2): an expert that is not in the GPU cache is computed on the CPU from the pinned host weights instead of being copied over PCIe, until it has missed
#                                 EDGELLM_MOE_CPU_ADMIT times, then it is admitted to the cache as before. A GPU kernel hands the missed experts to spinning CPU threads through a mapped mailbox (no host sync, CUDA-graph safe).
#                                 Costs EDGELLM_MOE_CPU_THREADS (6) busy CPU cores. RTX 4060 + i9-12900H, beta 0.5, prefill overlap on, 55 slots: decode 62.5 -> 75.9 tok/s (+21%, prompt about 10 tokens); prefill unchanged;
#                                 HumanEval 155/164 vs 154/164 without it (serve defaults; identical configurations differ by up to 3 problems run to run).
#  EDGELLM_MOE_CPU_INT8=1         int8 activations on the CPU (AVX-VNNI; falls back to fp32 without it): LOSSY, about 0.65% relative error on an expert's output (about 2% with outlier channels), no change to the stored weights;
#                                 +4% to +7% decode over the fp32 CPU path at its best admit. =0 keeps fp32 (also lossy against the GPU's fp16 arithmetic, but about 3e-7 relative error per expert).
#  EDGELLM_MOE_CPU_ADMIT=16       misses before an expert is admitted to the GPU cache: 8 is best for fp32, 16 for int8 (higher: the CPU becomes the bottleneck). EDGELLM_MOE_CPU_CORES=auto pins one worker per performance core on
#                                 Intel hybrid CPUs (unpinned elsewhere; the benefit of pinning has not been re-measured since the final fix).
SERVE_ENV="EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe EDGELLM_EMBEDDING_ON_HOST=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MOE_CACHE_SLOTS=auto EDGELLM_MOE_CACHE_RESERVE_MB=250 EDGELLM_MOE_PREFILL_OVERLAP=1 EDGELLM_MOE_CACHE_BIAS=${CACHE_BIAS:-0.5} EDGELLM_MOE_TAIL_SKIP=${TAIL_SKIP:-0} EDGELLM_MOE_CPU_MISS=${CPU_MISS:-1} EDGELLM_MOE_CPU_ADMIT=${CPU_ADMIT:-16} EDGELLM_MOE_CPU_INT8=${CPU_INT8:-1} EDGELLM_MOE_CPU_THREADS=${CPU_THREADS:-6} EDGELLM_MOE_CPU_CORES=${CPU_CORES:-auto}"
DRAFTER_INT4_MLP=0
# Chat defaults for a served endpoint (docker compose up), not for the benchmarks (they send temperature 0 and their own fields). Qwen's model-card "thinking, general tasks" settings, applied to requests that omit the field:
#  EDGELLM_DEFAULT_ENABLE_THINKING=1   the model's own default; a request can still send chat_template_kwargs {"enable_thinking": false}. Thinking is not free: HumanEval 21-30/33 against 31/33 without, and it can run away
#                                      (13 of 56 agent runs hit a 16,384-token cap in a repeat loop, METRICS R25dv, R25dw), so set max_tokens.
#  EDGELLM_DEFAULT_TEMPERATURE=1.0, TOP_P=0.95, TOP_K=20, PRESENCE_PENALTY=1.5   (presence penalty needs this profile's plain engine: the runtime refuses it with speculation)
#                                      The agent study's best arm (14/16 completed against 12/24) changed temperature and penalty together; their separate effect is not measured.
#  EDGELLM_PRESERVE_THINKING=1         keeps the reasoning of earlier turns in the rendered history (agent loops); costs prompt tokens and prefix-reuse hits.
CHAT_ENV="EDGELLM_DEFAULT_ENABLE_THINKING=${THINKING:-1} EDGELLM_DEFAULT_TEMPERATURE=${TEMPERATURE:-1.0} EDGELLM_DEFAULT_TOP_P=${TOP_P:-0.95} EDGELLM_DEFAULT_TOP_K=${TOP_K:-20} EDGELLM_DEFAULT_PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5} EDGELLM_PRESERVE_THINKING=${PRESERVE_THINKING:-1}"
# Prebuilt export published for this model: `docker compose up` downloads it and only builds the engine (no 26 GB-RAM export, no hours of GPTQ). QUANTIZE=1 redoes the quantisation locally from the original weights.
HF_ONNX_REPO=${HF_ONNX_REPO:-drakarah/Qwen3.6-35B-A3B-EdgeLLM-INT4}   # INT8-KV export (INT4 weights); downloaded instead of quantising
