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
#  EDGELLM_FORCE_GREEDY=1 / EDGELLM_JSON_MAX_WHITESPACE=20   as in every profile (README)
#  EDGELLM_MOE_CACHE_SLOTS=auto   per-layer GPU expert cache sized from the free VRAM at the first decode step; EDGELLM_MOE_CACHE_RESERVE_MB=250  VRAM kept free after sizing
#  EDGELLM_MOE_PREFILL_OVERLAP=1  the experts of layer L+1 are copied while layer L computes (prefill floor 1.9 s per pass; overlap reaches 92% of it)
#  EDGELLM_MOE_CACHE_BIAS=0.5     LOSSY (decode only): the router adds 0.5 to the logits of experts already in the GPU cache before the top-8 choice: 65% fewer cache misses for +0.09% perplexity,
#                                 needles 40/40 at 2k and 6k tokens; replay-60 363 -> 270 s, decode 49 -> 62 tok/s. 0 = exact routing, 1 = faster (245 s) and +1.5% perplexity.
SERVE_ENV="EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe EDGELLM_EMBEDDING_ON_HOST=1 EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MOE_CACHE_SLOTS=auto EDGELLM_MOE_CACHE_RESERVE_MB=250 EDGELLM_MOE_PREFILL_OVERLAP=1 EDGELLM_MOE_CACHE_BIAS=${CACHE_BIAS:-0.5}"
DRAFTER_INT4_MLP=0
# Prebuilt export published for this model: `docker compose up` downloads it and only builds the engine (no 26 GB-RAM export, no hours of GPTQ). QUANTIZE=1 redoes the quantisation locally from the original weights.
HF_ONNX_REPO=${HF_ONNX_REPO:-}      # set to the published repository once it exists
