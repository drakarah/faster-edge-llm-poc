# Gemma 4 26B-A4B (google/gemma-4-26B-A4B-it), mixture of experts (128 experts, top-8, 3.8B active parameters), on a discrete 8 GB GPU (measured on an RTX 4060 laptop-class
# card, PCIe 4.0 x8, 31 GB RAM). Loaded instead of gemma4-26b.sh when PROFILE=dgpu. Every value says why.
# The experts (11.4 GB as INT4) do not fit next to the rest in 8 GB: they stay in pinned host memory and the plugin streams what a step needs over PCIe, with a per-layer LRU
# expert cache on the GPU (the fork's int4MoePlugin; Orin has unified memory and needs none of this). NO MTP: with streaming, speculative verify steps touch more experts per step
# and lose (measured: MTP slower than plain decode on this card). No stock arm: stock keeps all experts on the GPU, has no streaming, and its exporter loads the whole 51 GB checkpoint
# (inferred from the code and the sizes, never tried).
KIND=gemma4moe          # model family handled by the builder: Gemma 4 mixture of experts (re-shard, dense GPTQ, expert-streaming export)
NO_STOCK=1              # no stock arm (see above); the table has the optimised arm only
NO_SPEC=1               # build a plain autoregressive engine: no MTP drafter, no verify tree (see above)
LONGCTX_DEFAULT=2048,4096,7500          # 7,500 and not 8,192: prompt + answer must fit CAP (8,192)
# Google's QAT checkpoint in BF16 ("unquantized"). Experts: INT4 RTN g64 on the QAT grid (the exporter packs each layer as it loads). Dense part (attention q/k/v/o, dense MLP, lm_head):
# INT4 g128 by layer-wise calibrated GPTQ (tools/gptq_dense_layerwise_26b.py, 128 x 1,024 generic nl/en/fr Wikipedia + code tokens, 5.5 min on this card): HumanEval 156/164 against
# 155 (RTN) and 158 (FP16 dense), same speed. Cost: exact retrieval at 2k tokens is weaker than FP16 dense (22/24 needle facts against 24/24 on the Orin's 8 prompts), HumanEval does not show it.
BASE_REPO=google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
ASSIST_REPO=            # empty: no MTP assistant is downloaded or exported (no speculation on this profile)
DENSE_GPTQ=1            # run the layer-wise GPTQ of the dense part in the builder (the Orin profile keeps FP16 dense: its 2k retrieval was the reason)
GPTQ_SAMPLES=128        # 128 sequences x 1,024 tokens of generic nl/en/fr Wikipedia + code: the setting used (output error 0.0006-0.0027 per layer); not swept
# INT8 KV cache with per-layer scales calibrated on this model (tools/calibrate_int8_kv_layerwise_26b.py, 141 s; the file ships in poc/data): decode +10/15/21% at 2k/4k/7.5k, idle VRAM -330 MiB,
# no retrieval loss visible. EDGELLM_GEMMA4_BOUNDED_KV_QUANT=1 keeps the sliding-window KV bounded for a quantised KV (otherwise the bound is dropped).
KV_SCALES=$POC/data/gemma4-26b-int8-kv-scales.json
# MAXIN 4,096 tokens per prefill pass (longer prompts are chunked), CAP 8,192: the largest pair that loads next to the expert cache in 8 GB (8,192 + 16,384 does not). POOL=0: prefix reuse
# comes from the host prefix cache below, not from extra KV pages.
MAXIN=4096; CAP=8192; POOL=0; TREE=1   # TREE unused (no speculation); the other three are explained above
# Export switches, one reason each: GELU_PLUGIN = fused GELU(gate)*up (prompt passes), FUSE_GEMMA4_INT4 = fused q+k+v and gate+up INT4 GEMMs (decode +1-4% with the plugin),
# XQA_MULTI_BLOCK = split-KV decode attention (build and export read it), INT8_KV_SCALES + BOUNDED_KV_QUANT as above.
EXPORT_ENV="EDGELLM_INT8_KV_SCALES=$KV_SCALES EDGELLM_GEMMA4_BOUNDED_KV_QUANT=1 EDGELLM_XQA_MULTI_BLOCK=auto EDGELLM_GEMMA4_GELU_PLUGIN=1 EDGELLM_FUSE_GEMMA4_INT4=1"
# int4_awq = the INT4 expert format; --externalize-weights keeps the expert weights (and the embedding table) as separate files next to the engine, which the plugin streams from host memory.
EXPORT_ARGS="--quantization int4_awq --externalize-weights int4_ffn int4_moe --skip-visual --skip-audio"
INT8_EMBEDDINGS=0       # the 26B has no per-layer embedding tables (the E4B's INT8 tables do not apply); the 1.5 GB token embedding goes to host memory at serve time instead
# EDGELLM_MOE_STAGE_EXPERTS=2 at engine build: the MoE plugin stages routed experts into private buffers so the prefill copy can overlap the previous layer's GEMMs (cold prefill 4k +59%).
BUILD_ENV="EDGELLM_MOE_STAGE_EXPERTS=2"
SERVE_ARGS="--tool-call-parser gemma4"   # the model's tool-call format (the PoC replay and the app use tool calls); no draft flags: no speculation
# Serve environment, one reason each:
#  EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe  experts live in pinned host memory (the point of this profile)       EDGELLM_EMBEDDING_ON_HOST=1  the 1.5 GB embedding table too
#  EDGELLM_FORCE_GREEDY=1 / EDGELLM_JSON_MAX_WHITESPACE=20   as in every profile (README)
#  EDGELLM_MOE_CACHE_SLOTS=auto   per-layer GPU expert cache sized from the free VRAM at the first decode step (45 slots of 128 here: 83.75% hit rate); decode 14 tok/s with no cache -> 54
#  EDGELLM_MOE_CACHE_RESERVE_MB=250   VRAM kept free after sizing the cache (the late +124 MiB growth of the process is unexplained)
#  EDGELLM_MOE_PREFILL_CHUNK=32   prefill prefetches the experts in chunks of 32 into a 3 x 64 MiB ring instead of two whole-layer buffers: +6 cache slots, replay-60 346.9 -> 326.0 s
#  EDGELLM_MOE_CACHE_BIAS=0   exact routing: the fork's cache-aware routing defaults to 0.5 and was only measured on Qwen3.6
#  EDGELLM_SWA_PREFIX_CACHE_MB=4096   host-resident prefix cache of the bounded sliding-window KV (replay-60 -7.5%; warm TTFT halves for prompts longer than one 4,096-token pass); not --enable-context-reuse (it
#                                     switches the sliding-window KV back to full storage)
SERVE_ENV="EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe EDGELLM_EMBEDDING_ON_HOST=1 EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MOE_CACHE_SLOTS=auto EDGELLM_MOE_CACHE_RESERVE_MB=250 EDGELLM_MOE_PREFILL_CHUNK=32 EDGELLM_SWA_PREFIX_CACHE_MB=4096 EDGELLM_MOE_CACHE_BIAS=0"
DRAFTER_INT4_MLP=0      # no drafter on this profile
