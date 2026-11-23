# Inkling 2 Mini, a looped mixture of experts (26B parameters: 24 layers visited 48 times per token, 128 routed experts per layer with top-8 sigmoid routing, short convolutions, 256K context),
# on a discrete 8 GB GPU (measured on an RTX 4060 laptop-class card, PCIe 4.0 x8, 31 GB RAM). Loaded instead of inkling2.sh when PROFILE=dgpu. Every value says why.
# The experts (13.1 GB as NVFP4) stay in pinned host memory; the plugin streams what a step needs over PCIe into a per-layer expert cache on the GPU (the cache takes everything the rest leaves free).
# Unlike the other profiles the engine is built straight from the checkpoint (experimental builder), not from an ONNX export, and there is no published export: the checkpoint is prepared here from the original download.
# DISK: about 20 GB for the download plus 15 GB for the prepared checkpoint (links to the original shard plus the NVFP4 expert sidecars and the FP8 dense file). RAM: the sidecar step needs about 20 GB.
KIND=inkling2
NO_STOCK=1              # stock Edge-LLM cannot load this model: no looped-layer support, and every expert would have to fit on the GPU
NO_SPEC=1               # plain autoregressive engine: the shipped drafter needs steps of more than 3 tokens, which skip the expert cache (about 2 s per step)
BASE_REPO=${BASE_REPO:-}   # the Hugging Face repository of the original Inkling 2 Mini checkpoint (not set here: pass BASE_REPO=<repo>, or put the files in $MODELS_DIR/<name> and pass BASE_REPO=<owner>/<name>)
[ -n "$BASE_REPO" ] || die "inkling2: set BASE_REPO to the Hugging Face repository of the Inkling 2 Mini checkpoint"
ASSIST_REPO=
LONGCTX_DEFAULT=2048,8000,30000   # the engine's window is 262,144 tokens; prefill passes are 2,048 (MAXIN)
# Preparation (poc/tools): NVFP4 experts of the shipped checkpoint repacked for the Marlin MoE plugin with one global scale per layer (inkling2_nvfp4_sidecar.py; the four MXFP8 layers are rounded to NVFP4 first),
# FP8 dense attention projections with the shipped activation calibration (data/inkling2-fp8-amax.json; inkling2_fp8_dense.py), and a config with the FP8 KV cache switched on (inkling2_merge_checkpoint.py).
# Quality against the bf16 reference model, teacher-forced decode on 16 x 96 forced tokens: mean KL divergence 0.0103, top-1 agreement 94.8% (the dense FP8 part costs about 0.004, the NVFP4 experts about 0.006).
# Window: 262,144 tokens (the sliding-window layers keep a bounded KV, the 8 full-attention visits a growable pool). MAXIN is the longest prefill pass: every pass streams all experts over PCIe (26 GB), so a longer pass is cheaper per token
# (8K-token prompt: 413 tok/s with 1,024-token passes, 784 with 2,048, 1,159 with 4,096) but its KV pool is larger and costs expert-cache slots (39 / 36 slots at 2,048 / 4,096 tokens, decode 41 / 45 ms with the cache overlay below).
MAXIN=${MAXIN:-2048}; CAP=262144; POOL=0
# Engine build environment, one reason each:
#  EDGELLM_MOE_STAGE_EXPERTS=2     the MoE plugin stages only the routed experts of a prefill pass through private buffers, so the copy of the next layer overlaps this layer's GEMMs
#  EDGELLM_MOE_PREFILL_OVERLAP=1   as for the other profiles
#  EDGELLM_INKLING_FUSED_RESIDUAL=1  one kernel for each block attention residual instead of a chain of small ops: 145 MiB less activation memory at 2,048 tokens (decode speed unchanged)
BUILD_ENV="EDGELLM_MOE_STAGE_EXPERTS=2 EDGELLM_MOE_PREFILL_OVERLAP=1 EDGELLM_INKLING_FUSED_RESIDUAL=1"
# The model's own chat template uses keyword arguments the runtime's template engine cannot parse; this one renders the same prompt (reasoning effort levels, tools, thinking). Copied over the engine's template after the build.
CHAT_TEMPLATE=${CHAT_TEMPLATE-$POC/data/inkling2-chat-template.jinja}
SERVE_ARGS="--tool-call-parser inkling --reasoning-parser inkling"
# Serve environment, one reason each:
#  EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe  the expert sidecars stay in pinned host memory (the point of this profile)
#  EDGELLM_HOST_CHECKPOINT_WEIGHTS=.mlp.experts.,__embedding__   checkpoint tensors kept in pinned host memory too: the experts and the 883 MiB token embedding table (one row is read per token; on the GPU it cost 9 cache slots per layer and 5 ms of decode)
#  EDGELLM_MOE_CACHE_SLOTS=auto            per-layer GPU expert cache sized from the free VRAM at the first decode step; EDGELLM_MOE_CACHE_RESERVE_MB=250  VRAM kept free after sizing
#  EDGELLM_MOE_CACHE_SHARE=1, EDGELLM_MOE_CACHE_LAYERS=24   the three passes of a looped layer use the same weights, so one cache per layer (24 caches, not 48)
#  EDGELLM_MOE_PREFILL_CHUNK=32            prefill copies the experts in chunks of 32 through a ring of three buffers: a pass never holds a whole layer of experts on the GPU
#  EDGELLM_MOE_EXEC_OVERLAY=1              the memory TensorRT reserves for the activations of a long prefill pass (421 MiB at 2,048 tokens) is lent to the expert cache while decoding, and the cache is invalidated after every prefill: decode only touches 3.3 MiB of it
#  EDGELLM_MOE_CPU_MISS=1                  CPU miss path (x86 only): a missed expert is computed on the CPU (int8 AVX-VNNI) from the pinned host weights instead of being copied; costs EDGELLM_MOE_CPU_THREADS (6) busy cores, pinned to the performance cores with CPU_CORES=auto;
#                                          decode 42.4 -> 36.8 ms/token, same quality. EDGELLM_MOE_CPU_ADMIT=16: misses before an expert is admitted to the cache
#  EDGELLM_MOE_TAIL_SKIP=0, EDGELLM_MOE_TAIL_MIN_WEIGHT=1e9   routing is exact. LOSSY, opt-in with TAIL_SKIP / TAIL_MIN_WEIGHT in .env (decode ms/token at 48 slots, teacher-forced KL divergence): exact 33.3 (0.0101); TAIL_SKIP=6 31.6 (+0.0022);
#                                          TAIL_SKIP=2 TAIL_MIN_WEIGHT=0.08 29.0 (+0.0038); 0.10 28.2 (+0.0083); TAIL_SKIP=4 28.0 (+0.0099, top-1 agreement -2.7 points)
#  EDGELLM_MOE_CACHE_BIAS=0                no cache-aware routing (opt-in elsewhere: -18% decode time for +0.0055 KL divergence)
#  EDGELLM_KV_INITIAL_TOKENS=20480         maps the KV cache on demand (grows to input + max_tokens per request) and gives the freed VRAM to the expert cache; clients must send max_tokens
#  EDGELLM_SWA_PREFIX_CACHE_MB=6144, EDGELLM_SWA_PREFIX_CAPTURE_INTERVAL=4096   prefix reuse for the sliding-window layers: host snapshots of the KV after the prompt and every 4,096 prompt tokens (agents resend a growing prefix: a 28K-token turn answers in 3-4 s once cached)
#  EDGELLM_JSON_MAX_WHITESPACE=20          as in every profile (README)
SERVE_ENV="EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe EDGELLM_HOST_CHECKPOINT_WEIGHTS=.mlp.experts.,__embedding__ EDGELLM_EMBEDDING_ON_HOST=1 EDGELLM_MOE_CACHE_SLOTS=auto EDGELLM_MOE_CACHE_RESERVE_MB=250 EDGELLM_MOE_STAGE_EXPERTS=2 EDGELLM_MOE_PREFILL_CHUNK=32 EDGELLM_MOE_EXEC_OVERLAY=1 EDGELLM_MOE_CPU_MISS=${CPU_MISS:-1} EDGELLM_MOE_CPU_INT8=1 EDGELLM_MOE_CPU_CORES=auto EDGELLM_MOE_CPU_THREADS=6 EDGELLM_MOE_CPU_ADMIT=16 EDGELLM_MOE_TAIL_SKIP=${TAIL_SKIP:-0} EDGELLM_MOE_TAIL_MIN_WEIGHT=${TAIL_MIN_WEIGHT:-1e9} EDGELLM_MOE_CACHE_SHARE=1 EDGELLM_MOE_CACHE_LAYERS=24 EDGELLM_MOE_CACHE_BIAS=0 EDGELLM_KV_INITIAL_TOKENS=20480 EDGELLM_SWA_PREFIX_CACHE_MB=${PREFIX_MB:-6144} EDGELLM_SWA_PREFIX_CAPTURE_INTERVAL=${CAPTURE_INTERVAL:-4096} EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_REQUEST_STATS=${REQUEST_STATS:-1}"
# Chat defaults for a served endpoint (not for the benchmarks): the model's recommended sampling, thinking on, reasoning-effort words passed through to the template (an unknown word falls back to medium), earlier turns' reasoning kept in the rendered history.
CHAT_ENV="EDGELLM_DEFAULT_ENABLE_THINKING=${THINKING:-1} EDGELLM_DEFAULT_TEMPERATURE=${TEMPERATURE:-1.0} EDGELLM_DEFAULT_TOP_P=${TOP_P:-0.95} EDGELLM_DEFAULT_TOP_K=${TOP_K:-20} EDGELLM_DEFAULT_PRESENCE_PENALTY=0.0 EDGELLM_REASONING_EFFORT_PASSTHROUGH=1 EDGELLM_IGNORE_UNSUPPORTED_PENALTIES=1 EDGELLM_PRESERVE_THINKING=1"
export BENCH_SERVER_DEFAULTS=1 BENCH_NEEDLE_MAX_TOKENS=4096 BENCH_HUMANEVAL_MAX_TOKENS=4096
