# Inkling 2 Mini, a looped mixture of experts (26B parameters: 24 layers visited 48 times per token, 128 routed experts per layer with top-8 sigmoid routing, short convolutions, 256K context),
# on a discrete 8 GB GPU (measured on an RTX 4060 laptop-class card, PCIe 4.0 x8, 31 GB RAM). Loaded instead of inkling2.sh when PROFILE=dgpu. Every value says why.
# The experts (13.1 GB as NVFP4) stay in pinned host memory; the plugin streams what a step needs over PCIe into a per-layer expert cache on the GPU (the cache takes everything the rest leaves free).
# Unlike the other profiles the engine is built straight from the checkpoint (experimental builder), not from an ONNX export, and there is no published export: the checkpoint is prepared here from the original download.
# DISK: about 20 GB for the download plus 15 GB for the prepared checkpoint (links to the original shard plus the NVFP4 expert sidecars and the FP8 dense file). RAM: the sidecar step needs about 20 GB.
KIND=inkling2
NO_STOCK=1              # stock Edge-LLM cannot load this model: no looped-layer support, and every expert would have to fit on the GPU
NO_SPEC=1               # plain autoregressive engine: the shipped drafter needs steps of more than 3 tokens, which skip the expert cache (about 2 s per step)
# The original checkpoint: thinkingmachines/Inkling-2-Mini (Apache-2.0, not gated; one 19.5 GB shard with the NVFP4 experts of layers 0-19, MXFP8 experts of layers 20-23 and BF16 for the rest, plus a 67 MB drafter that is not used).
# HF_REVISION pins the revision the preparation tools and the measurements below were validated on (a later revision only changed the model card; the weight shard is identical).
BASE_REPO=${BASE_REPO:-thinkingmachines/Inkling-2-Mini}
export HF_REVISION=${HF_REVISION:-b85c3cfa615fc413cc9af3379f491b963f3b586a}
ASSIST_REPO=
LONGCTX_DEFAULT=2048,8000,30000   # the engine's window is 262,144 tokens; prefill passes are 2,048 (MAXIN)
# Preparation (poc/tools): NVFP4 experts of the shipped checkpoint repacked for the Marlin MoE plugin with one global scale per layer (inkling2_nvfp4_sidecar.py; the four MXFP8 layers are rounded to NVFP4 first),
# FP8 dense attention projections with the shipped activation calibration (data/inkling2-fp8-amax.json; inkling2_fp8_dense.py), and a config with the FP8 KV cache switched on (inkling2_merge_checkpoint.py).
# Quality against the bf16 reference model, teacher-forced decode on 16 x 96 forced tokens (1,520 steps): mean KL divergence 0.0103, top-1 agreement 94.8%; in the 512-token prefill harness the NVFP4 engine reads 0.0096 and the FP8 dense projections alone cost 0.0044 against a floor of 0.0024 (README, Inkling 2 Mini results).
# Window: 262,144 tokens (the sliding-window layers keep a bounded KV, the 8 full-attention visits a growable pool). MAXIN is the longest prefill pass: every pass streams all experts over PCIe (26 GB), so a longer pass is cheaper per token
# (7,580-token prompt, one run each: 413 tok/s with 1,024-token passes, 784 with 2,048, 1,159 with 4,096) but the bounded sliding-window KV pool grows with the pass (830 / 1,150 MiB at 2,048 / 4,096) and costs expert-cache slots: each 1,000 tokens of pass length cost about 1.6 slots,
# about 1 ms of decode. 2,048 is the middle: 48 slots per layer and 33.3 ms/token with this file's settings (measured; the 4,096 engine was measured at 36 slots / 40.2 ms before the embedding table moved to host memory, not re-measured since).
MAXIN=${MAXIN:-2048}; CAP=262144; POOL=0
# Engine build environment (these three are baked into the engine, so changing them means a rebuild):
#  EDGELLM_MOE_STAGE_EXPERTS=2        the MoE plugin stages only the experts a prefill pass routes to through private buffers, so the PCIe copy of the next layer overlaps this layer's GEMMs. Must also be set when serving (below).
#  EDGELLM_MOE_PREFILL_OVERLAP=1      same overlap switch as in the other profiles.
#  EDGELLM_INKLING_FUSED_RESIDUAL=1   one kernel for each of the model's block attention residuals instead of a chain of small TensorRT ops: 145 MiB less activation memory at 2,048 tokens, quality unchanged (KL divergence 0.01026 against 0.01031),
#                                     decode speed unchanged (the no-fill floor measured 16.68 against 16.30 ms: memory, not speed, is the gain). Needs the plugin of this pin at serve time.
BUILD_ENV="EDGELLM_MOE_STAGE_EXPERTS=2 EDGELLM_MOE_PREFILL_OVERLAP=1 EDGELLM_INKLING_FUSED_RESIDUAL=1"
# The model's own chat template uses keyword arguments the runtime's template engine cannot parse; this one renders the same prompt (reasoning effort, tool declarations, thinking blocks). It always renders an assistant message's reasoning_content
# when the client sends it back (it has no preserve_thinking switch). Copied over the engine's template after the build.
CHAT_TEMPLATE=${CHAT_TEMPLATE-$POC/data/inkling2-chat-template.jinja}
# --tool-call-parser inkling / --reasoning-parser inkling: the model writes its tool calls and its reasoning in its own message formats; these parsers turn them into OpenAI tool_calls and reasoning_content.
SERVE_ARGS="--tool-call-parser inkling --reasoning-parser inkling"
# Serve environment, one reason each (all measured on the RTX 4060 laptop card named above unless it says otherwise):
#  EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe   the expert weight files stay in pinned host memory and the engine reads them over PCIe (UVA): 13.1 GB of experts cannot sit in 8 GB.
#  EDGELLM_HOST_CHECKPOINT_WEIGHTS=.mlp.experts.,__embedding__   checkpoint tensors kept in pinned host memory as well (comma-separated substrings of the tensor name): the experts and the 883 MiB token embedding table.
#                                           One embedding row is read per token, so on the GPU the table cost 9 expert-cache slots per layer and 5 ms of decode (39 -> 48 slots, 38.3 -> 33.3 ms/token when it moved), with identical output.
#  EDGELLM_EMBEDDING_ON_HOST=1              the same wish for engines that ship the embedding as a separate file; inert for this model (its table is a checkpoint tensor, handled by the line above), kept so an engine exported that way behaves the same.
#  EDGELLM_MOE_CACHE_SLOTS=auto             the per-layer GPU expert cache is sized from the free VRAM at the first decode step (the more it holds, the fewer experts cross PCIe: about 0.7 ms of decode per slot).
#  EDGELLM_MOE_CACHE_RESERVE_MB=250         VRAM kept free after sizing (graph capture, the KV pool's growth, driver); the earlier default of 1,450 left 24 slots and 57 ms/token, 250 left 35-38 slots and 42 ms (before the changes below).
#  EDGELLM_MOE_STAGE_EXPERTS=2              as in the build.
#  EDGELLM_MOE_PREFILL_CHUNK=32             prefill copies a layer's routed experts in chunks of 32 through a ring of three buffers instead of staging whole layers: +20% prefill on an 8K prompt and 2-4 more cache slots.
#  EDGELLM_MOE_EXEC_OVERLAY=1               TensorRT reserves 421 MiB for the activations of a 2,048-token pass; decode touches only 3.3 MiB of it (poison scan), so the rest is lent to the expert cache while decoding and the cache is invalidated after every prefill
#                                           (the prefill's activations overwrite it). 417 MiB more cache: 39 slots instead of 33 on the unfused engine, and it makes longer prefill passes cost no activation memory at decode time.
#  EDGELLM_MOE_CPU_MISS=1                   x86 only (off when the CPU lacks AVX2): an expert missing from the GPU cache is computed on the CPU from the pinned host weights, in parallel with the PCIe fills of the other misses, until it has missed
#                                           EDGELLM_MOE_CPU_ADMIT times, then it is admitted to the cache. Costs EDGELLM_MOE_CPU_THREADS busy cores. Decode 42.4 -> 36.8 ms/token at 38 slots, same quality (KL divergence 0.01012 against 0.01031).
#  EDGELLM_MOE_CPU_INT8=1                   int8 activations on the CPU (AVX-VNNI; falls back to fp32 without it): the fp32 CPU path gave only -5% at best, int8 -13%; about 0.65% relative error on an expert's output.
#  EDGELLM_MOE_CPU_CORES=auto               pins one worker per performance core on Intel hybrid CPUs (unpinned elsewhere): with eight threads of which only six were pinned the path was no faster than 5%.
#  EDGELLM_MOE_CPU_THREADS=6                the six performance cores of the i9-12900H; set it to the performance-core count of your CPU. Sweep at 39 slots (ms/token): 4 threads 40.2, 6 threads 38.3, 8 threads 38.0 (6 and 8 equal within noise).
#  EDGELLM_MOE_CPU_ADMIT=16                 misses before an expert is admitted to the cache (admit 0 is degenerate; higher makes the CPU the bottleneck). Sweep at 39 slots (ms/token): 8 -> 39.0, 12 -> 38.1 (5 threads), 16 -> 38.3, 24 -> 38.6, 32 -> 40.5.
#  EDGELLM_MOE_TAIL_SKIP=0, EDGELLM_MOE_TAIL_MIN_WEIGHT=1e9   routing is exact. LOSSY and opt-in (TAIL_SKIP / TAIL_MIN_WEIGHT in .env): a token's lower-ranked or low-weight experts are skipped. Decode ms/token at 48 slots (teacher-forced KL divergence added to the exact 0.0101):
#                                           TAIL_SKIP=6 31.6 (+0.0022); TAIL_SKIP=2 TAIL_MIN_WEIGHT=0.08 29.0 (+0.0038); TAIL_MIN_WEIGHT=0.10 28.2 (+0.0083); TAIL_SKIP=4 28.0 (+0.0099, top-1 agreement 94.8% -> 92.1%). Exact is 33.3. No task benchmark has been run on these.
#  EDGELLM_MOE_CACHE_SHARE=1, EDGELLM_MOE_CACHE_LAYERS=24   the model has 24 layers of weights visited 48 times (a loop of 12 layers, three passes): the visits of one layer share one expert cache, 24 caches instead of 48 (first host-streaming engine: 10.3 -> 16.1 tok/s).
#  EDGELLM_MOE_CACHE_BIAS=0                 no cache-aware routing: it is lossy (-18% decode time for +0.0055 KL divergence at a bias of 0.05) and was not stacked with the CPU path.
#  EDGELLM_KV_INITIAL_TOKENS=20480          the KV cache is mapped on demand: it starts at this many tokens (not swept), grows to input + max_tokens per request, trims to the peak of the last requests, and the memory it frees goes to the expert cache.
#                                           Clients must send max_tokens (without it the first request maps the whole pool).
#  EDGELLM_SWA_PREFIX_CACHE_MB=6144         host memory for prefix reuse of the sliding-window layers (the generic context reuse would need the full window KV, which does not fit): snapshots of the KV after each prompt and every
#  EDGELLM_SWA_PREFIX_CAPTURE_INTERVAL=4096 4,096 prompt tokens. Neither value was swept. In an agent session 28K-token prompts answered in 3-4 s once cached.
#  EDGELLM_JSON_MAX_WHITESPACE=20           as in every profile (README): bounds the whitespace a constrained JSON answer may emit.
#  EDGELLM_REQUEST_STATS=1                  the dashboard at /stats and its JSON at /api/stats (0 turns it off).
SERVE_ENV="EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe EDGELLM_HOST_CHECKPOINT_WEIGHTS=.mlp.experts.,__embedding__ EDGELLM_EMBEDDING_ON_HOST=1 EDGELLM_MOE_CACHE_SLOTS=auto EDGELLM_MOE_CACHE_RESERVE_MB=250 EDGELLM_MOE_STAGE_EXPERTS=2 EDGELLM_MOE_PREFILL_CHUNK=32 EDGELLM_MOE_EXEC_OVERLAY=1 EDGELLM_MOE_CPU_MISS=${CPU_MISS:-1} EDGELLM_MOE_CPU_INT8=1 EDGELLM_MOE_CPU_CORES=auto EDGELLM_MOE_CPU_THREADS=6 EDGELLM_MOE_CPU_ADMIT=16 EDGELLM_MOE_TAIL_SKIP=${TAIL_SKIP:-0} EDGELLM_MOE_TAIL_MIN_WEIGHT=${TAIL_MIN_WEIGHT:-1e9} EDGELLM_MOE_CACHE_SHARE=1 EDGELLM_MOE_CACHE_LAYERS=24 EDGELLM_MOE_CACHE_BIAS=0 EDGELLM_KV_INITIAL_TOKENS=20480 EDGELLM_SWA_PREFIX_CACHE_MB=${PREFIX_MB:-6144} EDGELLM_SWA_PREFIX_CAPTURE_INTERVAL=${CAPTURE_INTERVAL:-4096} EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_REQUEST_STATS=${REQUEST_STATS:-1}"
# Chat defaults for a served endpoint (not for the benchmarks, which send their own fields), applied to requests that omit the field:
#  EDGELLM_DEFAULT_ENABLE_THINKING=1        the model thinks by default (its template's default reasoning effort is 0.9); a request can send chat_template_kwargs {"enable_thinking": false}. Thinking costs tokens: set max_tokens.
#  EDGELLM_DEFAULT_TEMPERATURE=1.0          the checkpoint's own generation_config (do_sample, temperature 1.0).
#  EDGELLM_DEFAULT_TOP_P=${TOP_P:-0.9999}, EDGELLM_DEFAULT_TOP_K=${TOP_K:-0}   the vendor-equivalent sampling: llama.cpp users of Inkling 2 are given `--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0` (Unsloth's Inkling 2 page; the model card itself names no values).
#                                           The sampler cannot run untruncated: temperature 1.0 with top_p 1.0 and no top-k takes the greedy branch, so top_p is 0.9999 (the tail beyond 99.99% of the mass is cut) and top_k 0 is off.
#  EDGELLM_DEFAULT_PRESENCE_PENALTY=0.0     none: the model card asks for none.
#  EDGELLM_REASONING_EFFORT_PASSTHROUGH=1   the template takes the level names minimal / low / medium / high / xhigh / max or a number in [0, 0.99]; the server's default mapping would collapse them to Qwen's three levels. A word the template does not know
#                                           falls back to medium (and is logged), a number is clamped; none / off / disabled / false turn thinking off.
#  EDGELLM_IGNORE_UNSUPPORTED_PENALTIES=1   agent harnesses send a frequency_penalty that cannot be configured and the runtime has no kernel for it: drop it instead of answering HTTP 400.
#  EDGELLM_DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-16384}   bounds only requests that send no max_tokens. Without it such a request may generate up to the 262,144-token window, so the runtime grows the KV pool to that size by
#                                           taking the memory from the expert cache (48 -> 16 slots) and the cache does not grow back: decode fell from 30 to 15 tok/s after one such request (measured with the bench replay,
#                                           HumanEval greedy 14.3 tok/s). 16,384 plus a 2,048-token prompt stays inside the 20,480-token KV pool that is mapped at start. Clients that need more send max_tokens.
#  EDGELLM_PRESERVE_THINKING=1              inert for this model: it only affects templates that read preserve_thinking (Qwen3.5/3.6). Earlier turns' reasoning stays in the history when the client sends reasoning_content back, which this template always renders.
CHAT_ENV="EDGELLM_DEFAULT_ENABLE_THINKING=${THINKING:-1} EDGELLM_DEFAULT_TEMPERATURE=${TEMPERATURE:-1.0} EDGELLM_DEFAULT_TOP_P=${TOP_P:-0.9999} EDGELLM_DEFAULT_TOP_K=${TOP_K:-0} EDGELLM_DEFAULT_PRESENCE_PENALTY=0.0 EDGELLM_REASONING_EFFORT_PASSTHROUGH=1 EDGELLM_IGNORE_UNSUPPORTED_PENALTIES=1 EDGELLM_PRESERVE_THINKING=1 EDGELLM_DEFAULT_MAX_TOKENS=${DEFAULT_MAX_TOKENS:-16384}"
# The bench times the server as it runs normally (sampling defaults, thinking on); the output caps are raised so a long reasoning block does not dominate. 4,096 tokens: not tuned for this model.
export BENCH_SERVER_DEFAULTS=1 BENCH_NEEDLE_MAX_TOKENS=4096 BENCH_HUMANEVAL_MAX_TOKENS=4096
