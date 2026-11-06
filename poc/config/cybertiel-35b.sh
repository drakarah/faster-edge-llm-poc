# CyberTiel: huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated (Qwen3.5-MoE derived 35B-A3B: 40 layers = 30 gated delta-net + 10 full attention with 2 KV heads, 256 experts top-8 + shared expert), INT4, on the Jetson AGX Orin 32 GB.
# Loaded when PROFILE=orin (config/cybertiel-35b.dgpu.sh is the 8 GB discrete-GPU profile). All 256 experts stay on the GPU here (unified memory), so none of the dgpu profile's expert cache, CPU miss path or growable KV pool applies.
# THE MODEL ANSWERS REQUESTS AN ALIGNED MODEL WOULD DECLINE: run agents built on it in a sandbox (no network, no host mounts) and do not expose it to untrusted users.
# Only the published export is supported (code and security calibration, INT8 KV scales inside it): QUANTIZE=1 is refused.
[ "${QUANTIZE:-0}" = 1 ] && die "cybertiel-35b has no local quantisation path here: use the published export (HF_ONNX_REPO) or ONNX_DIR"
KIND=qwen36moe
NO_STOCK=1              # stock cannot load an engine of this model built by the fork, and has no INT8-KV / chunked MTP path
BASE_REPO=huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated
ASSIST_REPO=
HF_ONNX_REPO=${HF_ONNX_REPO:-drakarah/Huihui-Ornith-1.5-35B-A3B-abliterated-EdgeLLM-INT4}   # llm/ (plain) and mtp/ (speculative base graph + drafter)
LONGCTX_DEFAULT=2048,8192,24576,49000
# Engine: passes of 12,288 tokens (MAXIN; longer prompts run in several passes, chunked prefill, also with MTP since fork dev 7386f7c), KV capacity 262,144 (INT8 KV: 10 full-attention layers x 2 KV heads x 256 x K and V = 10 KiB per token, 2.7 GB at 262,144; the model's full window, as in the dgpu profile).
# POOL=0: no extra KV pool pages. EDGELLM_XQA_MULTI_BLOCK=auto (split-KV decode attention) is set by build_engines: +26% decode at 18k tokens.
# A smaller MAXIN saves activation memory but slows prefill passes; 4,096 was not tried.
MAXIN=12288; CAP=262144; POOL=0; TREE=8   # TREE sizes both the base's verify window and the draft's accept pass (build_engines): 8 rows leave room for the lookup windows below; the served window stays 3 rows (chain of 2)
BUILD_ENV=""
# MTP drafter: published in the export's `mtp/` folder (base graph + draft; uploaded 2026-11-05). A local ONNX_DIR that holds `llm/` and `mtp_draft/` (the layout of `mtp/` in the export: draft layer RTN INT4 g128 from the original bf16
# mtp.* tensors, 32,768 or 49,152 token domain draft vocabulary), works too. The engine is a speculative chain when either exists, otherwise plain autoregressive.
# Measured on the Orin: plain 55.8 tok/s, MTP chain 2/3 with the 49k vocabulary 75.8 tok/s on code and 62.9 on prose; HumanEval 146/164 both, 77.5 vs 52.1 tok/s generation.
# Tree drafting and EDGELLM_INT4_GEMV_MAX_M=1 (the Gemma / Qwen 4B settings) LOSE on this MoE: the verify reads the union of the drafted tokens' experts and the GEMM path is slower than the GEMV at 3 rows. So neither is set.
NO_SPEC=${NO_SPEC:-$([ -f "$ENGINE_DIR/fast/.plain" ] && echo 1 || echo 0)}   # .plain: written by lib.sh when the export had no mtp/; lib.sh falls back to the plain engine when the export has neither mtp/ nor mtp_draft/; NO_SPEC=1 forces it
INT8_EMBEDDINGS=0
EXPORT_ENV=""; EXPORT_ARGS=""
# The terse "Sharp" template of the model card (thinks and talks less); copied over the engine's template after the build. Its generation prompt ends with <think>, so --reasoning-parser qwen3 is REQUIRED:
# `auto` finds no config.json in a speculative engine directory (base_config.json only) and then returns "...\n</think>\n\n answer" as plain content.
CHAT_TEMPLATE=${CHAT_TEMPLATE-$POC/data/cybertiel-sharp-chat-template.jinja}
# Context reuse with host-memory snapshots (agents resend a growing prefix) and a tail checkpoint per request so the hybrid model can restore it.
REUSE="--enable-context-reuse --context-cache-snapshots-in-host-memory --context-cache-recurrent-snapshot-pool-bytes 268435456 --context-cache-partial-kv-snapshot-pool-bytes 67108864"
if [ "$NO_SPEC" = 1 ]; then
  SERVE_ARGS="--tool-call-parser generic --reasoning-parser qwen3 $REUSE"
else
  SERVE_ARGS="--tool-call-parser generic --reasoning-parser qwen3 --draft-top-k 1 --draft-step 2 $REUSE"
fi
SERVE_ENV="EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_HYBRID_PLAIN_TAIL_CHECKPOINT=1"
# N-gram lookup windows (fork branch feat/mtp-lookup-draft, e27bc20; speculative engine only). When the suffix cache finds a continuation of the context in earlier output or in the prompt and a policy expects more tokens per
# millisecond than the draft layer's own window, the round verifies up to 8 rows taken from the lookup instead. Output is unchanged (verification), the policy learns the costs online. CyberTiel on the Orin, same engine, lookup off -> on:
# HumanEval 134/164 -> 134/164, generation 73.1 -> 92.1 tok/s overall; a code-edit probe 78.5 -> 104-138 tok/s; prose unchanged. EDGELLM_MTP_LOOKUP=7: proposals of up to 7 tokens (windows of 8 rows, the engine's
# maximum). EDGELLM_MTP_LOOKUP_PROMPT=65536: also index up to that many prompt tokens (code edits copy long spans from the prompt). Chain mode, batch 1, no guided decoding; not combinable with the packed K8V4 KV cache.
[ "$NO_SPEC" = 0 ] && SERVE_ENV="$SERVE_ENV EDGELLM_MTP_LOOKUP=${MTP_LOOKUP:-7} EDGELLM_MTP_LOOKUP_PROMPT=${MTP_LOOKUP_PROMPT:-65536}"
DRAFTER_INT4_MLP=0
# Chat defaults for a served endpoint (not the benchmarks): the model card's thinking settings, applied to requests that omit the field. WITH THE MTP ENGINE temperature, top-p and top-k have no effect: the runtime logs "Spec-decode active: overriding sampling params to greedy" and decodes greedy (only the plain engine samples). presence_penalty is only set for the plain engine:
# the runtime answers 400 to it with speculative decoding. A client must not send it to the MTP engine.
CHAT_ENV="EDGELLM_DEFAULT_ENABLE_THINKING=${THINKING:-1} EDGELLM_DEFAULT_TEMPERATURE=${TEMPERATURE:-1.0} EDGELLM_DEFAULT_TOP_P=${TOP_P:-0.95} EDGELLM_DEFAULT_TOP_K=${TOP_K:-20} EDGELLM_PRESERVE_THINKING=${PRESERVE_THINKING:-1} EDGELLM_REQUEST_STATS=${REQUEST_STATS:-1}"
[ "$NO_SPEC" = 1 ] && CHAT_ENV="$CHAT_ENV EDGELLM_DEFAULT_PRESENCE_PENALTY=${PRESENCE_PENALTY:-1.5}"
# The bench times the server as it runs (server sampling defaults, thinking on), with raised output caps and a reasoning budget like the dgpu profile.
export BENCH_SERVER_DEFAULTS=1 BENCH_NEEDLE_MAX_TOKENS=12288 BENCH_HUMANEVAL_MAX_TOKENS=12288
# The reasoning budget is implemented for plain decoding only (fork dev: "reasoning budget for vanilla decoding"), so the speculative engine gets none.
if [ "$NO_SPEC" = 1 ]; then export BENCH_EXTRA_BODY=${BENCH_EXTRA_BODY-'{"reasoning_budget_tokens": 8192}'}; fi
