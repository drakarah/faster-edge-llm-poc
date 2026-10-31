# CyberTiel: huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated (a Qwen3.5-MoE derived 35B-A3B with its refusals removed), INT4 export, on a discrete 8 GB GPU.
# Same architecture, quantisation recipe and engine settings as qwen36-35b.dgpu.sh (sourced below); this file only changes what differs on the live server it was measured on.
# THE MODEL ANSWERS REQUESTS AN ALIGNED MODEL WOULD DECLINE: run agents built on it in a sandbox (no network, no host mounts) and do not expose it to untrusted users.
# Only the published export is supported (the calibration mix, code + security, and its scripts are in the notes repository); QUANTIZE=1 is refused because it would use the Qwen3.6 KV scales and calibration text.
[ "${QUANTIZE:-0}" = 1 ] && die "cybertiel-35b has no local quantisation path here: use the published export (HF_ONNX_REPO) or ONNX_DIR"
_user_onnx_repo=${HF_ONNX_REPO:-}
REQUEST_STATS=${REQUEST_STATS:-1}               # per-request timing at GET /stats (the live server runs it)
KV_INITIAL_TOKENS=${KV_INITIAL_TOKENS:-12288}   # growable KV pool, see qwen36-35b.dgpu.sh (this profile maps a 262,144-token pool on demand)
source "$(dirname "${BASH_SOURCE[0]}")/qwen36-35b.dgpu.sh"
BASE_REPO=huihui-ai/Huihui-Ornith-1.5-35B-A3B-abliterated
HF_ONNX_REPO=${_user_onnx_repo:-drakarah/Huihui-Ornith-1.5-35B-A3B-abliterated-EdgeLLM-INT4}   # INT8 KV scales ship inside the export
# A 262,144-token window: passes stay at 12,288 tokens (MAXIN), the KV capacity is 262,144. A static pool of that size leaves 16 expert slots (59.1 tok/s); the growable pool keeps 56 (84.1 tok/s), METRICS R25ev.
MAXIN=12288; CAP=262144
# The terse "Sharp" template of the model card (thinks and talks less); the checkpoint's own template is the alternative (leave CHAT_TEMPLATE empty). Copied over the engine's template after the build.
CHAT_TEMPLATE=${CHAT_TEMPLATE-$POC/data/cybertiel-sharp-chat-template.jinja}
# Context reuse with host-memory snapshots (agents resend a growing prefix: 14 turns at up to 68k tokens 23.3 s/turn without, 3.6 with; a live 116-request session reached 97.6% of prompt tokens from reuse, 88k tokens at most).
# EDGELLM_HYBRID_PLAIN_TAIL_CHECKPOINT=1 keeps one checkpoint per request tail so a hybrid (linear attention) model can restore it.
SERVE_ENV="$SERVE_ENV EDGELLM_HYBRID_PLAIN_TAIL_CHECKPOINT=1"
SERVE_ARGS="--tool-call-parser generic --enable-context-reuse --context-cache-snapshots-in-host-memory --context-cache-recurrent-snapshot-pool-bytes 268435456 --context-cache-partial-kv-snapshot-pool-bytes 67108864"
# CHAT_ENV is the one of qwen36-35b.dgpu.sh (thinking on, temperature 1.0, top_p 0.95, top_k 20, presence penalty 1.5, preserved thinking) plus the growable pool above.
# Clients must send max_tokens (without it the first request maps the whole pool); the model's reasoning can run away (3 of 24 agent runs hit 16,384 tokens with the Qwen3.6 arm at 12).
