# Gemma 4 26B-A4B (google/gemma-4-26B-A4B-it), a mixture-of-experts model (128 experts, top-8, 3.8B active parameters) with its MTP assistant.
# Orin only (measured there: METRICS 109-112 of the research repo). There is NO stock arm: stock Edge-LLM cannot load an engine built by the fork
# ("engine bundles are cache artifacts") and its own exporter loads the whole 51 GB BF16 checkpoint, which does not fit in 32 GB. The table has the optimised arm only.
KIND=gemma4moe
NO_STOCK=1
LONGCTX_DEFAULT=2048,4096,7500
# Google's QAT checkpoint in BF16 ("unquantized"): the exporter packs each layer's experts to INT4 (RTN, group 64, the checkpoint's own QAT grid) while it loads them.
# Attention, dense MLP and lm_head stay FP16 (dense INT4 is the next step, not in this profile).
BASE_REPO=google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
ASSIST_REPO=google/gemma-4-26B-A4B-it-assistant
GPTQ_SAMPLES=0
KV_SCALES=""
MAXIN=2048; CAP=8192; POOL=0; TREE=16
# EDGELLM_GEMMA4_BOUNDED_KV_SPEC=1 keeps the sliding-window KV bounded for the MTP base and draft (it is off for MTP exports otherwise); the host prefix cache below needs it.
EXPORT_ENV="EDGELLM_GEMMA4_DRAFT_VOCAB_MAP=$POC/data/vocab/gemma4 EDGELLM_GEMMA4_BOUNDED_KV_SPEC=1"
EXPORT_ARGS="--quantization int4_awq --externalize-weights int4_ffn int4_moe --skip-visual --skip-audio --mtp"
INT8_EMBEDDINGS=0
SERVE_ARGS="--draft-top-k 5 --draft-step 3 --verify-tree-size 16 --tool-call-parser gemma4"
# EDGELLM_MOE_SMALL_THRESHOLD=16: the 16-token verify step puts about one row on each expert; the 8-row Marlin block beats the 32-row one
# (generation 51.6 -> 60.2 tok/s, replay-60 313 -> 276 s, HumanEval 158/164 both).
# EDGELLM_SWA_PREFIX_CACHE_MB=4096: host-resident prefix cache of the bounded sliding-window KV (METRICS 115: replay-60 -10.9%, warm TTFT at 7.5k 4.7 -> 1.0 s; cold prefill 5-9% slower). Needs the fork at 147af64 or later (one-token chunked-prefill fix).
SERVE_ENV="EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MTP_SUFFIX_DRAFT=19 EDGELLM_MTP_SUFFIX_MIN=3 EDGELLM_MTP_SUFFIX_PROMPT=16384 EDGELLM_MOE_SMALL_THRESHOLD=16 EDGELLM_SWA_PREFIX_CACHE_MB=4096"
BUILD_ENV=""
DRAFTER_INT4_MLP=0
