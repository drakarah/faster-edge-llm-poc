# Gemma 4 26B-A4B (google/gemma-4-26B-A4B-it), a mixture-of-experts model (128 experts, top-8, 3.8B active parameters) with its MTP assistant.
# Orin only (measured there: METRICS 109-115 of the research repo). Every value below says whether it was measured on this model or carried over from the 12B/E4B.
# There is NO stock arm: stock Edge-LLM cannot load an engine built by the fork ("engine bundles are cache artifacts") and its own exporter loads the whole
# 51 GB BF16 checkpoint, which does not fit in 32 GB. The table has the optimised arm only.
KIND=gemma4moe
NO_STOCK=1
# Long-context prompt lengths (tokens). 7,500 and not 8,192: the prompt plus the generated answer must fit the KV capacity (CAP) of 8,192.
LONGCTX_DEFAULT=2048,4096,7500
# Google's QAT checkpoint in BF16 ("unquantized"): the exporter packs each layer's experts to INT4 (RTN, group 64, the checkpoint's own QAT grid) while it loads them,
# so the 51 GB model never has to exist in memory as INT4 files first. Attention, dense MLP and lm_head stay FP16: dense INT4 (GPTQ) is +33% generation speed but
# costs retrieval (2k-token needles 8/8 -> 6/8 prompts, METRICS 113); INT4 only for MLP gate/up + lm_head keeps retrieval and gives +11% (METRICS 114, needs the
# GPTQ checkpoint from the RTX 4060 session and tools/dense_int4_subset_26b.py, not part of this profile yet).
BASE_REPO=google/gemma-4-26B-A4B-it-qat-q4_0-unquantized
ASSIST_REPO=google/gemma-4-26B-A4B-it-assistant
GPTQ_SAMPLES=0          # no GPTQ calibration in this profile (experts: RTN on the QAT grid, dense part: FP16)
KV_SCALES=""            # KV cache stays FP16: INT8 KV scales exist (docs repo results/g26_int8_kv_scales.json) but are not measured on the Orin yet
# MAXIN: prompt tokens per prefill pass (longer prompts are chunked); also the unit of prefix reuse. 2,048 was chosen, not swept.
# CAP: KV capacity. 8,192 was chosen, not swept: the engine is 17-18 GB and the 4 GiB host cache comes out of the same 32 GB; longer was not tried. POOL=0: no extra KV pool pages,
# prefix reuse comes from the host-resident cache (SERVE_ENV) instead of the full-KV pool.
# TREE: verify-tree size 16 = one 16-row tile of the INT4/Marlin GEMMs on the Orin (see the 12B config for the tile-ladder measurement); a 26B tree sweep was started, not finished.
MAXIN=2048; CAP=8192; POOL=0; TREE=16
# EDGELLM_GEMMA4_DRAFT_VOCAB_MAP: the drafter's reduced candidate vocabulary (same file as the 12B). EDGELLM_GEMMA4_BOUNDED_KV_SPEC=1: keeps the sliding-window KV bounded
# (1,152 slots instead of the full context) for the MTP base and draft; it is off for MTP exports otherwise, and the host prefix cache below needs it.
EXPORT_ENV="EDGELLM_GEMMA4_DRAFT_VOCAB_MAP=$POC/data/vocab/gemma4 EDGELLM_GEMMA4_BOUNDED_KV_SPEC=1"
# int4_awq: the INT4 format of the experts; --externalize-weights keeps the INT4 weights as separate files next to the engine (same as the other profiles); --mtp adds the drafter.
EXPORT_ARGS="--quantization int4_awq --externalize-weights int4_ffn int4_moe --skip-visual --skip-audio --mtp"
INT8_EMBEDDINGS=0       # the 26B has no per-layer embedding tables (the E4B's INT8 tables do not apply)
# --draft-top-k 5 --draft-step 3: the 12B's Orin tree shape (5 candidates, 3 steps, verified in one 16-node tree); not swept for the 26B. --tool-call-parser gemma4: the model's tool-call format.
SERVE_ARGS="--draft-top-k 5 --draft-step 3 --verify-tree-size 16 --tool-call-parser gemma4"
# Serve environment, one reason each:
#  EDGELLM_FORCE_GREEDY=1             the MTP tree verifies greedily only (README: correctness workaround, no speed effect)
#  EDGELLM_JSON_MAX_WHITESPACE=20     bounds whitespace runs in constrained JSON (README: without it the model can emit indentation until max_tokens)
#  EDGELLM_MTP_SUFFIX_DRAFT=19 / _MIN=3 / _PROMPT=16384   suffix draft in the verify tree and prompt lookup (values of the 12B profile, README rows "Suffix draft"/"Prompt lookup"); not tuned for the 26B
#  EDGELLM_MOE_SMALL_THRESHOLD=16     the 16-token verify step puts about one row on each expert; the 8-row Marlin block beats the 32-row one
#                                     (generation 51.6 -> 60.2 tok/s, replay-60 313 -> 276 s, HumanEval 158/164 both; larger blocks and a register cap were slower, METRICS 112)
#  EDGELLM_SWA_PREFIX_CACHE_MB=4096   host-resident prefix cache of the bounded sliding-window KV (METRICS 115: replay-60 -10.9%, warm TTFT at 7.5k 4.7 -> 1.0 s;
#                                     cold prefill 5-9% slower; no gain for prompts under one 2,048-token pass). Takes 4 GiB of the unified memory. Needs the fork at 147af64 or later.
# Not set on purpose: EDGELLM_INT4_GEMV_MAX_M=1 and the W4A8 prompt GEMM (they act on the dense INT4 GEMMs; this profile's dense part is FP16, W4A8 did not help the INT4 variant either, METRICS 113);
# --enable-context-reuse (switches the sliding-window KV back to full storage and is not the host cache).
SERVE_ENV="EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_MTP_SUFFIX_DRAFT=19 EDGELLM_MTP_SUFFIX_MIN=3 EDGELLM_MTP_SUFFIX_PROMPT=16384 EDGELLM_MOE_SMALL_THRESHOLD=16 EDGELLM_SWA_PREFIX_CACHE_MB=4096"
BUILD_ENV=""            # EDGELLM_XQA_MULTI_BLOCK=auto is already read from the build script's environment for every gemma4 build
DRAFTER_INT4_MLP=0      # INT4 drafter MLP/head (a 12B gain) not measured on the 26B
