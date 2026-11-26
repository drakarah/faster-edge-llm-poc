# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini): a looped mixture of experts (24 layers, the middle 12 run three times = 48 visits, 128 experts top-8, hidden 2,304, NVFP4 experts).
# SETTINGS-ONLY PROFILE, Orin only: the values below are the ones the measured engine used (README, "inkling2-mini"). It is NOT wired into the docker compose pipeline:
# the engine needs a converted checkpoint (NVFP4 experts, dense weights in bf16/fp16) that this repo cannot produce, and stock Edge-LLM has no Inkling support (no stock arm).
# Build and run it by hand with poc/inkling2/build_engine.sh and poc/inkling2/serve.sh, which read this file.
die "inkling2-mini is a settings profile (see poc/inkling2/README.md): use poc/inkling2/build_engine.sh, not the compose pipeline"
KIND=inkling2
NO_STOCK=1
BASE_REPO=thinkingmachines/Inkling-2-Mini
DRAFTER_DEFAULT=dflash2
DRAFTER_REPO=incoai/Inkling-2-Mini-DFlash2   # 5 layers, 0.5B, apache-2.0, 990 MB; converted to a 1.06 GB INT8 file by poc/tools/inkling2_dflash2_int8.py
# Engine: one 1,024-token prefill pass (longer prompts are chunked), KV capacity 32,768 (FP16 KV, 8 full-attention visits; the sliding-window visits keep a bounded 1,120-slot ring).
MAXIN=1024; CAP=${CAP:-32768}   # CAP=262144 KV_INT8=1 builds the 262,144-token engine (INT8 KV, METRICS 192 and 245): serve it WITHOUT CONTEXT_REUSE (reuse keeps the sliding-window layers at full length and exhausts the Orin's memory)
# Build environment, one reason each:
#  EDGELLM_INT8_GEMV=attn+head        the dense attention projections and the LM head run as INT8 weight-only GEMVs (per-row RTN): -8% decode, quality in METRICS 194 of the docs repo
#  EDGELLM_INKLING_FUSED_GATE=1       q/gate/rel/sink share one input: fused gate kernels
#  EDGELLM_INKLING_FUSED_QRSG=1       one INT8 GEMV for q, gate, rel and sink (-0.5 ms per token)
#  EDGELLM_INKLING_VERIFY=8           generation profile of 1..8 tokens: chain verification of up to 7 drafted tokens. Adds per-token conv states (the rollback after a partly accepted window) and
#                                     the four drafter taps as engine outputs; plain decode costs +2.4 ms for them (36.5 vs 34.0 ms per token).
#  EDGELLM_INT8_MMA=1                 the INT8 dense weights are stored in mma order and run on tensor cores with split-K and batched loads for 1..8 tokens: a verify round of 5-8 rows is
#                                     about 10 ms cheaper, 3-4 rows 2-3 ms; hybrid K=3 +2.7% (36.75 against 35.77 tok/s), plain 1-token decode +1.25 ms (3.4%) and prefill +3.9% slower (METRICS 208)
BUILD_ENV="EDGELLM_INT8_GEMV=attn+head EDGELLM_INKLING_FUSED_GATE=1 EDGELLM_INKLING_FUSED_QRSG=1 EDGELLM_INKLING_VERIFY=8 EDGELLM_INKLING_FUSED_RESIDUAL=1 EDGELLM_INT8_MMA=1 EDGELLM_INKLING_CONV_POST=1 EDGELLM_INKLING_NORM_ROUTE=1"   # the last two (short-conv epilogue, fused norm + router plugin) are the engine the numbers below were measured on (METRICS 227-229, 243)
[ "${KV_INT8:-0}" = 1 ] && BUILD_ENV="$BUILD_ENV EDGELLM_INKLING_KV_INT8=1"
# Drafter: the DSpark block drafter that ships in the checkpoint (mtp.safetensors: 3 layers, rank-256 recurrent Markov head over the target's four taps), converted by poc/tools/inkling2_drafter_int8.py
# to an 884 MB INT8 file (tensor-core INT8 GEMV, Markov correction folded into the vocabulary space). EDGELLM_INKLING_SPEC picks the draft source:
#   lookup    n-gram lookup only (no drafter file needed): +18.7% on the 36 code/prose prompts, +4.9% on the reasoning set
#   drafter   the drafter's EDGELLM_INKLING_DRAFT_K drafts (DSpark: 3 is best: 2 -> 32.3, 3 -> 33.7, 4 -> 29.6 tok/s; DFlash2 with the mma engine: 4 is best): +29.7% / +19.7% for DSpark
#   hybrid    the drafter, except a lookup match of 6+ tokens takes a window of up to 8 rows: +37.6% on the 36 prompts (35.77 vs 25.99 tok/s)
# Greedy and sampled single-sequence requests (sampled: every verify row is sampled from the target and a draft is accepted when the sample equals it; with a seed the output equals plain decoding's for short generations);
# requests with logprobs, penalties, logit bias, guided decoding, a reasoning budget or a batch above one run the plain step. Output is not byte-identical run to run for long generations (the engine itself is not deterministic).
SERVE_ENV="EDGELLM_INKLING_FUSED_RESIDUAL=1 EDGELLM_INKLING_SPEC=hybrid EDGELLM_INKLING_DRAFT_K=4 EDGELLM_INKLING_DRAFTER=\$DRAFTER_FILE"
# Drafter choice: build_engine.sh writes drafter_int8.safetensors from the community DFlash2 drafter incoai/Inkling-2-Mini-DFlash2 (DEFAULT) or, with DRAFTER=dspark, from the DSpark drafter of the checkpoint
# (5 layers, one parallel pass, top-16 selector): hybrid K=4 44.35 tok/s (K=3 42.70, K=5 43.78, K=6 42.65) on the engine whose fused q/gate/rel/sink projection also runs on the mma path, 25.99 plain (METRICS 209, 210); the lookup threshold (6) is flat between 2 and 6.
# Not set on purpose: EDGELLM_INKLING_DRAFT_K=auto (33.4 tok/s with DSpark, below fixed K=3).
