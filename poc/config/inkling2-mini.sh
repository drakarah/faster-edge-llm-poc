# Inkling 2 Mini (thinkingmachines/Inkling-2-Mini): a looped mixture of experts (24 layers, the middle 12 run three times = 48 visits, 128 experts top-8, hidden 2,304, NVFP4 experts).
# SETTINGS-ONLY PROFILE, Orin only: the values below are the ones the measured engine used (README, "inkling2-mini"). It is NOT wired into the docker compose pipeline:
# the engine needs a converted checkpoint (NVFP4 experts, dense weights in bf16/fp16) that this repo cannot produce, and stock Edge-LLM has no Inkling support (no stock arm).
# Build and run it by hand with poc/inkling2/build_engine.sh and poc/inkling2/serve.sh, which read this file.
die "inkling2-mini is a settings profile (see poc/inkling2/README.md): use poc/inkling2/build_engine.sh, not the compose pipeline"
KIND=inkling2
NO_STOCK=1
BASE_REPO=thinkingmachines/Inkling-2-Mini
# Engine: one 1,024-token prefill pass (longer prompts are chunked), KV capacity 32,768 (FP16 KV, 8 full-attention visits; the sliding-window visits keep a bounded 1,120-slot ring).
MAXIN=1024; CAP=32768
# Build environment, one reason each:
#  EDGELLM_INT8_GEMV=attn+head        the dense attention projections and the LM head run as INT8 weight-only GEMVs (per-row RTN): -8% decode, quality in METRICS 194 of the docs repo
#  EDGELLM_INKLING_FUSED_GATE=1       q/gate/rel/sink share one input: fused gate kernels
#  EDGELLM_INKLING_FUSED_QRSG=1       one INT8 GEMV for q, gate, rel and sink (-0.5 ms per token)
#  EDGELLM_INKLING_VERIFY=8           generation profile of 1..8 tokens: chain verification of up to 7 drafted tokens. Adds per-token conv states (the rollback after a partly accepted window) and
#                                     the four drafter taps as engine outputs; plain decode costs +2.4 ms for them (36.5 vs 34.0 ms per token).
BUILD_ENV="EDGELLM_INT8_GEMV=attn+head EDGELLM_INKLING_FUSED_GATE=1 EDGELLM_INKLING_FUSED_QRSG=1 EDGELLM_INKLING_VERIFY=8 EDGELLM_INKLING_FUSED_RESIDUAL=1"
# Drafter: the DSpark block drafter that ships in the checkpoint (mtp.safetensors: 3 layers, rank-256 recurrent Markov head over the target's four taps), converted by poc/tools/inkling2_drafter_int8.py
# to an 884 MB INT8 file (tensor-core INT8 GEMV, Markov correction folded into the vocabulary space). EDGELLM_INKLING_SPEC picks the draft source:
#   lookup    n-gram lookup only (no drafter file needed): +18.7% on the 36 code/prose prompts, +4.9% on the reasoning set
#   drafter   the drafter's EDGELLM_INKLING_DRAFT_K drafts (3 is best: 2 -> 32.3, 3 -> 33.7, 4 -> 29.6 tok/s): +29.7% / +19.7%
#   hybrid    the drafter, except a lookup match of 6+ tokens takes a window of up to 8 rows: +37.6% on the 36 prompts (35.77 vs 25.99 tok/s)
# Greedy and sampled single-sequence requests (sampled: every verify row is sampled from the target and a draft is accepted when the sample equals it; with a seed the output equals plain decoding's for short generations);
# requests with logprobs, penalties, logit bias, guided decoding, a reasoning budget or a batch above one run the plain step. Output is not byte-identical run to run for long generations (the engine itself is not deterministic).
SERVE_ENV="EDGELLM_INKLING_FUSED_RESIDUAL=1 EDGELLM_INKLING_SPEC=hybrid EDGELLM_INKLING_DRAFT_K=3 EDGELLM_INKLING_DRAFTER=\$DRAFTER_FILE"
# Not set on purpose: EDGELLM_INT8_MMA=1 at build (tensor-core dense path: +3 ms plain decode, only -6% at an 8-row verify) and EDGELLM_INKLING_DRAFT_K=auto (33.4 tok/s, below fixed K=3).
