# Qwen3.5-4B (Qwen/Qwen3.5-4B): hybrid model (gated delta-net + attention) with its built-in MTP layer.
KIND=qwen
LONGCTX_DEFAULT=2048,4096,8192,16384,24576,30000   # the optimised engine holds 32,768 tokens (12,288 on the 8 GB card, where larger engines do not load); stock holds 10,240 and is recorded as not reached above that
BASE_REPO=Qwen/Qwen3.5-4B
ASSIST_REPO=
# MAXIN = CAP: the hybrid engine cannot chunk its prompt, so the input limit is the longest prompt it takes. Orin 32,768 (fits in unified memory; 4,096-token passes refuse every longer prompt, one app prompt alone is 8,901 tokens).
# RTX 4060 8 GB: 32,768, 24,576 and 16,384 do not load (cudaMalloc out of memory at runtime init), 12,288 does (7,669 MiB); 10,240 refused 16k.
# STOCK_CAP: stock keeps its 10,240-token engine (it was built there for the tables; larger stock engines were not tried).
MAXIN=32768; CAP=32768; STOCK_CAP=10240; POOL=320; TREE=12
if [ "$PROFILE" = dgpu ]; then MAXIN=12288; CAP=12288; LONGCTX_DEFAULT=2048,4096,8192,12000; fi
EXPORT_ENV=""
EXPORT_ARGS="--mtp --draft-reduced-vocab-dir $POC/data/vocab/qwen35"
INT8_EMBEDDINGS=0
if [ "$PROFILE" = orin ]; then SNAP=1610612736; else SNAP=536870912; fi        # recurrent-state snapshot pool for prefix reuse (bytes)
# SNAP: Orin 1.5 GB, dgpu 0.5 GB. The sizes are not tied to a recorded sweep  (a gap, not a tuned value): the dgpu figure is the smaller of the two, and the snapshot pool only bounds how many
# recurrent-state checkpoints prefix reuse can keep; INT4_GEMV_MAX_M=1 below is Orin-only because the GEMV/GEMM switch was only measured there (the dgpu keeps upstream's switch at 4 tokens).
# Tree: 1 candidate x 3 steps, 4 verify nodes = a chain, with the 32k reduced-vocabulary drafter head (--draft-reduced-vocab-dir, worth 13%). The tree export cannot use the reduced head and a tree buys at most 1.8% over a chain
# with the same head (replay-60: reduced chain 102.5 s, full-vocabulary tree 2x5/12 116.0 s, full chain 118.1 s, single runs), so tree drafting stays off on Qwen.
# The two pool-bytes flags size the recurrent-state snapshots (1.5 GB Orin / 0.5 GB dgpu) and partial-KV snapshots (128 MB) that make prefix reuse work for a hybrid model.
SERVE_ARGS="--draft-top-k 1 --draft-step 3 --verify-tree-size 4 --enable-context-reuse --context-cache-recurrent-snapshot-pool-bytes $SNAP --context-cache-partial-kv-snapshot-pool-bytes 134217728 --tool-call-parser generic"
# HYBRID_DIVERGENCE_CHECKPOINT=256: checkpoint the recurrent state every 256 tokens so a hybrid model can reuse cross-request prefixes (replay-396 -34%, upstream refuses it).
# GDN_CHUNK_PREFILL=1: chunked prompt pass for the gated delta-net layers (replay-60 104.2 -> 102.5 s, -1.6%, it is also the fork's default). INT4_GEMV_MAX_M=1 (Orin): GEMM from 2 tokens (+11% decode).
SERVE_ENV="EDGELLM_FORCE_GREEDY=1 EDGELLM_JSON_MAX_WHITESPACE=20 EDGELLM_HYBRID_DIVERGENCE_CHECKPOINT=256 EDGELLM_GDN_CHUNK_PREFILL=1"
[ "$PROFILE" = orin ] && SERVE_ENV="$SERVE_ENV EDGELLM_INT4_GEMV_MAX_M=1"
# Stock gets exactly the optimised arm's settings where it has the same knob: drafting tree (top-k 1, step 3, verify 4), the engine's build maximum for the
# tree ($TREE), context length, the prefix-reuse flag. Fork-only features (recurrent-state snapshot pool for hybrid reuse, suffix draft, GDN chunked prefill,
# fused projections, plugins) are what is being measured, so stock does not get them.
# If stock rejects --enable-context-reuse for this hybrid model, it runs without: that is an upstream feature gap (the fork's hybrid prefix reuse exists for
# that reason), not a setting withheld from it; the README must say so next to the result.
STOCK_TREE="--draft-top-k 1 --draft-step 3 --verify-tree-size 4"
# Stock refuses --enable-context-reuse on this hybrid model at startup ("Hybrid context reuse requires at least one recurrent snapshot slot", contextCacheCoordinator.cpp:916):
# the upstream gap that the fork's hybrid prefix reuse exists for. So stock runs without prefix reuse; everything else is the optimised arm's setting.
STOCK_ARGS="--max-input-len $STOCK_CAP --max-kv-cache-capacity $STOCK_CAP --enable-auto-tool-choice $STOCK_TREE --max-verify-tree-size $TREE --max-draft-tree-size $TREE"
STOCK_SPEC='{"method":"mtp"}'
