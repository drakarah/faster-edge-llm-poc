# Flint (RecursiveSI/flint, 40B-A2.5B mixture of experts coding model: 49 layers, 256 experts top-6 + an ungated shared expert, sliding window 513 with learned attention sinks, QAT INT4) on the Jetson AGX Orin.
# SETTINGS-ONLY PROFILE: the values below are the ones the measured engine used (METRICS 119/120 of the docs repo). It is NOT wired into the docker compose pipeline: the checkpoint is
# RecursiveSI/flint-int4 (MLX affine INT4, group 64, 22.6 GB) and stock Edge-LLM has no Flint support (no stock arm). Export, build and serve it by hand with poc/flint/*.sh, which read this file.
# Needs fork dev >= a85d5e7 (Flint merged into dev, 119ac0c: model class, int4AffineGroupwiseGemmPlugin for the asymmetric dense layers, int4MoeAffinePlugin for the experts, prefill attention sinks).
die "flint is a settings profile (see poc/flint/README.md): use poc/flint/export.sh, build_engine.sh and serve.sh, not the compose pipeline"
KIND=flint
NO_STOCK=1
BASE_REPO=RecursiveSI/flint-int4
# Engine, one reason each:
#  llm_build --maxBatchSize 1       one sequence (the Orin serves one agent at a time)
#  --maxInputLen 12288              one prefill pass takes up to 12,288 tokens; a longer prompt is accepted and prefilled in chunks (13,222 tokens ran, not expected from a non-hybrid engine)
#  --maxKVCacheCapacity 16384       FP16 KV, every layer keeps a full KV cache (bounded sliding-window KV is not wired for Flint): 14 full-attention layers x 2 KiB x 16,384 plus the window layers
# No build environment variables; the dense GEMM/GEMV tactics come from the plugin's own autotune.
MAXBATCH=1; MAXIN=12288; CAP=16384
EXPORT_ENV="PYTHONPATH=/opt/edge-llm LLM_SDK_DIR=/opt/edge-llm"
BUILD_ARGS="--maxBatchSize $MAXBATCH --maxInputLen $MAXIN --maxKVCacheCapacity $CAP"
# Server: the generic tool-call parser only accepts the tools array. Flint's own format (<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>) has NO parser in the fork yet,
# and the <think> reasoning parser was not checked: a real agent cannot use tool calls through this server until one exists.
SERVE_ARGS="--enable-auto-tool-choice --tool-call-parser generic"
# Measured (one Orin, METRICS 120, no prefix reuse, temperature 0, 256 tokens): prefill about 2.1k tokens/s flat to 10k tokens of prompt, decode 50.7 tok/s at 0.4k context falling to 42.8 at 13k.
# KNOWN LIMITS: (1) the C++ tokenizer fails every non-ASCII character for Flint (HTTP 500 "Pretokenizer failed: invalid codepoint": the second pre-tokenizer split has literal CJK ranges), so only ASCII
# prompts work; (2) no Flint tool-call parser; (3) peak resident memory about 26-27 GB (the engine is 22 GB): run nothing next to it and keep earlyoom's threshold in mind (SIGTERM below 10% available);
# (4) --enable-context-reuse was never tried with Flint; (5) a discrete 8 GB GPU is not supported: 21 GB of experts would need the expert-offload path, which was built for the symmetric int4_moe plugin and never tried with the affine one.
