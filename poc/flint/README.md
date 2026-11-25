# Flint on the Jetson AGX Orin (settings profile, not run through compose)

`config/flint.sh` holds the settings of the engine behind the docs repo's METRICS 119/120; `export.sh`, `build_engine.sh` and `serve.sh` reproduce it by hand inside the builder image
(fork `dev` >= a85d5e7, which contains Flint).

```
export.sh  CHECKPOINT_DIR ONNX_DIR      # RecursiveSI/flint-int4 (MLX affine INT4) -> ONNX + external expert weights, about 22 GB
build_engine.sh ONNX_DIR ENGINE_DIR     # llm_build --maxBatchSize 1 --maxInputLen 12288 --maxKVCacheCapacity 16384 (FP16 KV)
serve.sh ENGINE_DIR [PORT]              # python -m experimental.server ... --enable-auto-tool-choice --tool-call-parser generic
```

Measured on the Orin (one run, no prefix reuse, temperature 0, 256 tokens): prefill about 2.1k tokens/s flat to 10k tokens, decode 50.7 tok/s at 0.4k context falling to 42.8 at 13k.

Known limits: the C++ tokenizer fails every non-ASCII character for Flint (HTTP 500), there is no Flint tool-call parser, peak memory is 26-27 GB (nothing else can run), context reuse was never tried, and a discrete 8 GB GPU is not supported (the expert-offload path was never tried with the affine MoE plugin).
