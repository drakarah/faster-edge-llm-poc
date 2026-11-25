# Inkling 2 Mini on the Jetson AGX Orin (settings profile, not run through compose)

`config/inkling2-mini.sh` holds the settings of the engine behind the README's "inkling2-mini" table; `build_engine.sh` and `serve.sh` reproduce it by hand.

Prerequisites this repo does not provide: the converted Inkling 2 checkpoint (NVFP4 experts, dense weights in 16 bit) and the original `mtp.safetensors` / target safetensors next to it
(`HF_DIR`). The engine and the drafter file are built from them; a prompt file in the `llm_inference` JSON format (`max_generate_length`, `temperature: 0`, `top_k: 1`, no logprobs) is the workload.
Speculative decoding applies to greedy single-sequence requests only; with `num_logprobs` set a request runs the plain step.
