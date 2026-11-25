# Inkling 2 Mini on the Jetson AGX Orin (settings profile, not run through compose)

`config/inkling2-mini.sh` holds the settings of the engine behind the README's "inkling2-mini" table; `build_engine.sh` and `serve.sh` reproduce it by hand.

Prerequisites this repo does not provide: the converted Inkling 2 checkpoint (NVFP4 experts, dense weights in 16 bit) and the original `mtp.safetensors` / target safetensors next to it
(`HF_DIR`). The engine and the drafter file are built from them; a prompt file in the `llm_inference` JSON format (`max_generate_length`, `temperature: 0`, `top_k: 1`, no logprobs) is the workload.
Speculative decoding applies to greedy and sampled single-sequence requests; with `num_logprobs`, penalties, logit bias, guided decoding or a reasoning budget set, a request runs the plain step.

`serve_http.sh ENGINE_DIR [PORT]` starts the OpenAI-compatible server (`experimental.server`) on the engine with the same settings; sampled requests through it decode at 33.6 tok/s against 25.8 plain (temperature 0.7, 12 prompts, five hybrid runs, METRICS 205 of the docs repo). Context reuse is on by default (`CONTEXT_REUSE=0` turns it off): the drafter resumes from the previous request's rows when the reused prompt tokens match, reused turns decode at 30.25 tok/s against 24.83 plain (METRICS 206); with a different conversation in between the request runs without the drafter (one drafter slot).
