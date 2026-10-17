# faster-llm-poc

Reproduce the speed-ups of our [TensorRT Edge-LLM fork](https://github.com/drakarah/TensorRT-Edge-LLM) on your own machine with one command.

It builds **two stacks side by side** from source, on your GPU, and times them on the same quantised checkpoint with the same prompts:

| | what it is |
|---|---|
| **stock** | [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) `main` at a pinned commit, used as upstream documents it: `tensorrt-edgellm-serve <checkpoint>` with MTP speculative decoding and context reuse, building its own engine |
| **optimised** | the fork's `dev` branch at a pinned commit (upstream plus the optimisations: fused INT4 projections, INT8 KV cache, split-KV decode attention, tree + suffix drafting, prefix reuse for hybrid models, a faster prompt-pass GEMM and attention, ...) with engine and server settings tuned per GPU |

Nothing is downloaded prebuilt except the base image and the model weights. Engines are built on your machine (a TensorRT engine only loads on the GPU and TensorRT version that built it).

## Run it

```bash
git clone git@github.com:drakarah/faster-llm-poc.git && cd faster-llm-poc
cp .env.example .env        # set PROFILE, MODEL and HF_TOKEN (Gemma is gated: accept its licence on Hugging Face first)
docker compose up
```

That is all. `docker compose up` builds the image, then runs three services in order:

1. **builder**: downloads the model, quantises it (GPTQ INT4 for Gemma, Edge-LLM's own INT4-AWQ for Qwen), exports and builds the optimised engine, and lets the stock server build its own.
2. **bench**: starts stock, times it, stops it, starts the optimised stack, times it, and repeats (`BENCH_ROUNDS`, default 2, so clock or thermal drift cancels). Prints a table and writes it to `data/results/`.
3. **serve**: leaves the optimised server running at `http://localhost:8000/v1` (OpenAI-compatible) so you can try it yourself.

Every step resumes from `./data`, so an interrupted run continues where it stopped.

### Settings

| variable | values | |
|---|---|---|
| `PROFILE` | `orin` (default), `dgpu` | `orin`: Jetson AGX Orin (sm_87). `dgpu`: discrete GPU, x86_64 or arm64; set `GPU_ARCH` (default 89 = RTX 40-series) |
| `MODEL` | `gemma4-e4b` (default), `gemma4-12b`, `qwen35-4b` | the three models we measured |
| `HF_TOKEN` | token | needed for Gemma only |
| `BENCH_CALLS`, `BENCH_ROUNDS` | 60, 2 | replayed requests per round, rounds per stack |
| `BENCH_HUMANEVAL` | 164 | HumanEval problems per run (0 skips); public benchmark, pass@1 and tokens/s |
| `DATA_DIR`, `PORT`, `BUILD_JOBS` | `./data`, 8000, 8 | |

`PROFILE` picks the build target and the tuned settings that differ between the two GPU classes (all in `poc/config/`): a Jetson shares one DRAM between CPU and GPU, so host-memory tricks are off there, and FP16-accumulate prefill (twice the tensor rate on GeForce parts) is on only for `dgpu`.

### Requirements

* Docker with Compose v2.20+ and the NVIDIA container runtime (`runtime: nvidia`; on a Jetson this is the default JetPack setup, elsewhere `nvidia-ctk runtime configure --runtime=docker`).
* **Jetson:** run `sudo nvpmodel -m 0 && sudo jetson_clocks` on the host first. An unpinned GPU clock ramps with load and distorts short benchmarks (the bench prints a warning).
* RAM + swap of **at least 40 GB** for the quantisation and 28 GB for the export (a 32 GB Jetson needs about 24 GB of swap; the builder stops with the exact commands if it is short). The 12B needs more (about 50 GB) because the bf16 model is expanded for calibration.
* Free disk: about 60 GB (gemma4-e4b) to 120 GB (gemma4-12b), including the 25 GB base image.
* Time on first run (AGX Orin): image build about 1-2 h, E4B quantisation and engines a few hours, the benchmark about 20 min. Later runs only repeat the benchmark.

## How the comparison is made fair

* Same weights: both stacks load the same quantised checkpoint, built once by the builder.
* **HumanEval** (`poc/humaneval.py`, `BENCH_HUMANEVAL`): the public 164-problem code benchmark, greedy, pass@1 plus generated tokens/s, same prompts for both stacks. The problems are fetched from the Hugging Face dataset `openai/openai_humaneval` on first use. Answers are scored by running the problem's unit tests in a child python process (empty environment, 20 s and 2 GiB limits) inside the bench container, so generated code is executed there.
* Same prompts: the three workloads in `poc/bench.py` are identical for both: a fresh ~1.5k-token prompt for **prefill** (a nonce defeats prefix caching), a free-text generation for **decode**, and a **replay** of recorded requests (`poc/data/workload.jsonl`: 396 chat requests with JSON-schema answers, recorded from an application test-suite on synthetic data; the first `BENCH_CALLS` are sent one after another).
* Same client: timings are taken from the HTTP client, the same way for both servers; greedy decoding.
* Alternating order, medians over rounds; the first request after a start is a warm-up.
* Stock gets its own best documented settings (MTP, context reuse where upstream supports it), not a crippled configuration. If the stock stack cannot run a model on your GPU (for example out of memory) the table says so.
* Answers are not forced to be identical: the optimised stack also quantises the KV cache to INT8, which can change wording. The report prints how many greedy answers match and how similar they are, and `data/results/*.json` holds every answer so you can read them.

## What is in this repo

| path | |
|---|---|
| `Dockerfile` | builds both stacks from pinned commits for `PROFILE` / `GPU_ARCH` |
| `docker-compose.yml` | builder, bench, serve |
| `poc/run.sh`, `poc/lib.sh` | the pipeline (download, quantise, export, build, benchmark, serve) |
| `poc/config/` | engine and server settings per model and profile |
| `poc/bench.py` | the timing harness and report |
| `poc/tools/` | the quantisation tools (GPTQ for Gemma, INT8 embeddings, reduced draft vocabulary) |
| `poc/data/` | recorded workload, INT8 KV-cache scales, draft vocabulary maps |

## Status

**First end-to-end run done on the Orin only.** What has been checked and what has not:

| | state |
|---|---|
| Image build for `PROFILE=orin` (both stacks compile from the pinned commits on a Jetson AGX Orin) | done |
| Fork changes the comparison rests on (measured separately on the Orin, kernel level and replay level) | e.g. the exact INT8 KV conversion in the decode attention: replay of 60 recorded requests 121.4 s to 116.75 s, decode +3.5%, bit-exact |
| INT8 tensor-core prompt pass (W4A8) for `gemma4-e4b` (prompts of 64+ tokens) | on: Orin runs it on all projections except the two with outlier inputs (per-layer-embedding gate, down projection); measured prompt speed +10-12% and HumanEval 89.0% vs 88.4% (W4A16), but on structured-output requests it leans to slightly longer answers (about +3% output tokens, paired replay). The discrete-GPU profile runs it on every projection and has **not** been checked: the kernel was only run on sm_87 |
| INT4 decode GEMM with the group scale applied to FP32 partial sums (fewer instructions per tensor op) | on in the optimised build: kernel 6-21% faster at the decode batch size, whole-replay effect flat within noise; only run on sm_87 |
| Builder (export, engine build) -> bench -> serve through `docker compose up` on the Orin, `MODEL=gemma4-e4b` | **done end to end** (one run, 2026-10-18: engines built inside the containers, 2 benchmark rounds, optimised server started) |
| Quantisation from scratch inside the builder (`gptq`, needs `HF_TOKEN`) | the run used the checkpoint prepared by the same tools beforehand; the from-scratch path has not been run through the container |
| The stock stack loading our quantised checkpoint (`gemma4_text`, MTP assistant) and building its own engine | **works** (builds in about 8 minutes, input limit set to the full context because the stock engine has no chunked prompt pass) |
| `PROFILE=dgpu` (RTX 40-series etc.) | **untested**: no such machine was available when this was written; the settings come from separate runs on an RTX 4060 |
| `MODEL=gemma4-12b`, `MODEL=qwen35-4b` | **untested** through this pipeline |

### First complete run (Jetson AGX Orin 32 GB, GPU pinned at 930 MHz, `gemma4-e4b`, 2 rounds, medians)

| | stock | optimised | ratio |
|---|---:|---:|---:|
| Prefill, tokens/s | 1,405 | 2,129 | 1.52x |
| Decode, tokens/s (probe prompt) | 21.4 | 105.2 | 4.9x |
| HumanEval generation, tokens/s | 27.1 | 164.5 | 6.1x |
| HumanEval pass@1 (greedy), per round | 138/164, 138/164 | 139/164, 138/164 | same within noise |
| Replay of 60 recorded requests, seconds | 666.5 | 109.1 | 6.1x |
| Replay output tokens | 27,882 | 10,298 | |

Read the replay seconds together with the output tokens: the stock stack answers structured (JSON) requests with 2.7 times as many tokens, because it lets the model pad the JSON with long whitespace runs (51,000 whitespace characters in the stock answers against 6,000 in the optimised ones); the optimised stack caps those runs (`EDGELLM_JSON_MAX_WHITESPACE=20`). Tokens per second (decode, HumanEval) is the fair speed comparison; the replay seconds mix speed with that length difference. The stock arm also has no `gemma4` tool-call parser, so tool-call answers differ in format; only 5 of 60 greedy replay answers are identical to stock. The decode probe of the optimised stack varied between rounds (122.2 and 105.2 tok/s): it uses a short prompt and the first request after a server start; the HumanEval and replay numbers repeat to within 1%.

Caveats of what is switched on in the optimised stack for this model: INT8 tensor-core prompt pass (W4A8, all projections except two on the Orin), group-scale-on-partial-sums INT4 GEMM, exact INT8 KV conversion in the decode attention. The two GEMM changes were only run on the Orin (sm_87).
