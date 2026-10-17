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

**Work in progress: not yet validated end to end.** What has been checked and what has not:

| | state |
|---|---|
| Image build for `PROFILE=orin` (both stacks compile from the pinned commits on a Jetson AGX Orin) | done |
| Fork changes the comparison rests on (measured separately on the Orin, kernel level and replay level) | e.g. the exact INT8 KV conversion in the decode attention: replay of 60 recorded requests 121.4 s to 116.75 s, decode +3.5%, bit-exact |
| Builder (export, engine build) -> bench -> serve through `docker compose up` on the Orin, `MODEL=gemma4-e4b` | first run started and then stopped by hand during the export; **not completed yet** |
| Quantisation from scratch inside the builder (`gptq`, needs `HF_TOKEN`) | the steps are the tools that produced our checkpoint, but this pipeline has not been run through the container yet |
| The stock stack loading our quantised checkpoint (`gemma4_text`, MTP assistant) and building its own engine | **untested** |
| `PROFILE=dgpu` (RTX 40-series etc.) | **untested**: no such machine was available when this was written; the settings come from separate runs on an RTX 4060 |
| `MODEL=gemma4-12b`, `MODEL=qwen35-4b` | **untested** through this pipeline |

Measured numbers from this repository's own benchmark will be added here once the first complete run has finished; until then treat everything above the table as the intended behaviour, not a result.

