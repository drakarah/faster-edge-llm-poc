# Proof of concept for faster Nvidia Edge LLM engines

**Run language models fast on a small NVIDIA GPU or a Jetson, behind an OpenAI-compatible API you can plug into any chat UI.**

This is a proof of concept for a set of optimisations in a fork of NVIDIA's TensorRT Edge-LLM, measured on several model families and two kinds of hardware: Qwen3.6-35B-A3B and a derivative of it (CyberTiel), Gemma 4 (26B-A4B, 12B and E4B) and Qwen3.5-4B, on an RTX 4060 laptop GPU (8 GB) and a Jetson AGX Orin (32 GB). The smaller models are compared side by side with stock Edge-LLM. The large mixture-of-experts models have no stock arm (stock cannot load them in 8 GB; they fit at all because of the streaming described below), so they are compared with llama.cpp where a baseline was run. Per-model numbers, caveats and everything that did not work are in the [Models and GPUs](#models-and-gpus) table and the results further down.

## The same 60-request workload on every model

Replay of 60 recorded requests, seconds (lower is better), median of two rounds. "Stock" is NVIDIA's unmodified Edge-LLM with the same settings; where it cannot load the model, the baseline is llama.cpp or there is none.

| model | Jetson AGX Orin 32 GB: stock, then this project | RTX 4060 8 GB: stock, then this project |
|---|---|---|
| Gemma 4 E4B | 511.0 s, then **103.7 s** (4.93x) | stock did not run (out of memory), then **56.6 s** |
| Gemma 4 12B | 557.2 s, then **256.2 s** (2.18x) | stock did not run (out of memory), then **141.5 s** |
| Qwen3.5-4B | 325.6 s, then **160.8 s** (2.02x) | stock did not run (out of memory), then **86.1 s** |
| Gemma 4 26B-A4B (mixture of experts) | no stock arm, **246.0 s** | no stock arm, **323.8 s** |
| Qwen3.6-35B-A3B (mixture of experts) | not run | llama.cpp 1,367 s, then **212-215 s** (6.4x), see below |
| CyberTiel (a Qwen3.6-based 35B-A3B) | not run | no 60-request replay; see [long context and agents](#long-context-and-agents-qwen36-and-cybertiel-on-the-rtx-4060) |

The 26B's two engines differ per machine (the 4060 streams the experts from host memory and has no draft model), so its two figures are not a hardware comparison. Decode speed, prompt processing, long-context behaviour, quality checks and what did not work are per model in [Models and GPUs](#models-and-gpus) and in the results further down; a tokens-per-second figure means nothing without the prompt length it was measured at, so those tables name it.

The workload is a recording of a real app that turns user requests into SQL and chart specifications (60 independent requests, prompts of 300 to 7,500 tokens, median about 2,700, JSON output). The requests share no growing conversation, so it says nothing about agentic coding, where each turn extends the previous prompt; that needs prefix reuse and long context and is measured separately. A tokens-per-second figure means nothing without the prompt length it was measured at, so both columns name it; decode speed at longer contexts is in the long-context table further down. How the numbers were measured, what is not apples to apples, and everything that did not work are further down.

### The hard case: a 35B model on an 8 GB laptop GPU

A 35-billion-parameter mixture-of-experts model does not fit an 8 GB card, so most of its weights live in system RAM and the CPU normally does much of the work. Here the same workload on Qwen3.6-35B-A3B and an RTX 4060 laptop (8 GB, 31 GB RAM):

| Qwen3.6-35B-A3B on an RTX 4060 laptop (8 GB, 31 GB RAM) | time for the 60 requests | decode, 300 tokens after a ~10-token prompt | prompt processing, one 3,000-token prompt |
|---|---:|---:|---:|
| llama.cpp (Q4_K_M, experts on the CPU, `--n-cpu-moe` tuned) | 1,367 s | 17-23 tok/s | 350-460 tok/s |
| **this project, exact routing** (lossless apart from the INT4 weights) | **235 s** (5.8x) | 64 tok/s | ~1,700 tok/s |
| **this project, default** (cache-aware routing, +0.09% perplexity) | **212-215 s** (6.4x) | 68-74 tok/s | ~1,700 tok/s |

The numbers in this table were measured with the Linux kernel's default DMA mapping. On a boot with **`iommu=pt`** (see Requirements) the same default setting needs **187 s** for the 60 requests (one run), the cold prefill of a 1.4k-token prompt goes from 726 to 1,232 tok/s and the warm time to first token from 1.9 to 1.1 s; decode after the cache is warm does not change (97.8 tok/s). Exact routing and llama.cpp were not run again on that boot, so the ratios in the table belong to the default boot.

![Replay time, Qwen3.6-35B-A3B on an RTX 4060 laptop](docs/replay-qwen36.svg)

Statistics of a Cybertiel reverse engineering session in Pi with compaction:

<img width="951" height="1226" alt="image" src="https://github.com/user-attachments/assets/3fbb33d0-815d-4aa2-a1d1-293527f044f3" />


## Quick start

You need Docker with the NVIDIA container runtime and an NVIDIA GPU with 8 GB or more (the default model, Qwen3.6-35B-A3B, needs about 31 GB of system RAM and 60 GB of free disk; the small Gemma and Qwen3.5 models need less, see [Models and GPUs](#models-and-gpus)).

```bash
git clone https://github.com/drakarah/faster-llm-poc.git && cd faster-llm-poc
docker compose up
```

The first start downloads the model and builds the engine for your GPU (it says what it is doing; a TensorRT engine only loads on the GPU that built it). Later starts take about a minute. When it prints that the server is ready, the API is at `http://localhost:8000/v1`. A live dashboard of speeds (time to first token, prefill, decode, reuse) and memory (GPU, KV cache, expert cache, host RAM) is at **`http://localhost:8000/stats`** for the Qwen3.6 and CyberTiel profiles; its JSON is at `/api/stats` (`REQUEST_STATS=0` in `.env` turns it off).

## Use it from your chat UI

Any client that speaks the OpenAI API works: Open WebUI, LM Studio's remote mode, SillyTavern, Continue, Cline, `curl`. Set:

| setting | value |
|---|---|
| base URL | `http://localhost:8000/v1` |
| API key | anything (not checked) |
| model name | `local` |

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}]}'
```

Tool calls, JSON-schema output and streaming are supported.

## Measure it yourself

With the server running:

```bash
docker compose run --rm bench
```

It replays the recorded workload, runs the public HumanEval set and a few long prompts through the running server, and prints a table. It does not load a second copy of the model, so it works on an 8 GB card.

To compare against llama.cpp on your machine: stop the server (`docker compose stop serve`), start llama.cpp's server with the settings in [the technical details](#how-the-llamacpp-baseline-was-run), and point the same benchmark at it:

```bash
BENCH_URL=http://host.docker.internal:8080 BENCH_LABEL=stock docker compose run --rm --no-deps bench
```

## Models and GPUs

| model | tested on | what you get |
|---|---|---|
| **Qwen3.6-35B-A3B** (default; hybrid mixture of experts) | RTX 4060 laptop 8 GB | the numbers above |
| **CyberTiel** (Huihui-Ornith-1.5-35B-A3B-abliterated, a Qwen3.6-architecture 35B-A3B with its refusals removed; INT4 export, `MODEL=cybertiel-35b`) | RTX 4060 8 GB (mini PC), 31 GB RAM | a 262,144-token window with a growable KV pool, context reuse and the terse Sharp chat template, see [long context and agents](#long-context-and-agents-qwen36-and-cybertiel-on-the-rtx-4060). Cold prefill about 1,300 tok/s on 1,600-token prompts; decode 68-78 tok/s at 5k-65k tokens of context (model card, one run per cell). **No refusals: run agents on it in a sandbox** |
| Gemma 4 26B-A4B (mixture of experts) | RTX 4060 laptop 8 GB, Jetson AGX Orin 32 GB | measured, see the results below |
| Gemma 4 12B and E4B, Qwen3.5-4B | RTX 4060 laptop 8 GB, Jetson AGX Orin 32 GB | measured, side by side with stock NVIDIA Edge-LLM |

Pick another model with `MODEL=` in `.env` (see `.env.example`). Gemma needs a Hugging Face token of an account that accepted its licence. All measurements were taken on one RTX 4060 box (i9-12900H, 31 GB RAM, PCIe 4.0 x8) and one Jetson AGX Orin; other cards are untested, and the 4060 numbers depend on the card's 8 GB (they are labelled 4060-specific where it matters).

## Long context and agents (Qwen3.6 and CyberTiel on the RTX 4060)

The 60-request replay above has no growing conversation. An agent resends a longer prompt every turn, so what matters is context reuse and the cost of a long window. These numbers were measured on the 4060 box with engines built the way the profiles build them (`config/qwen36-35b.dgpu.sh`, `config/cybertiel-35b.dgpu.sh`), **not** re-run from a clean data directory through `docker compose`. Sources: METRICS R25eu-R25ex and the live-session log, in the notes repository.

| measurement (RTX 4060 only) | result |
|---|---|
| decode, short prompts, static 128k KV pool / static 256k pool / growable 256k pool | 81.4 tok/s with 42 expert slots / 59.1 tok/s with 16 / **84.1 tok/s with 56** |
| decode at 128k tokens of context, warm: static 128k pool against growable 256k | 81.4 against 83.5 tok/s (+2.5%, not significant) |
| 14-turn agent conversation up to 68k tokens, 256k engine: no reuse / reuse with a static pool / reuse with the growable pool | 23.29 / 3.21 / 3.60 s per turn (turns 8-14); first-token log-probabilities agree on 14 of 14 turns (largest difference 0.023) |
| live pi coding-agent session, CyberTiel, reuse on: 116 requests, up to 88,597 prompt tokens | 97.6% of the prompt tokens served from reuse (3.95 M of 4.04 M), median 300 new tokens prefilled per request, no errors |
| against PolyStrata (Qwen3.6 on both; PolyStrata ran the Unsloth UD-Q4_K_M GGUF, ours is our own INT4 export; 12k engines, thinking off, R25bt) | cold prefill 1,112 against 555 tok/s at 2k tokens and 2,416 against 1,049 at 6k; decode 72-74 against 35-55 tok/s (PolyStrata default and exact arms). The replay totals are not comparable: PolyStrata refused 11-12 of the 60 requests (structured output with tools) |

What is not shown. A needle test at 222k tokens passed on the static pool and failed on both growable runs; the log-probabilities of the first token are identical in top-1 across the two pools (12 of 12), so it is unresolved whether the failure is the pool or the model's own uncertainty at that length. Reuse was exercised to 88k tokens only in one live session, and interleaved long and short requests (which evict cached prefixes above the mapped pages) were not tested. A client that does not send `max_tokens` makes the first request map the whole pool; send it. Greedy output is not reproducible on this stack, so text identity cannot be tested.

## Why it is fast, in six lines

1. **Experts stay in RAM, the GPU keeps a cache of the ones in use.** Each layer has a small cache of experts on the GPU; a step copies only what is missing.
2. **Prompts overlap compute and copy.** While layer N computes, the experts of layer N+1 are already on their way, and the ones that are already in the decode cache are copied inside the GPU instead of over PCIe.
3. **The router prefers experts that are already cached** (optional, `CACHE_BIAS`): about 65% fewer cache misses for about +0.1 to +0.3% perplexity (two measurements: +0.09% and +0.27%; the test noise is about 0.15%). Perplexity understates the change: against the exact model the top-1 token differs at about 5% of positions (KL 0.022, teacher-forced on 32 x 1,024 tokens; many are near-ties). Plain exact routing remains one setting away (`CACHE_BIAS=0`). A second option, `TAIL_SKIP`, does not fetch low-ranked experts that are missing from the cache: with exact routing otherwise it gives +18% decode for +0.04% perplexity.
4. **Everything is INT4 with calibrated quantisation** (GPTQ), an INT8 KV cache and fused kernels, so more of the cache fits in 8 GB.
5. **The KV cache is mapped on demand** (optional, `KV_INITIAL_TOKENS`): a static 256k-token KV pool leaves the expert cache 16 slots per layer and costs 27% of the decode speed; the growable pool starts small, grows with each request and keeps 56 slots (details below).
6. **It is a fork of NVIDIA's TensorRT Edge-LLM**, so all of it runs as ordinary TensorRT plugins; no custom runtime.

The fork lives at [drakarah/TensorRT-Edge-LLM](https://github.com/drakarah/TensorRT-Edge-LLM); every change below links to the branch that holds it.

---

## Technical details

Everything below is reference material: settings, the full list of changes with measurements, results per machine, and what did not work.

<details>
<summary><b>Settings, requirements and the side-by-side comparison with stock Edge-LLM</b></summary>

### The side-by-side comparison with stock NVIDIA Edge-LLM

```bash
git clone https://github.com/drakarah/faster-llm-poc.git && cd faster-llm-poc
cp .env.example .env        # set MODEL, PROFILE and HF_TOKEN (Gemma is gated)
docker compose --profile compare run --rm compare
```

The `compare` service builds the image with both stacks, then runs three steps:

1. **builder**: downloads the model, quantises it (GPTQ INT4 for Gemma, Edge-LLM's own INT4-AWQ for Qwen3.5), exports and builds the optimised engine, and lets the stock server build its own.
2. **bench**: starts stock, times it, stops it, starts the optimised stack, times it, and repeats (`BENCH_ROUNDS`, default 2, so clock or thermal drift cancels). Prints a table and writes it to `data/results/`.
3. **serve**: leaves the optimised server running.

Every step resumes from `./data`, so an interrupted run continues where it stopped. A new benchmark run moves earlier results into `data/results/previous-<time>/` instead of deleting them.

### Settings

| variable | values | |
|---|---|---|
| `PROFILE` | `dgpu` (default), `orin` | `dgpu`: discrete GPU, x86_64 or arm64; set `GPU_ARCH` (default 89 = RTX 40-series). `orin`: Jetson AGX Orin (sm_87) |
| `MODEL` | `qwen36-35b` (default, dgpu only), `cybertiel-35b` (dgpu only: [drakarah/Huihui-Ornith-1.5-35B-A3B-abliterated-EdgeLLM-INT4](https://huggingface.co/drakarah/Huihui-Ornith-1.5-35B-A3B-abliterated-EdgeLLM-INT4), refusals removed, run it sandboxed; 262,144-token window, growable KV pool, context reuse and the terse Sharp chat template, the settings of the live 4060 server in METRICS R25ev-R25ex; engine build is not re-run end to end from this repo yet), `gemma4-e4b`, `gemma4-12b`, `qwen35-4b`, `gemma4-26b` (optimised arm only: stock Edge-LLM cannot load an engine built by the fork nor export the 51 GB checkpoint in 32 GB of RAM; `PROFILE=dgpu` uses `config/gemma4-26b.dgpu.sh` and the dgpu pin, and has not been run end to end from a clean data dir yet) | the models we measured |
| `HF_TOKEN` | token | needed for Gemma only |
| `REQUEST_STATS` | 1 (Qwen3.6 and CyberTiel dgpu) | the dashboard at `/stats` and its JSON at `/api/stats` (`EDGELLM_REQUEST_STATS`); 0 turns the per-request part off; `POST /api/stats/reset` clears it |
| `KV_INITIAL_TOKENS` | unset (Qwen3.6 dgpu), 12288 (CyberTiel) | growable KV pool: tokens mapped at the start (`EDGELLM_KV_INITIAL_TOKENS`); clients must send `max_tokens` |
| `BENCH_CALLS`, `BENCH_ROUNDS` | 60, 2 | replayed requests per round, rounds per stack |
| `BENCH_HUMANEVAL` | 164 | HumanEval problems per run (0 skips); public benchmark, pass@1 and tokens/s |
| `BENCH_LONGCTX` | per model (gemma4-12b: `4096,8192,11500,16384,24576,30000`; e4b and qwen: `2048,4096,8192`); empty skips | long-context test: prompt lengths in tokens, ascending. Per length: cold time to first token and prompt tokens/s (a unique nonce defeats prefix reuse), the same prompt again (warm), exact retrieval of 3 planted facts (10/50/90% depth), and decode tokens/s with that context in the KV cache. A stack that refuses or fails a length is recorded as such and not tried on longer ones (stock builds its engine for the replay's context only: no chunked prefill, no bounded sliding-window KV, so on the 12B it stops at 12,288 tokens). Both engines are built for a fixed capacity (`CAP` in `config/`): lengths above it are refused by design |
| `DATA_DIR`, `PORT`, `BUILD_JOBS` | `./data`, 8000, 8 | |

`PROFILE` picks the build target and the tuned settings that differ between the two GPU classes (all in `poc/config/`): a Jetson shares one DRAM between CPU and GPU, so host-memory tricks are off there, and FP16-accumulate prefill (twice the tensor rate on GeForce parts) is on only for `dgpu`.

### Requirements

* Docker with Compose v2.20+ and the NVIDIA container runtime (`runtime: nvidia`; on a Jetson this is the default JetPack setup, elsewhere `nvidia-ctk runtime configure --runtime=docker`).
* **Discrete GPU on Linux (the `dgpu` profile): add `iommu=pt` to the kernel command line.** The experts stream from pinned host memory over PCIe all the time, and with the IOMMU translating the GPU's DMA (the default of many distro kernels) the link only reaches about 8.6 GB/s. Check: `cat $(readlink -f /sys/bus/pci/devices/<GPU address>/iommu_group)/type` prints `DMA-FQ` or `DMA` when the GPU's DMA is translated; `identity` means it already is not. To change it: `GRUB_CMDLINE_LINUX="iommu=pt"` in `/etc/default/grub`, `sudo update-grub`, reboot. Measured on the RTX 4060 laptop (PCIe 4.0 x8): pinned host-to-device copy 8.62 -> 12.91 GB/s, device-to-host 8.09 -> 13.18, replay of the 60 requests 211.6 / 214.5 -> 186.9 s. It does not affect the Docker NVIDIA runtime; it does change how the kernel maps the DMA of the devices it owns, so leave it out if you rely on IOMMU isolation for those devices. Not needed on a Jetson (one shared DRAM).
* **Jetson:** run `sudo nvpmodel -m 0 && sudo jetson_clocks` on the host first. An unpinned GPU clock ramps with load and distorts short benchmarks (the bench prints a warning).
* RAM + swap of **at least 40 GB** for the quantisation and 28 GB for the export (a 32 GB Jetson needs about 24 GB of swap; the builder stops with the exact commands if it is short). The 12B needs more (about 50 GB) because the bf16 model is expanded for calibration.
* Free disk: about 60 GB (gemma4-e4b) to 120 GB (gemma4-12b), including the 25 GB base image; the 26B needs about 150 GB (52 GB download, 49 GB re-sharded copy while the download is deleted, 18 GB engine with sidecars; the dgpu profile: 13 GB of expert and embedding files twice during the build, engine about 13 GB).
* Time on first run (AGX Orin): image build about 1-2 h, E4B quantisation and engines a few hours, the benchmark about 20 min. Later runs only repeat the benchmark.
### Quantising yourself (`QUANTIZE=1`)

By default `docker compose up` downloads the published export of the model and only builds the engine. `QUANTIZE=1` (in `.env` or the environment) ignores that and runs the whole pipeline on your machine from the original weights: the layer-wise GPTQ of the experts and of the dense layers (`poc/tools/`), the export and the engine build. It needs the full download (67 GB for Qwen3.6-35B-A3B), 40 GB of RAM + swap and a few hours. Models without a published export (the Gemma models, whose licence does not allow redistributing derivatives) always take this path. The published Qwen3.6 export's INT4 weights were calibrated on a Dutch-weighted text mix (nl 40%, en 25%, fr 10%, code 25%), made for the Dutch application it comes from; a code and security mix gave the same HumanEval-think pass rate (59/66 against 58/66, one run per task, not significant). `QUANTIZE=1` with your own calibration text is the way to tune the weights for another workload.

### If the download fails (Hugging Face rate limits)

The prebuilt export is about 18 GB. The free Hugging Face tier rate-limits (HTTP 429), so the downloader uses two parallel connections, retries with backoff (honouring `Retry-After`) for up to 12 attempts, and resumes partial files, so running `docker compose up` again continues where it stopped. In order, the fallbacks are:

1. **A free Hugging Face token** (`HF_TOKEN=` in `.env`) raises the limits.
2. **A mirror**: `ONNX_MIRROR_URL=https://your.host/path` for any plain HTTPS host with range requests (object store, web server, release page) holding a copy of the export plus a `manifest.txt` made with `poc/tools/make_mirror_manifest.py`. Files are resumed with curl and verified by SHA-256.
3. **Your own download**: fetch the files by any means (browser, `git lfs`, torrent, a USB stick) into a directory and set `ONNX_DIR=` to it (it must contain `llm/`). Nothing is downloaded then.
4. **`QUANTIZE=1`**: quantise on your machine from the original weights. That download also comes from Hugging Face (67 GB), so it only helps if the limit was specific to the export repository.

### How the llama.cpp baseline was run

`ghcr.io/ggml-org/llama.cpp:server-cuda`, Unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22 GB), `-ngl 99 --n-cpu-moe N --flash-attn on` with a q8_0 KV cache, context 12,288, `--threads 6 --no-mmap`, `--jinja --chat-template-kwargs '{"enable_thinking":false}'` (the thinking stream otherwise empties the benchmark's content stream). `N` was swept (30 and 40 gave the best results, within noise of each other). The recorded replay of 60 requests took 1,367 s. Sustained decode was 17-23 tok/s: the first decode after loading is faster (up to 35 tok/s, probably thermal headroom, not verified) and the same prompt twice can differ by up to 40%. Prompt processing was 350-460 tok/s in probes; llama.cpp's repeated-prompt numbers of 1,600-2,100 tok/s are its prompt cache, not prompt processing.

What is not apples to apples: llama.cpp ran Unsloth's mixed Q4_K_M quantisation, this project runs its own GPTQ INT4 (experts and all dense layers). Quality checks for ours: HumanEval 155/164 (150-156 for every variant tried), +0.09% perplexity for the default cache-aware routing (+1.5% at the most aggressive setting, which is faster still: 245 s), and two retrieval tests of planted facts: a thinking-off needle test (40 prompts per length) found every fact at 2k, 4k and 6k tokens, while the stricter test that `docker compose run --rm bench` runs (3 facts per prompt, a fresh random filler each run) found 23 of 27 with exact routing and 22 of 27 with the default routing, so the difference between them is noise. On the same test llama.cpp's Q4_K_M found all 27 of 27, so our INT4 weights do lose some exact retrieval at these lengths; we are isolating which part (the dense layers or the experts) causes it. Exact routing (`CACHE_BIAS=0`) is lossless apart from the weights.

</details>

<details>
<summary><b>What the fork changes, with every measurement (and the branch that holds each change)</b></summary>

## What the optimised build changes compared with stock NVIDIA TensorRT-Edge-LLM

"Stock" is NVIDIA/TensorRT-Edge-LLM `main`; "optimised" is the `dev` branch of the fork ([`dev`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/dev)), which is upstream plus the changes below. Every change is listed with what was measured. Unless a row says otherwise, numbers are from a Jetson AGX Orin 32 GB (14 SMs, MAXN, GPU pinned at 930 MHz), Gemma 4 E4B, batch 1. "Replay-60 / replay-396" is a replay of the first 60 / all 396 recorded requests of a structured-output (JSON, SQL, tool call) workload, in order, from a fresh server; "71-test suite" is the application test suite used as a quality gate (its run-to-run spread is about +-2-3 tests, so one test more or less is noise). Greedy answers change slightly with any numerics change, so speed is quoted together with how many answers changed where that matters. Rows marked RTX 4060 were measured on an 8 GB RTX 4060 (x86, sm_89) and rows marked Qwen3.5-4B on a hybrid linear-attention model; neither helps the Gemma 4 E4B Orin stack unless the last column says so.

### Whole journey on the engine level (Jetson AGX Orin 32 GB, Gemma 4 E4B)

The end-to-end stock-versus-optimised numbers of this benchmark are in the results at the end of this file. The stepwise ladder behind them was measured on the engine level, one change at a time, with a Dutch-prose decode probe (400 tokens, greedy) and the 396-request replay. Rows are cumulative; the starting point is Edge-LLM v0.11.0 with the same INT4 weights, no speculative decoding. The last row is the engine this benchmark builds.

Two decode columns, never to be mixed up. **Drafter only** is the probe with the suffix draft switched off: it exists for every stage, because the suffix draft was only added on 2026-10-14 (after the first rows were measured), and it isolates the engine and the MTP drafter. **As served** is what the engine does with the suffix draft on, measured *cold*: different prompts, one request each, a fresh server (never the same prompt repeated, which lets the suffix cache copy the earlier answer and reads 150-200 tok/s). The as-served column can only be filled for the engines still on disk.

| stage | decode, drafter only (tok/s) | decode, as served, cold (tok/s) | prefill (tok/s, 5.2k prompt) | GPU memory (nvmap) | replay-396 |
|---|---:|---:|---:|---:|---:|
| llama.cpp Q4_K_M reference | 21.0 | n/a | 842 | 6.0 GB | |
| Edge-LLM v0.11.0, INT4 weights + INT4 head, fp16 embedding tables | 43.8 | n/a | 1,834 | 10.4 GB | |
| + MTP tree drafting with the GEMV/GEMM threshold fix | 56.5 | n/a | | 11.1 GB | |
| + INT8 embedding tables | 56.3 | n/a | | 7.78 GB | |
| + fused q/k/v and gate/up (with a reduced output vocabulary that was later rejected) | 63.4 | n/a | | | |
| engine of the final design at that time (full vocabulary, GPTQ INT4 head, INT8 KV cache, split-KV attention, chunked prefill 2048, 320-page pool) | 57.2 | n/a | 1,833 | 7.44 GB | 846 s, 628 s after tree and pool tuning |
| + centroid head, sparse draft logits | 61.2 | n/a | 1,830-1,840 | 7.43 GB | 619 s |
| + suffix draft | not re-measured | not measured on that image | 1,830-1,840 | 7.43 GB | 545 s (same session as the 619 s: -11.9%) |
| served-equivalent engine, re-measured in the session of the next row (adds the exact INT8 attention conversion) | 62.9 | Dutch 65.5, 61.5, 60.5; English 83.1 | 1,814 | 7.43 GB | 508.5 s |
| **this benchmark's engine: + W4A8 prompt GEMM with per-op filter, + group-scale decode GEMM** | **64-66** | **Dutch 69.7, 66.7, 64.2; English 91.7** | **1,980-2,008** | **7.44 GB** | **482.9 s** |

Reading the last two rows (same session, same server settings, suffix draft on for the replay): the benchmark's engine decodes 6-10% faster cold, prefills 9-11% faster, and replays the 396 calls 5.0% faster (W4A8 prompt GEMM 4.3%, group-scale GEMM about 0.7%). Cold decode on Dutch prose with the suffix draft on is about the same as drafter-only (it has nothing to copy yet); the suffix draft earns its gain on structured and repetitive output (replay -11.9%, code, JSON), not on the first essay. English prose decodes faster than Dutch (the drafter accepts more tokens). The sessions of the 545 s and 508.5 s rows differ by a few percent, so compare within a session. The 63.4 row used a reduced output vocabulary that failed the quality gate. Memory is unchanged to within 10 MB: W4A8 needs a small scratch buffer only.

### Changes

Legend for the last-but-one column: **yes** = switched on in the benchmark's optimised `gemma4-e4b` stack; **no (opt-in)** = exists, off; **no (other GPU)** / **no (Qwen)** = only measured to help another GPU or model.

| change | what it does | effect on speed | effect on memory | in the optimised stack | where |
|---|---|---|---|---|---|
| **Weights, embeddings** | | | | | |
| Fused INT4 q+k+v and gate+up projections (`EDGELLM_FUSE_GEMMA4_INT4=1` at export) | concatenates the GPTQ tensors so each layer runs one wider INT4 GEMM instead of three / two | decode probe (MTP tree 3x2/8) 59.3 -> 63.4 tok/s (+7%), English 67.3 -> 71.9; verify pass 24.6 -> 22.7 ms; answers 100% identical to unfused | not measured separately | yes | [`feat/gemma4-int4-fusion`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/gemma4-int4-fusion) |
| INT8 embedding tables (token embedding and per-layer table, per-row/slice scales; loader in the runtime, converter `poc/tools/quantize_ple.py` in this repo) | looks up INT8 rows and dequantises on the fly instead of reading fp16 | none by design (decode 43.8 -> 43.4, prefill 1,834 -> 1,821; MTP English acceptance 1.93 -> 1.83) | GPU 10.4 -> 7.74 GB (plain engine), 11.1 -> 7.78 GB (with MTP); tables on disk 6.98 -> 3.53 GB; relative error 4e-5 | yes | [`feat/int8-embedding-tables`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/int8-embedding-tables), [`feat/int8-ple-tables`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/int8-ple-tables) |
| INT4 output head calibrated with GPTQ (builder step, `poc/tools/gptq_head.py`; not a fork change) | quantises the 262k-row output head to INT4 | Edge-LLM engine, bf16 head -> GPTQ INT4 head: decode 48.3 -> 57.3 tok/s (+19%); (vLLM, same change: 36.2 -> 44.3, +22%) | GPU 9.03 -> 8.07 GB; 71-test suite 49 -> 51 passed (vLLM experiment: a plain RTN INT4 head cost answers, 13 -> 5 of 53 re-run tests) | yes (builder) | this repo, `poc/tools` |
| Externalised INT4 FFN weights (`--externalize-weights int4_ffn`; upstream exporter option, not a fork change) | keeps FFN weights outside the TensorRT engine file | not measured separately | split: external FFN 1.70 GB, base engine 0.75 GB | yes | upstream |
| **GEMM kernels** | | | | | |
| INT4 GEMV/GEMM threshold (`EDGELLM_INT4_GEMV_MAX_M=1`, Orin profile) | upstream uses the GEMV for up to 4 tokens, the tensor-core GEMM above; on Orin the GEMM is faster from 2 tokens | MTP, 1 draft token, Dutch: 45.5 -> 50.5 tok/s (+11%), English 48.7 -> 53.8; verify pass 28.0 -> 24.9 ms; forcing the GEMM at 1 token is 45% slower (English probe 23.8 vs 43.5). No effect on the RTX 4060 | none | yes (Orin only) | [`perf/int4-gemv-threshold-sm87`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/int4-gemv-threshold-sm87) |
| INT4 decode GEMM with the group scale applied to FP32 partial sums (build switch `EDGELLM_INT4_GEMM_GROUP_FLUSH=1`, whole-group split rule) | fewer instructions per tensor op in the decode / verify GEMM | kernel at 16 tokens: gate+up 180 -> 169 us (-6%), down 99.6 -> 90.7 us (-9%), qkv -16%, o -11%; over 300 captured launches 18.95 -> 17.98 ms (-5%); **replay-60 116.55 -> 116.05 s (-0.4%, inside noise)**; the smallest launches got 5-9% slower. Tiny numerics change (summation order) | none | yes (build switch) | [`exp/int4-gemm-group-flush`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/int4-gemm-group-flush) ([113dbe5](https://github.com/drakarah/TensorRT-Edge-LLM/commit/113dbe5), [ade06d3](https://github.com/drakarah/TensorRT-Edge-LLM/commit/ade06d3)) |
| W4A8 prompt-pass GEMM, first version (INT8 tensor cores, INT4 weights widened in registers; `EDGELLM_INT4_W4A8_MIN_M`) | quantises activations to INT8 per row for prompts of 64+ tokens | RTX 4060: GEMM 1.58x, 12B prefill +52%. Orin as first shipped: GEMM 0.96-1.00x of the W4A16 kernel, replay-60 123.1 -> 127.2 s (+3.3% slower), 36 of 60 answers changed | small scratch buffer, not measured on Orin | superseded by the three rows below | [`rtx4060/w4a8-gemm`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/w4a8-gemm) ([37baedb](https://github.com/drakarah/TensorRT-Edge-LLM/commit/37baedb), [4bef0c2](https://github.com/drakarah/TensorRT-Edge-LLM/commit/4bef0c2)) |
| W4A8: flush once per quant group, loop unrolled | the shipped kernel issued its ~190-instruction flush every K-tile (predicated off, still issued) | GEMM 20.1 -> 24.0 TOPS (+19%, +22% on the down shape); prefill probe 1,885 -> 2,010 tok/s (+6.6%); replay-60 122.3 -> 123.4 s (flat) | none | yes | [`perf/w4a8-flush-unroll`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/w4a8-flush-unroll) ([be98bcf](https://github.com/drakarah/TensorRT-Edge-LLM/commit/be98bcf), [86b9275](https://github.com/drakarah/TensorRT-Edge-LLM/commit/86b9275)) |
| W4A8: flush by Abel summation (no accumulator clear) | removes the per-group clear | kernel 4,359 -> 4,187 us (-3.9%) | none | yes | [`exp/w4a8-flush-abel`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/w4a8-flush-abel) ([3dd2875](https://github.com/drakarah/TensorRT-Edge-LLM/commit/3dd2875)) |
| W4A8: INT4 -> INT8 widening with shift + two PRMT | cheaper widening of weight words | kernel (with Abel) 4,359 -> 3,744 us (-14%, 24.6 -> 28.7 TOPS) | none | yes | [`exp/w4a8-widen-prmt`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/w4a8-widen-prmt) ([1713dc5](https://github.com/drakarah/TensorRT-Edge-LLM/commit/1713dc5)) |
| W4A8 per-op filter (`EDGELLM_INT4_W4A8_MAX_K=8192`, `_MIN_N=512`; selectors `_ONLY_N/_SKIP_N/_MAX_N`) | keeps the two ops with outlier-heavy inputs (per-layer-embedding gate, down projection) on the FP16 path | prefill probe 1,819 / 1,809 -> 2,001 / 2,026 tok/s (+10% / +12%); replay-60 wall -0.1% / -2.2% because answers come out about 3% longer on structured requests (paired p 0.006-0.03 in one pair of runs, 0.57 in a later repeat; pooled 59 longer vs 27 shorter; cause not found); decode unchanged. Quality (HumanEval pass@1, 4,096-token cap): W4A16 88.4%, W4A8 all ops 85.4%, with the filter 90.2% (89.0% with the final kernel). Honest summary: a prompt-speed gain, flat on whole-replay time | small scratch buffer, not measured on Orin | yes (Orin profile; the discrete-GPU profile uses all ops and was not checked) | [`exp/w4a8-op-filter`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/w4a8-op-filter) ([36269ed](https://github.com/drakarah/TensorRT-Edge-LLM/commit/36269ed), [18108f6](https://github.com/drakarah/TensorRT-Edge-LLM/commit/18108f6)) |
| FP16-accumulate INT4 prefill GEMM (`EDGELLM_INT4_GEMM_FP16_ACC_MIN_BM=64` at kernel build) | halves the tensor-core cost of the prompt-pass GEMM on GeForce parts | **RTX 4060 only**: GEMM 6.51 -> 3.84 ms, prefill 3,090 -> 4,448 tok/s (+44%), replay-60 68.3 -> 62.2 s (-8.9%); quality gate not cleanly passed (71-test suite about -2 tests). **Orin: no gain** (FP16 tensor peak 26.6 TFLOPS with either accumulator) | none | no (other GPU) | [`rtx4060/int4-gemm-fp16-acc`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/int4-gemm-fp16-acc) ([9d25394](https://github.com/drakarah/TensorRT-Edge-LLM/commit/9d25394)) |
| **Attention** | | | | | |
| INT8 KV cache (`EDGELLM_INT8_KV_SCALES` at export; FP8 KV is not available on sm_87) | stores K/V as INT8 with calibrated static scales; kernels read and write it | speed neutral (71 tests 15:28 -> 15:38) | KV pool at 240 pages 1,680 -> 840 MiB; GPU 9.87 -> 9.03 GB; 71-test suite 49 -> 49 | yes | [`feat/int8-kv-cache`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/int8-kv-cache) (scales option: [`feat/int8-kv-calibration-scales`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/int8-kv-calibration-scales); tests: [`test/int8-kv-cache-kernels`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/test/int8-kv-cache-kernels)) |
| Split-KV (multi-block) XQA decode/verify attention (`EDGELLM_XQA_MULTI_BLOCK=auto` at engine build) with a NaN fix for contiguous-query sliding windows and a bounded-window workspace fix | spreads one sequence over several CTAs | decode at 7.2k-token context 50.6 -> 57.5 tok/s (+14%), 5.3k +4%, short prompts neutral (57.3 -> 56.5); 71-test suite 51 -> 51 | adds a fp16 split workspace: negligible at 10k capacity, +488 MB (base) and +683 MB (draft engine) at 128k capacity | yes | [`perf/xqa-multi-block`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/xqa-multi-block) (NaN fix: [`fix/xqa-multi-block-nan-rows`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/fix/xqa-multi-block-nan-rows); tests: [`test/xqa-multi-block-int8-tree`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/test/xqa-multi-block-int8-tree)) |
| Exact INT8 -> FP16 conversion of the KV cache in XQA (no I2F instruction) | cheaper bit-exact widening in the decode / verify attention | attention kernels -20% to -37% (sliding verify 73 -> 58 us, global 135 -> 87 us); decode round 28.9 -> 27.6 ms (-4.5%); **replay-60 121.4 -> 116.75 s (-3.8%)**, decode probe +3.5%; bit-exact; engines must be rebuilt | engine file 3.4 MB smaller | yes | [`perf/xqa-int8-fast-convert`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/xqa-int8-fast-convert) ([7587cc9](https://github.com/drakarah/TensorRT-Edge-LLM/commit/7587cc9)); NVRTC option hook [093f65f](https://github.com/drakarah/TensorRT-Edge-LLM/commit/093f65f) |
| Warp-specialized head-512 prompt attention (Gemma 4 global layers) | one CTA computes Q.K once per query tile instead of twice | kernel (2,048 x 16,384) 62.4 -> 40.7 ms (-35%); prompt pass 8k -3%, 32k -11%, **64k 61.1 -> 51.2 s (-16%)**; on the 396-request replay only -0.5% (prompts are about 10k tokens); measured on Orin only; skipped on GPUs with 99 KB shared memory | none | yes (built by default on sm_87) | [`perf/d512-warp-specialized`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/d512-warp-specialized) ([1f2b318](https://github.com/drakarah/TensorRT-Edge-LLM/commit/1f2b318), [c34ae84](https://github.com/drakarah/TensorRT-Edge-LLM/commit/c34ae84)) |
| INT8 Q.K^T head-512 prompt attention (`EDGELLM_FMHA_INT8_QK=1`) | SageAttention-style INT8 Q and K in the global-layer prompt attention | Orin: kernel 41.1 -> 30.0 ms (-27%); prompt pass 32k -4 to -5%, 64k 51.9 -> 47.5 s (-8.4%); **nothing at the application's prompt lengths (up to 10k)**; quality unchanged. RTX 4060 (single-P-stage kernel): 12B 23k-token prompt 18.6 -> 13.1 s (-29%) | none | no (opt-in) | [`exp/int8-qk-attention`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/int8-qk-attention) ([0330b16](https://github.com/drakarah/TensorRT-Edge-LLM/commit/0330b16), [3eae272](https://github.com/drakarah/TensorRT-Edge-LLM/commit/3eae272), [2f6a4bd](https://github.com/drakarah/TensorRT-Edge-LLM/commit/2f6a4bd), [82a1317](https://github.com/drakarah/TensorRT-Edge-LLM/commit/82a1317)); RTX 4060 [`rtx4060/fmha-d512-i8-sm89`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/fmha-d512-i8-sm89) ([224c265](https://github.com/drakarah/TensorRT-Edge-LLM/commit/224c265)) |
| Bounded paged-KV fp16 gather | gathers only up to the live length / window instead of the full KV capacity before each prompt pass | replay-30 78.0 -> 77.4 s (-0.8%); 5.2k prefill 1,704 -> 1,711 tok/s; cost no longer grows with capacity (about 15 GB of writes per pass avoided at 128k, estimated) | none | yes | [8eed88a](https://github.com/drakarah/TensorRT-Edge-LLM/commit/8eed88a) |
| **Speculative decoding** | | | | | |
| Tree shape and size (2 x 5 / 12; tree drafting itself is upstream, this is a tuned setting) | deeper, narrower draft tree | same binary, 120 replayed requests: 3x2/8 325.6 s -> 2x3/8 294.2 s (-9.6%); 396 requests: 2x3/8 677.4 s -> 2x5/12 628.2 s (-7.3%, includes a larger prefix pool); 2x6/16 is 0.8% worse over all 396; a 16-node verify tree costs about 2% by itself | not measured separately | yes | `--draft-top-k 2 --draft-step 5 --verify-tree-size 12` |
| Centroid-ordered assistant LM head (fused gather + dot) | the drafter's head table is stored in centroid order so the candidate rows are one contiguous block | per-step kernels 149 -> 102 us; -0.9% per output token (13.35 -> 13.24 ms), decode probe 56.9 -> 57.5 tok/s | none (the original table leaves the graph) | yes | [`perf/assistant-head-centroid-order`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/assistant-head-centroid-order) ([c259d88](https://github.com/drakarah/TensorRT-Edge-LLM/commit/c259d88)) |
| Sparse draft logits (`EDGELLM_GEMMA4_SPARSE_DRAFT_LOGITS=1` at export) | the drafter returns its 4,096 candidate logits plus ids instead of a 262,144-entry row | per draft step 102 -> 82 us; tree selection 194 -> 76 us per round; ms per output token 13.23 / 13.34 -> 13.11 / 13.09 (about -1%); decode probe 57.6 -> 58.3 | GPU 7.44 -> 7.43 GB | yes | [`perf/sparse-draft-logits`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/sparse-draft-logits) ([e0c42fb](https://github.com/drakarah/TensorRT-Edge-LLM/commit/e0c42fb)) |
| Split DDTree candidate selection | selects tree candidates with 32 CTAs per row over the 262k vocabulary instead of one | 472 -> 81 + 43 us per round; replay-30 78.4 -> 78.0 s (-0.5%) | none | yes | [`perf/ddtree-candidate-split`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/ddtree-candidate-split) ([4d93221](https://github.com/drakarah/TensorRT-Edge-LLM/commit/4d93221)) |
| Suffix draft merged into the MTP verify tree (`EDGELLM_MTP_SUFFIX_DRAFT=6`, `EDGELLM_MTP_SUFFIX_MIN=2`; cache capped at 2^18 tokens) | proposes tokens that followed the last occurrence of the current context in earlier outputs, verified in the same pass | replay-60 (cold cache) 134.6 -> 123.9 s (-6.1% paired on identical calls); **replay-396 619.1 -> 545.4 s (-11.9%)**, paired -11.1%, gain grows as the cache fills; 71-test suite unchanged. A repeated-prompt probe reads 90 tok/s only because the cache copies the answer | host index about 45 MB (the first design, 2^22 tokens, could reach about 0.7 GB) | yes | [06bc3dc](https://github.com/drakarah/TensorRT-Edge-LLM/commit/06bc3dc), [fe1eeb0](https://github.com/drakarah/TensorRT-Edge-LLM/commit/fe1eeb0) (branch [`exp/hybrid-suffix-draft`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/hybrid-suffix-draft) is stacked) |
| Prompt lookup (`EDGELLM_MTP_SUFFIX_PROMPT`, chain/accept depth capped at the KV-commit path length of 32) | also indexes the request's own prompt for the suffix draft | **Gemma 4 E4B: no gain** (replay-60 123.1 -> 122.9 s; code edits 287.7 -> 292.5 tok/s, +1.7%; a long chain takes verify slots from the MTP tree, -14%). RTX 4060, 12B: code edits 57.0 tok/s (plain MTP) / 62.5 (with suffix draft) -> 146.7 (2.6x over plain MTP), up to about 207 tok/s with a 19-token chain; prose unchanged | host index only | no (opt-in; no gain for E4B) | [`rtx4060/gemma4-mtp-prompt-lookup`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/gemma4-mtp-prompt-lookup) ([ccda699](https://github.com/drakarah/TensorRT-Edge-LLM/commit/ccda699), [009e66e](https://github.com/drakarah/TensorRT-Edge-LLM/commit/009e66e)) |
| Variable verify size (`EDGELLM_MTP_VERIFY_SMALL`) | verifies fewer nodes in rounds without a long suffix chain | RTX 4060, 12B code edits +4% (166.4 -> 172.7 tok/s) against a static 20-node tree, prose neutral. Not run for E4B (needs an engine built larger than the served tree, which the E4B does not have) | none | no (other model) | [`rtx4060/gemma4-mtp-variable-verify`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/gemma4-mtp-variable-verify) ([95b0039](https://github.com/drakarah/TensorRT-Edge-LLM/commit/95b0039)) |
| Fixed draft candidate set for assistants without a centroid head (12B) | scores a fixed list of 32k token ids in the drafter | RTX 4060, 12B: decode 57.4 -> 59.1 tok/s (+3%) | draft engine 848 -> 378 MB (-0.47 GB VRAM), which lets 4,096-token capacity load | no (other model; the E4B assistant has a centroid head) | [`rtx4060/gemma4-draft-fixed-candidates`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/gemma4-draft-fixed-candidates) ([0267085](https://github.com/drakarah/TensorRT-Edge-LLM/commit/0267085), [993c3df](https://github.com/drakarah/TensorRT-Edge-LLM/commit/993c3df)) |
| INT4 candidate head of the 12B drafter (`EDGELLM_GEMMA4_DRAFT_HEAD_INT4=1` at export; RTN group 128, Orin profile) | the drafter scores its 32k fixed candidates with an INT4 GEMM instead of FP16 (a quarter of the bytes, read three times per decode round) | Orin, 12B, tree 16: head kernel 1.22 -> 0.44 ms per round, round -0.7%, per output token -0.7%, replay-60 -1.1% / -1.7% wall; HumanEval 150/164 vs 150 and 147 (same noise as repeats) | -50 MB | yes (Orin profile) | [`perf/drafter-head-int4`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/drafter-head-int4) |
| INT4 MLP of the 12B drafter (`scripts/quantize_gemma4_assistant.py --mlp-only`, builder step; q/o stay bf16, Orin profile) | the assistant's MLP weights as INT4 (RTN, group 128), read by the same INT4 GEMM as the target | Orin, 12B, tree 16, on top of the INT4 head: round 60.2 -> 57.8 ms (-4.0%), tokens per round -1.1%, per output token -3.3%; replay-60 270.7 -> 265.1 s (-2.1%); HumanEval 148 / 148 vs 149 / 148 of 164, generation time -3.7%. Quantising q/o as well lost 4.2% acceptance | draft engine 328 -> 179 MB | yes (Orin profile) | [`perf/drafter-mlp-int4`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/drafter-mlp-int4) |
| W4A8 prompt GEMM with 4 pipeline stages (default; `EDGELLM_INT4_W4A8_64_STAGES=3` restores the old depth) | one more cp.async stage hides more latency (registers allow 2 CTAs per SM at 4, not more) | Orin: kernel -6.6%, prefill-only replay (max_tokens 1) 119.9 -> 111.6 s (-6.9%); the mixed replay hides it (about 3%); the shared-memory wavefront drop the profiler shows is not less work, it is better latency hiding. No gain on the RTX 4060 (tensor-pipe bound) | none | yes (default) | [`perf/w4a8-int4-int8-stage4`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/w4a8-int4-int8-stage4) |
| Split-K 3 and 5 decode GEMM variants (autotune candidates) | two more split counts for the 16-row decode tile, picked per layer by the build | Orin, 12B: in-engine qkv -3.2%, N=512 layers -13%, down -0.6%; decode round -0.5% (replay flat, below noise). RTX 4060: round -0.7% | none | yes (autotune) | [`perf/int4-gemm-splitk35-dev`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/int4-gemm-splitk35-dev) |
| **KV and prefix cache** | | | | | |
| No KV pool for KV-sharing layers | 18 of 42 Gemma 4 layers read a donor layer's cache and no longer get their own pool | not measured separately | pool for 240 pages 2,940 -> 1,680 MiB (-43%) | yes | [`feat/kv-sharing-no-alloc`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/kv-sharing-no-alloc) |
| Prefix reuse with a sized KV pool (`--enable-context-reuse`, `--maxKVPoolPages 320`) | default pool holds one request, so reuse never hit; a larger pool keeps earlier prefixes | reuse 0 of 25 requests -> 106 of 164 (240 pages); 12 re-run tests 223 -> 165 s (-26%); 320 pages: reused prompt tokens 72.7% -> 75.9%. (Stock also accepts the flag; its hit rate with its default pool was not measured in this benchmark) | GPU 7.14 -> 7.44 GB for 320 pages (pool 1,120 MiB) | yes | [`dev`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/dev) |
| Bounded sliding-window KV with INT8 KV and tree speculative decoding (`EDGELLM_GEMMA4_BOUNDED_KV_SPEC=1`, `_BOUNDED_KV_QUANT=1`; plus a fix for garbage output on prompts longer than the engine input length) | sliding layers keep a ring of the last 1,152 positions instead of the full capacity | no per-step speed cost (63.4 vs 63.4, 71.2 vs 73.0, 55.8 vs 55.9 tok/s on three prompts). Disables ordinary prefix reuse (see next row). **For the E4B on Orin it saves only about 190 MiB at 10k capacity** (about 1.35 GB at 70k, estimated from the model config) | RTX 4060, 12B: capacity 8,192 in 7,347 MiB (the full layout did not load at 6,144), 32,768 in 7,647 MiB | no (E4B on Orin; mainly helps the 12B on 8 GB) | [`rtx4060/gemma4-bounded-kv-spec`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/gemma4-bounded-kv-spec) ([5c2f632](https://github.com/drakarah/TensorRT-Edge-LLM/commit/5c2f632), [3203e24](https://github.com/drakarah/TensorRT-Edge-LLM/commit/3203e24), [df5c581](https://github.com/drakarah/TensorRT-Edge-LLM/commit/df5c581)) |
| Host-resident prefix cache for bounded sliding-window KV (`EDGELLM_SWA_PREFIX_CACHE_MB`) | snapshots KV into pinned host memory so bounded engines get prefix reuse | RTX 4060, 12B: replay-396 1,667.8 -> 1,191.3 s (-28.6%); repeated 4.7k-token prompt 2.67 -> 0.46 s; 296 of 396 answers byte-identical | no VRAM; host memory up to the configured budget | no (needs bounded engine) | [`rtx4060/gemma4-bounded-prefix-cache`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/gemma4-bounded-prefix-cache) ([b5363a2](https://github.com/drakarah/TensorRT-Edge-LLM/commit/b5363a2)) |
| Verify size that fits the INT4 GEMM tile ladder (`--verify-tree-size 16` instead of 20 for the 12B on Orin) | the GEMM runs in 16-row tiles and is tuned once at the engine's largest verify size; 20 rows pad to a 32-row tile (37.5% idle) | Orin, 12B: decode round 72.8 -> 60.7 ms, -16% per output token for 3.5% fewer tokens per round; HumanEval unchanged | none | yes (12B, Orin) | settings only (`poc/config/gemma4-12b.sh`) |
| GELU(gate) * up fusion plugin (`EDGELLM_GEMMA4_GELU_PLUGIN=1`) | one kernel instead of TensorRT's generated chain (cast, split, gelu, mul, clip, cast), 16-byte accesses, exp2-form tanh | Orin: 12B prompt passes -9%, decode round -2.4%; E4B prompt passes -11%, prefill +11%, decode round -2.9%. Output within one fp16 ulp of the FP32 reference | none | yes (12B) | [`perf/gelu-mul-plugin`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/gelu-mul-plugin) |
| Drop identity activation permutations at export (no-op Gather in front of every INT4 GEMM) | the exporter emitted `index_select` with an arange for GPTQ checkpoints without act-order; 201 Gathers in the 12B graph | included in the plugin branch's numbers above (the Gather alone moved fusion boundaries and was net zero) | none | yes (12B) | [`perf/gelu-mul-plugin`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/gelu-mul-plugin) |
| **Runtime and host** | | | | | |
| Chunked prefill (`--maxInputLen 2048`; strategies opt in) | prompts longer than the engine's input length run in several passes appended to the KV cache. Stock needs an input limit equal to the full context | prefill 1,683 -> 1,704 tok/s (unchanged); decode unchanged | activations 899 -> 225 MB; GPU 8.07 -> 7.14 GB | yes | [`feat/chunked-prefill`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/chunked-prefill) ([fe561b9](https://github.com/drakarah/TensorRT-Edge-LLM/commit/fe561b9)) |
| TensorRT binding cache | re-sets only the bindings that changed between steps (was about 270 per round) | verify launch 3.5 -> 1.1 ms after round start; replay-30 82.5 -> 80.0 s (-3.5%); with the tree-selection change prose decode at 7.2k context 56.7 -> 62.2 tok/s | none | yes | [`perf/executor-binding-cache`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/executor-binding-cache) ([a99a2e1](https://github.com/drakarah/TensorRT-Edge-LLM/commit/a99a2e1)) |
| O(n log n) byte-pair merge in the tokenizer | replaces a quadratic merge loop for long pieces | 138 -> 9.6 ms per 11.7k-character prompt (Hugging Face 7.8 ms); ids identical on 300 of 300 prompts; replay-30 about -4.5% | none | yes | [`perf/tokenizer-bpe-heap`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/tokenizer-bpe-heap) ([aaa2b6a](https://github.com/drakarah/TensorRT-Edge-LLM/commit/aaa2b6a)); upstream PR #248 |
| XGrammar v0.2.1 -> v0.2.8 | newer grammar compiler | with the JSON whitespace cap, 9 schemas compiled to 2,272 MB at 301 ms each -> 96 MB at 62 ms; replay-396 with the default 64 MB cache 651.4 -> 629.7 s (-3.3%) | host RAM 2.27 GB -> 96 MB for the compiled grammars | yes | [38f73e4](https://github.com/drakarah/TensorRT-Edge-LLM/commit/38f73e4) |
| JSON whitespace cap (`EDGELLM_JSON_MAX_WHITESPACE=20`) | bounds whitespace runs inside constrained JSON | not a per-token speed change: without it the model can emit indentation until max_tokens. In this benchmark's stock run the stock stack wrote 51k whitespace characters against 6k, which accounts for much of its 2.7x output tokens on the replay | none | yes | [`feat/json-whitespace-cap`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/json-whitespace-cap) |
| `EDGELLM_FORCE_GREEDY` | forces greedy sampling (the MTP tree is greedy-only; the discrete-GPU profiles do not speculate and do not set it, so a request keeps its own temperature) | none (a correctness workaround) | none | yes | [`feat/force-greedy`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/force-greedy) |
| Server correctness fixes (JSON-schema property order kept, Gemma 4 tool parser, end tokens stripped from tool output, chat-template string content, status codes) | API behaviour, not speed | JSON-schema property order alone moved the 71-test suite from 19 to 46 of 71 re-run tests passing (upstream PR #234) | none | yes | [`fix/json-schema-property-order`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/fix/json-schema-property-order) (PR #234), [`feat/gemma4-tool-parser`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/gemma4-tool-parser), [`dev`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/dev) |
| `EDGELLM_EMBEDDING_ON_HOST` | keeps the token embedding table in pinned host memory | RTX 4060, 12B: costs nothing measurable. **Not useful on a Jetson** (CPU and GPU share one DRAM; reasoned, not measured) | RTX 4060: the 2.0 GB fp16 table leaves VRAM (the 12B engine did not load without it) | no (Jetson shares DRAM) | [`rtx4060/embedding-on-host`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/embedding-on-host) ([3f08600](https://github.com/drakarah/TensorRT-Edge-LLM/commit/3f08600)) |
| **Mixture of experts (Gemma 4 26B-A4B; none of these affect the dense models)** | | | | | |
| Quantise each layer's BF16 experts to INT4 as they load (`--externalize-weights int4_ffn int4_moe` export) | the exporter packs the fused expert tensors of a layer to Marlin INT4 as soon as both are loaded and frees them, so the 51 GB checkpoint is never resident in full | not a speed change: the 26B export fits a 32 GB Orin (10 min, container RAM peak 23.0 GiB, swap 0.9 GiB); the resulting engine is 5.5 GB plus 13 GB of sidecar weights, 20.9 GiB of server memory | none | yes | [`feat/gemma4-moe-layerwise-expert-quant`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/gemma4-moe-layerwise-expert-quant) ([ff05062](https://github.com/drakarah/TensorRT-Edge-LLM/commit/ff05062)) |
| MTP drafter and verify tree 16 for the 26B (`--mtp` accepted by the MoE export, assistant checkpoint) | the speculative path of the dense models also runs on the MoE | Orin: 300-token code probe 25.1 -> 40.9 tok/s (1.63x); HumanEval generation 24.7 -> 50.8 tok/s, 158/164 against 159/164 (one round each, suffix draft and JSON cap in the second number) | draft engine 0.37 GB | yes | upstream MTP path, no extra branch |
| 8-row Marlin block for the 16-token verify step (`EDGELLM_MOE_SMALL_THRESHOLD=16`) | the MoE plugin used the 8-row block only up to 8 tokens; a verify of 16 tokens spreads about one row per expert and was padded to 32 rows | Orin, 26B: decode probe 29.3-30.9 -> 33.8-34.9 tok/s (+13%); HumanEval generation 51.6 -> **60.2 tok/s (+17%)**; replay-60 313.0 -> **276.3 s (-11.7%)**; prefill unchanged; larger prefill blocks (48, 64) and a register cap for 3 blocks per SM were slower | none | HumanEval 158/164 both, needles 3/3 | [`perf/moe-small-block-threshold`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/perf/moe-small-block-threshold) ([6460200](https://github.com/drakarah/TensorRT-Edge-LLM/commit/6460200)) |
| Bounded sliding-window KV + host prefix cache for the 26B (`EDGELLM_GEMMA4_BOUNDED_KV_SPEC=1` at export, `EDGELLM_SWA_PREFIX_CACHE_MB=4096` at serve; the 26B's MTP export keeps full KV otherwise, so the cache had nothing to attach to) | the sliding layers keep 1,152 positions; the host cache snapshots them between prefill passes of 2,048 tokens | Orin, 26B (same engine, one round each): replay-60 262.7 -> **234.0 s (-10.9%)**; warm time to first token at 4k / 7.5k prompts 2.43 / 4.73 -> **1.32 / 1.02 s**; no gain for prompts under one 2,048-token pass; cold prefill 5-9% slower (1,712 -> 1,565 tok/s at 1.4k) | 4 GiB of the unified memory; KV pool smaller | HumanEval 158 -> 157/164, needles 3/3/3 (within noise) | [`rtx4060/gemma4-bounded-prefix-cache`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/gemma4-bounded-prefix-cache) ([b5363a2](https://github.com/drakarah/TensorRT-Edge-LLM/commit/b5363a2)) |
| **Qwen3.5-4B and other hybrid (GDN) models** (none of these affect Gemma 4) | | | | | |
| GDN prefill kernel for sm_80-class GPUs, state in registers | sequential token loop with the state in registers instead of shared memory | kernel 40.2 -> 9.2 ms at 2,048 tokens (4.4x); engine prefill 1,121 -> 1,788-1,842 tok/s (+62%); replay-60 293 -> 231 s (-21%) | none | no (Qwen) | [`exp/gdn-prefill-sm80`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/gdn-prefill-sm80) ([029ff0d](https://github.com/drakarah/TensorRT-Edge-LLM/commit/029ff0d)) |
| GDN multi-token verify kernel | state read once into registers per verify | 230.7 -> 84 us per layer call; decode probe 45.2 -> 52.5 tok/s (+16%); replay-60 -4.4% | none | no (Qwen) | [`exp/gdn-mtp-verify-sm80`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/gdn-mtp-verify-sm80) ([a0812cf](https://github.com/drakarah/TensorRT-Edge-LLM/commit/a0812cf)) |
| GDN verify reads the resident state pool directly | drops a 2 MB per-layer copy before every verify | decode +4.9% (66.3 -> 69.5 tok/s); replay-60 198.0 -> 193.7 s (-2.2%) | none | no (Qwen) | [`exp/gdn-verify-pool-read`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/gdn-verify-pool-read) ([b01100d](https://github.com/drakarah/TensorRT-Edge-LLM/commit/b01100d)) |
| Causal conv1d prefill fills the GPU | more blocks per SM, two channels per thread | prefill +2.7% (1,840 -> 1,888 tok/s); replay-60 -1.2% | none | no (Qwen) | [`exp/conv1d-prefill-grid`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/conv1d-prefill-grid) ([a322d64](https://github.com/drakarah/TensorRT-Edge-LLM/commit/a322d64)) |
| Linear MTP replay stash and register-resident replay commit | replaces per-step state snapshots | verify 84 -> 33 us per layer; decode +3.0% (69.7 -> 71.8 tok/s); replay-60 -1.4%; bit-identical | less snapshot scratch (about 16 KB per step against 4 x 2 MB per layer per round) | no (Qwen) | [`exp/gdn-linear-replay`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/gdn-linear-replay) ([020c3f5](https://github.com/drakarah/TensorRT-Edge-LLM/commit/020c3f5), [f937812](https://github.com/drakarah/TensorRT-Edge-LLM/commit/f937812)) |
| Chunked tensor-core GDN prefill (opt-in; default on sm_80-89 since [fee11c7](https://github.com/drakarah/TensorRT-Edge-LLM/commit/fee11c7)) | chunked (WY) form on WMMA | 9.1 -> 3.6 ms per layer call at 2,048 tokens; prefill 1,890 -> 2,146 tok/s (+13.5%); replay-60 189.1 -> 178.2 s (-5.8%); RTX 4060 replay-60 104.2 -> 102.5 s | about 34 MB scratch | no (Qwen) | [`exp/gdn-chunk-prefill`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/gdn-chunk-prefill) ([e2d9057](https://github.com/drakarah/TensorRT-Edge-LLM/commit/e2d9057)), [`rtx4060/gdn-chunk-x86`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/gdn-chunk-x86) ([fee11c7](https://github.com/drakarah/TensorRT-Edge-LLM/commit/fee11c7)) |
| Prefix reuse for hybrid models with MTP (divergence checkpoints, `EDGELLM_HYBRID_DIVERGENCE_CHECKPOINT=256`) | lets hybrid models keep cross-request prefixes (upstream refuses it) | Qwen3.5-4B replay-396 1,447.6 -> 903.7 s (-34% paired), 70.8% of prompt tokens reused; all Qwen changes together 1,892.6 -> 723.4 s | 5.8 -> 8.6 GB with a 1.5 GB recurrent snapshot pool (7.58 GB with 512 MB) | no (Qwen) | [`exp/hybrid-context-reuse`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/exp/hybrid-context-reuse) ([5805006](https://github.com/drakarah/TensorRT-Edge-LLM/commit/5805006), [bad64ce](https://github.com/drakarah/TensorRT-Edge-LLM/commit/bad64ce), [d457c6d](https://github.com/drakarah/TensorRT-Edge-LLM/commit/d457c6d)) |
| Quantiser fits a 32 GB board | lower calibration memory for Qwen3.5-4B INT4-AWQ | tooling | n/a | no (Qwen) | [`fix/quantize-low-memory`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/fix/quantize-low-memory) ([4976719](https://github.com/drakarah/TensorRT-Edge-LLM/commit/4976719)) |
| **Expert streaming on a discrete 8 GB GPU (Gemma 4 26B-A4B and Qwen3.6-35B-A3B; none of these affect the dense models or the Orin)** | | | | | |
| Routed experts stay in pinned host memory; per-layer GPU expert cache (`EDGELLM_HOST_EXTERNAL_WEIGHTS=int4_moe`, `EDGELLM_MOE_CACHE_SLOTS=auto`, `EDGELLM_MOE_STAGE_EXPERTS=2` at build) | the MoE plugin copies the experts a step routes to into private device buffers (prefill) or a per-layer slot cache (decode) | RTX 4060, 26B: decode 14 tok/s with no cache -> 54 with 45 slots of 128 (83.75% hit rate) | the 11.4 GB of INT4 experts leave VRAM; the cache size is derived from the free VRAM at the first decode step | yes (dgpu profile) | [`rtx4060/moe-prefill-stage-overlap`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-prefill-stage-overlap) ([6359ac5](https://github.com/drakarah/TensorRT-Edge-LLM/commit/6359ac5), [25238c2](https://github.com/drakarah/TensorRT-Edge-LLM/commit/25238c2)) |
| Prefill expert prefetch overlap (`EDGELLM_MOE_PREFILL_OVERLAP=1`) | copies the experts of layer L+1 while layer L computes | RTX 4060, 26B: cold prefill 4k +59%; Qwen3.6: a prefill pass floors at 1.9 s (all experts over PCIe at about 8.5 GB/s), overlap reaches 92% of it. It copies the whole layer whatever the prompt length (15.5 GB at the measured 8.6 GB/s = 1.8 s), so time to first token is flat at 1.84 s from 162 to 1,012 tokens where plain staging takes 0.79 to 1.13 s; it wins only above about 4,200 tokens (7,012 tokens 2.69 -> 2.16 s) and loses at 3,012 (1.64 -> 1.96 s). With the routed chunking below on, only passes longer than the token tile still use it | one second 396 MiB staging buffer (about 5 cache slots per layer on Qwen3.6) | yes (dgpu profile) | [`rtx4060/moe-prefill-stage-overlap`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-prefill-stage-overlap) ([6359ac5](https://github.com/drakarah/TensorRT-Edge-LLM/commit/6359ac5)) |
| Prefill chunk ring (`EDGELLM_MOE_PREFILL_CHUNK=32`) | prefetches the experts in chunks into a 3 x 64 MiB ring instead of two whole-layer buffers | RTX 4060, 26B: +6 cache slots, replay-60 346.9 -> 326.0 s. **Not used for Qwen3.6**: replay -12% but decode -17% (cause not found) | 3 x 64 MiB instead of 2 x 386 MiB | 26B only | [`rtx4060/moe-prefill-stage-overlap`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-prefill-stage-overlap) ([d25ebf5](https://github.com/drakarah/TensorRT-Edge-LLM/commit/d25ebf5)) |
| Cache-aware expert routing (`EDGELLM_MOE_CACHE_BIAS`, default 0.5) **lossy, decode only** | the router adds beta to the logits of experts that are already in the GPU cache before choosing the top 8; the combine weights stay the softmax of the original logits | Qwen3.6, RTX 4060: 65% fewer cache misses at beta 0.5 for +0.09% perplexity (beta 1: 85% fewer, +1.5%); replay-60 363.3 -> 269.7 s (beta 1: 245.2 s; a second perplexity measurement at 55 slots gave +0.27% for beta 0.5, test noise about 0.15%); decode 49.4 -> 62.1 tok/s; needles 40/40 at 2k and 6k tokens at beta 0.5 (beta 1: 40/40 and 39/40). Measured on Qwen3.6 only: set 0 for exact routing on other models | none | yes (Qwen3.6, beta 0.5) | [`rtx4060/moe-cache-aware-routing`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-cache-aware-routing) ([cf3d3cf](https://github.com/drakarah/TensorRT-Edge-LLM/commit/cf3d3cf), [dc94344](https://github.com/drakarah/TensorRT-Edge-LLM/commit/dc94344)) |
| CPU miss path (`EDGELLM_MOE_CPU_MISS=1`, `_ADMIT`, `_INT8`, `_THREADS`, `_CORES=auto`; x86 only) **decode only** | an expert that is not in the GPU cache is computed on the CPU from the pinned host weights (a GPU doorbell kernel hands it to spinning threads through a mapped mailbox: no host synchronisation, CUDA-graph safe) until it has missed `ADMIT` times, then it is cached as before; the CPU kernel reads the Marlin layout in place, with fp32 or int8 activations (AVX-VNNI) | Qwen3.6, RTX 4060 + i9-12900H, beta 0.5, prefill overlap on, 55 slots (decode = 300 tokens after a ~10-token prompt, four warm readings): 62.5 -> 75.9 tok/s (+21%, int8, admit 16); fp32 admit 8: 71.2 (+14%); prefill unchanged; uses 6 busy CPU cores | int8 activations: about 0.65% relative error per expert output (about 2% with outlier channels); fp32: about 3e-7 against float64 but not bit-identical to the GPU path; HumanEval 155/164 vs 154/164 without it (serve defaults; identical runs differ by up to 3 problems) | yes (`CPU_MISS=0` turns it off; AVX2 required, int8 needs AVX-VNNI, otherwise it falls back) | [`rtx4060/moe-cpu-miss`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-cpu-miss) ([ee3825b](https://github.com/drakarah/TensorRT-Edge-LLM/commit/ee3825b)) |
| Expert-cache fill and lookup launch floors (always on, no setting) | an empty fill call launched 512 blocks that all exit at once (17 us per layer) and the lookup walked all experts in one thread (9 us per layer); now 64 blocks looping over the miss list and a warp-compacted lookup | Qwen3.6, RTX 4060, serve defaults (beta 0.5, overlap off, 61 slots; decode = 300 tokens after a 24-token prompt): CPU miss path on 77.5 -> 81.9 tok/s (+6%); CPU off about 65 -> 67.3 tok/s; greedy output byte-identical with the CPU path off; prefill unchanged | none | yes | [`rtx4060/moe-cache-floor`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-cache-floor) ([cc7e450](https://github.com/drakarah/TensorRT-Edge-LLM/commit/cc7e450)) |
| Routed-expert chunked prefill (`EDGELLM_MOE_PREFILL_ROUTED_CHUNK`, off by default, `16` turns it on; engines built with `EDGELLM_MOE_STAGE_EXPERTS=2`) | after the router has run, only the experts the pass routes to are copied (DMA, 16 experts per chunk, into a ring of 4 small buffers) while the previous chunk's expert GEMM runs; the whole layer is no longer staged in front of the GEMMs. Costs one host round trip of the routed list per layer. A pass longer than the engine's token tile (4,096 with `EDGELLM_MOE_TOKEN_TILE=4096`) keeps the previous staging: copying per tile re-copies the experts the tiles share | Qwen3.6, RTX 4060, time to first token, random text, serve defaults, 6 requests per cell: 32 tokens 0.63 -> 0.56 s, 162: 0.81 -> 0.70, 424: 0.94 -> 0.80, 1,012: 1.12 -> 0.93, 3,012: 1.66 -> 1.28 s (-11% to -23%); no change from 4,102 tokens up; decode unchanged; prompt-processing probe (3,016 tokens) 1,420 -> 1,770 tok/s. A prefill touches only 26% (32 tokens) to 49% (6,012 tokens) of an MoE layer's 256 experts (one random-text prompt per length), which is why copying only the routed ones pays there. **Application prompts route to far more experts: 79% (705 tokens), 85% (864) and 91% (3,087) of the 256 per layer, so on the 49-request application replay it is 7-10% slower (193-195 s against 179 s for the same engine with it off; time to first token of 2,500-4,096-token prompts 1.64 -> 2.10 s) and it is therefore off by default.** Gemma 4 26B-A4B (default against off, 3,013 tokens): time to first token 1.63 -> 1.26 s, 413 tokens 0.84 -> 0.75 s. | one ring of about 80 MiB (about 1 cache slot per layer: 61 -> 60). HumanEval (164 problems, two serves each, run-to-run noise about 5 problems): 154 and 151 without, 156 and 153 with; no task fails in both routed runs and passes in both plain runs | yes (opt-in) | [786a71b](https://github.com/drakarah/TensorRT-Edge-LLM/commit/786a71b) |
| Prefill experts staged from the decode cache (`EDGELLM_MOE_PREFILL_FROM_CACHE`, default on, acts with `EDGELLM_MOE_PREFILL_OVERLAP=1`) | the prefetch of layer L+1 copies the experts its decode cache already holds device-to-device from their slot and only the others over PCIe (contiguous runs coalesced) | Qwen3.6, RTX 4060, tiled 12k engine, overlap on, beta 0.5: replay-20 79.3 -> 74.8 s (-5.7%), warm time to first token 1,926 -> 1,681 ms (-12.7%); decode and cold prefill unchanged | none, lossless (identical weights); needles 30/30, 30/30, 30/30 at 2,048 / 6,000 / 8,500 tokens | yes | [`rtx4060/prefill-stage-from-cache`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/prefill-stage-from-cache) ([3b6f86d](https://github.com/drakarah/TensorRT-Edge-LLM/commit/3b6f86d)) |
| Cache-aware router in one warp and a warp-parallel victim search in the expert-cache lookup (always on, no setting) | the router kernel used one block per token and the lookup walked its slots one by one; now one warp per token (up to 512 experts) and a warp-parallel LRU search | micro-timings, launches back to back (1 token, 256 experts, top 8): router 5.88 -> 4.46 us, lookup 8.74 -> 6.63 us per call. Routing bit-identical to the old kernel on 2,880 of 2,880 cases, lookups identical on 63,000 calls. **No end-to-end gain measurable**: decode 79.3-82.7 against 79.8-82.0 tok/s (the run-to-run spread is about 2 tok/s) | none | yes | [786a71b](https://github.com/drakarah/TensorRT-Edge-LLM/commit/786a71b) |
| Expert-cache sizing and diagnostics: `EDGELLM_MOE_CACHE_RESERVE_MB` (VRAM kept free after sizing the cache, 250), `EDGELLM_MOE_MAX_VRAM_MB` (cap on the total device memory the server may use, so the slot count does not depend on other GPU tenants), `EDGELLM_MOE_STATS=<prefix>` (per-layer, per-expert selection counters, prefill and decode, written every 15 s) | settings, not speed changes | one cache slot costs about 62 MiB across the 40 layers; decode against slots (CPU path on, beta 0.5): 30 -> 72-74 tok/s, 40 -> 76-77, 50 -> 78-80, 61 -> 80-82; with the CPU path off 30 -> 49, 45 -> 59-60, 61 -> 67. The counters show how uneven the routing is: in a 20-call replay (66,700 prompt tokens) the busiest quarter of the experts take 64% of the picks (42-74% by layer) and 21 of the 10,240 layer-expert pairs were never picked. Decode steps served by the cache do not reach the counters | none | yes | [786a71b](https://github.com/drakarah/TensorRT-Edge-LLM/commit/786a71b) |
| Growable KV pool (`EDGELLM_KV_INITIAL_TOKENS`, off unless set; trim window `EDGELLM_KV_SHRINK_WINDOW_REQUESTS` 4 / `_SECONDS` 300 / `EDGELLM_KV_SHRINK_SLACK_TOKENS` 16,384) | maps the KV cache through CUDA virtual memory on demand: at the start of a request it grows to input + `max_tokens` + headroom, taking the VRAM from the expert cache, and it trims back to the peak need of the last requests; the context cache works on the mapped pages (lowest pages first, records on unmapped pages are evicted) | **RTX 4060 only**, Qwen3.6-35B-A3B, 256k window: a static 256k pool leaves 16 expert slots and decodes 59.1 tok/s, the growable pool keeps 56 slots and decodes 84.1 tok/s (short requests, METRICS R25ev); against a static 128k pool 83.5 vs 81.4 tok/s at 128k (not significant). Tested with reuse to 88k tokens in a live agent session; 222k needle retrieval passed on the static pool and failed on both growable runs (cause unresolved, R25ew) | KV 120 MiB mapped at the start (12,288 tokens) instead of 2.5 GiB | yes, off unless the variable is set; clients must send `max_tokens` (without it the first request maps the whole pool) | [10e2bf7](https://github.com/drakarah/TensorRT-Edge-LLM/commit/10e2bf7) |
| Per-request timing (`EDGELLM_REQUEST_STATS=1`, ring size `EDGELLM_REQUEST_STATS_CAPACITY`, default 500) | `GET /api/stats` returns the last requests and a token-weighted summary (`/stats` is the dashboard page that charts it); per request: prompt, reused and completion tokens, time to first token, decode time and the prefill and decode rates (streamed requests carry the split, a non-streamed one only its total); reused tokens come from the context-cache counters and are left out when another request ran at the same time; the prefill rate is only given for 256+ uncached prompt tokens; `POST /api/stats/reset` clears the ring | none (bookkeeping per request). Example on the 4060: a cold 5,731-token prompt of random words 2,261 tok/s prefill, 143 ms first token when 5,727 tokens were reused (METRICS R25ey) | none | yes, on in the dgpu profiles (`REQUEST_STATS=0` turns it off) | [2c4e12e](https://github.com/drakarah/TensorRT-Edge-LLM/commit/2c4e12e) |
| Stats dashboard at `/stats` | one self-contained page, no extra service: tiles (median and p90 first-token time, decode, prefill, reuse), charts of first-token time and prefill speed by new tokens, decode speed by context, and each of those over the requests, a table of the latest requests, and the memory bars below. It reloads `/api/stats` every 5 s | none (the page is static; the JSON is a snapshot of a bounded ring) | none | yes, on in the dgpu profiles; open `http://localhost:8000/stats` | [c3eb0eb](https://github.com/drakarah/TensorRT-Edge-LLM/commit/c3eb0eb) |
| Memory figures in `/api/stats` and stacked bars on `/stats` | GPU (nvidia-smi: KV cache, expert cache, everything else, free), KV cache in tokens (pages in use, mapped and free, not mapped), experts per layer on the GPU against all of them, host RAM (this server, other, available) and the reuse snapshot pools. The runtime's own figures (KV, pools, expert cache) are captured by the engine thread at request start and end, because its live structures are not safe to read from another thread; GPU and host are read on each refresh | none | none | yes, on in the dgpu profiles | [b1b5eca](https://github.com/drakarah/TensorRT-Edge-LLM/commit/b1b5eca) |
| Idle sleep stage of the CPU miss workers (`EDGELLM_MOE_CPU_IDLE_MS`, default 1000, 0 = never) | the worker threads that compute cache-missing experts on the CPU polled every 50 us after 20 ms idle but still kept a core about 60% busy; after the idle time they now sleep 1 ms at a time, so the first doorbell afterwards waits up to 1 ms | no visible cost: first token 125 ms after 5 s idle against 122-133 ms back to back, decode 92-95 tok/s both (few runs, CyberTiel, 176-token prompt) | none; **idle CPU of the server 416% -> about 18%** (docker stats, 6 workers) | yes, on by default | [37510e5](https://github.com/drakarah/TensorRT-Edge-LLM/commit/37510e5) |
| Environment defaults for request fields (`EDGELLM_DEFAULT_TEMPERATURE`, `_TOP_P`, `_TOP_K`, `_ENABLE_THINKING`) | the server applies them to requests that leave the field out; out-of-range or unparsable values fall back to the built-in 0.7 / 0.9 / 50 / thinking off | none | none | yes (set in the dgpu Qwen3.6 profile for a served endpoint, not for the benchmarks) | [rtx4060/server-sampling-defaults](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/server-sampling-defaults) |
| Presence penalty (`presence_penalty`, default from `EDGELLM_DEFAULT_PRESENCE_PENALTY`) | a sampler kernel penalises tokens already in the output; refused with speculative decoding or several request slots | not measured separately. The agent study's best arm changed temperature and penalty together: 14 of 16 runs completed against 12 of 24 (METRICS R25dv, R25dw); their separate effects are not known | none | yes (dgpu Qwen3.6 profile, 1.5) | [rtx4060/presence-penalty](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/presence-penalty) |
| Preserved thinking (`EDGELLM_PRESERVE_THINKING=1`) | the chat template keeps the reasoning of earlier turns in the rendered history (agent loops) | costs prompt tokens and prefix-reuse hits; not measured separately | none | yes (dgpu Qwen3.6 profile, on) | [rtx4060/preserve-thinking](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/preserve-thinking) |
| Hybrid checkpoints in pinned host memory (`--context-cache-snapshots-in-host-memory`) and the plain-tail checkpoint (`EDGELLM_HYBRID_PLAIN_TAIL_CHECKPOINT=1`) | the recurrent, conv and partial-KV snapshots of a hybrid model (Qwen3.6, CyberTiel) live in pinned host memory: capture is a device-to-host copy, restore a host-to-device copy, and the two snapshot pool budgets count host bytes. The tail flag now takes effect on a plain hybrid (it only did with a draft model) so a request tail has a checkpoint to restore | 14-turn agent conversation up to 68k tokens on the 256k engine: 3.60 s per turn with reuse (growable pool, turns 8-14) against 23.29 s without; 3.21 s with a static pool (METRICS R25ex). A live 116-request session got 97.6% of its prompt tokens from reuse | snapshot pools of 256 MiB (recurrent) and 64 MiB (partial KV) moved off the GPU | yes, off unless the flags are given; the Qwen3.6 dgpu profile keeps reuse off (it costs an extra expert-streaming pass on the replay), the CyberTiel profile turns it on | [rtx4060/hybrid-host-checkpoint-store](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/hybrid-host-checkpoint-store) |
| Reasoning budget (`reasoning_budget_tokens`, `reasoning_budget_message`; defaults `EDGELLM_DEFAULT_REASONING_BUDGET`, `EDGELLM_DEFAULT_REASONING_BUDGET_MESSAGE`) | counts the tokens of the reasoning block and, when the budget runs out, forces the optional message and the end-of-thinking token; only with thinking on, plain decoding, one request, full vocabulary. Named as in llama.cpp | not measured; the model card of CyberTiel advises against a budget below about 16,000 tokens | none | yes (off unless the field or default is set) | [rtx4060/reasoning-budget](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/reasoning-budget) |
| Control-token strings in message content stay text (`EDGELLM_ALLOW_CONTROL_TOKENS_IN_CONTENT=1` restores the old behaviour) | tool results and user text containing `<\|im_start\|>`, `<think>` or an image placeholder were tokenised as the real tokens (forged turns, a closed or reopened thinking block, or a placeholder without an image); such strings now get a zero-width space after their first character | none (a correctness and safety fix) | none | yes (always on) | [rtx4060/defang-control-tokens](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/defang-control-tokens) |
| Server: a client that disconnects cancels its non-streaming request; `EDGELLM_DEFAULT_MAX_TOKENS` and `EDGELLM_LOG_CANCELLED_DIR` | the middleware hid the disconnect from the route, so a request ran to its end after the client had gone, and a request without `max_tokens` (the app sends none) could generate until the whole KV capacity was used. Now the route cancels the generation and answers 499, an optional token cap applies to requests that send none, and the bodies of cancelled requests can be dumped for diagnosis | not a speed change. In one app run on Qwen3.6 five or six requests ran about 12 min each (62-65k tokens) and three hit the app's 300 s client timeout; two reruns did not repeat it and the cause was not found. With the fix the engine is free about 0.1 s after the disconnect; 20 server tests, 8 of them fail on the previous server | none | yes | [786a71b](https://github.com/drakarah/TensorRT-Edge-LLM/commit/786a71b) |
| Finite-value probe (`EDGELLM_FINITE_WATCH=1`), hardening from a code audit, and unit tests for the Qwen path | probe: for the input and output of every MoE layer a host thread counts non-finite values, the largest |x| and values from 30,000 up (fp16 limit 65,504). Hardening: router logits clamped and the cache-aware top-k limited to 16, expert ids outside the table sent to slot 0, the expert cache refused when an expert's bytes are not a multiple of 16, and the CPU miss path refuses layers with one scale group (a different Marlin scale permutation, found by an edge-case test; Qwen's 2,048 and 512 shapes are not affected) | over a full 1,405-test application run: 0 non-finite values, largest |x| 56.5. Unit tests: CPU kernel 12 (line coverage of the kernel header 184 of 184), router 12, cache lookup and fill 14, probe 10, guided-decoding whitespace 3, server disconnect 20; gaps recorded in METRICS R25da | probe about 1% of decode, off by default | probe off, the fixes on | [786a71b](https://github.com/drakarah/TensorRT-Edge-LLM/commit/786a71b) |
| Tail skip in the cache-aware router (`EDGELLM_MOE_TAIL_SKIP=N`, off by default; the idea of PolyStrata's `--tail-skip`) **lossy, decode only** | an expert that is not in the GPU cache and ranks N+1 or lower of the top 8 is not fetched; its slot gets the top expert's id with weight 0, nothing is renormalised. A lower N skips more | Qwen3.6, RTX 4060, 55 slots, 64k-context engine (decode = 300 tokens after a ~10-token prompt; replay-20 = the first 20 recorded requests): exact routing 47.5 tok/s, 98.5 s; N=6: 55.9 tok/s, 82.7 s (+0.04% perplexity); N=5: 61.5 tok/s, 80.7 s (+0.26%); beta 0.5 alone 60.3 tok/s, 77.2 s (+0.27%); beta 0.5 + N=6 65.6 tok/s, 75.7 s (+0.83%); beta 0.5 + N=5 68.6 tok/s, 73.9 s. Needles 27/27 at 2k, 4k and 6k tokens for exact + N=6 and beta 0.5 + N=6. Perplexity from an offline study on 32 sequences of 1,024 tokens against the BF16 model | none | no (opt-in `TAIL_SKIP`) | [`rtx4060/moe-tail-skip`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-tail-skip) ([9d3679c](https://github.com/drakarah/TensorRT-Edge-LLM/commit/9d3679c)) |
| Chunked prefill for hybrid (gated-delta-net) models, on by default (`EDGELLM_HYBRID_CHUNKED_PREFILL=0` turns it off) | a prompt longer than the engine's input length is prefilled in passes of at most that length, also under `--enable-context-reuse`; a prompt that fits is still one pass. The recurrent and convolution state carries over between passes | Qwen3.6 with a 12,288-token input limit and 65,536 KV capacity (INT8 KV): prompts of 16k and 32k tokens answered, needles 45/45 (2k to 32k, three runs, exact routing, reuse on); before, any prompt above the input limit was refused. Prompts that fit one pass take the same path as before | none | yes | [`rtx4060/hybrid-chunked-prefill-v2`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/hybrid-chunked-prefill-v2) ([e0839df](https://github.com/drakarah/TensorRT-Edge-LLM/commit/e0839df)) |
| Expert-cache fill grid sized by the possible misses (tokens x top-k) instead of the slot count | the fill kernel launched one block per slot even when at most 8 experts can miss | Qwen3.6: empty-call floor 137 -> 19 us per layer; decode 45.8 -> 49.7 tok/s with exact routing | none | yes | [`rtx4060/moe-cache-aware-routing`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-cache-aware-routing) ([0edb7ee](https://github.com/drakarah/TensorRT-Edge-LLM/commit/0edb7ee)) |
| MoE token tiling (`EDGELLM_MOE_TOKEN_TILE=4096` at engine build) | stages the experts once per layer as before, then runs top-k, expert GEMMs and aggregation over tiles of 4,096 tokens, so the workspace holds one tile instead of the whole 12,288-token pass | Qwen3.6 12k engine: 55 -> 64 cache slots per layer, decode 60.3 -> 63.3 tok/s (+5%), replay-20 flat (76.8 vs 77.5 s) | MoE scratch at 12k input about 810 -> 270 MiB (arithmetic from the buffer sizes, not read from a profile) | yes (Qwen3.6) | [`rtx4060/moe-token-tiling`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/moe-token-tiling) ([e25265a](https://github.com/drakarah/TensorRT-Edge-LLM/commit/e25265a)) |
| Lazy per-tensor dtype fix and compaction of the ONNX external data at export | the exporter converted all tensors of a 35B model in one pass | not a speed change: lets the 35B export run in 26 GB of RAM + 42 GB of swap | export peak memory | yes (Qwen3.6) | [`rtx4060/export-lazy-dtype-fixup`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/export-lazy-dtype-fixup) ([487b30b](https://github.com/drakarah/TensorRT-Edge-LLM/commit/487b30b), [312e391](https://github.com/drakarah/TensorRT-Edge-LLM/commit/312e391)) |
| **Qwen3.6-35B-A3B (hybrid GDN + attention mixture of experts)** | | | | | |
| INT8 KV cache for Qwen3.5/3.6 (`EDGELLM_INT8_KV_SCALES`, scales calibrated by `poc/tools/calibrate_int8_kv_layerwise_qwen36.py`, only the 10 full-attention layers own a KV cache) | same INT8 KV as the Gemma 4 export | RTX 4060: replay-20 99.1 -> 96.3 s (-2.8%) with prefill overlap, decode unchanged; needles 89/90 | KV of the full-attention layers stored as INT8 | yes (Qwen3.6) | [`rtx4060/qwen35-int8-kv`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/qwen35-int8-kv) ([8890df2](https://github.com/drakarah/TensorRT-Edge-LLM/commit/8890df2)) |
| Prefix reuse for non-speculative hybrid models (`--enable-context-reuse`) | checkpoints the recurrent state at the shared-prefix point so a plain hybrid engine reuses a prefix (before, only the MTP path had a working checkpoint) | RTX 4060: a request sharing a 3,456-token prefix takes 1.5-1.8 s instead of 3.0-3.3 s; **but** each checkpoint split costs an extra expert-streaming pass, replay-20 77.5 -> 100.0 s on independent requests, so it stays off for the replay. On an agent-style session (a website built with file tools and five follow-ups, prompts growing from 600 to about 60,000 tokens, 64k engine, one run each) it cut the session from 1,253 s to 721 s (-42%), 33.9 -> 20.7 s per 1,000 output tokens, and the median short tool-call turn from 19.0 s to 2.7 s | recurrent snapshot pool | no (opt-in) | [`rtx4060/hybrid-prefix-reuse`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/rtx4060/hybrid-prefix-reuse) ([6c768ce](https://github.com/drakarah/TensorRT-Edge-LLM/commit/6c768ce), [0c79b34](https://github.com/drakarah/TensorRT-Edge-LLM/commit/0c79b34)) |
| Layer-wise GPTQ of the routed experts and of every dense linear (builder step, `poc/tools/gptq_experts_layerwise_qwen36.py`, `poc/tools/gptq_dense_layerwise_qwen36.py`; not a fork change) | one decoder layer at a time, calibration on generic text; dense = shared expert, lm_head, GDN projections, attention | dense INT4 for all linears: replay-60 682 -> 397 s against FP16 dense; HumanEval 155/164 against 150-156 for the other variants; needles 80/80 at 2k and 4k | the 4.4 GB of FP16 dense weights shrink to INT4, which is what leaves room for 60 cache slots per layer | yes (Qwen3.6) | this repo |
| **Build and tooling** | | | | | |
| Timing cache and averaged timing iterations for the engine builder | `EDGELLM_TIMING_CACHE` makes tactic choices reproducible | averaging iterations did not help; a timing cache reproduced a build exactly. Build-to-build decode variance of about 7% was traced to tuning at the maximum tree: build with the maximum equal to the served tree (12) | none | yes (via build settings) | [1c44945](https://github.com/drakarah/TensorRT-Edge-LLM/commit/1c44945) |
| NVTX ranges for request setup; kernel unit tests (INT8 KV, XQA INT8 tree verify, dense head-512 prefill accuracy) | profiling and tests | none | none | n/a | [6c7120c](https://github.com/drakarah/TensorRT-Edge-LLM/commit/6c7120c), [84bf152](https://github.com/drakarah/TensorRT-Edge-LLM/commit/84bf152), [d09afb2](https://github.com/drakarah/TensorRT-Edge-LLM/commit/d09afb2), [8b35109](https://github.com/drakarah/TensorRT-Edge-LLM/commit/8b35109) |

</details>

<details>
<summary><b>What did not work (negative and neutral results, all models)</b></summary>

**Qwen3.6-35B-A3B (RTX 4060)**

* **A static 256k KV pool costs 27% of the decode speed.** Its 2.5 GiB of KV leaves 16 expert slots per layer instead of 42 (59.1 against 81.4 tok/s, METRICS R25eu); the growable pool is the answer (table above).
* **Trimming the KV mapping too eagerly or too lazily both cost.** With a trim window of 32 requests one 222k-token request left decode at 65.6 tok/s afterwards (the mapping stayed large, the expert cache small); trimming immediately cost about 2.2 s each time a long and a short request alternated. The default is 4 requests / 300 s; that setting was chosen from these two ends and was not run as an arm of its own.
* **Growing the KV cache under context reuse crashed first.** The context cache's circular page free list handed out pages above the mapped range (illegal memory access in the RoPE kernel); fixed by a lowest-first page allocator with a usable limit. A guard that disabled growth whenever reuse was on was written and dropped, because the combination has to work.
* **Qwen3.6's own MTP head (speculative decoding) loses badly with streamed experts.** Decode 45.8 -> 17.5 tok/s (-62%), replay-20 110.7 -> 213.2 s, cold prefill of a 3k prompt 1,358 -> 1,080 tok/s; a re-test on the current build with exact routing gave 19.1 tok/s decode (replay-20 184.7 s), so the loss is structural. A verify pass touches the experts of 4 tokens (20.3 distinct experts per layer against 8 for one token in a routing trace) and cache fills fall only 5% per token at 4 tokens, so every miss still crosses PCIe, and the draft layer takes cache slots. The one useful by-product was that prefix reuse works with it (warm time to first token 129 ms against 2,160 ms), which is why plain hybrid models got their own checkpoint path. MTP with the cache-aware bias crashes (illegal memory access on the first request, under investigation), so that combination is untested.
* **Hybrid chunked prefill with small chunks is a loss on this MoE; with the engine's own input length it is free.** A 1,024-token-input engine got 37 cache slots per layer instead of 14 and decode 28.4-28.7 against 22.0-23.5 tok/s, but replay-60 went 682.4 -> 815.2 s (+19.5%); a 4,096-token-input engine with chunking (75 slots) made replay-20 110.7 -> 122.3 s (+10.5%) for +0.4% decode. Each chunk touches nearly every expert, so a chunked pass streams the non-resident experts again once per chunk (inferred, not profiled). Chunking at the full input length (12,288) only affects prompts that would otherwise be refused, and is on by default for that reason (see the table above). It pays for the dense hybrid (see Qwen3.5-4B).
* **The first design of CPU compute for expert-cache misses was parked; the second one is on by default (table above).** The host round trip alone (router ids to the host plus an empty stream-blocking host node, no CPU math) cost +3.9 ms per token for 40 layers (about 98 us per layer): decode 45.9 -> 38.9 tok/s, and the arithmetic at the time left a net of about +4%. What paid later was handing the misses to spinning threads through a mapped mailbox from a GPU kernel, with no host synchronisation (decode 62.5 -> 75.9 tok/s, +21%).
* **Further tuning of the CPU miss kernel moved nothing in the engine.** A software prefetch of the next k-group left the gate_up phase at 71.4 -> 70.0 us and decode at 77.3 -> 77.0 tok/s. A flat split of the gate_up phase cut the host time per doorbell from 126 to 113 us but decode stayed at 77.4 -> 77.5 tok/s. Twelve workers on both hardware threads of the six performance cores were 3% slower (74.7-76.7 against 77.0-77.6 tok/s) although the best arm of the microbenchmark: the barrier waits show stragglers. The box reads 55-63 GB/s from DRAM at best and a doorbell reaches about 29 GB/s (inferred); the headroom to free misses is 8-10% (82-84 tok/s).
* **The GPU's own memory compression and entropy coding give almost nothing on INT4 experts.** Compressible allocations are granted (a block of zeros reads at 2,871 GB/s against 258 GB/s), but real INT4 weights read at 256.5 against 258.4 GB/s. The nibble entropy of the experts is 3.32 of 4 bits, so even free software decoding would save at most 17% of the expert bytes, on kernels that already run at 80-86% of the DRAM roof (dense INT4 GEMV 223 GB/s = 86%, Marlin MoE 207 GB/s = 80% of 259.6 GB/s).
* **A different number of expert-cache slots per layer instead of a uniform count gives nothing at the serve defaults.** Slots split by the miss curves of each layer (experiment, not in the fork's dev) gave +3.6% decode at 40 slots on average and +1.5% at 45 with the CPU path off, and nothing with it on (77.1-77.4 against 75.9-78.8 tok/s at 40; 76.9-78.4 against 75.6-78.4 at 45).
* **Routed chunking per token tile is worse than staging once per layer.** With a pass of several tiles each tile copied its own routed experts, so an expert shared by tiles was copied again. Time to first token, plain staging against routed per tile: 7,012 tokens 2.71 -> 2.73 s, 11,012 tokens 4.04 -> 4.46 s (+10%); a smaller tile made it worse (tile 1,024: 3,012 tokens 2.45 s against 1.63 s). Plain staging costs about 1% for tile 2,048 against 4,096. So routed chunking applies only to passes that fit one tile.
* **The whole-layer prefill overlap is copy-bound and needs room for its buffer.** It copies 15.5 GB per pass whatever the prompt length (1.8 s at the measured 8.6 GB/s host-to-device rate, measured with a pinned-memory copy), so it loses below about 4,200 tokens (table above). With a fixed cache size that leaves no 396 MiB free the second buffer cannot be allocated and the plugin silently falls back to plain staging with one warning per pass; sweeps with overlap on at a fixed 61 slots therefore measured nothing.
* **Serve-time flags that do nothing for this model on this card.** On the dense-INT4 engine (reference: decode 45.8 tok/s, replay-20 110.7 s): `EDGELLM_INT4_GEMV_MAX_M` 2 / 8 gives 45.8 / 45.9 tok/s and 113.6 / 110.8 s; the W4A8 prompt GEMM (`EDGELLM_INT4_W4A8_MIN_M=64`) 45.8 tok/s, 113.6 s (it costs ~12 MiB of VRAM); `EDGELLM_MOE_SMALL_THRESHOLD=16` 45.6 tok/s, 109.3 s; PDL 45.7 tok/s, 110.0 s. Sequential instead of chunked GDN prefill: prefill 1,331 against 1,358 tok/s, replay-20 112.5 s, so the chunked kernel is worth about 2% here. All inside the ~2 s replay-20 noise.
* **INT8 / FP8 activations through the whole MoE are quality-neutral, but the W4A8 Marlin kernel was not built.** Fake-quantised in the full-precision stream on 32 x 1,024 held-out tokens (exact NLL 1.47822): INT8 expert input -0.00005, INT8 input plus INT8 SwiGLU intermediate -0.00098, per-128-channel intermediate +0.00028, FP8 E4M3 input -0.00081, FP8 input plus intermediate +0.00022 (all within the ~0.001 noise). Not built because tensor math is off the critical path here (prefill is bound by the 1.9 s expert copy, decode by weight reads) and the gain would be about 350 MB of MoE scratch at 12k tokens (inferred).
* **GPTAQ (residual-corrected GPTQ) for the dense linears is worse than plain GPTQ.** Hidden-state deviation after layer 11 (held-out): GPTQ 0.1171, GPTAQ 0.1968 (damping 0.01) and 0.1991 (damping 0.10); per-layer GPTAQ/GPTQ ratio 1.14 at layer 6, 1.44 at layer 9, 1.68 at layer 11. A tenfold damping does not help. Probable cause: the paper's gains come after a rotation that removes outlier channels, which this unrotated residual stream does not have (inferred).
* **Gate-weighted (affinity) Hessians for the expert GPTQ give no held-out gain.** On layers 0-7 with 32 of 256 calibration sequences held out, held-out error is within 0.003 of plain GPTQ in every layer and slightly worse in 4 of 8; the lower error on the calibration data is fitting to it. Plain GPTQ's held-out error is 10-35% above its calibration error, so calibration size or variety is the open lever (inferred).
* **Cache-aware routing beyond beta 1 breaks the model.** Perplexity on 32 x 1,024 held-out tokens (exact 4.385): beta 2 gives 5.425 (+23.7%) at 0.6% cache misses, beta 4 gives 30.6 (x7) at 0.1%.
* **A DMA copy engine does not beat the zero-copy fill kernel for decode misses.** Pinned `cudaMemcpyAsync` reaches 8.1-8.6 GB/s at any size; the in-kernel fill reaches 11.5 GB/s for one expert per launch (11.3 / 9.9 / 8.8 / 8.0 / 7.7 for 2 / 4 / 8 / 32 / 256 experts). DMA is only 12% faster for whole-layer copies (8.57 against 7.66 GB/s), so it helps prefill, not decode; the decode fill at 9.4 GB/s is already ~82% of what the kernel can reach, at most ~1.8 ms per token to gain.
* **llama.cpp tuning moves nothing beyond noise.** On top of the `--n-cpu-moe` sweep: pinning 6 threads to the performance cores, 12 pinned threads and 8 unpinned threads give first/second decode of 27.6/20.1, 24.4/17.4 and 19.0/16.3 tok/s at `--n-cpu-moe 40` (24.8/22.5, 35.1/22.8, 35.4/20.9 at 30); a repeated run of the first variant gave 20.4/16.8. The first decode after a load is fast, later ones are 17-23 tok/s whatever the setting; 28 and 24 layers on the CPU do not start for lack of VRAM.

**Gemma 4 26B-A4B (RTX 4060 and Orin)**

* **MTP loses when the experts are streamed (RTX 4060).** With the INT4 dense part and no cache, tree 8 decodes 7.5 tok/s (prose) / 8.1 (probe) against 14.0 for plain decode, and replay-15 takes 242.3 s against 187.5 s; with a 36-slot expert cache and a 3-token verify tree it is still 20.4 / 24.1-25.1 against 37.9-38.9 / 46.5 tok/s, and replay-15 126.3 s against about 113 s (estimated from the 60-call replay). A verify step of k tokens selects up to 8k distinct experts per layer (24 for 3 tokens) and evicts the cache, while only ~1.5 tokens are accepted per round. The speculative base engine also did not fit with the FP16 dense part (6.98 GB of weights against 5.53 GB, `cudaMalloc` out of memory).
* **Letting the Marlin kernel read experts over PCIe directly was the wrong design (RTX 4060).** Decode 9.3 tok/s and prefill about 360 tok/s; staging only the routed experts through VRAM gave 11.27 tok/s and 1,228-1,600 tok/s (ceiling for decode over PCIe: 18 tok/s).
* **INT8 Q.K^T gave nothing on the 26B (RTX 4060).** Prose decode 37.6-39.0 against 37.7-39.0 tok/s without it, long-context decode 48.5 / 49.3 / 47.5 against 48.7-49.4 / 48.8-49.3 / 47.3-47.6 tok/s at 2k / 4k / 7.5k, needles 5/6 against 6/6 (one 2/3 in one run). The macro belongs to the speculative-verify kernel and the plain one-token decode probably never takes that path (inferred): "no effect observed", not proven.
* **More expert-cache slots can make decode slower (RTX 4060).** Prose decode 43.0-44.5 tok/s at 44 slots, 41.2-42.4 at 46 and 39.2-40.2 at 48: the cache is allocated per layer at the first decode step and a layer whose `cudaMalloc` fails silently falls back to staging (28 of 30 layers had a cache at 48 slots). 44 is the maximum on this card.
* **A static hot-expert set is workload specific (RTX 4060).** The 36 most-used of 128 experts per layer cover 65.1% of the decode reads on the replay; the set chosen on the replay covers only 52.5% of the reads on HumanEval code (77.0% for its own set), 25 points lower. A per-layer LRU cache was used instead.
* **A pool for the host prefix cache's pinned buffers did not help (RTX 4060).** Replay-60 325.2 s against 326.0 s without it, inside the ~1% noise and below the 2.5% ceiling estimated from 163-187 `cudaMallocHost` calls of ~35-40 ms; not merged.
* **Dense INT4 costs exact retrieval (Orin and RTX 4060).** With the GPTQ dense part, eight 2k-token prompts with three planted facts each find 22 of 24 facts (6 of 8 prompts complete) against 24/24 for FP16 dense on the Orin; on the 4060 the same test gives 20/24 (FP16 KV) and 22/24 (INT8 KV). At 3k tokens it is 24/24 again and HumanEval does not see it (155-157 against 158), while the speed is real (Orin: generation +33%, replay-60 -13%). Keeping attention in FP16 and quantising only the MLP gate/up and the lm_head removes the loss (158/164, 8/8 at 2k and 3k) and keeps generation +11%, replay-60 -4.9%.
* **The W4A8 prompt GEMM made the dense-INT4 retrieval worse (Orin).** On top of dense INT4: 20 of 24 facts (4 of 8 prompts) at 2k and 23/24 at 3k, against 22/24 and 24/24 without it, for replay-60 -14.7% instead of -13.3%. Not adopted.
* **GPTQ and RTN are indistinguishable for the 26B dense INT4 (RTX 4060).** HumanEval 156 against 155 of 164 (FP16 dense 158), identical speed; GPTQ stays because it costs nothing at serve time (5.5 min of calibration).
* **Marlin MoE block sizes and a register cap (Orin).** Reference decode 29.3-30.9 tok/s, prefill 1,639-1,644 tok/s. A 16-row block for prefill: 1,410 tok/s (-14%); 48 rows: decode 24.6, prefill 1,584; 64 rows: decode 17.0, prefill 1,200 (-27%). Compiling the kernels with `__launch_bounds__(128, 3)` (168 registers, 550 B of spills) gave 15.1 / 14.4 tok/s decode and 739 / 726 tok/s prefill; with the 8-row block it was neutral (32.4-33.5 against 33.8-34.9). A forced 256-thread config is invalid for N = 1,408 (HTTP 500). Why the 48- and 64-row kernels lose was not measured.
* **INT8 KV is not a long-context decode win on the Orin.** Replay-60 improved 234.0 -> 218.2 s (-6.8%), but single long-context probes fell from 38.9 / 44.1 / 41.6 to 37.6 / 36.2 / 34.5 tok/s at 2k / 4k / 7.5k (-17 to -22% at 4k and above), while the 4060 gained +10 / +15 / +21% there. Probably because the KV is already read from fast unified memory and the dequantisation adds work (inferred, not profiled).
* **llama.cpp expert offload knobs on the 26B (RTX 4060, baseline).** Decode 38.3 tok/s and prefill 420 tok/s at `--n-cpu-moe 21`; 12 / 14 / 20 threads 38.7 / 39.5 / 41.1 tok/s; `--no-op-offload` (host experts computed on the CPU) 34.7 tok/s and 130 tok/s prefill (40.2 s for a 5.2k prompt against 12.45 s); `-ub 4096` or 8192 does not start. Prefill wants big micro-batches (a large ubatch with N 30: 1,312 tok/s) and decode wants experts on the GPU, so no single setting wins: every combination came to 8.3-8.9 s per application call.

**Gemma 4 12B**

* **Split-KV count is not a lever for the 12B decode attention (Orin and RTX 4060).** Orin: forcing 4 splits on the sliding layers moved a launch 150 -> 156 us, 3 / 6 splits on the target's global layers 445 -> 441 / 482 us, cold decode stayed inside +-1 tok/s (31.5 against 32 / 30.8 / 30.8 for 32 / 48 / 64 CTAs). RTX 4060 at 24k tokens: global-layer launch 621 us at 3 splits, 646 / 691 / 846 us at 6 / 12 / 24 (round 35.97 -> 38.03 ms). Launch time does not scale with positions per block; the head-512 decode attention is tensor-bound at about 20 TFLOPS, 67% of the 30.2 TFLOPS FP32-accumulate peak (inferred from the arithmetic, the profiler capture failed).
* **The Orin's verify-tree and tile tuning mostly vanish on the RTX 4060.** Tree 16 instead of 20: replay-60 139.3 / 139.4 s against 141.3 / 142.5 s (-1.5%) against -16.6% per token on the Orin (hypothesis: the 4060 decode GEMM is already near its read roof). W4A8 with 4 pipeline stages: prefill 2,353 against 2,353 tok/s, because the kernel is tensor-bound at 79% of the INT8 peak (95.5 of 120.8 TOPS). Split-K 3 / 5 variants: decode round -0.7%.
* **The GELU plugin buys little on the RTX 4060.** The generated kernel was already near-efficient on sm_89: 324.7 -> 268.6 us per call (-18%, against -65% on the Orin), prompt passes -4.4%, wall time per output token -0.8% to -1.2%.
* **The RMSNorm plugin is neutral (Orin).** A plugin for one norm plus the optional residual add: decode round 59.44 -> 59.57 ms (+0.2%), prompt passes 36.4 -> 36.1 s, total kernel time +0.9%. The norm family costs 1.82 s of 75 s; the plugin saves about 0.8 s of that, because TensorRT already fused norm, residual add and the next norm into one launch (190 us) while the plugin runs one launch per norm. Not merged.
* **The decode INT4 GEMM has no headroom left in tuning, on either card.** RTX 4060: 190 configurations (tile 16 / 32 rows, split-K 1-6, stages 2-6) on the four 12B decode shapes: the five best of gate+up and of down lie within 0.2% and the engine's picks are the optimum; the whole decode GEMM runs at 92% of the measured 259.6 GB/s read floor (about 2 ms of the 34.4 ms round at most). Orin: stages 5 / 6 give -3.5% on the down shape and 0-0.5% on the other three (about 0.6% of a round), so no 5- or 6-stage variants were baked.
* **A register cap on the decode INT4 GEMM (Orin).** `min_blocks_per_mp` 5 / 6 (96 / 80 registers, 5-6 CTAs per SM): best gain 1-2%, inside the 2.7% drift between sessions, many capped variants much slower (down shape 235 -> 250-270 us), and one (split-K 5, 4 stages, cap 6) produces wrong output on K = 15,360 (max abs diff 10.94, 5.9% relative) while its time looks normal. Not used.
* **W4A8 tiles and stages beyond 64x128x64 / 4 stages (Orin).** Gate/up 8.23 ms at 4 stages; 5 and 6 stages never beat 4 and stage 2 is ~1.75x slower; 128x128 or 64x256 tiles with 128 threads take 255-370 ms (40 registers, 109 M local loads and stores); a 256-thread 128x128 tile 9.12 ms (+11%); 32x128 11.33 ms; a 2x2x1 warp layout with 24% fewer shared-memory wavefronts is 4.2% slower because it executes 4.2% more instructions. Kernel time followed instructions per MMA, not wavefronts or bank conflicts.
* **A shiftless INT4 dequantisation is not worth building (Orin).** An upper-bound probe in the real decode GEMM (dequantisation instructions removed, bias assumed free, results wrong on purpose) gave no speed-up: 168.9 -> 185.1 us on the E4B gate/up shape and 379.4 -> 388.3 us on the 12B's; the kernel is DRAM-bound at 88% of the roof, and after paying for the bias the net instruction saving is only about 5%.
* **No 2-bit decode beats INT4 on the Orin.** A bench kernel (M = 16, correctness-checked against the same decode rules, random weights): INT4 streams 328 G weights/s (93% of the DRAM roof); a QTIP-style table-free trellis 164, a shared-memory vector-quantisation table 213 (bank conflicts: 5.8 wavefronts per load), the cheapest imaginable pair trellis 329 (a tie at half the bytes). Even a free decode would reach only 569-600 G weights/s, 1.7x. Trellis formats are also lossy, so nothing was built.
* **Lossless entropy coding of the INT4 weights can save at most 12% (RTX 4060 measurement, Orin sample agrees).** All 96 INT4 FFN tensors of the 12B: nibble entropy 3.517 bit per weight, so at most 513 MB of 4,246.7 MB (12.1%); per-tensor Huffman 11.6%; modelling the neighbouring nibble gains nothing; the scales could save at most ~70 MB. Variable-length decoding needs a table lookup per symbol, which does not fit the 2-instruction decode of the INT4 kernel (inferred from the 2-bit bench above); not built.
* **An s4 tensor-core decode GEMM is parked (Orin).** The hardware has `IMMA.16864.S4` at 106.2 TOPS (4x FP16), but after the tree-16 fix the decode GEMM runs at 85% of the 186 GB/s roofline (39.2 ms of a 60.7 ms round for 6.2 GB), so any GEMM idea can win at most 5.9 ms (10% of the round).
* **A static 40-node verify tree is no better than 20 nodes (RTX 4060).** Prose 58.7 against 63.4 tok/s (-7%) because every round verifies 40 positions, and on fresh code-edit runs 163.6 / 161.5 / 161.7 against 166.1 / 166.6 tok/s. Only the variable size (20 nodes unless a long suffix chain is proposed) helped: 172.7 tok/s.
* **Deeper or wider MTP trees lose on the 12B too (RTX 4060).** On the tree-24 engine: 6x3/20 63.0 tok/s, 6x4/20 55.2, 4x4/20 55.1, 8x2/20 57.4, 8x2/24 57.9, 8x3/24 56.2; each draft step reads the whole drafter, so extra depth and width cost more than the acceptance they add (±2 tok/s run-to-run, single prompt).
* **MTP chain mode produced garbled answers on the 12B (RTX 4060).** Chains 1x3/4 and 1x2/3 dropped spans of tokens mid-answer ("...bandwidth,2. Analyze the3. Check for high4. Monitor the..."), while tree mode reproduces the INT8-KV model. The chain verify path was not investigated and is not used.
* **A 128k-token context does not fit in 8 GB (RTX 4060).** A 131,072-position engine fails at load with `cudaMalloc` out of memory: the global-layer KV alone is 1,024 MiB (+768 MiB over 32k), so about 8.4 GB would be needed against 8.19 GB, with the 32k engine already at 7,647 MiB. Roughly 64-80k positions fit on paper. Every lever that would make room costs something: INT4 global KV (-512 MiB, quality risk), INT3 (see above), a reduced vocabulary (-450 MiB, rejected on the E4B), no drafter (-360 MiB, decode 63 -> 37 tok/s).
* **A pooled pinned-memory allocator for the prefix-cache snapshots changed nothing (RTX 4060).** Replay-60 wall per output token -1.4% (170.3 / 170.4 s against 172.0 / 169.8 s with a 1 GiB cache), inside the noise; with a 16 GiB budget nothing is evicted, so the pool is never used. The 9% GPU idle came from the per-chunk snapshots themselves, which the capture interval (fewer snapshots) addressed instead.

**Gemma 4 E4B**

* **Round-to-nearest requantisation to group 128 cost application tests.** Of 53 tests that regressed against llama.cpp, Google's QAT group-32 checkpoint as shipped passed 26, an RTN requantisation to group 128 with a bf16 head 13, and RTN group 128 with an INT4 head 5; GPTQ group 128 with a bf16 head passed 27, and a GPTQ-calibrated INT4 head cost no answers (71 tests: 51 against 49 with the bf16 head). Both conversion steps cost quality, which is why every checkpoint here is calibrated.
* **The Orin's tuning is flat or worse on the RTX 4060 (E4B, replay-60 68.3 s base).** `EDGELLM_INT4_GEMV_MAX_M` 0 / 1 / 8: 68.4 / 68.0 / 68.4 s; all 12-node tree shapes 67.8-69.5 s; a maximum tree of 16: 67.7-68.8 s; of 24: 70.2-71.9 s (3-5% slower); pure chains 1x5/6 76.8 s and 1x4/5 79.2 s; suffix draft length 3: 70.2 s, 6 / 10 / 16: 68.3 / 68.0 / 68.9 s. A larger prefix-cache pool is worth under 1% (60% of prompt tokens are already reused, an unlimited cache would reuse about 66%). Forcing the host's real CUDA driver instead of the container's compatibility layer: 68.4 s either way.
* **A pure chain tree loses on prose (Orin).** 1x4/5 matched the 2x3/8 tree on the recorded JSON-schema calls (297.1 against 294.2 s) but lost 9% on prose (52.5 / 50.7 against 56.8 / 56.1 tok/s).
* **Bisecting the W4A8 answer-length lean by operation class found nothing.** Enabling W4A8 for one class at a time (replay-60, paired by call): gate+up +2.6% tokens (11 longer / 8 shorter), qkv +4.5% (15 / 11), q only +0.6% (7 / 7), o plus the per-layer projection +1.4% (14 / 11); only all together was significant (22 longer / 9 shorter, then 21 / 6, then 16 / 12 in a third pair, so the 60-call metric is noisy; 59 / 27 pooled). The lean is workload dependent (HumanEval mean lengths 414 against 409 tokens for all-ops W4A8).
* **The RMSNorm plugin is neutral on the E4B too (Orin).** Decode step 24.6 against 24.7 ms, prompt passes 12.4 against 13.0 s, replay-60 107.2 / 106.4 against 108.2 / 110.1 s; TensorRT already fuses norm, add and norm.
* **A graph of the whole draft chain gave nothing outside the profiler (Orin).** The six draft steps leave about 0.7 ms of GPU idle per round (~2%); capturing them as one graph showed no measurable gain in the replay.
* **Deep split-K variants (8 and 16) for the INT4 decode GEMM (Orin).** Replay-60 117.55 against 116.25 s (+1.1%) and cold decode 80.9 against 83.1 tok/s: the tiny-N launches improve in isolation but they are launch-latency bound.
* **Nothing to win in the E4B drafter (Orin).** The sparse candidate head is 81 us per round (0.3% of the round) and the drafter's layer GEMVs 0.84 ms per round (3.2%), mostly at the launch floor, so INT4 could save under 0.2 ms (<1%); the INT4 head and INT4 MLP drafter are 12B items.
* **K smoothing for INT8 Q.K^T attention changed almost nothing.** Subtracting the K mean (SageAttention-style) moved the relative error 0.40 -> 0.37%, and a single static per-head K scale gave 0.42-0.60%, so smoothing was dropped (Gemma's Q/K norms keep the channels tame) and K is read straight from the INT8 cache. INT8 or FP8 P.V was rejected: sm_87 has no FP8 MMA and the O side is not the bottleneck.
* **The newest toolchain changed nothing measurable.** TensorRT 11.3 and then everything newest (CUTLASS DSL 4.8.0, cupy 14.2, onnx 1.23): replay-60 135.5 -> 134.8 -> 134.5 s, prompt 1,848 / 1,808 / 1,856 tok/s, 71-test suite 52 / 50 / 50 (spread 45-52). Newer `transformers` (5.15+) breaks the Gemma 4 export (the config no longer exposes the global head size, and the assistant silently gets head size 256 instead of 512).
* **Pure-spin barrier waits in the XQA decode kernel are not worth a flag (Orin).** Replacing the `nanosleep` back-off gains 5.7% on the sliding-window and 6-8% on the global attention kernels, about 0.3 ms of a 29 ms round; left off.
* **Overlapping decode with prefill is not pursued (RTX 4060).** In the 396 recorded calls none starts while another is running (318 of 395 gaps between a response and the next request are under 50 ms); the engine is batch 1, in-flight batching refuses speculative decoding at startup, and a second execution context would not fit (7,577 of 8,188 MiB used).

**Qwen3.5-4B**

* **The first chunked GDN prefill kernel was slower than the sequential one.** 2,048 tokens: 10.7 against 9.1 ms per layer call; 8,192 tokens 42.9 against 37.0 ms (a column-serial triangular solve and 57% long-scoreboard stalls). The blocked inverse, cp.async and 2 CTAs per SM version reached 3.6 and 14.6 ms. Computing the Gram matrices per k head instead of per v head was no better (3.72 against 3.63 ms).
* **Chunked GDN prefill is worth much less on the RTX 4060 than on the Orin.** Prefill 3,388 -> 3,483 tok/s (+2.8%) and replay-60 119.0 -> 117.1 s (-1.6%), against +13.5% prefill on the Orin; the INT4 prefill GEMMs dominate there (inferred, not profiled).
* **Tree drafting stays off.** Orin: replay-60 284.3 s for a 2x5/12 tree against 292.8 s for the chain, probe 48.8 against 49.7 tok/s. RTX 4060 with the same full-vocabulary head: 116.0 s (2x5/12) and 118.6 s (2x3/6) against 118.1 s for the chain. The tree export cannot use the reduced draft vocabulary, which is worth 13% (102.5 against 118.1 s).
* **A hybrid model could not take a prompt longer than its engine input length (fixed).** The input limit used to be the longest prompt: with a 32,768-token KV capacity, input limits of 16,384, 24,576 and 32,768 do not load on 8 GB (`cudaMalloc` out of memory; the prefill scratch is about 60 KB per input token, 7.9 GB at 131,072), 12,288 does; on the Orin an input limit of 4,096 refused 16 of the 60 replay requests. Chunked prefill for hybrid models (table above) removes the limit: a 12,288-token input engine with 65,536 KV capacity answers 32k-token prompts. The E4B reaches 30,000 tokens with a 2,048-token input limit thanks to chunked prefill.

**All models / tooling**

* **A standalone variant sweep mispredicted the in-engine best (Orin).** Forced into the engine, the standalone-best variants were slower than the autotuner's picks: qkv 118.0 against 112.2 us (the standalone benchmark had said 106 against 119, an 11% error) and gate/up 393.3 against 384.3 us (a 5% error with the sign reversed). TensorRT's own timing agreed with the engine. A sweep is a candidate generator; the decision needs a trace of the engine in the same session.
* **Buffer placement does not explain run-to-run kernel speed (Orin).** A pure read of the 16.2 MB (qkv) and 60.8 MB (gate/up) weights from 11 start offsets (16 B to 4 KiB alignment, positions up to 333 MiB) varied by under 0.5% (97.1-97.8 and 346.3-349.6 us).
* **Stock Edge-LLM's default speculative setting is slower than no speculation (Orin, E4B).** Default MTP (chain of 6, verify size 7) 19.2 tok/s, 3 steps / verify 4 24.0, 2 / 3 25.9, no speculation 29.0, and the tree flags used here (top-k 2, 5 steps, tree 12) 32.3 tok/s; the benchmark's stock arm uses the last.

### Earlier items (Jetson AGX Orin unless stated)


| negative result |
|---|
| Reduced output vocabulary (98k of 262k rows) for the E4B base model: 71-test suite 49 -> 41 passed and no speed gain, because the drafter proposed out-of-vocabulary tokens in 336 of 423 requests (rejected). |
| W4A8 as first shipped (all ops, one-tile flush): GEMM 0.96-1.00x of the FP16 kernel, replay-60 123.1 -> 127.2 s (+3.3% slower), 36 of 60 answers changed. |
| W4A8 with all ops (even after the kernel fixes): HumanEval 88.4% -> 85.4%; fixed by the per-op filter (90.2%), but an answer-length lean of about +3% on structured requests remains (significant in two of three paired replay runs; it could not be located or removed). |
| Dithered (unbiased) INT8 activation rounding to remove that length lean: lean unchanged (+2.1% tokens with the filter, +3.7% all ops). |
| W4A8 flush with a magic-number INT32 -> FP32 conversion: kernel +4.8% slower (4,356 -> 4,566 us) and 10% more instructions. |
| W4A8 hoisting the B-tile copy addresses: 2.2% fewer instructions but kernel +2.2% slower (3,744 -> 3,828 us). |
| W4A8 register cap for higher occupancy (`min_blocks_per_mp` 3 / 4): spills, and wrong results on a long K loop. |
| FP16-accumulate prefill GEMM on the Orin: no gain (26.6 vs 26.5 TFLOPS peak); only the RTX 4060 gains. |
| INT8-native paged prompt attention (INT8 K/V loaded and widened inside the kernel): head-512 kernel +21% slower (63.8 -> 77.2 ms), head-256 +74% slower. |
| 64-row warp-specialized head-512 kernel: 41.6 ms against 40.7 ms for the 32-row kernel, not faster. |
| INT8 Q.K^T for head 256 (sliding layers): 31.7 ms against 31.9 ms for the generic kernel, only at par. |
| MTP draft tree with only the top-1 chain expanded: at most 0.4% either way, noise. |
| Coalesced state access in the GDN verify and replay commit: bit-identical, standalone commit 9% faster, no change in the engine (71.8 tok/s and 188.9 s both). |
| Using the tree-replay kernel for the linear GDN commit: eats the verify saving, replay 0.7% slower. |
| Group-flush decode GEMM at engine level: -5% summed weight-GEMM time but the 16- and 32-wide launches got 5-9% slower and replay-60 moved only -0.4%. |
| Prompt lookup on the E4B: replay-60 123.1 -> 122.9 s (Orin); with an 11-token chain code edits -14% (RTX 4060, 287.7 -> 246.4 tok/s). |
| Wider or deeper trees: a 16-node verify tree costs about 2% by itself; 2x6/16 loses 0.8% over all 396 requests; Qwen draft chain length 3-6 is flat (at best -1.9%). |
| GEMM instead of GEMV for single-token decode (threshold 0): decode 45% slower (23.8 vs 43.5 tok/s). |
| Averaging builder timing iterations: no effect on decode or prompt speed; newer CuTe DSL 4.8.0 against 4.7.0: no gain. |
| INT3 MLP weights on the 12B (RTX 4060, fake-quant test): HumanEval 92.1% -> 75.0%. |

</details>

<details>
<summary><b>What is in this repo</b></summary>

## What is in this repo

| path | |
|---|---|
| `Dockerfile` | builds the optimised stack (and, with `WITH_STOCK=1`, stock Edge-LLM) from pinned commits for `PROFILE` / `GPU_ARCH` |
| `docker-compose.yml` | `serve` (default), `bench`, `compare` |
| `poc/run.sh`, `poc/lib.sh` | the pipeline: prepare (download export or quantise, build engine), serve, benchmark |
| `poc/config/` | engine and server settings per model and profile, each value with its reason |
| `poc/bench.py` | the timing harness and report; works against any OpenAI-compatible server |
| `poc/tools/` | the quantisation tools (layer-wise GPTQ for Gemma and Qwen3.6, INT8 embeddings, reduced draft vocabulary, INT8 KV calibration) |
| `poc/data/` | recorded workload, INT8 KV-cache scales, draft vocabulary maps, the Sharp chat template of the CyberTiel profile (Apache-2.0, with its notice) |

</details>

<details>
<summary><b>Results per machine and model (Jetson AGX Orin and RTX 4060, stock vs optimised)</b></summary>

## Results

Two rounds, medians. Long-context rows: prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts.

### Jetson AGX Orin 32 GB (GPU pinned at 930 MHz; fork `dev` 2e02476, the 26B at `ff5e1fe`)

**Caveats** (both machines)
* **The Gemma checkpoints were prepared with the same quantisation tools outside the container** (the from-scratch `gptq` path of the builder, which needs `HF_TOKEN`, was not run through it; the Qwen quantisation was). 
* **Qwen stock runs without prefix reuse** (upstream gap: stock refuses `--enable-context-reuse` on this hybrid model at start with "Hybrid context reuse requires at least one recurrent snapshot slot"; the fork's hybrid prefix reuse closes it). The Qwen AWQ checkpoint was calibrated with 64 samples (an attempt with 256 was killed by the kernel for memory on the Orin's shared 30 GB pool); the 4060 reuses that checkpoint.
* **The 12B stock engine was built for 12,288 tokens, and stock cannot go much further.** Chunked prefill (a long prompt run in several passes, each appended to the KV cache) is a fork feature and is **on in the optimised arm**: its 12B engine takes prompts in 1,024-token passes and holds 32,768 tokens on the Orin. Stock Edge-LLM has no chunked prefill, so a stock engine must take a whole prompt in one pass: its input limit is its whole context, and the activation memory of every pass grows with it. We built the stock engine for the 12,288 tokens the replay needs (same settings as the optimised arm otherwise) and did not try a larger stock build on the Orin. The long-context rows therefore show "refused / failed" for stock from 16,384 tokens: the table states it, it is not hidden.
* **The 12B warm rows at 11,500 tokens and above equal the cold ones on the Orin**: the host prefix cache is 4 GiB and keeps one snapshot of the bounded window per 1,024-token prefill chunk, so the three needle prompts of one length do not all fit; this is a cache-size effect, not a limit of the technique (fork branch `rtx4060/swa-prefix-capture-interval` keeps 2.5-3x more prompts; not part of this pin). Warm prefill at short lengths is slower than stock's for the 12B (the snapshot restore).
* **HumanEval uses a 1024-token answer cap.** On the E4B most of the optimised arm's lower pass count comes from answers cut off at that cap (failures not ending in a closed code block: stock 7 and 9, optimised 11 and 18 across the two rounds; a proxy, `finish_reason` is not recorded), and the optimised arm varies more between rounds (8 problems flip between its two rounds, 2 for stock). With a 4096-token cap an earlier A/B showed no truncation and mean lengths of 414 vs 409 tokens. We have not re-run the E4B pair with a larger cap.
* **Needle retrieval (3 planted facts) is 2 of 3 at several lengths on both arms** (stock 8,192 tokens, optimised 8,192 and up to 30,000 on the 12B); not investigated.
* The replay output lengths differ between the stacks (the stock stack pads structured JSON with long whitespace runs on the E4B and Qwen), so tokens per second is the fair speed number there and the replay seconds mix speed with answer length.
* **The 26B has no stock arm and its two engines differ per machine** (Orin: FP16 attention, dense MLP and lm_head, INT4 experts, FP16 bounded KV, MTP tree 16; RTX 4060: INT4 dense part, INT8 KV, experts streamed from host memory, no MTP), so the two 26B tables are not a hardware comparison. On the Orin the BF16 base checkpoint was copied from the 4060 box instead of downloaded. The decode rate in its long-context rows is a short answer drafted from the prompt, not a steady rate.

#### gemma4-12b
| | stock | optimised | speed-up |
|---|---:|---:|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | 740 | 1,012 | **1.37x** |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | 0.276 | 0.404 | **0.68x** |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | 27.2 | 43.0 | **1.58x** |
| Decode, same prompt again (warm), tokens/s (higher is better) | 27.1 | 49.3 | **1.82x** |
| Replay of 60 recorded requests, seconds (lower is better) | 557.2 | 256.2 | **2.18x** |
| Replay output tokens in total (the stacks do not answer with the same length) | 10,090 | 10,402 |  |
| HumanEval generation, tokens/s (higher is better) | 35.9 | 76.5 | **2.13x** |

HumanEval pass@1 (greedy, public benchmark), per round: stock 152/164 / 153/164; optimised 149/164 / 148/164.

Greedy answers identical to stock: 3/60 (mean text similarity 0.618). Quantised weights are the same; the optimised stack also uses an INT8 KV cache and fused kernels, so some answers differ in wording. Read the two answers of the first request in `results/`.

**Long context**
| prompt tokens | metric | stock | optimised |
|---:|---|---:|---:|
| 4,096 (actual 4,046) | cold time to first token, s | 5.69 | 4.10 |
|  | cold prompt tokens/s | 711 | 988 |
|  | warm time to first token, s | 0.29 | 1.04 |
|  | decode tokens/s at this context | 25.9 | 36.1 |
|  | needles correct of 3 | 3 | 3 |
| 8,192 (actual 8,184) | cold time to first token, s | 12.73 | 8.56 |
|  | cold prompt tokens/s | 643 | 956 |
|  | warm time to first token, s | 12.58 | 1.14 |
|  | decode tokens/s at this context | 24.8 | 38.9 |
|  | needles correct of 3 | 2 | 2 |
| 11,500 (actual 11,500) | cold time to first token, s | 18.66 | 12.51 |
|  | cold prompt tokens/s | 616 | 919 |
|  | warm time to first token, s | 18.57 | 12.45 |
|  | decode tokens/s at this context | 23.5 | 35.0 |
|  | needles correct of 3 | 3 | 3 |
| 16,384 (actual 16,376) | cold time to first token, s | refused / failed | 18.60 |
|  | cold prompt tokens/s | refused / failed | 881 |
|  | warm time to first token, s | refused / failed | 18.52 |
|  | decode tokens/s at this context | refused / failed | 33.8 |
|  | needles correct of 3 | refused / failed | 2 |
| 24,576 (actual 24,536) | cold time to first token, s | not reached | 30.11 |
|  | cold prompt tokens/s | not reached | 815 |
|  | warm time to first token, s | not reached | 29.99 |
|  | decode tokens/s at this context | not reached | 37.2 |
|  | needles correct of 3 | not reached | 2 |
| 30,000 (actual 29,964) | cold time to first token, s | not reached | 38.63 |
|  | cold prompt tokens/s | not reached | 776 |
|  | warm time to first token, s | not reached | 38.51 |
|  | decode tokens/s at this context | not reached | 31.2 |
|  | needles correct of 3 | not reached | 2 |

Long-context refusals / failures: stock 16384: RuntimeError no tokens streamed

#### gemma4-e4b
| | stock | optimised | speed-up |
|---|---:|---:|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | 1,403 | 2,423 | **1.73x** |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | 0.163 | 0.075 | **2.17x** |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | 31.7 | 89.8 | **2.84x** |
| Decode, same prompt again (warm), tokens/s (higher is better) | 32.2 | 111.3 | **3.46x** |
| Replay of 60 recorded requests, seconds (lower is better) | 511.0 | 103.7 | **4.93x** |
| Replay output tokens in total (the stacks do not answer with the same length) | 27,844 | 10,242 |  |
| HumanEval generation, tokens/s (higher is better) | 39.2 | 169.7 | **4.33x** |

HumanEval pass@1 (greedy, public benchmark), per round: stock 143/164 / 141/164; optimised 139/164 / 133/164.

Greedy answers identical to stock: 7/60 (mean text similarity 0.495). Quantised weights are the same; the optimised stack also uses an INT8 KV cache and fused kernels, so some answers differ in wording. Read the two answers of the first request in `results/`.

**Long context**
| prompt tokens | metric | stock | optimised |
|---:|---|---:|---:|
| 2,048 (actual 2,016) | cold time to first token, s | 1.55 | 0.86 |
|  | cold prompt tokens/s | 1,301 | 2,345 |
|  | warm time to first token, s | 0.14 | 0.07 |
|  | decode tokens/s at this context | 30.7 | 87.2 |
|  | needles correct of 3 | 3 | 3 |
| 4,096 (actual 4,088) | cold time to first token, s | 2.98 | 1.74 |
|  | cold prompt tokens/s | 1,372 | 2,348 |
|  | warm time to first token, s | 2.88 | 0.08 |
|  | decode tokens/s at this context | 29.6 | 130.9 |
|  | needles correct of 3 | 3 | 3 |
| 8,192 (actual 8,180) | cold time to first token, s | 6.27 | 3.61 |
|  | cold prompt tokens/s | 1,306 | 2,265 |
|  | warm time to first token, s | 6.12 | 0.09 |
|  | decode tokens/s at this context | 28.1 | 85.7 |
|  | needles correct of 3 | 3 | 3 |

#### qwen35-4b
| | stock | optimised | speed-up |
|---|---:|---:|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | 1,128 | 1,828 | **1.62x** |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | 1.237 | 0.053 | **23.13x** |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | 46.6 | 88.0 | **1.89x** |
| Decode, same prompt again (warm), tokens/s (higher is better) | 47.5 | 88.0 | **1.85x** |
| Replay of 60 recorded requests, seconds (lower is better) | 325.6 | 160.8 | **2.02x** |
| Replay output tokens in total (the stacks do not answer with the same length) | 9,521 | 13,144 |  |
| HumanEval generation, tokens/s (higher is better) | 64.5 | 118.0 | **1.83x** |

HumanEval pass@1 (greedy, public benchmark), per round: stock 129/164 / 129/164; optimised 129/164 / 130/164.

Greedy answers identical to stock: 12/60 (mean text similarity 0.623). Quantised weights are the same; the optimised stack also uses an INT8 KV cache and fused kernels, so some answers differ in wording. Read the two answers of the first request in `results/`.

**Long context**
| prompt tokens | metric | stock | optimised |
|---:|---|---:|---:|
| 2,048 (actual 2,007) | cold time to first token, s | 1.91 | 1.13 |
|  | cold prompt tokens/s | 1,052 | 1,779 |
|  | warm time to first token, s | 1.78 | 0.05 |
|  | decode tokens/s at this context | 43.7 | 88.3 |
|  | needles correct of 3 | 3 | 3 |
| 4,096 (actual 4,088) | cold time to first token, s | 3.79 | 2.20 |
|  | cold prompt tokens/s | 1,079 | 1,858 |
|  | warm time to first token, s | 3.63 | 0.06 |
|  | decode tokens/s at this context | 44.5 | 84.3 |
|  | needles correct of 3 | 3 | 3 |
| 8,192 (actual 8,183) | cold time to first token, s | 7.65 | 4.46 |
|  | cold prompt tokens/s | 1,069 | 1,835 |
|  | warm time to first token, s | 7.40 | 0.07 |
|  | decode tokens/s at this context | 43.3 | 77.6 |
|  | needles correct of 3 | 3 | 3 |

#### gemma4-26b (optimised arm only)
| | optimised |
|---|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | 1,556 |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | 0.821 |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | 34.4 |
| Decode, same prompt again (warm), tokens/s (higher is better) | 48.2 |
| Replay of 60 recorded requests, seconds (lower is better) | 246.0 |
| Replay output tokens in total (the stacks do not answer with the same length) | 9,842 |
| HumanEval generation, tokens/s (higher is better) | 60.8 |

HumanEval pass@1 (greedy, public benchmark), per round: optimised 160/164 / 157/164.

**Long context**
| prompt tokens | metric | optimised |
|---:|---|---:|
| 2,048 (actual 2,011) | cold time to first token, s | 1.29 |
|  | cold prompt tokens/s | 1,559 |
|  | warm time to first token, s | 1.16 |
|  | decode tokens/s at this context | 29.2 |
|  | needles correct of 3 | 3 |
| 4,096 (actual 4,092) | cold time to first token, s | 2.61 |
|  | cold prompt tokens/s | 1,565 |
|  | warm time to first token, s | 1.30 |
|  | decode tokens/s at this context | 50.1 |
|  | needles correct of 3 | 3 |
| 7,500 (actual 7,496) | cold time to first token, s | 5.13 |
|  | cold prompt tokens/s | 1,462 |
|  | warm time to first token, s | 1.02 |
|  | decode tokens/s at this context | 75.0 |
|  | needles correct of 3 | 3 |

### RTX 4060 8 GB (`PROFILE=dgpu`; GPU clock locked at 2,460 MHz; fork `dev` 2e02476, the 26B at 481a34c content)

**Stock Edge-LLM did not run on this card for any of the three models** (same settings as the optimised arm): the stock server builds its engine and then fails at start with `CUDA runtime error in cudaMalloc(...): out of memory`; for the 12B its engine alone needs 7.07 GB of TensorRT activation memory at its 12,288-token input limit (stock has no chunked prefill, so the input limit is its whole context). Smaller contexts were tried, down to 2,048 tokens: 12B out of memory at 4,096 and 2,048; E4B out of memory at 8,192, 4,096 and 2,048, engine build fails at 6,144; Qwen engine build fails at 8,192, 6,144 and 4,096 (TensorRT cannot allocate an 8.1 GB scratch buffer), out of memory at 2,048 and at 10,240. So the table has an optimised column only; the fork features that make the models fit are the embedding table in host memory, chunked prefill, bounded sliding-window KV, INT8 KV and PLE, and stripped multimodal weights.

**Note on the long-context warm rows:** the host prefix cache keeps one snapshot per 1,024-token prefill chunk, so five prompts of 30k+ tokens overflow its 16 GiB budget before the warm repeat; the warm time at 30k, 40k and 46k tokens therefore equals the cold time (a cache-size artefact, not a limit: `EDGELLM_SWA_PREFIX_CAPTURE_INTERVAL` on fork `dev` 2b67583 keeps 2.5-3x more prompts). The needle rows are 2/3 at most lengths from 8k up on the 12B (the misses are the model's extra-zero errors; with no stock arm there is nothing to compare them to).

#### gemma4-12b
| | stock | optimised | speed-up |
|---|---:|---:|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | did not run | 1,776 |  |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | did not run | 0.212 |  |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | did not run | 73.3 |  |
| Decode, same prompt again (warm), tokens/s (higher is better) | did not run | 89.0 |  |
| Replay of 60 recorded requests, seconds (lower is better) | did not run | 141.5 |  |
| Replay output tokens in total (the stacks do not answer with the same length) | did not run | 10,366 |  |
| HumanEval generation, tokens/s (higher is better) | did not run | 123.5 |  |
| GPU memory used, MiB | did not run | 7,739 |  |

HumanEval pass@1 (greedy, public benchmark), per round: stock did not run; optimised 153/164 / 154/164.

**Long context**
| prompt tokens | metric | stock | optimised |
|---:|---|---:|---:|
| 4,096 (actual 4,045) | cold time to first token, s | not reached | 2.36 |
|  | cold prompt tokens/s | not reached | 1,711 |
|  | warm time to first token, s | not reached | 0.54 |
|  | decode tokens/s at this context | not reached | 61.4 |
|  | needles correct of 3 | not reached | 3 |
| 8,192 (actual 8,188) | cold time to first token, s | not reached | 4.87 |
|  | cold prompt tokens/s | not reached | 1,680 |
|  | warm time to first token, s | not reached | 0.58 |
|  | decode tokens/s at this context | not reached | 85.2 |
|  | needles correct of 3 | not reached | 2 |
| 11,500 (actual 11,496) | cold time to first token, s | not reached | 7.13 |
|  | cold prompt tokens/s | not reached | 1,613 |
|  | warm time to first token, s | not reached | 0.19 |
|  | decode tokens/s at this context | not reached | 67.5 |
|  | needles correct of 3 | not reached | 3 |
| 16,384 (actual 16,374) | cold time to first token, s | not reached | 10.41 |
|  | cold prompt tokens/s | not reached | 1,573 |
|  | warm time to first token, s | not reached | 0.65 |
|  | decode tokens/s at this context | not reached | 63.8 |
|  | needles correct of 3 | not reached | 2 |
| 24,576 (actual 24,538) | cold time to first token, s | not reached | 16.54 |
|  | cold prompt tokens/s | not reached | 1,484 |
|  | warm time to first token, s | not reached | 0.71 |
|  | decode tokens/s at this context | not reached | 61.2 |
|  | needles correct of 3 | not reached | 2 |
| 30,000 (actual 29,964) | cold time to first token, s | not reached | 21.13 |
|  | cold prompt tokens/s | not reached | 1,418 |
|  | warm time to first token, s | not reached | 21.10 |
|  | decode tokens/s at this context | not reached | 62.7 |
|  | needles correct of 3 | not reached | 2 |
| 40,000 (actual 39,962) | cold time to first token, s | not reached | 30.10 |
|  | cold prompt tokens/s | not reached | 1,327 |
|  | warm time to first token, s | not reached | 30.08 |
|  | decode tokens/s at this context | not reached | 55.7 |
|  | needles correct of 3 | not reached | 2 |
| 46,000 (actual 45,975) | cold time to first token, s | not reached | 35.89 |
|  | cold prompt tokens/s | not reached | 1,281 |
|  | warm time to first token, s | not reached | 36.11 |
|  | decode tokens/s at this context | not reached | 56.8 |
|  | needles correct of 3 | not reached | 2 |


#### gemma4-e4b
| | stock | optimised | speed-up |
|---|---:|---:|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | did not run | 7,151 |  |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | did not run | 0.028 |  |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | did not run | 146.0 |  |
| Decode, same prompt again (warm), tokens/s (higher is better) | did not run | 154.5 |  |
| Replay of 60 recorded requests, seconds (lower is better) | did not run | 56.6 |  |
| Replay output tokens in total (the stacks do not answer with the same length) | did not run | 10,290 |  |
| HumanEval generation, tokens/s (higher is better) | did not run | 280.6 |  |
| GPU memory used, MiB | did not run | 7,583 |  |

HumanEval pass@1 (greedy, public benchmark), per round: stock did not run; optimised 137/164 / 136/164.

**Long context**
| prompt tokens | metric | stock | optimised |
|---:|---|---:|---:|
| 2,048 (actual 2,027) | cold time to first token, s | not reached | 0.31 |
|  | cold prompt tokens/s | not reached | 6,583 |
|  | warm time to first token, s | not reached | 0.03 |
|  | decode tokens/s at this context | not reached | 142.8 |
|  | needles correct of 3 | not reached | 3 |
| 4,096 (actual 4,089) | cold time to first token, s | not reached | 0.66 |
|  | cold prompt tokens/s | not reached | 6,185 |
|  | warm time to first token, s | not reached | 0.03 |
|  | decode tokens/s at this context | not reached | 145.8 |
|  | needles correct of 3 | not reached | 3 |
| 8,192 (actual 8,182) | cold time to first token, s | not reached | 1.52 |
|  | cold prompt tokens/s | not reached | 5,382 |
|  | warm time to first token, s | not reached | 0.04 |
|  | decode tokens/s at this context | not reached | 146.4 |
|  | needles correct of 3 | not reached | 3 |


#### qwen35-4b
| | stock | optimised | speed-up |
|---|---:|---:|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | did not run | 4,748 |  |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | did not run | 0.019 |  |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | did not run | 163.7 |  |
| Decode, same prompt again (warm), tokens/s (higher is better) | did not run | 162.3 |  |
| Replay of 60 recorded requests, seconds (lower is better) | did not run | 86.1 |  |
| Replay output tokens in total (the stacks do not answer with the same length) | did not run | 13,006 |  |
| HumanEval generation, tokens/s (higher is better) | did not run | 221.0 |  |
| GPU memory used, MiB | did not run | 7,393 |  |

HumanEval pass@1 (greedy, public benchmark), per round: stock did not run; optimised 128/164 / 128/164.

**Long context**
| prompt tokens | metric | stock | optimised |
|---:|---|---:|---:|
| 2,048 (actual 2,008) | cold time to first token, s | not reached | 0.45 |
|  | cold prompt tokens/s | not reached | 4,452 |
|  | warm time to first token, s | not reached | 0.02 |
|  | decode tokens/s at this context | not reached | 157.3 |
|  | needles correct of 3 | not reached | 3 |
| 4,096 (actual 4,088) | cold time to first token, s | not reached | 0.90 |
|  | cold prompt tokens/s | not reached | 4,557 |
|  | warm time to first token, s | not reached | 0.02 |
|  | decode tokens/s at this context | not reached | 145.9 |
|  | needles correct of 3 | not reached | 3 |
| 8,192 (actual 8,182) | cold time to first token, s | not reached | 2.39 |
|  | cold prompt tokens/s | not reached | 3,421 |
|  | warm time to first token, s | not reached | 0.03 |
|  | decode tokens/s at this context | not reached | 137.1 |
|  | needles correct of 3 | not reached | 3 |

#### gemma4-26b (optimised arm only)
| | optimised |
|---|---:|
| Prefill of a fresh ~1.5k-token prompt (cold), tokens/s (higher is better) | 992 |
| Same prompt again (warm, prefix reused), time to first token, seconds (lower is better) | 1.361 |
| Decode, first request of a prompt (cold), tokens/s (higher is better) | 54.1 |
| Decode, same prompt again (warm), tokens/s (higher is better) | 55.6 |
| Replay of 60 recorded requests, seconds (lower is better) | 323.8 |
| Replay output tokens in total (the stacks do not answer with the same length) | 9,598 |
| HumanEval generation, tokens/s (higher is better) | 35.3 |
| GPU memory used, MiB | 7,581 |

HumanEval pass@1 (greedy, public benchmark), per round: optimised 153/164 / 155/164.

**Long context**
| prompt tokens | metric | optimised |
|---:|---|---:|
| 2,048 (actual 2,011) | cold time to first token, s | 1.51 |
|  | cold prompt tokens/s | 1,333 |
|  | warm time to first token, s | 1.40 |
|  | decode tokens/s at this context | 54.1 |
|  | needles correct of 3 | 3 |
| 4,096 (actual 4,092) | cold time to first token, s | 1.67 |
|  | cold prompt tokens/s | 2,455 |
|  | warm time to first token, s | 1.54 |
|  | decode tokens/s at this context | 54.8 |
|  | needles correct of 3 | 3 |
| 7,500 (actual 7,482) | cold time to first token, s | 3.49 |
|  | cold prompt tokens/s | 2,144 |
|  | warm time to first token, s | 1.76 |
|  | decode tokens/s at this context | 53.7 |
|  | needles correct of 3 | 3 |

</details>
