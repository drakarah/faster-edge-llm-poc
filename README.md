# faster-llm-poc

Reproduce the speed-ups of our [TensorRT Edge-LLM fork](https://github.com/drakarah/TensorRT-Edge-LLM) on your own machine with one command.

It builds **two stacks side by side** from source, on your GPU, and times them on the same quantised checkpoint with the same prompts:

| | what it is |
|---|---|
| **stock** | [NVIDIA/TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) `main` at a pinned commit, used as upstream documents it: `tensorrt-edgellm-serve <checkpoint>` with MTP speculative decoding and context reuse, building its own engine; its draft-tree flags are set to the best values we found (see below), not to its default |
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

Every step resumes from `./data`, so an interrupted run continues where it stopped. To start only the optimised server (no benchmark) use `docker compose up --no-deps serve`: plain `docker compose up serve` starts the builder and the benchmark first. A new benchmark run moves earlier results into `data/results/previous-<time>/` instead of deleting them.

### Settings

| variable | values | |
|---|---|---|
| `PROFILE` | `orin` (default), `dgpu` | `orin`: Jetson AGX Orin (sm_87). `dgpu`: discrete GPU, x86_64 or arm64; set `GPU_ARCH` (default 89 = RTX 40-series) |
| `MODEL` | `gemma4-e4b` (default), `gemma4-12b`, `qwen35-4b`, `gemma4-26b` (Orin only, optimised arm only: stock Edge-LLM cannot load an engine built by the fork nor export the 51 GB checkpoint on 32 GB) | the models we measured |
| `HF_TOKEN` | token | needed for Gemma only |
| `BENCH_CALLS`, `BENCH_ROUNDS` | 60, 2 | replayed requests per round, rounds per stack |
| `BENCH_HUMANEVAL` | 164 | HumanEval problems per run (0 skips); public benchmark, pass@1 and tokens/s |
| `BENCH_LONGCTX` | per model (gemma4-12b: `4096,8192,11500,16384,24576,30000`; e4b and qwen: `2048,4096,8192`); empty skips | long-context test: prompt lengths in tokens, ascending. Per length: cold time to first token and prompt tokens/s (a unique nonce defeats prefix reuse), the same prompt again (warm), exact retrieval of 3 planted facts (10/50/90% depth), and decode tokens/s with that context in the KV cache. A stack that refuses or fails a length is recorded as such and not tried on longer ones (stock builds its engine for the replay's context only: no chunked prefill, no bounded sliding-window KV, so on the 12B it stops at 12,288 tokens). Both engines are built for a fixed capacity (`CAP` in `config/`): lengths above it are refused by design |
| `DATA_DIR`, `PORT`, `BUILD_JOBS` | `./data`, 8000, 8 | |

`PROFILE` picks the build target and the tuned settings that differ between the two GPU classes (all in `poc/config/`): a Jetson shares one DRAM between CPU and GPU, so host-memory tricks are off there, and FP16-accumulate prefill (twice the tensor rate on GeForce parts) is on only for `dgpu`.

### Requirements

* Docker with Compose v2.20+ and the NVIDIA container runtime (`runtime: nvidia`; on a Jetson this is the default JetPack setup, elsewhere `nvidia-ctk runtime configure --runtime=docker`).
* **Jetson:** run `sudo nvpmodel -m 0 && sudo jetson_clocks` on the host first. An unpinned GPU clock ramps with load and distorts short benchmarks (the bench prints a warning).
* RAM + swap of **at least 40 GB** for the quantisation and 28 GB for the export (a 32 GB Jetson needs about 24 GB of swap; the builder stops with the exact commands if it is short). The 12B needs more (about 50 GB) because the bf16 model is expanded for calibration.
* Free disk: about 60 GB (gemma4-e4b) to 120 GB (gemma4-12b), including the 25 GB base image; the 26B needs about 150 GB (52 GB download, 49 GB re-sharded copy while the download is deleted, 18 GB engine with sidecars).
* Time on first run (AGX Orin): image build about 1-2 h, E4B quantisation and engines a few hours, the benchmark about 20 min. Later runs only repeat the benchmark.

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
| `EDGELLM_FORCE_GREEDY` | forces greedy sampling (the MTP tree is greedy-only) | none (a correctness workaround) | none | yes | [`feat/force-greedy`](https://github.com/drakarah/TensorRT-Edge-LLM/tree/feat/force-greedy) |
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
| **Build and tooling** | | | | | |
| Timing cache and averaged timing iterations for the engine builder | `EDGELLM_TIMING_CACHE` makes tactic choices reproducible | averaging iterations did not help; a timing cache reproduced a build exactly. Build-to-build decode variance of about 7% was traced to tuning at the maximum tree: build with the maximum equal to the served tree (12) | none | yes (via build settings) | [1c44945](https://github.com/drakarah/TensorRT-Edge-LLM/commit/1c44945) |
| NVTX ranges for request setup; kernel unit tests (INT8 KV, XQA INT8 tree verify, dense head-512 prefill accuracy) | profiling and tests | none | none | n/a | [6c7120c](https://github.com/drakarah/TensorRT-Edge-LLM/commit/6c7120c), [84bf152](https://github.com/drakarah/TensorRT-Edge-LLM/commit/84bf152), [d09afb2](https://github.com/drakarah/TensorRT-Edge-LLM/commit/d09afb2), [8b35109](https://github.com/drakarah/TensorRT-Edge-LLM/commit/8b35109) |

### Notable negative results (all measured on the Jetson AGX Orin unless stated)

| negative result |
|---|
| Reduced output vocabulary (98k of 262k rows) for the E4B base model: 71-test suite 49 -> 41 passed and no speed gain, because the drafter proposed out-of-vocabulary tokens in 336 of 423 requests (rejected). |
| W4A8 as first shipped (all ops, one-tile flush): GEMM 0.96-1.00x of the FP16 kernel, replay-60 123.1 -> 127.2 s (+3.3% slower), 36 of 60 answers changed. |
| W4A8 with all ops (even after the kernel fixes): HumanEval 88.4% -> 85.4%; fixed by the per-op filter (90.2%), but a +3% answer-length lean on structured requests remains. |
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
| Prompt lookup on the E4B: replay-60 123.1 -> 122.9 s; with a 11-token chain code edits -14%. |
| Wider or deeper trees: a 16-node verify tree costs about 2% by itself; 2x6/16 loses 0.8% over all 396 requests; Qwen draft chain length 3-6 is flat (at best -1.9%). |
| GEMM instead of GEMV for single-token decode (threshold 0): decode 45% slower (23.8 vs 43.5 tok/s). |
| Averaging builder timing iterations: no effect on decode or prompt speed; newer CuTe DSL 4.8.0 against 4.7.0: no gain. |
| INT3 MLP weights on the 12B (RTX 4060, fake-quant test): HumanEval 92.1% -> 75.0%. |

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

## Results

> **Pin note (2026-10-22, updated):** the image now pins the fork's `dev` at `ff5e1fe`: `07382e4` plus the RTX 4060 session's merge of its MoE expert cache, host-pinned expert staging and prefill overlap (all off unless an env variable switches them on, and aimed at experts streamed over PCIe, so they should do nothing on the Orin's unified memory; the 26B run below was done on this pin: it ran with all of them unset and gave the expected numbers, which is consistent with a no-op, though it is not an A/B against `07382e4`). `07382e4` is `2e02476` plus six commits that change nothing for the three models below (an opt-in prefix-snapshot interval, the fix for a chunked prompt that ends in a one-token pass, and the 26B-A4B additions: the 8-row Marlin block threshold and layer-wise expert quantisation, both off unless a model's config sets them). The three result tables below were measured at `2e02476` (INT4 drafter head and MLP for the 12B on the Orin, W4A8 stage 4, split-K 3/5 decode variants, and the optional INT8 Q.K^T decode attention that only the 4060 profile switches on). The result tables below were measured at that earlier pin with the PoC files of commit `c5d0f39` (the Orin image was built from the Dockerfile of `3eeb42e`; the only difference, a dgpu-only line fixed in `6cf1fd2`, is empty for `PROFILE=orin`). Later commits change only this README. Results from the earlier pin `e4fd0ef` (2026-10-19) are replaced.

### Jetson AGX Orin 32 GB, 2026-10-21/22 (GPU pinned at 930 MHz, 2 rounds, medians)

**Setup.** One frozen commit for all runs on both machines: poc `main` c5d0f39, fork `dev` 2e02476, stock NVIDIA 0f52705. The Gemma checkpoints and the Qwen AWQ checkpoint are the same files on both machines (the 4060 copied them from the Orin; sha256 of the two files compared equal). Stock gets the optimised arm's settings wherever it has the same knob (drafting tree, engine build maximum for the tree, context length, the drafter checkpoint), not its best settings. Fork-only features (suffix draft, plugins, W4A8 prompt pass, INT8 KV cache, INT4 drafter head and MLP for the 12B, hybrid prefix reuse, ...) are what is being measured. Two rounds, stock and optimised alternated per round, medians.

**Caveats that apply to the tables below.**
* **The Gemma checkpoints were prepared with the same quantisation tools outside the container** (the from-scratch `gptq` path of the builder, which needs `HF_TOKEN`, was not run through it; the Qwen quantisation was). 
* **Qwen stock runs without prefix reuse** (upstream gap: stock refuses `--enable-context-reuse` on this hybrid model at start with "Hybrid context reuse requires at least one recurrent snapshot slot"; the fork's hybrid prefix reuse closes it). The Qwen AWQ checkpoint was calibrated with 64 samples (an attempt with 256 was killed by the kernel for memory on the Orin's shared 30 GB pool); the 4060 reuses that checkpoint.
* **The 12B stock engine was built for 12,288 tokens, and stock cannot go much further.** Chunked prefill (a long prompt run in several passes, each appended to the KV cache) is a fork feature and is **on in the optimised arm**: its 12B engine takes prompts in 1,024-token passes and holds 32,768 tokens on the Orin. Stock Edge-LLM has no chunked prefill, so a stock engine must take a whole prompt in one pass: its input limit is its whole context, and the activation memory of every pass grows with it. We built the stock engine for the 12,288 tokens the replay needs (same settings as the optimised arm otherwise) and did not try a larger stock build on the Orin. The long-context rows therefore show "refused / failed" for stock from 16,384 tokens: the table states it, it is not hidden.
* **The 12B warm rows at 11,500 tokens and above equal the cold ones on the Orin**: the host prefix cache is 4 GiB and keeps one snapshot of the bounded window per 1,024-token prefill chunk, so the three needle prompts of one length do not all fit; this is a cache-size effect, not a limit of the technique (fork branch `rtx4060/swa-prefix-capture-interval` keeps 2.5-3x more prompts; not part of this pin). Warm prefill at short lengths is slower than stock's for the 12B (the snapshot restore).
* **HumanEval uses a 1024-token answer cap.** On the E4B most of the optimised arm's lower pass count comes from answers cut off at that cap (failures not ending in a closed code block: stock 7 and 9, optimised 11 and 18 across the two rounds; a proxy, `finish_reason` is not recorded), and the optimised arm varies more between rounds (8 problems flip between its two rounds, 2 for stock). With a 4096-token cap an earlier A/B showed no truncation and mean lengths of 414 vs 409 tokens. We have not re-run the E4B pair with a larger cap.
* **Needle retrieval (3 planted facts) is 2 of 3 at several lengths on both arms** (stock 8,192 tokens, optimised 8,192 and up to 30,000 on the 12B); not investigated.
* The replay output lengths differ between the stacks (the stock stack pads structured JSON with long whitespace runs on the E4B and Qwen), so tokens per second is the fair speed number there and the replay seconds mix speed with answer length.

#### gemma4-12b on orin: stock Edge-LLM vs optimised fork (2 rounds, medians)

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

#### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)

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

#### gemma4-e4b on orin: stock Edge-LLM vs optimised fork (2 rounds, medians)

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

#### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)

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

#### qwen35-4b on orin: stock Edge-LLM vs optimised fork (2 rounds, medians)

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

#### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)

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

#### Gemma 4 26B-A4B on the Jetson AGX Orin (2026-10-22, pin `ff5e1fe`, optimised arm only)

Run from a clean data dir through the compose `builder` and `bench` services (re-shard 2 min, export 11 min, engine builds 6 min, then the bench; 23 minutes for the whole builder). **The BF16 base checkpoint was copied from another machine instead of downloaded** (the Orin's Wi-Fi gave 6.5 MB/s; same files, `rsync -c` verified), so the download stage was not part of this run. There is no stock arm (stock cannot load a fork-built engine, and its exporter does not fit 32 GB). Engine: INT4 experts, FP16 attention, dense MLP and lm_head, FP16 bounded KV, MTP with tree 16, host prefix cache 4 GiB (see `poc/config/gemma4-26b.sh` for the reason behind every setting).

**Reading the numbers.** The HumanEval rounds differ (160 and 157 of 164; greedy output is not bit-reproducible between identical runs). The decode rate in the long-context rows is *not* a steady decode speed: the answer is a few tokens long and prompt-lookup drafting accepts almost all of it, so it rises with the prompt (29 -> 75 tok/s); use the replay and the HumanEval generation rows for speed. The warm rows at 4,096 tokens and above show the prefix cache (cold 2.61 s -> warm 1.30 s at 4k; 5.13 -> 1.02 s at 7.5k); a 2,048-token prompt is one pass and gets no reuse.

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

#### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)

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

### RTX 4060 8 GB (`PROFILE=dgpu`), optimised arm only

**Setup:** RTX 4060 8 GB (sm_89), `PROFILE=dgpu GPU_ARCH=89`, GPU clock locked at 2,460 MHz, CPU governor performance, poc main f792bab + c5d0f39, fork `dev` 2e02476, 2 rounds, medians. Gemma checkpoints and the Qwen AWQ checkpoint (64 samples) are the Orin's, copied unchanged (sha256 equal). The 12B uses the 4060 recipe: tree 16, INT4 drafter MLP + head, INT8 Q.K^T in prompt and decode attention, KV capacity 49,152.

**Stock Edge-LLM did not run on this card for any of the three models** (same settings as the optimised arm): the stock server builds its engine and then fails at start with `CUDA runtime error in cudaMalloc(...): out of memory`; for the 12B its engine alone needs 7.07 GB of TensorRT activation memory at its 12,288-token input limit (stock has no chunked prefill, so the input limit is its whole context). Smaller contexts were tried, down to 2,048 tokens: 12B out of memory at 4,096 and 2,048; E4B out of memory at 8,192, 4,096 and 2,048, engine build fails at 6,144; Qwen engine build fails at 8,192, 6,144 and 4,096 (TensorRT cannot allocate an 8.1 GB scratch buffer), out of memory at 2,048 and at 10,240. So the table has an optimised column only; the fork features that make the models fit are the embedding table in host memory, chunked prefill, bounded sliding-window KV, INT8 KV and PLE, and stripped multimodal weights.

**Note on the long-context warm rows:** the host prefix cache keeps one snapshot per 1,024-token prefill chunk, so five prompts of 30k+ tokens overflow its 16 GiB budget before the warm repeat; the warm time at 30k, 40k and 46k tokens therefore equals the cold time (a cache-size artefact, not a limit: `EDGELLM_SWA_PREFIX_CAPTURE_INTERVAL` on fork `dev` 2b67583 keeps 2.5-3x more prompts). The needle rows are 2/3 at most lengths from 8k up on the 12B (the misses are the model's extra-zero errors; with no stock arm there is nothing to compare them to).

#### gemma4-12b on dgpu: stock Edge-LLM vs optimised fork (2 rounds, medians)

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

##### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)

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


#### gemma4-e4b on dgpu: stock Edge-LLM vs optimised fork (2 rounds, medians)

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

##### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)

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


#### qwen35-4b on dgpu: stock Edge-LLM vs optimised fork (2 rounds, medians)

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

##### Long context (prompts of the given length, greedy; cold = nothing to reuse, warm = same prompt again, needles = exact retrieval of 3 planted facts)

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
