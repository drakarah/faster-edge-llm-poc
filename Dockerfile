# faster-llm-poc: one image with two builds of TensorRT Edge-LLM, so a stock and an optimised stack can be timed on the
# same machine, on the same checkpoint, with the same prompts.
#   /opt/stock  NVIDIA/TensorRT-Edge-LLM, upstream main at a pinned commit, built exactly as upstream documents
#   /opt/fast   drakarah/TensorRT-Edge-LLM, branch `dev` at a pinned commit (upstream + the optimisations measured here)
# Nothing GPU-specific is prebuilt: the CuTe DSL kernels and the plugin are compiled here for GPU_ARCH, and the TensorRT
# engines are built on the user's machine by the `builder` service (an engine only loads on the GPU and TensorRT version
# that built it, and `docker build` has no GPU).
#
#   PROFILE=orin  Jetson AGX Orin (sm_87), Jetson toolchain
#   PROFILE=dgpu  discrete GPU (x86_64 or sbsa); GPU_ARCH defaults to 89 (RTX 4060 / 40-series), e.g. 86, 80, 120
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.09-py3
FROM ${BASE_IMAGE}

ARG PROFILE=dgpu
ARG GPU_ARCH=
ARG BUILD_JOBS=8
ARG STOCK_REPO=https://github.com/NVIDIA/TensorRT-Edge-LLM.git
ARG STOCK_REF=0f52705723abfa8c6702654bfdcc082a728ed353
ARG FAST_REPO=https://github.com/drakarah/TensorRT-Edge-LLM.git
ARG FAST_REF=ff5e1febe674aeaa18204b6d774dc046a320f72f
# The discrete-GPU profile pins a newer commit: the 26B dgpu engine streams its experts from host memory (expert cache, prefill chunk ring, cache-aware routing, token tiling, Qwen3.6 export, prefix reuse, hybrid chunked prefill, tail skip, CPU miss path, cache fill floor fix, routed-expert chunked prefill (opt-in), cache-fill from decode cache: fork dev 786a71b; the 26B config sets EDGELLM_MOE_CACHE_BIAS=0).
# The Orin pin stays where its measurements were taken.
# Pin moved from dev 786a71b to rtx4060/qwen-defaults-merge bb8d0a9 (786a71b plus presence penalty, preserved thinking, environment defaults for request fields, hybrid checkpoints in host memory): the served-endpoint defaults of the dgpu Qwen3.6 profile need them. The headline numbers below were measured on 786a71b; the added features are switched by environment variables and the speed numbers were not re-measured on the new pin.
# Moved again to fork dev 10e2bf7 (bb8d0a9 plus the growable KV pool, EDGELLM_KV_INITIAL_TOKENS, which composes with context reuse; also reasoning budget and control-token defanging). Off unless the variable is set.
ARG FAST_REF_DGPU=10e2bf7a5b8497a4ea86753ad2de8291b778a12e

# Python dependencies of the exporter, the server and the quantisation tools (torch and numpy come from the base image).
COPY poc/build/pins.txt /tmp/pins.txt
RUN pip install --no-cache-dir -r /tmp/pins.txt

COPY poc/build/build_edgellm.sh /usr/local/bin/build_edgellm.sh
# The stock stack is only built for the side-by-side comparison (WITH_STOCK=1, the `compare` compose profile); serving does not need it.
ARG WITH_STOCK=0
RUN if [ "${WITH_STOCK}" = 1 ]; then git clone "${STOCK_REPO}" /opt/stock && git -C /opt/stock checkout "${STOCK_REF}" \
    && git -C /opt/stock submodule update --init --recursive; fi
RUN FAST_PIN=$( [ "${PROFILE}" = dgpu ] && echo "${FAST_REF_DGPU}" || echo "${FAST_REF}" ) && echo "${FAST_PIN}" > /opt/fast_ref \
    && git clone "${FAST_REPO}" /opt/fast && git -C /opt/fast checkout "${FAST_PIN}" \
    && git -C /opt/fast submodule update --init --recursive

# Both stacks are compiled the same way. Only the fast stack gets two extra build options: the INT4 GEMM that applies the group
# scale to FP32 partial sums once per quantisation group (EDGELLM_INT4_GEMM_GROUP_FLUSH, fewer instructions per tensor op), and on
# the discrete GPU FP16 accumulation in the INT4 GEMM (twice the tensor rate on GeForce parts; no gain on a Jetson, so it is left off there).
RUN if [ "${WITH_STOCK}" = 1 ]; then PROFILE=${PROFILE} GPU_ARCH=${GPU_ARCH} BUILD_JOBS=${BUILD_JOBS} build_edgellm.sh /opt/stock; fi
RUN env PROFILE=${PROFILE} GPU_ARCH=${GPU_ARCH} BUILD_JOBS=${BUILD_JOBS} \
    EDGELLM_INT4_GEMM_GROUP_FLUSH=1 $( [ "${PROFILE}" = dgpu ] && echo EDGELLM_INT4_GEMM_FP16_ACC_MIN_BM=64 || true ) build_edgellm.sh /opt/fast

COPY poc /poc
ENV PYTHONDONTWRITEBYTECODE=1 PROFILE=${PROFILE} STOCK_REF=${STOCK_REF}
WORKDIR /poc
ENTRYPOINT ["/poc/run.sh"]
CMD ["up"]
