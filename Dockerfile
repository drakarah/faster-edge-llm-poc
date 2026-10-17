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

ARG PROFILE=orin
ARG GPU_ARCH=
ARG BUILD_JOBS=8
ARG STOCK_REPO=https://github.com/NVIDIA/TensorRT-Edge-LLM.git
ARG STOCK_REF=0f52705723abfa8c6702654bfdcc082a728ed353
ARG FAST_REPO=https://github.com/drakarah/TensorRT-Edge-LLM.git
ARG FAST_REF=ade06d3e59a451a1d47ff7e9353166355bcfa487

# Python dependencies of the exporter, the server and the quantisation tools (torch and numpy come from the base image).
COPY poc/build/pins.txt /tmp/pins.txt
RUN pip install --no-cache-dir -r /tmp/pins.txt

COPY poc/build/build_edgellm.sh /usr/local/bin/build_edgellm.sh
RUN git clone "${STOCK_REPO}" /opt/stock && git -C /opt/stock checkout "${STOCK_REF}" \
    && git -C /opt/stock submodule update --init --recursive
RUN git clone "${FAST_REPO}" /opt/fast && git -C /opt/fast checkout "${FAST_REF}" \
    && git -C /opt/fast submodule update --init --recursive

# Both stacks are compiled the same way. Only the fast stack gets two extra build options: the INT4 GEMM that applies the group
# scale to FP32 partial sums once per quantisation group (EDGELLM_INT4_GEMM_GROUP_FLUSH, fewer instructions per tensor op), and on
# the discrete GPU FP16 accumulation in the INT4 GEMM (twice the tensor rate on GeForce parts; no gain on a Jetson, so it is left off there).
RUN PROFILE=${PROFILE} GPU_ARCH=${GPU_ARCH} BUILD_JOBS=${BUILD_JOBS} build_edgellm.sh /opt/stock
RUN PROFILE=${PROFILE} GPU_ARCH=${GPU_ARCH} BUILD_JOBS=${BUILD_JOBS} \
    EDGELLM_INT4_GEMM_GROUP_FLUSH=1 $( [ "${PROFILE}" = dgpu ] && echo EDGELLM_INT4_GEMM_FP16_ACC_MIN_BM=64 ) build_edgellm.sh /opt/fast

COPY poc /poc
ENV PYTHONDONTWRITEBYTECODE=1 PROFILE=${PROFILE}
WORKDIR /poc
ENTRYPOINT ["/poc/run.sh"]
CMD ["all"]
