#!/bin/bash
# Build one Edge-LLM checkout in place: CuTe DSL kernels (ahead of time, no GPU needed), then the plugin, runtime bindings and llm_build.
# Usage: PROFILE=orin|dgpu [GPU_ARCH=87] [BUILD_JOBS=8] [EDGELLM_INT4_GEMM_FP16_ACC_MIN_BM=64] build_edgellm.sh SRC_DIR
set -euo pipefail
SRC=$1; PROFILE=${PROFILE:-orin}; JOBS=${BUILD_JOBS:-8}
case "$PROFILE" in
  orin) ARCH=${GPU_ARCH:-87} ;;
  dgpu) ARCH=${GPU_ARCH:-89} ;;
  *) echo "PROFILE must be orin or dgpu" >&2; exit 2 ;;
esac
cd "$SRC"
# build_cutedsl.py insists on one exact CUTLASS DSL and cupy version: take both from the pins file.
export CUTE_DSL_VERSION="$(grep -oP 'nvidia-cutlass-dsl\[cu13\]==\K[^ ]+' /tmp/pins.txt)"
CUPY_VERSION="$(grep -oP 'cupy-cuda13x==\K[^ ]+' /tmp/pins.txt)"
sed -i "s/(\"cupy-cuda13x\", \"13.6.0\")/(\"cupy-cuda13x\", \"${CUPY_VERSION}\")/" kernelSrcs/build_cutedsl.py
python kernelSrcs/build_cutedsl.py --gpu_arch "sm_${ARCH}" --jobs "$JOBS"

case "$(uname -m)" in x86_64) TGT=x86_64-linux ;; aarch64) TGT=sbsa-linux ;; esac
COMMON=(-DCMAKE_BUILD_TYPE=Release -DTRT_PACKAGE_DIR=/usr -DCUDA_CTK_VERSION=13.4 -DENABLE_CUTE_DSL=ALL -DBUILD_PYTHON_BINDINGS=ON
        -DCUDA_DRIVER_LIB=/usr/local/cuda/targets/$TGT/lib/stubs/libcuda.so -Dpybind11_DIR="$(python -m pybind11 --cmakedir)")
if [ "$PROFILE" = orin ]; then
  PROFILE_ARGS=(-DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake -DEMBEDDED_TARGET=jetson-orin)
else
  PROFILE_ARGS=(-DCMAKE_CUDA_ARCHITECTURES="$ARCH")
fi
cmake -S . -B build "${COMMON[@]}" "${PROFILE_ARGS[@]}"
cmake --build build -j"$JOBS" --target NvInfer_edgellm_plugin _edgellm_runtime llm_build
echo "built $SRC for sm_$ARCH ($PROFILE)"
