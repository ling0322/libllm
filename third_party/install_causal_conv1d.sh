#!/usr/bin/env bash

set -euo pipefail

# Builds the vendored causal-conv1d kernels. Unlike install_flash_attn.sh there is nothing to
# fetch: the sources are in the tree and need only the CUDA toolkit, so this just configures and
# builds them.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/causal-conv1d/build}"
SOURCE_DIR="${SCRIPT_DIR}/causal-conv1d/csrc"

echo "==> Configuring causal_conv1d in ${BUILD_DIR}"
cmake_args=(-S "${SOURCE_DIR}" -B "${BUILD_DIR}")
if [[ -n "${CAUSAL_CONV1D_CUDA_ARCH:-}" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${CAUSAL_CONV1D_CUDA_ARCH}")
fi
cmake "${cmake_args[@]}"

echo "==> Building causal_conv1d"
cmake --build "${BUILD_DIR}" --target causal_conv1d --parallel

echo "==> causal_conv1d is available at ${BUILD_DIR}"
exit 0
