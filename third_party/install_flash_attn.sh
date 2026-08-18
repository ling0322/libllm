#!/usr/bin/env bash

set -euo pipefail

FLASH_ATTN_CUTLASS_REVISION="dc4817921edda44a549197ff3a9dcf5df0636e7b"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${1:-${SCRIPT_DIR}/flash-attention/build}"
FLASH_ATTN_DIR="${SCRIPT_DIR}/flash-attention/csrc/flash_attn"
FLASH_ATTN_CUTLASS_DIR="${SCRIPT_DIR}/flash-attention/csrc/cutlass"

if [[ ! -d "${FLASH_ATTN_CUTLASS_DIR}" ]]; then
  echo "==> Cloning FlashAttention CUTLASS ${FLASH_ATTN_CUTLASS_REVISION}"
  git init --quiet "${FLASH_ATTN_CUTLASS_DIR}"
  git -C "${FLASH_ATTN_CUTLASS_DIR}" remote add origin https://github.com/NVIDIA/cutlass.git
  git -C "${FLASH_ATTN_CUTLASS_DIR}" fetch --quiet --depth 1 origin "${FLASH_ATTN_CUTLASS_REVISION}"
  git -C "${FLASH_ATTN_CUTLASS_DIR}" checkout --quiet --detach FETCH_HEAD
elif [[ ! -d "${FLASH_ATTN_CUTLASS_DIR}/.git" ]] || \
     [[ "$(git -C "${FLASH_ATTN_CUTLASS_DIR}" rev-parse HEAD)" != "${FLASH_ATTN_CUTLASS_REVISION}" ]]; then
  echo "error: ${FLASH_ATTN_CUTLASS_DIR} is not CUTLASS ${FLASH_ATTN_CUTLASS_REVISION}" >&2
  exit 1
fi

echo "==> FlashAttention CUTLASS ${FLASH_ATTN_CUTLASS_REVISION} is available at ${FLASH_ATTN_CUTLASS_DIR}"

# A single kernel needs several GB, so never build these in parallel.
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAKEFLAGS="-j1"
export NINJAFLAGS="-j1"

echo "==> Configuring FlashAttention in ${BUILD_DIR}"
cmake_args=(-S "${FLASH_ATTN_DIR}" -B "${BUILD_DIR}")
if [[ -n "${FLASH_ATTN_CUDA_ARCH:-}" ]]; then
  cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="${FLASH_ATTN_CUDA_ARCH}")
fi
cmake "${cmake_args[@]}"

echo "==> Building flash_attn (serial)"
# Cap the build so a runaway compile cannot freeze the machine.
if command -v systemd-run >/dev/null 2>&1 && [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
  systemd-run --user --scope --quiet \
    -p MemoryMax="${FLASH_ATTN_MEMORY_MAX:-80%}" \
    -p MemorySwapMax=2G \
    cmake --build "${BUILD_DIR}" --target flash_attn --parallel 8
else
  cmake --build "${BUILD_DIR}" --target flash_attn --parallel 1
fi

echo "==> flash_attn is available at ${BUILD_DIR}"
exit 0
