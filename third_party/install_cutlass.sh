#!/usr/bin/env bash

set -euo pipefail

CUTLASS_VERSION="v4.1.0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CUTLASS_DIR="${SCRIPT_DIR}/cutlass"

if [[ ! -d "${CUTLASS_DIR}" ]]; then
  echo "==> Cloning CUTLASS ${CUTLASS_VERSION}"
  git clone \
    --branch "${CUTLASS_VERSION}" \
    --depth 1 \
    https://github.com/NVIDIA/cutlass.git \
    "${CUTLASS_DIR}"
elif [[ ! -d "${CUTLASS_DIR}/.git" ]] || \
     [[ "$(git -C "${CUTLASS_DIR}" describe --tags --exact-match 2>/dev/null)" != "${CUTLASS_VERSION}" ]]; then
  echo "error: ${CUTLASS_DIR} is not CUTLASS ${CUTLASS_VERSION}" >&2
  exit 1
fi

echo "==> CUTLASS ${CUTLASS_VERSION} is available at ${CUTLASS_DIR}"
exit 0
