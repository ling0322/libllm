#!/usr/bin/env bash

set -euo pipefail

MODEL_URL="https://huggingface.co/ling0322/llama3.2-libllm/resolve/main/llama3.2-3b-instruct-fp16.llmpkg"
TEST_CASE_URL="https://huggingface.co/ling0322/libllm_test_data/resolve/main/llama3.2-3b-instruct-fp16_test.llmpkg"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$(dirname "${SCRIPT_DIR}")/models"

if ! command -v curl >/dev/null; then
  echo "error: curl is required" >&2
  exit 1
fi

download() {
  local url="$1"
  local name
  name="$(basename "${url}")"
  local dest="${MODELS_DIR}/${name}"

  if [[ -f "${dest}" ]]; then
    echo "==> ${name} already downloaded"
    return
  fi

  echo "==> Downloading ${name}"

  # download to a temporary name so an interrupted run leaves no partial file behind.
  curl -fL --progress-bar -o "${dest}.part" "${url}"
  mv "${dest}.part" "${dest}"
}

mkdir -p "${MODELS_DIR}"
download "${MODEL_URL}"
download "${TEST_CASE_URL}"

echo "==> Models are available at ${MODELS_DIR}"
