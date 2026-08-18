#!/usr/bin/env bash
# build_unwind.sh — build NonGNU libunwind (static, no lzma) in place, no install step.

set -euo pipefail

VERSION=1.6.2
TARBALL="libunwind-${VERSION}.tar.gz"
URL="https://github.com/libunwind/libunwind/releases/download/v1.6.2/libunwind-1.6.2.tar.gz"
WORKDIR="$(pwd)/libunwind"

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

if [ ! -f "${TARBALL}" ]; then
  echo "==> Downloading ${URL}"
  curl -L "${URL}" -o "${TARBALL}"
fi

if [ ! -d "libunwind-${VERSION}" ]; then
  echo "==> Extracting"
  tar xf "${TARBALL}"
fi

cd "libunwind-${VERSION}"
mkdir -p build && cd build

../configure \
  --disable-minidebuginfo \
  --enable-static \
  --disable-shared \
  CFLAGS="-O2 -fPIC" \
  CXXFLAGS="-O2 -fPIC"

make -j"$(nproc || echo 4)"

echo
echo "==> Done."