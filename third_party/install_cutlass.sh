#!/bin/bash

set -e

if [ ! -d cutlass ]; then
  wget -nv https://github.com/NVIDIA/cutlass/archive/refs/tags/v2.11.0.tar.gz
  tar xzf v2.11.0.tar.gz

  rm v2.11.0.tar.gz
  mv cutlass-2.11.0 cutlass
fi

