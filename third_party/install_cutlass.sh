#!/bin/bash

set -e

if [ ! -d cutlass ]; then
  wget -nv https://github.com/NVIDIA/cutlass/archive/refs/tags/v4.1.0.tar.gz
  tar xzf v4.1.0.tar.gz

  rm v4.1.0.tar.gz
  mv cutlass-4.1.0 cutlass
fi

