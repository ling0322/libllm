// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <cuda_fp16.h>
#include "llyn/operators/cuda/cuda_common.h"

namespace llyn {
namespace cuda {

constexpr int GroupSizeQ4 = 32;

__device__ inline void dequantQ4ToHalf(
    int n,
    const uint8_t *data,
    const half *pscale,
    const int8_t *pzeroPoint,
    half *tgt) {
  int nb = n / GroupSizeQ4;

  const uint8_t *psrc;
  half *ptgt = tgt;

  for (int j = 0; j < nb; ++j) {
    int zeroPoint = pzeroPoint[j];
    half scale = pscale[j];

    #pragma unroll
    for (int i = 0; i < GroupSizeQ4; ++i) {
      *ptgt++ = scale * __int2half_rd(static_cast<int>(*psrc >> 4) - zeroPoint);
      *ptgt++ = scale * __int2half_rd(static_cast<int>(*psrc & 0xf) - zeroPoint);
      ++psrc;
    }
  }
}

}  // cuda
}  // llyn
