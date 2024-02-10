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

#include <assert.h>
#include <stdlib.h>
#include <algorithm>
#include "libllm/cpu/kernel/common.h"
#include "libllm/cpu/kernel/hkernel.h"
#include "libllm/cpu/kernel/q4kernel.h"
#include "libllm/cpu/kernel/skernel.h"
#include "libllm/cpu/kernel/util.h"
#include "libllm/lut/half.h"
#include "libllm/lut/log.h"


namespace lymath {


// -- fallback micro-kernels ---------

void SGemm6x16DefaultKernel::apply(int64_t kc, PFp32 a, PFp32 b, PFp32 c, int64_t rs_c) {
  // a: kc x MR
  // b: kc x NR
  constexpr int64_t MR = 6;
  constexpr int64_t NR = 16;

  for (int k = 0; k < kc; ++k) {
    float *Ak = a + k * MR;
    for (int m = 0; m < MR; ++m) {
      float *Cm = c + m * rs_c;
      float Akm = Ak[m];
      float *Bk = b + k * NR;
      
      for (int n = 0; n < NR; ++n) {
        Cm[n] += Akm * Bk[n];
      }
    }
  }
}

void DequantQ4FallbackKernel::apply(int n, DataQ4 x, int64_t offsetX, PFp32 y) {
  int64_t groupIdx = offsetX / GroupSizeQ4;
  int64_t nb = n / GroupSizeQ4;
  assert(offsetX % GroupSizeQ4 == 0 && n % GroupSizeQ4 == 0);

  for (int i = groupIdx; i < groupIdx + nb; ++i) {
    float scale = lut::cvtsh_ss(x.getScaleValByGroup(i));
    UInt8 zero = x.getZeroValByGroup(i);
    PCQ4x2 p = x.getDataByGroup(i);
    for (int j = 0; j < GroupSizeQ4 / 2; ++j) {
      *y++ = scale * (static_cast<int>(*p & 0xf) - zero);
      *y++ = scale * ((static_cast<int>(*p) >> 4) - zero);
      ++p;
    }
  }
}

float DotQ4FallbackKernel::apply(int64_t n, PCFp32 x, DataQ4 y, int64_t offsetY) {
  int64_t groupIdx = offsetY / GroupSizeQ4;
  int64_t nb = n / GroupSizeQ4;
  assert(offsetY % GroupSizeQ4 == 0 && n % GroupSizeQ4 == 0);

  float sum = 0.0f;

  PCQ4x2 py = y.getDataByGroup(groupIdx);
  PCUInt8 zeroY = y.getZeroByGroup(groupIdx);
  for (int64_t i = groupIdx; i < groupIdx + nb; ++i) {
    float scale = lut::cvtsh_ss(y.getScaleValByGroup(i));
    UInt8 zero = y.getZeroValByGroup(i);
    for (int j = 0; j < GroupSizeQ4 / 2; ++j) {
      sum += *x++ * scale * (static_cast<int>(*py & 0xf) - zero);
      sum += *x++ * scale * ((static_cast<int>(*py) >> 4) - zero);
      ++py;
    }
  }

  return sum;
}

float DotQ4FallbackKernel::applyRow(const Q4GemvArgs &args, int row) {
  int64_t offset = args.N * row;
  return apply(args.N, args.x, args.A, offset);
}

void SAxpyFallbackKernel::apply(int64_t n, float a, PCFp32 x, PFp32 y) {
  const float *px = x;
  float *py = y;
  for (int i = 0; i < n; ++i) {
    *py = a * *px;
    ++px;
    ++py;
  }
}

void CvtHalfToFloatFallbackKernel::apply(int64_t n, PCFp16 x, PFp32 y) {
  for (int i = 0; i < n; ++i) {
    y[i] = lut::cvtsh_ss(x[i]);
  }
}

}  // namespace libllmmath
