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
#include "lymath/common.h"
#include "lymath/q4kernel.h"
#include "lymath/skernel.h"
#include "lymath/util.h"
#include "lyutil/half.h"
#include "lyutil/log.h"


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

void DequantQ4FallbackKernel::apply(int n, PCQ4x2 src, PCFp16 scales, PCUInt8 zeros, PFp32 tgt) {
  CHECK(n % Q4GroupSize == 0);
  int nb = n / Q4GroupSize;

  for (int i = 0; i < nb; ++i) {
    float scale = lut::cvtsh_ss(scales[i]);
    UInt8 zero = i % 2 == 0 ? zeros[i / 2] & 0xf : zeros[i / 2] >> 4;
    PCQ4x2 p = src + i * Q4GroupSize / 2;
    PFp32 pt = tgt + i * Q4GroupSize;
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      *pt++ = scale * (static_cast<int>(*p & 0xf) - zero);
      *pt++ = scale * ((static_cast<int>(*p) >> 4) - zero);
      ++p;
    }
  }
}

float DotQ4FallbackKernel::apply(int64_t n, PCFp32 x, PCQ4x2 y, PCFp16 scaleY, PCUInt8 zerosY) {
  int64_t nb = n / Q4GroupSize;
  assert(n % Q4GroupSize == 0);

  float sum = 0.0f;

  PCQ4x2 py = y;
  for (int64_t i = 0; i < nb; ++i) {
    float scale = lut::cvtsh_ss(scaleY[i]);
    UInt8 zero = i % 2 == 0 ? zerosY[i / 2] & 0xf : zerosY[i / 2] >> 4;
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      sum += *x++ * scale * (static_cast<int>(*py & 0xf) - zero);
      sum += *x++ * scale * ((static_cast<int>(*py) >> 4) - zero);
      ++py;
    }
  }

  return sum;
}

float DotQ4FallbackKernel::applyRow(const Q4GemvArgs &args, int row) {
  PCQ4x2 data = args.A + row * args.N / 2;
  PCFp16 scale = args.scaleA + row * args.N / Q4GroupSize;
  PCUInt8 zeroPoint = args.zeroA + row * args.N / Q4GroupSize / 2;

  return apply(args.N, args.x, data, scale, zeroPoint);
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

}  // namespace lymath
