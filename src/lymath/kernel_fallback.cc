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
#include "lymath/q4sym_kernel.h"
#include "lymath/q8kernel.h"
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

void DequantQ4SymFallbackKnl::apply(int n, PCQ4x2 src, PCFp16 scale, PFp32 tgt) {
  CHECK(n % Q4GroupSize == 0);
  int nb = n / Q4GroupSize;

  for (int i = 0; i < nb; ++i) {
    float s = lut::cvtsh_ss(scale[i]);
    PCQ4x2 p = src + i * Q4GroupSize / 2;
    PFp32 pt = tgt + i * Q4GroupSize;
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      *pt++ = s * (static_cast<int>(*p >> 4) - 8);
      *pt++ = s * ((static_cast<int>(*p) & 0xf) - 8);
      ++p;
    }
  }
}

void DequantQ4FallbackKernel::apply(int n, PCQ4x2 src, PCFp16 scale, PCInt8 zero, PFp32 tgt) {
  CHECK(n % Q4GroupSize == 0);
  int nb = n / Q4GroupSize;

  for (int i = 0; i < nb; ++i) {
    float s = lut::cvtsh_ss(scale[i]);
    Int8 zeroPoint = zero[i];
    PCQ4x2 p = src + i * Q4GroupSize / 2;
    PFp32 pt = tgt + i * Q4GroupSize;
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      *pt++ = s * (static_cast<int>(*p >> 4) - zeroPoint);
      *pt++ = s * ((static_cast<int>(*p) & 0xf) - zeroPoint);
      ++p;
    }
  }
}

float DotQ4SymFallbackKernel::apply(int64_t n, PCFp32 x, PCQ4x2 y, PCFp16 scaleY) {
  int64_t nb = n / Q4GroupSize;
  assert(n % Q4GroupSize == 0);

  float sum = 0.0f;

  const uint8_t *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    float scale = lut::cvtsh_ss(scaleY[i]);
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      sum += *x++ * scale * (static_cast<int>(*py >> 4) - 8);
      sum += *x++ * scale * ((static_cast<int>(*py) & 0xf) - 8);
      ++py;
    }
  }

  return sum;
}

float DotQ4FallbackKernel::apply(int64_t n, PCFp32 x, PCQ4x2 y, PCFp16 scaleY, PCInt8 zpY) {
  int64_t nb = n / Q4GroupSize;
  assert(n % Q4GroupSize == 0);

  float sum = 0.0f;

  PCQ4x2 py = y;
  PCInt8 pyzp = zpY;
  for (int64_t i = 0; i < nb; ++i) {
    float scale = lut::cvtsh_ss(scaleY[i]);
    Int8 zp = *pyzp;
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      sum += *x++ * scale * (static_cast<int>(*py >> 4) - zp);
      sum += *x++ * scale * ((static_cast<int>(*py) & 0xf) - zp);
      ++py;
    }
    
    ++pyzp;
  }

  return sum;
}

float DotQ4FallbackKernel::applyRow(const Q4GemvArgs &args, int row) {
  PCQ4x2 data = args.A + row * args.N / 2;
  PCFp16 scale = args.scaleA + row * args.N / Q4GroupSize;
  PCInt8 zeroPoint = args.zeroPointA + row * args.N / Q4GroupSize;

  return apply(args.N, args.x, data, scale, zeroPoint);
}

void AxpyQ4SymFallbackKernel::apply(int64_t n, float a, PCQ4x2 x, PCFp16 xscale, PFp32 y) {
  int64_t nb = n / Q4GroupSize;
  assert(n % Q4GroupSize == 0);

  const uint8_t *px = x;
  float *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    float scale = lut::cvtsh_ss(xscale[i]);
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      *py++ += a * scale * (static_cast<int>(*px >> 4) - 8);
      *py++ += a * scale * ((static_cast<int>(*px) & 0xf) - 8);
      ++px;
    }
  }
}

void AxpyQ4SymFallbackKernel::applyColumn(const QGEMVInt4AArgs &args, int col, float *y) {
  PCQ4x2 data = args.A + col * args.N / 2;
  PCFp16 scale = args.scaleA + col * args.N / 32;
  apply(args.N, args.x[col], data, scale, y);
}

void AxpyQ4FallbackKernel::apply(int64_t n, float a, PCQ4x2 x, PCFp16 scaleX, PCInt8 zpX, PFp32 y) {
  int64_t nb = n / Q4GroupSize;
  assert(n % Q4GroupSize == 0);

  const uint8_t *px = x;
  float *py = y;
  for (int64_t i = 0; i < nb; ++i) {
    float scale = lut::cvtsh_ss(scaleX[i]);
    int8_t zeroPoint = zpX[i];
    for (int j = 0; j < Q4GroupSize / 2; ++j) {
      *py++ += a * scale * (static_cast<int>(*px >> 4) - zeroPoint);
      *py++ += a * scale * ((static_cast<int>(*px) & 0xf) - zeroPoint);
      ++px;
    }
  }
}

void AxpyQ4FallbackKernel::applyColumn(const Q4GemvArgs &args, int col, float *y) {
  PCQ4x2 data = args.A + col * args.N / 2;
  PCFp16 scale = args.scaleA + col * args.N / Q4GroupSize;
  PCInt8 zp = args.zeroPointA + col * args.N / Q4GroupSize;
  apply(args.N, args.x[col], data, scale, zp, y);
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

// real_value = A * quantized_value + B
void DequantInt8BFallbackKernel::apply(
    int64_t n, const uint8_t *data, const float *scaleZp, int64_t offset, float *tgt) {
  int64_t hBegin = offset % Int8bScaleGroupSize;
  int64_t tEnd = (offset + n - 1) % Int8bScaleGroupSize + 1;
  int64_t nb = (n + Int8bScaleGroupSize - 1) / Int8bScaleGroupSize + (hBegin < tEnd ? 0 : 1);

  const float *pscaleZp = scaleZp;
  const uint8_t *pdata = data + offset;
  float *ptgt = tgt;
  for (int i = 0; i < nb; ++i) {
    float scale = *pscaleZp++;
    float zeroPoint = *pscaleZp++;

    int64_t begin = i == 0 ? hBegin : 0;
    int64_t end = i == nb - 1 ? tEnd : Int8bScaleGroupSize;
    assert(end - begin > 0);
    for (int64_t j = begin; j < end; ++j) {
      *ptgt++ = scale * (*pdata++) + zeroPoint;
    }
  }
}

}  // namespace lymath
