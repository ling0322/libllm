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
#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/cpu/kernel/kernel_h.h"
#include "libllm/cpu/kernel/kernel_hq4.h"
#include "libllm/cpu/kernel/kernel_sq4.h"
#include "libllm/cpu/kernel/kernel_s.h"
#include "libllm/cpu/kernel/util.h"
#include "libllm/lut/half.h"
#include "libllm/lut/log.h"


namespace libllm {
namespace op {
namespace cpu {
namespace kernel {


// -- fallback micro-kernels ---------

void SGemm6x16DefaultKernel::apply(int64_t kc, float *a, float *b, float *c, int64_t rs_c) {
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

void DequantQInt4FallbackKernel::apply(int n, DataQInt4 x, int64_t offsetX, float *y) {
  int64_t groupIdx = offsetX / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetX % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  for (int i = groupIdx; i < groupIdx + nb; ++i) {
    float scale = cvt_h2s(x.getScaleValByGroup(i));
    uint8_t zero = x.getZeroValByGroup(i);
    const UInt4x2 *p = x.getDataByGroup(i);
    for (int j = 0; j < GroupSizeQInt4 / 2; ++j) {
      *y++ = scale * (static_cast<int>(p->b & 0xf) - zero);
      *y++ = scale * ((static_cast<int>(p->b) >> 4) - zero);
      ++p;
    }
  }
}

float DotQ4FallbackKernel::apply(int64_t n, const float *x, DataQInt4 y, int64_t offsetY) {
  int64_t groupIdx = offsetY / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetY % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  float sum = 0.0f;

  const UInt4x2 *py = y.getDataByGroup(groupIdx);
  const UInt4x2 *zeroY = y.getZeroByGroup(groupIdx);
  for (int64_t i = groupIdx; i < groupIdx + nb; ++i) {
    float scale = cvt_h2s(y.getScaleValByGroup(i));
    uint8_t zero = y.getZeroValByGroup(i);
    for (int j = 0; j < GroupSizeQInt4 / 2; ++j) {
      sum += *x++ * scale * (static_cast<int>(py->b & 0xf) - zero);
      sum += *x++ * scale * ((static_cast<int>(py->b) >> 4) - zero);
      ++py;
    }
  }

  return sum;
}

float DotQ4FallbackKernel::applyRow(const Q4GemvArgs &args, int row) {
  int64_t offset = args.N * row;
  return apply(args.N, args.x, args.A, offset);
}

void SAxpyFallbackKernel::apply(int64_t n, float a, const float *x, float *y) {
  const float *px = x;
  float *py = y;
  for (int i = 0; i < n; ++i) {
    *py += a * *px;
    ++px;
    ++py;
  }
}

void SAxpyFallbackKernel::applyColumn(const SGEMVArgs &args, int column, float *y) {
  apply(args.N, args.x[column], args.A + column * args.lda, y);
}

float SDotFallbackKernel::apply(int64_t n, const float *x, const float *y) {
  float sum = 0;
  for (int64_t i = 0; i < n; ++i) {
    sum += x[i] * y[i];
  }

  return sum;
}

float SDotFallbackKernel::applyRow(const SGEMVArgs &args, int row) {
  return apply(args.N, args.A + row * args.lda, args.x);
}

void CvtHalfToFloatFallbackKernel::apply(int64_t n, const Float16 *x, float *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = cvt_h2s(x[i]);
  }
}

void AxpyHalfFallbackKernel::apply(int64_t n, Float16 a, const Float16 *x, Float16 *y) {
  const Float16 *px = x;
  Float16 *py = y;
  for (int i = 0; i < n; ++i) {
    *py = cvt_s2h(cvt_h2s(*py) + cvt_h2s(a) * cvt_h2s(*px));
    ++px;
    ++py;
  }
}

Float16 DotHalfFallbackKernel::apply(int64_t n, const Float16 *x, const Float16 *y) {
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += cvt_h2s(x[i]) * cvt_h2s(y[i]);
  }

  return cvt_s2h(sum);
}

template<int MR, int NR>
void GemmHalfFallbackKernel<MR, NR>::apply(
    int64_t kc, Float16 *a, Float16 *b, Float16 *c, int64_t rs_c) {
  for (int64_t m = 0; m < MR; ++m) {
    for (int64_t n = 0; n < NR; ++n) {
      float sum = cvt_h2s(c[m * rs_c + n]);
      for (int64_t k = 0; k < kc; ++k) {
        sum += cvt_h2s(a[k * MR + m]) * cvt_h2s(b[k * NR + n]);
      }
      c[m * rs_c + n] = cvt_s2h(sum);
    }
  }
}

void DequantQInt4ToHalfFallbackKernel::apply(int n, DataQInt4 x, int64_t offsetX, Float16 *y) {
  int64_t groupIdx = offsetX / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetX % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  for (int i = groupIdx; i < groupIdx + nb; ++i) {
    float scale = cvt_h2s(x.getScaleValByGroup(i));
    uint8_t zero = x.getZeroValByGroup(i);
    const UInt4x2 *p = x.getDataByGroup(i);
    for (int j = 0; j < GroupSizeQInt4 / 2; ++j) {
      *y++ = cvt_s2h(scale * (static_cast<int>(p->b & 0xf) - zero));
      *y++ = cvt_s2h(scale * ((static_cast<int>(p->b) >> 4) - zero));
      ++p;
    }
  }
}

template struct GemmHalfFallbackKernel<12, 16>;

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
