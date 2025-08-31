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

#include "lynn/cpu/kernel/fallback.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <algorithm>

#include "lutil/attributes.h"
#include "lutil/half.h"
#include "lutil/log.h"
#include "lynn/cpu/kernel/abstract.h"
#include "lynn/cpu/kernel/util.h"

namespace ly {
namespace op {
namespace cpu {
namespace kernel {

void saxpyFallbackKernel(int64_t n, float a, const float *x, float *y) {
  const float *px = x;
  float *py = y;
  for (int i = 0; i < n; ++i) {
    *py += a * *px;
    ++px;
    ++py;
  }
}

float sdotFallbackKernel(int64_t n, const float *x, const float *y) {
  float sum = 0;
  for (int64_t i = 0; i < n; ++i) {
    sum += x[i] * y[i];
  }

  return sum;
}

void hscvtFallbackKernel(int64_t n, const Float16 *x, float *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = cvtf<float>(x[i]);
  }
}

void shcvtFallbackKernel(int64_t n, const float *x, Float16 *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = cvtf<Float16>(x[i]);
  }
}

void haxpyFallbackKernel(int64_t n, Float16 a, const Float16 *x, float *y) {
  const Float16 *px = x;
  float *py = y;
  for (int i = 0; i < n; ++i) {
    *py += cvt_h2s(a) * cvt_h2s(*px);
    ++px;
    ++py;
  }
}

Float16 hdotFallbackKernel(int64_t n, const Float16 *x, const Float16 *y) {
  float sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += cvtf<float>(x[i]) * cvtf<float>(y[i]);
  }

  return cvtf<Float16>(sum);
}

template<typename T, int MR, int NR>
void gemmFallbackKernel(int64_t kc, const T *a, const T *b, T *c, int64_t rs_c) {
  for (int64_t m = 0; m < MR; ++m) {
    for (int64_t n = 0; n < NR; ++n) {
      float sum = cvtf<float>(c[m * rs_c + n]);
      for (int64_t k = 0; k < kc; ++k) {
        sum += cvtf<float>(a[k * MR + m]) * cvtf<float>(b[k * NR + n]);
      }
      c[m * rs_c + n] = cvtf<T>(sum);
    }
  }
}

void sgemm6x16DefaultKernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c) {
  return gemmFallbackKernel<float, 6, 16>(kc, a, b, c, rs_c);
}

void sgemm12x32DefaultKernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c) {
  return gemmFallbackKernel<float, 12, 32>(kc, a, b, c, rs_c);
}

void hgemm12x16FallbackKernel(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c) {
  return gemmFallbackKernel<Float16, 12, 16>(kc, a, b, c, rs_c);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace ly
