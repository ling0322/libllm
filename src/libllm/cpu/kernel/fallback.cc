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

#include "libllm/cpu/kernel/fallback.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <algorithm>

#include "libllm/cpu/kernel/abstract.h"
#include "libllm/cpu/kernel/util.h"
#include "lutil/attributes.h"
#include "lutil/half.h"
#include "lutil/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

template<typename T>
void qfcvtFallbackKernel(int n, const QInt4x32 *x, int64_t offsetX, T *y) {
  int64_t groupIdx = offsetX / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetX % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  for (int i = groupIdx; i < groupIdx + nb; ++i) {
    T scale = cvtf<T>(x[i].scale);
    T zero = cvtf<T>(x[i].zero);
    const uint8_t *p = x[i].data;
    for (int j = 0; j < GroupSizeQInt4 / 2; ++j) {
      uint8_t b = *p;
      T b0 = static_cast<int>(b & 0xf);
      T b1 = static_cast<int>(b >> 4);
      *y++ = cvtf<T>(cvtf<T>(scale * b0) - zero);
      *y++ = cvtf<T>(cvtf<T>(scale * b1) - zero);
      ++p;
    }
  }
}

void qscvtFallbackKernel(int n, const QInt4x32 *x, int64_t offsetX, float *y) {
  qfcvtFallbackKernel<float>(n, x, offsetX, y);
}

#if LUT_CPU_ARCH == LUT_AARCH64
void qhcvtFallbackKernel(int n, const QInt4x32 *x, int64_t offsetX, Float16 *y) {
  qfcvtFallbackKernel<Float16>(n, x, offsetX, y);
}
#endif

template<typename T>
T fqdotFallbackKernel(int64_t n, const T *x, const QInt4x32 *y, int64_t offsetY) {
  int64_t groupIdx = offsetY / GroupSizeQInt4;
  int64_t nb = n / GroupSizeQInt4;
  assert(offsetY % GroupSizeQInt4 == 0 && n % GroupSizeQInt4 == 0);

  float sum = 0.0f;
  for (int64_t i = groupIdx; i < groupIdx + nb; ++i) {
    float scale = cvtf<float>(y[i].scale);
    float zero = cvtf<float>(y[i].zero);
    const uint8_t *py = y[i].data;
    for (int j = 0; j < GroupSizeQInt4 / 2; ++j) {
      uint8_t b = *py;
      sum += cvtf<float>(*x++) * (scale * static_cast<int>(b & 0xf) - zero);
      sum += cvtf<float>(*x++) * (scale * static_cast<int>(b >> 4) - zero);
      ++py;
    }
  }

  return cvtf<T>(sum);
}

float sqdotFallbackKernel(int64_t n, const float *x, const QInt4x32 *y, int64_t offsetY) {
  return fqdotFallbackKernel<float>(n, x, y, offsetY);
}

Float16 hqdotFallbackKernel(int64_t n, const Float16 *x, const QInt4x32 *y, int64_t offsetY) {
  return fqdotFallbackKernel<Float16>(n, x, y, offsetY);
}

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

template<typename T>
void fqcvtFallbackKernel(int64_t n, const T *x, QInt4x32 *y, int64_t offsetY) {
  int64_t nb = n / GroupSizeQInt4;
  int64_t groupOffset = offsetY / GroupSizeQInt4;
  CHECK(n % GroupSizeQInt4 == 0 && offsetY % GroupSizeQInt4 == 0);

  for (int i = 0; i < nb; ++i) {
    int begin = i * GroupSizeQInt4;
    int end = (i + 1) * GroupSizeQInt4;

    float minVal = *std::min_element(x + begin, x + end);
    float maxVal = *std::max_element(x + begin, x + end);

    float scale = (maxVal - minVal) / 15.0f;
    float zero = -minVal;

    for (int j = 0; j < GroupSizeQInt4; j += 2) {
      float dlFp32 = roundf((x[begin + j] - minVal) / scale);
      float dhFp32 = roundf((x[begin + j + 1] - minVal) / scale);
      CHECK(dlFp32 >= 0.0f && dlFp32 <= 15.0f && dhFp32 >= 0.0f && dhFp32 <= 15.0f);

      uint8_t dl = static_cast<uint8_t>(dlFp32);
      uint8_t dh = static_cast<uint8_t>(dhFp32);
      y[groupOffset + i].data[j / 2] = (dh << 4) | dl;
    }

    y[groupOffset + i].scale = cvtf<Float16>(scale);
    y[groupOffset + i].zero = cvtf<Float16>(zero);
  }
}

void sqcvtFallbackKernel(int64_t n, const float *x, QInt4x32 *y, int64_t offsetY) {
  fqcvtFallbackKernel<float>(n, x, y, offsetY);
}

#if LUT_APU_ARCH == LUT_AARCH64
void hqcvtFallbackKernel(int64_t n, const Float16 *x, QInt4x32 *y) {
  fqcvtFallbackKernel<Float16>(n, x, y);
}
#endif

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
