// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include <stdint.h>

#include "lten/cpu/kernel/abstract.h"
#include "lutil/log.h"

namespace lten {
namespace op {
namespace cpu {
namespace kernel {

// fallback kernels.
void qscvtFallbackKernel(int n, const QInt4x32 *x, int64_t offsetX, float *y);
void sqcvtFallbackKernel(int64_t n, const float *x, QInt4x32 *y, int64_t offsetY);
void qhcvtFallbackKernel(int n, const QInt4x32 *x, int64_t offsetX, Float16 *y);
void hscvtFallbackKernel(int64_t n, const Float16 *x, float *y);
void shcvtFallbackKernel(int64_t n, const float *x, Float16 *y);
void sgemm6x16DefaultKernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c);
void sgemm12x32DefaultKernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c);
void hgemm12x16FallbackKernel(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c);
float sqdotFallbackKernel(int64_t n, const float *x, const QInt4x32 *y, int64_t offsetY);
Float16 hqdotFallbackKernel(int64_t n, const Float16 *x, const QInt4x32 *y, int64_t offsetY);
float sdotFallbackKernel(int64_t n, const float *x, const float *y);
Float16 hdotFallbackKernel(int64_t n, const Float16 *x, const Float16 *y);
void haxpyFallbackKernel(int64_t n, Float16 a, const Float16 *x, float *y);
void saxpyFallbackKernel(int64_t n, float a, const float *x, float *y);

template<>
inline void cvtKernel<QInt4x32, float, CpuMathBackend::FALLBACK>(
    int n,
    const QInt4x32 *x,
    int64_t offsetX,
    float *y,
    int64_t offsetY) {
  return qscvtFallbackKernel(n, x, offsetX, y + offsetY);
}
template<>
inline void cvtKernel<float, QInt4x32, CpuMathBackend::FALLBACK>(
    int n,
    const float *x,
    int64_t offsetX,
    QInt4x32 *y,
    int64_t offsetY) {
  return sqcvtFallbackKernel(n, x + offsetX, y, offsetY);
}
template<>
inline void cvtKernel<QInt4x32, Float16, CpuMathBackend::FALLBACK>(
    int n,
    const QInt4x32 *x,
    int64_t offsetX,
    Float16 *y,
    int64_t offsetY) {
  return qhcvtFallbackKernel(n, x, offsetX, y + offsetY);
}
template<>
inline void cvtKernel<Float16, float, CpuMathBackend::FALLBACK>(
    int n,
    const Float16 *x,
    int64_t offsetX,
    float *y,
    int64_t offsetY) {
  return hscvtFallbackKernel(n, x + offsetX, y + offsetY);
}
template<>
inline void cvtKernel<float, Float16, CpuMathBackend::FALLBACK>(
    int n,
    const float *x,
    int64_t offsetX,
    Float16 *y,
    int64_t offsetY) {
  return shcvtFallbackKernel(n, x + offsetX, y + offsetY);
}
template<>
inline void gemmKernel<float, float, float, 6, 16, CpuMathBackend::FALLBACK>(
    int64_t kc,
    const float *a,
    const float *b,
    float *c,
    int64_t rs_c) {
  return sgemm6x16DefaultKernel(kc, a, b, c, rs_c);
}
template<>
inline void gemmKernel<float, float, float, 12, 32, CpuMathBackend::FALLBACK>(
    int64_t kc,
    const float *a,
    const float *b,
    float *c,
    int64_t rs_c) {
  return sgemm12x32DefaultKernel(kc, a, b, c, rs_c);
}
template<>
inline void gemmKernel<Float16, Float16, Float16, 12, 16, CpuMathBackend::FALLBACK>(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c) {
  return hgemm12x16FallbackKernel(kc, a, b, c, rs_c);
}
template<>
inline float dotKernel<float, float, QInt4x32, CpuMathBackend::FALLBACK>(
    int64_t n,
    const float *x,
    const QInt4x32 *y,
    int64_t offsetY) {
  return sqdotFallbackKernel(n, x, y, offsetY);
}
template<>
inline Float16 dotKernel<Float16, Float16, QInt4x32, CpuMathBackend::FALLBACK>(
    int64_t n,
    const Float16 *x,
    const QInt4x32 *y,
    int64_t offsetY) {
  return hqdotFallbackKernel(n, x, y, offsetY);
}
template<>
inline float dotKernel<float, float, float, CpuMathBackend::FALLBACK>(
    int64_t n,
    const float *x,
    const float *y,
    int64_t offsetY) {
  return sdotFallbackKernel(n, x, y + offsetY);
}
template<>
inline Float16 dotKernel<Float16, Float16, Float16, CpuMathBackend::FALLBACK>(
    int64_t n,
    const Float16 *x,
    const Float16 *y,
    int64_t offsetY) {
  return hdotFallbackKernel(n, x, y + offsetY);
}
template<>
inline void axpyKernel<float, float, float, CpuMathBackend::FALLBACK>(
    int64_t n,
    float a,
    const float *x,
    int64_t offsetX,
    float *y) {
  return saxpyFallbackKernel(n, a, x + offsetX, y);
}

template<>
inline void axpyKernel<Float16, Float16, float, CpuMathBackend::FALLBACK>(
    int64_t n,
    Float16 a,
    const Float16 *x,
    int64_t offsetX,
    float *y) {
  return haxpyFallbackKernel(n, a, x + offsetX, y);
}
template<>
inline void axpyKernel<float, QInt4x32, float, CpuMathBackend::FALLBACK>(
    int64_t n,
    float a,
    const QInt4x32 *x,
    int64_t offsetX,
    float *y) {
  NOT_IMPL();
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace lten
