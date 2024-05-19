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

#include "libllm/cpu/kernel/abstract.h"
#include "libllm/lut/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

void qhcvtAsimdhpKernel(int n, const QInt4x32 *x, int64_t offsetX, Float16 *y);
void hscvtAsimdhpKernel(int64_t n, const Float16 *x, float *y);
void shcvtAsimdhpKernel(int64_t n, const float *x, Float16 *y);
void hgemm12x16AsimdhpKernel(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c);
Float16 hdotAsimdhpKernel(int64_t n, const Float16 *x, const Float16 *y);
Float16 hqdotAsimdhpKernel(int64_t n, const Float16 *x, const QInt4x32 *y, int64_t offsetY);
void hsaxpyAsimdhpKernel(int64_t n, Float16 a, const Float16 *x, float *y);

template<>
inline void cvtKernel<QInt4x32, Float16, CpuMathBackend::ASIMDHP>(
    int n,
    const QInt4x32 *x,
    int64_t offsetX,
    Float16 *y,
    int64_t offsetY) {
  return qhcvtAsimdhpKernel(n, x, offsetX, y + offsetY);
}
template<>
inline void cvtKernel<Float16, float, CpuMathBackend::ASIMDHP>(
    int n,
    const Float16 *x,
    int64_t offsetX,
    float *y,
    int64_t offsetY) {
  return hscvtAsimdhpKernel(n, x + offsetX, y + offsetY);
}
template<>
inline void cvtKernel<float, Float16, CpuMathBackend::ASIMDHP>(
    int n,
    const float *x,
    int64_t offsetX,
    Float16 *y,
    int64_t offsetY) {
  return shcvtAsimdhpKernel(n, x + offsetX, y + offsetY);
}
template<>
inline void gemmKernel<Float16, Float16, Float16, 12, 16, CpuMathBackend::ASIMDHP>(
    int64_t kc,
    const Float16 *a,
    const Float16 *b,
    Float16 *c,
    int64_t rs_c) {
  return hgemm12x16AsimdhpKernel(kc, a, b, c, rs_c);
}

template<>
inline Float16 dotKernel<Float16, Float16, Float16, CpuMathBackend::ASIMDHP>(
    int64_t n,
    const Float16 *x,
    const Float16 *y,
    int64_t offsetY) {
  return hdotAsimdhpKernel(n, x, y + offsetY);
}
template<>
inline Float16 dotKernel<Float16, Float16, QInt4x32, CpuMathBackend::ASIMDHP>(
    int64_t n,
    const Float16 *x,
    const QInt4x32 *y,
    int64_t offsetY) {
  return hqdotAsimdhpKernel(n, x, y, offsetY);
}
template<>
inline void axpyKernel<Float16, Float16, float, CpuMathBackend::ASIMDHP>(
    int64_t n,
    Float16 a,
    const Float16 *x,
    int64_t offsetX,
    float *y) {
  return hsaxpyAsimdhpKernel(n, a, x + offsetX, y);
}
template<>
inline void axpyKernel<Float16, QInt4x32, float, CpuMathBackend::ASIMDHP>(
    int64_t n,
    Float16 a,
    const QInt4x32 *x,
    int64_t offsetX,
    float *y) {
  NOT_IMPL();
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
