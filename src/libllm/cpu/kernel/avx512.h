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
#include "libllm/cpu/kernel/avx2.h"
#include "libllm/lut/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

void sgemm12x32Avx512Kernel(int64_t kc, const float *a, const float *b, float *c, int64_t rs_c);

template<>
inline void cvtKernel<QInt4x32, float, CpuMathBackend::AVX512>(
    int n,
    const QInt4x32 *x,
    int64_t offsetX,
    float *y) {
  return qscvtAvx2Kernel(n, x, offsetX, y);
}
template<>
inline void cvtKernel<Float16, float, CpuMathBackend::AVX512>(
    int n,
    const Float16 *x,
    int64_t offsetX,
    float *y) {
  return hscvtAvx2Kernel(n, x + offsetX, y);
}
template<>
inline void gemmKernel<float, float, float, 12, 32, CpuMathBackend::AVX512>(
    int64_t kc,
    const float *a,
    const float *b,
    float *c,
    int64_t rs_c) {
  return sgemm12x32Avx512Kernel(kc, a, b, c, rs_c);
}

template<>
inline float dotKernel<float, float, float, CpuMathBackend::AVX512>(
    int64_t n,
    const float *x,
    const float *y,
    int64_t offsetY) {
  return sdotAvx2Kernel(n, x, y + offsetY);
}
template<>
inline float dotKernel<float, float, QInt4x32, CpuMathBackend::AVX512>(
    int64_t n,
    const float *x,
    const QInt4x32 *y,
    int64_t offsetY) {
  return sqdotAvx2Kernel(n, x, y, offsetY);
}
template<>
inline void axpyKernel<float, float, float, CpuMathBackend::AVX512>(
    int64_t n,
    float a,
    const float *x,
    int64_t offsetX,
    float *y) {
  return saxpyAvx2Kernel(n, a, x + offsetX, y);
}
template<>
inline void axpyKernel<float, QInt4x32, float, CpuMathBackend::AVX512>(
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
}  // namespace libllm
