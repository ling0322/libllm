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

#include <stdint.h>

#include "libllm/lut/attributes.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

#ifdef LUT_ARCH_AARCH64
typedef _Float16 Float16;
#else
struct Float16 {
  uint16_t h;
};
#endif

struct QInt4x32 {
  Float16 zero;
  Float16 scale;
  uint8_t data[16];
};
static_assert(sizeof(QInt4x32) == 20, "invalid size of QInt4x32");

enum class Mode { OMP, SingleThread };
enum class CpuMathBackend { DEFAULT, AVX2, AVX512, ASIMDHP, FALLBACK, UNKNOWN };

void init();
void destroy();
void setAllowSlowKernel(bool allow);

void gemmFloat(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

void gemmHalf(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const Float16 *B,
    int ldb,
    Float16 *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

void gemmHalfQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const QInt4x32 *B,
    Float16 *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

void dequantQInt4ToFloat(
    int n,
    const QInt4x32 *data,
    int offset,
    float *tgt,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

void quantFloatToQInt4(
    int n,
    const float *data,
    int offset,
    QInt4x32 *tgt,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

void dequantQInt4ToHalf(
    int n,
    const QInt4x32 *data,
    int offset,
    Float16 *tgt,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

// GEMM: A is a float32 matrix, B is a matrix with 4-bit asymmetric quantization. C is a float32
// matrix.
void gemmFloatQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const QInt4x32 *B,
    float *C,
    int ldc,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

void convertHalfToFloat(
    int n,
    const Float16 *x,
    float *y,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

void convertFloatToHalf(
    int n,
    const float *x,
    Float16 *y,
    Mode mode,
    CpuMathBackend backendType = CpuMathBackend::DEFAULT);

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
