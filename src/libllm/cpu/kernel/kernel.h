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

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

struct WORD {
  uint16_t h;
};

struct BYTE {
  uint8_t b;
};

#ifdef __aarch64__
typedef _Float16 Float16;
#else
typedef WORD Float16;
#endif

typedef BYTE UInt4x2;

enum class Mode {
  OMP,
  SingleThread,
  Auto
};

void init();
void destroy();

void sgemm(
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
    Mode mode = Mode::Auto);

void hgemm(
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
    Mode mode = Mode::Auto);

void dequantQ4(
    int n,
    const UInt4x2 *data,
    const Float16 *scale,
    const UInt4x2 *zeroPoint,
    int offset,
    float *tgt,
    Mode mode = Mode::Auto);

// GEMM: A is a float32 matrix, B is a matrix with 4-bit asymmetric quantization. C is a float32
// matrix.
void gemmQ4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const UInt4x2 *B,
    const Float16 *scaleB,
    const UInt4x2 *zeroPointB,
    float *C,
    int ldc,
    Mode mode = Mode::Auto);

void convertHalfToFloat(int n, const Float16 *x, float *y, Mode mode = Mode::Auto);

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
