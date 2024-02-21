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

typedef uint16_t Fp16;
typedef int8_t Int8;
typedef uint8_t UInt8;
typedef float Fp32;
typedef uint8_t Q4x2;

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
    const Fp32 *A,
    int lda,
    const Fp32 *B,
    int ldb,
    Fp32 *C,
    int ldc,
    Mode mode = Mode::Auto);

void dequantQ4(
    int n,
    const Q4x2 *data,
    const Fp16 *scale,
    const UInt8 *zeroPoint,
    int offset,
    Fp32 *tgt,
    Mode mode = Mode::Auto);

// GEMM: A is a float32 matrix, B is a matrix with 4-bit asymmetric quantization. C is a float32
// matrix.
void gemmQ4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Fp32 *A,
    int lda,
    const Q4x2 *B,
    const Fp16 *scaleB,
    const UInt8 *zeroPointB,
    Fp32 *C,
    int ldc,
    Mode mode = Mode::Auto);

void convertHalfToFloat(int n, const Fp16 *x, Fp32 *y, Mode mode = Mode::Auto);

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
