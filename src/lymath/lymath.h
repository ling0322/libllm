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

typedef uint8_t lymath_q4x2_t;
typedef int16_t lymath_float16_t;

void lymath_init();
void lymath_destroy();

void lymath_sgemm(
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
    int ldc);

void lymath_sgemm_omp(
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
    int ldc);

void lymath_dequant_q4(
    int n,
    const lymath_q4x2_t *data,
    const lymath_float16_t *scale,
    const int8_t *zeroPoint,
    float *tgt);

// GEMM: A is a float32 matrix, B is a matrix with 4-bit asymmetric quantization. C is a float32
// matrix.
void lymath_q4gemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const lymath_q4x2_t *B,
    const lymath_float16_t *scaleB,
    const int8_t *zeroPointB,
    float *C,
    int ldc);
