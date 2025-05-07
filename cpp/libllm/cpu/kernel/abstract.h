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

#include "libllm/cpu/kernel/interface.h"

#define LIBLLM_KERNEL_MSVC (_MSC_VER && !__INTEL_COMPILER)

#if LIBLLM_KERNEL_MSVC
#define LIBLLM_KERNEL_FORCE_INLINE __forceinline
#else
#define LIBLLM_KERNEL_FORCE_INLINE __attribute__((always_inline)) inline
#endif

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

constexpr int GEMVMinRowsPerThread = 128;
constexpr int CvtMinElemPerThread = 1024;
constexpr int DequantMinElemPerThread = 1024;
constexpr int GroupSizeQInt4 = 32;

template<typename ElementA, typename ElementC, CpuMathBackend TYPE>
void cvtKernel(int n, const ElementA *x, int64_t offsetX, ElementC *y, int64_t offsetY);

template<typename ElementA, typename ElementX, typename ElementY, CpuMathBackend TYPE>
ElementA dotKernel(int64_t n, const ElementX *x, const ElementY *y, int64_t offsetY);

template<typename ElementA, typename ElementX, typename ElementY, CpuMathBackend TYPE>
void axpyKernel(int64_t n, ElementA a, const ElementX *x, int64_t offsetX, ElementY *y);

template<
    typename ElementA,
    typename ElementB,
    typename ElementC,
    int MR,
    int NR,
    CpuMathBackend TYPE>
void gemmKernel(int64_t kc, const ElementA *a, const ElementB *b, ElementC *c, int64_t rs_c);

template<typename ElementA, typename ElementX, typename ElementY>
struct GemvArgs {
  bool transA;
  int M;
  int N;
  const ElementA *A;
  int lda;
  const ElementX *x;
  int incX;
  ElementY *y;
  int incY;
};

template<typename ElementA, typename ElementB, typename ElementC>
struct GemmArgs {
  bool transA;
  bool transB;
  int M;
  int N;
  int K;
  const ElementA *A;
  int lda;
  const ElementB *B;
  int ldb;
  ElementC *C;
  int ldc;
};

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
