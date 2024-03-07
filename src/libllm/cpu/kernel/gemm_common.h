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
#include <memory>
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/kernel/gemm_kernel.h"
#include "libllm/cpu/kernel/gemv_s.h"
#include "libllm/cpu/kernel/kernel_s.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

template<typename T>
class Gemm {
 public:
  virtual ~Gemm() = default;

  virtual void apply(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const T *A,
      int lda,
      const T *B,
      int ldb,
      T *C,
      int ldc) const = 0;
};

/// @brief Provides GEMM interface with dispatcher for GEMM/GEMV.
template<class TGemmKernel, class TGemvKernel, typename T>
class GemmImpl : public Gemm<T> {
 public:
  void apply(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const T *A,
      int lda,
      const T *B,
      int ldb,
      T *C,
      int ldc) const override {
    if (M == 1) {
      applyGemvRowVectorA(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else if (N == 1) {
      applyGemvColumnVectorB(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else {
      TGemmKernel().Apply(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    }
  }

 private:
  // row vector and matrix multiplication using SGEMV.
  void applyGemvRowVectorA(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const T *A,
      int lda,
      const T *B,
      int ldb,
      T *C,
      int ldc) const;

  // row vector and matrix multiplication using SGEMV.
  void applyGemvColumnVectorB(
      bool transA,
      bool transB,
      int M,
      int N,
      int K,
      const T *A,
      int lda,
      const T *B,
      int ldb,
      T *C,
      int ldc) const;
};

template<class TGemmKernel, class TGemvKernel, typename T>
void GemmImpl<TGemmKernel, TGemvKernel, T>::applyGemvRowVectorA(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    T *C,
    int ldc) const {
  CHECK(M == 1);

  // fill C with zero.
  std::fill(C, C + N, 0.0f);

  TGemvKernel().apply(GemvArgs<T>{
    !transB,
    transB ? N : K,
    transB ? K : N,
    B,
    ldb,
    A,
    transA ? lda : 1,
    C,
    1});
}

template<class TGemmKernel, class TGemvKernel, typename T>
void GemmImpl<TGemmKernel, TGemvKernel, T>::applyGemvColumnVectorB(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    T *C,
    int ldc) const {
  CHECK(N == 1);

  bool needPackC = ldc != 1;
  if (ldc != 1) {
    NOT_IMPL();
  } else {
    std::fill(C, C + M, 0.0f);
  }

  TGemvKernel().apply(GemvArgs<T>{
      transA,
      transA ? K : M,
      transA ? M : K,
      A,
      lda,
      B,
      transB ? 1 : ldb,
      C,
      ldc});
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
