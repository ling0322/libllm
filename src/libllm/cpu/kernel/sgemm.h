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
#include "libllm/cpu/kernel/gemm_common.h"
#include "libllm/cpu/kernel/sgemv.h"
#include "libllm/cpu/kernel/skernel.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {


// -- class SGEMM ----------

typedef GEMMCommon<288, 512, 4096, SGemm6x16DefaultKernel, Mode::SingleThread> SGEMMKernelDefault;
typedef GEMMCommon<288, 512, 4096, SGemm6x16Avx2Kernel, Mode::SingleThread> SGEMMKernelAvx2;
typedef GEMMCommon<576, 512, 4096, SGemm12x32Avx512Kernel, Mode::SingleThread> SGEMMKernelAvx512;

typedef GEMMCommon<288, 512, 4096, SGemm6x16DefaultKernel, Mode::OMP> SGEMMKernelDefaultOMP;
typedef GEMMCommon<288, 512, 4096, SGemm6x16Avx2Kernel, Mode::OMP> SGEMMKernelAvx2OMP;
typedef GEMMCommon<576, 512, 4096, SGemm12x32Avx512Kernel, Mode::OMP> SGEMMKernelAvx512OMP;

class SGEMM {
 public:
  virtual ~SGEMM() = default;

  virtual void apply(
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
      int ldc) const = 0;
  
};

template<class TGEMMKernel, class TGEMVKernel>
class SGEMMImpl : public SGEMM {
 public:
  void apply(
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
      int ldc) const override {
    if (M == 1) {
      applyRowVectorA(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else if (N == 1) {
      applyColumnVectorB(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    } else {
      TGEMMKernel().Apply(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
    }
  }

 private:
  // row vector and matrix multiplication using SGEMV.
  void applyRowVectorA(
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
      int ldc) const;

  // row vector and matrix multiplication using SGEMV.
  void applyColumnVectorB(
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
      int ldc) const;
};

typedef SGEMMImpl<SGEMMKernelAvx512OMP, SGEMVImplAvx512OMP> SGEMMImplAvx512OMP;
typedef SGEMMImpl<SGEMMKernelAvx2OMP, SGEMVImplAvx2OMP> SGEMMImplAvx2OMP;
typedef SGEMMImpl<SGEMMKernelDefaultOMP, SGEMVImplDefaultOMP> SGEMMImplDefaultOMP;

typedef SGEMMImpl<SGEMMKernelAvx512, SGEMVImplAvx512> SGEMMImplAvx512;
typedef SGEMMImpl<SGEMMKernelAvx2, SGEMVImplAvx2> SGEMMImplAvx2;
typedef SGEMMImpl<SGEMMKernelDefault, SGEMVImplDefault> SGEMMImplDefault;

template<class TGEMMKernel, class TGEMVKernel>
void SGEMMImpl<TGEMMKernel, TGEMVKernel>::applyRowVectorA(
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
    int ldc) const {
  CHECK(M == 1);

  // fill C with zero.
  std::fill(C, C + N, 0.0f);

  TGEMVKernel().apply(SGEMVArgs{
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

template<class TGEMMKernel, class TGEMVKernel>
void SGEMMImpl<TGEMMKernel, TGEMVKernel>::applyColumnVectorB(
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
    int ldc) const {
  CHECK(N == 1);

  bool needPackC = ldc != 1;
  if (ldc != 1) {
    NOT_IMPL();
  } else {
    std::fill(C, C + M, 0.0f);
  }

  TGEMVKernel().apply(SGEMVArgs{
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
