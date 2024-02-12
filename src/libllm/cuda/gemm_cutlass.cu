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

#include "libllm/cuda/gemm_cutlass.h"

#include <cuda_fp16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>
#include "libllm/dtype.h"
#include "libllm/cpu/common/common.h"
#include "libllm/cpu/common/matmul.h"
#include "libllm/cuda/common.h"
#include "libllm/lut/error.h"

namespace libllm {
namespace op {
namespace cuda {

using cutlass::layout::RowMajor;
using cutlass::layout::ColumnMajor;

std::shared_ptr<Gemm> CutlassGemm::create() {
  std::shared_ptr<CutlassGemm> mm = std::make_shared<CutlassGemm>();
  return mm;
}

template<class LayoutA, class layoutB>
lut::ErrorCode hgemmT(
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *A,
    int lda,
    const cutlass::half_t *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *C,
    int ldc) {
  using CutlassGemm = cutlass::gemm::device::Gemm<
      cutlass::half_t, LayoutA,
      cutlass::half_t, layoutB,
      cutlass::half_t, RowMajor,
      float,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm61>;
  CutlassGemm gemmOperator;
  typename CutlassGemm::Arguments args(
      {m, n, k},
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta});
  cutlass::Status status = gemmOperator(args);
  if (status != cutlass::Status::kSuccess) {
    return lut::ErrorCode::Aborted;
  }
}

lut::ErrorCode cutlassHgemm(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *A,
    int lda,
    const cutlass::half_t *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *C,
    int ldc) {
  if (transA == false && transB == false) {
    return hgemmT<RowMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == false) {
    return hgemmT<ColumnMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == false && transB == true) {
    return hgemmT<RowMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == true) {
    return hgemmT<ColumnMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}

template<class LayoutA, class layoutB>
lut::ErrorCode hgemmArrayT(
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *const *A,
    int lda,
    const cutlass::half_t *const *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *const *C,
    int ldc,
    int batchSize) {
  using CutlassGemm = cutlass::gemm::device::GemmArray<
      cutlass::half_t, LayoutA,
      cutlass::half_t, layoutB,
      cutlass::half_t, RowMajor,
      float,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm61>;
  CutlassGemm gemmOperator;

  typename CutlassGemm::Arguments args(
      {m, n, k},
      A, lda,
      B, ldb,
      C, ldc,
      C, ldc,
      {alpha, beta},
      batchSize);
  cutlass::Status status = gemmOperator(args);
  if (status != cutlass::Status::kSuccess) {
    return lut::ErrorCode::Aborted;
  }
}

lut::ErrorCode cutlassHgemmArray(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    cutlass::half_t alpha,
    const cutlass::half_t *const *A,
    int lda,
    const cutlass::half_t *const *B,
    int ldb,
    cutlass::half_t beta,
    cutlass::half_t *const *C,
    int ldc,
    int batchSize) {
  int bs = batchSize;
  if (transA == false && transB == false) {
    return hgemmArrayT<RowMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == true && transB == false) {
    return hgemmArrayT<ColumnMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == false && transB == true) {
    return hgemmArrayT<RowMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == true && transB == true) {
    return hgemmArrayT<ColumnMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  }
}


lut::ErrorCode CutlassGemm::hgemm(
      bool transA,
      bool transB,
      int m,
      int n,
      int k,
      __half alpha,
      const __half *A, 
      int lda,
      const __half *B,
      int ldb,
      __half beta,
      __half *C,
      int ldc) {
  cutlass::half_t alphaH = *reinterpret_cast<cutlass::half_t *>(&alpha);
  cutlass::half_t betaH = *reinterpret_cast<cutlass::half_t *>(&beta);
  return cutlassHgemm(
      transA,
      transB,
      m,
      n,
      k,
      alphaH,
      reinterpret_cast<const cutlass::half_t *>(A),
      lda,
      reinterpret_cast<const cutlass::half_t *>(B),
      ldb,
      betaH,
      reinterpret_cast<cutlass::half_t *>(C),
      ldc);
}


lut::ErrorCode CutlassGemm::hgemmArray(
    bool transA,
    bool transB,
    int m,
    int n,
    int k,
    __half alpha,
    const __half *const *arrayA,
    int lda,
    const __half *const *arrayB,
    int ldb,
    __half beta,
    __half *const *arrayC,
    int ldc,
    int batchSize) {
  cutlass::half_t alphaH = *reinterpret_cast<cutlass::half_t *>(&alpha);
  cutlass::half_t betaH = *reinterpret_cast<cutlass::half_t *>(&beta);
  return cutlassHgemmArray(
      transA,
      transB,
      m,
      n,
      k,
      alphaH,
      reinterpret_cast<const cutlass::half_t *const *>(arrayA),
      lda,
      reinterpret_cast<const cutlass::half_t *const *>(arrayB),
      ldb,
      betaH,
      reinterpret_cast<cutlass::half_t *const *>(arrayC),
      ldc,
      batchSize);
}

}  // cuda
}  // op
}  // ly
