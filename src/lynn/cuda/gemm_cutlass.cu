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

#define CUTLASS_DEBUG_TRACE_LEVEL 2

#include <cuda_fp16.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>

#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "lutil/error.h"
#include "lynn/cpu/common.h"
#include "lynn/cpu/matmul.h"
#include "lynn/cuda/common.h"
#include "lynn/cuda/gemm_cutlass.h"
#include "lynn/dtype.h"

#define CUTLASS_CHECK(x)                                                                     \
  {                                                                                          \
    cutlass::Status status = x;                                                              \
    if (status != cutlass::Status::kSuccess) {                                               \
      LOG(ERROR) << "Error while calling: " << #x << ": " << cutlassGetStatusString(status); \
      throw lut::AbortedError(cutlassGetStatusString(status));                               \
    }                                                                                        \
  }

namespace ly {
namespace op {
namespace cuda {

using namespace cute;

using cutlass::layout::ColumnMajor;
using cutlass::layout::RowMajor;

template<class LayoutA, class LayoutB>
struct Sm80Gemm {
  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      LayoutB,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<256, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>,
      cutlass::epilogue::thread::
          LinearCombination<cutlass::half_t, 8, cutlass::half_t, cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>>;
};

template<class LayoutA, class LayoutB, class ArchTag>
void hgemmT(
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
  using Gemm = typename Sm80Gemm<LayoutA, LayoutB>::Gemm;
  Gemm gemmOperator;

  typename Gemm::Arguments args{{m, n, k}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}};

  CUTLASS_CHECK(gemmOperator(args));
}

template<class ArchTag>
void cutlassHgemmArch(
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
    return hgemmT<RowMajor, RowMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == false) {
    return hgemmT<ColumnMajor, RowMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == false && transB == true) {
    return hgemmT<RowMajor, ColumnMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (transA == true && transB == true) {
    return hgemmT<ColumnMajor, ColumnMajor, ArchTag>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  } else {
    NOT_IMPL();
  }
}

void cutlassHgemm(
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
  cutlassHgemmArch<
      cutlass::arch::Sm90>(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<class LayoutA, class layoutB>
void hgemmArrayT(
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
  using Gemm = cutlass::gemm::device::GemmArray<
      cutlass::half_t,
      LayoutA,
      cutlass::half_t,
      layoutB,
      cutlass::half_t,
      RowMajor,
      float>;
  Gemm gemmOperator;

  typename Gemm::Arguments
      args({m, n, k}, A, lda, B, ldb, C, ldc, C, ldc, {alpha, beta}, batchSize);
  CUTLASS_CHECK(gemmOperator(args));
}

void cutlassHgemmArray(
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
    hgemmArrayT<RowMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == true && transB == false) {
    hgemmArrayT<ColumnMajor, RowMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == false && transB == true) {
    hgemmArrayT<RowMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else if (transA == true && transB == true) {
    hgemmArrayT<ColumnMajor, ColumnMajor>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, bs);
  } else {
    NOT_IMPL();
  }
}

void CutlassGemm::hgemm(
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
  cutlassHgemm(
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

void CutlassGemm::hgemmArray(
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
  cutlassHgemmArray(
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

std::shared_ptr<Gemm> CutlassGemm::create() {
  std::shared_ptr<CutlassGemm> mm = std::make_shared<CutlassGemm>();
  return mm;
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
