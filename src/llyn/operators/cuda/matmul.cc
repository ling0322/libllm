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

#include "llyn/operators/cuda/matmul.h"

#include <cublas_v2.h>
#include "llyn/dtype.h"
#include "llyn/operators/common/matmul.h"
#include "llyn/operators/cuda/common.h"

#define CHECK_CUBLAS_STATUS(x) { \
      cublasStatus_t status = x; \
      if (status != CUBLAS_STATUS_SUCCESS) { \
        LOG(ERROR) << "Error while calling: " << #x; \
        throw ly::AbortedError(cublasGetStatusString(status)); \
      } \
    }

namespace llyn {
namespace op {
namespace cuda {

std::shared_ptr<MatMul> MatMul::create() {
  std::shared_ptr<MatMul> mm = std::make_shared<MatMul>();
  mm->_handle = {nullptr, safeDestroyCublas};
  CHECK_CUBLAS_STATUS(cublasCreate(mm->_handle.get_pp()));

  return mm;
}

void MatMul::safeDestroyCublas(cublasHandle_t handle) {
  CHECK_CUBLAS_STATUS(cublasDestroy(handle));
}

Tensor MatMul::apply(const Tensor &A, const Tensor &B) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(B.getDevice().getType() == Device::kCuda);

  if (B.getDType() == DType::kFloat16) {
    return matmulHalf(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor MatMul::matmulHalf(const Tensor &A, const Tensor &B) {
  CHECK(A.getDType() == B.getDType() && A.getDType() == DType::kFloat16);
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmHalf(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor MatMul::gemmHalf(const Tensor &A, const Tensor &B) {
  CHECK(A.getDim() == B.getDim() && A.getDim() == 2);
  Tensor C = createCudaTensorHalf({A.getShape(0), B.getShape(1)});

  half alpha = 1.0;
  half beta = 0.0;

  common::GEMMArgs gemmArgs = common::generateGemmArgs(A, B, C);
  CHECK_CUBLAS_STATUS(cublasGemmEx(
      _handle.get(),
      gemmArgs.transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      gemmArgs.transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      gemmArgs.N,
      gemmArgs.M,
      gemmArgs.K,
      &alpha,
      B.getData<half>(),
      CUDA_R_16F,
      gemmArgs.ldb,
      A.getData<half>(),
      CUDA_R_16F,
      gemmArgs.lda,
      &beta,
      C.getData<half>(),
      CUDA_R_16F,
      gemmArgs.ldc,
      CUBLAS_COMPUTE_16F,
      CUBLAS_GEMM_DEFAULT));

  return C;
}

}  // cuda
}  // op
}  // llyn
