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
#include "llyn/operators/common/common.h"
#include "llyn/operators/common/matmul.h"
#include "llyn/operators/cuda/common.h"
#include "llyn/operators/cuda/dequant.h"

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
  cublasStatus_t status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Error while calling cublasDestroy(): " << cublasGetStatusString(status);
  }
}

Tensor MatMul::apply(const Tensor &A, const Tensor &B) {
  CHECK(A.getDevice().getType() == Device::kCuda);
  CHECK(B.getDevice().getType() == Device::kCuda);

  if (A.getDType() == DType::kFloat16 && B.getDType() == DType::kFloat16) {
    return matmulHalf(A, B);
  } else if (A.getDType() == DType::kFloat16 && B.getDType() == DType::kQInt4Group32) {
    return matmulQ4(A, B);
  } else {
    NOT_IMPL();
  }
}

Tensor MatMul::matmulQ4(const Tensor &A, const Tensor &B) {
  CHECK(B.getDType() == DType::kQInt4Group32);
  CHECK(A.getDType() == DType::kFloat16);

  Tensor xB = dequantQ4ToHalf(B);
  return matmulHalf(A, xB);
}

Tensor MatMul::matmulHalf(const Tensor &A, const Tensor &B) {
  CHECK(A.getDType() == B.getDType() && A.getDType() == DType::kFloat16);
  if (A.getDim() == 2 && B.getDim() == 2) {
    return gemmHalf(A, B);
  } else if (A.getDim() > 2 && B.getDim() == 2 && A.isContiguous()) {
    return bmmToGemmHalf(A, B);
  } else if (A.getDim() >= 2 && B.getDim() >= 2 && A.getDim() >= B.getDim()) {
    return bmmHalf(A, B);
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
  cudaDeviceSynchronize();

  return C;
}


template<int DIM>
std::vector<const half *> getBatchImpl(const Tensor &A);

template<>
std::vector<const half *> getBatchImpl<1>(const Tensor &A) {
  const half *base = A.getData<half>();

  int stride0 = A.getStride(0);
  std::vector<const half *> batch;
  for (int i = 0; i < A.getShape(0); ++i) {
    batch.push_back(base + i * stride0);
  }
  return batch;
}

template<>
std::vector<const half *> getBatchImpl<2>(const Tensor &A) {
  const half *base = A.getData<half>();

  int stride0 = A.getStride(0);
  int stride1 = A.getStride(1);
  std::vector<const half *> batch;
  for (int i = 0; i < A.getShape(0); ++i) {
    for (int j = 0; j < A.getShape(1); ++j) {
      batch.push_back(base + i * stride0 + j * stride1);
    }
  }
  return batch;
}

std::vector<const half *> MatMul::getBatch(const Tensor &A, int nBatchDim) {
  if (nBatchDim == 1) return getBatchImpl<1>(A);
  if (nBatchDim == 2) return getBatchImpl<2>(A);

  NOT_IMPL();
}

Tensor MatMul::bmmHalf(const Tensor &A, const Tensor &B) {
  Tensor xB = B;
  if (A.getDim() != B.getDim()) xB = common::broadcastTensor(B, A);

  std::vector<int> shapeC = common::getBmmOutputShape(A, xB);
  Tensor C = createCudaTensorHalf(shapeC);

  int nBatchDim = A.getDim() - 2;

  common::GEMMArgs gemmArgs = common::generateGemmArgs(A, xB, C);
  std::vector<const half *> batchA = getBatch(A, nBatchDim);
  std::vector<const half *> batchB = getBatch(xB, nBatchDim);
  std::vector<const half *> batchC = getBatch(C, nBatchDim);
  CHECK(batchA.size() == batchB.size() && batchA.size() == batchC.size());

  int64_t nb = batchA.size();
  ly::c_ptr<const void *> arrayA = llynCudaAlloc<const void *>(nb);
  ly::c_ptr<const void *> arrayB = llynCudaAlloc<const void *>(nb);
  ly::c_ptr<void *> arrayC = llynCudaAlloc<void *>(nb);

  int64_t nc = sizeof(void *) * nb;
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayA.get(), batchA.data(), nc, cudaMemcpyHostToDevice));
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayB.get(), batchB.data(), nc, cudaMemcpyHostToDevice));
  LL_CHECK_CUDA_STATUS(cudaMemcpy(arrayC.get(), batchC.data(), nc, cudaMemcpyHostToDevice));

  half alpha = 1.0;
  half beta = 0.0;
  CHECK_CUBLAS_STATUS(cublasGemmBatchedEx(
      _handle.get(),
      gemmArgs.transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      gemmArgs.transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      gemmArgs.N,
      gemmArgs.M,
      gemmArgs.K,
      &alpha,
      arrayB.get(),
      CUDA_R_16F,
      gemmArgs.ldb,
      arrayA.get(),
      CUDA_R_16F,
      gemmArgs.lda,
      &beta,
      arrayC.get(),
      CUDA_R_16F,
      gemmArgs.ldc,
      nb,
      CUBLAS_COMPUTE_16F,
      CUBLAS_GEMM_DEFAULT));

  cudaDeviceSynchronize();
  return C;
}

Tensor MatMul::bmmToGemmHalf(const Tensor &A, const Tensor &B) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(-1)});
  Tensor xC = gemmHalf(xA, B);

  shape.back() = B.getShape(1);
  return xC.view(shape);
}

}  // cuda
}  // op
}  // llyn
