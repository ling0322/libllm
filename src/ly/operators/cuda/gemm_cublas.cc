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

#include "ly/operators/cuda/gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace ly {
namespace op {
namespace cuda {

std::shared_ptr<Gemm> CublasGemm::create() {
  std::shared_ptr<CublasGemm> mm = std::make_shared<CublasGemm>();
  mm->_handle = {nullptr, safeDestroyCublas};
  if (CUBLAS_STATUS_SUCCESS != cublasCreate(mm->_handle.get_pp())) {
    return nullptr;
  } else {
    return mm;
  }
}

void CublasGemm::safeDestroyCublas(cublasHandle_t handle) {
  cublasStatus_t status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    LOG(ERROR) << "Error while calling cublasDestroy(): " << cublasGetStatusString(status);
  }
}

lut::ErrorCode CublasGemm::hgemm(
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
  float alphaFp32 = static_cast<float>(alpha);
  float betaFp32 = static_cast<float>(beta);

  cublasStatus_t status = cublasGemmEx(
      _handle.get(),
      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      n,
      m,
      k,
      &alphaFp32,
      B,
      CUDA_R_16F,
      ldb,
      A,
      CUDA_R_16F,
      lda,
      &betaFp32,
      C,
      CUDA_R_16F,
      ldc,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  cudaDeviceSynchronize();

  if (status == CUBLAS_STATUS_SUCCESS)
    return lut::ErrorCode::OK;
  else
    return lut::ErrorCode::Aborted;
}

lut::ErrorCode CublasGemm::hgemmArray(
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
  float alphaFp32 = static_cast<float>(alpha);
  float betaFp32 = static_cast<float>(beta);

  cublasStatus_t status = cublasGemmBatchedEx(
      _handle.get(),
      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      n,
      m,
      k,
      &alphaFp32,
      reinterpret_cast<const void *const *>(arrayB),
      CUDA_R_16F,
      ldb,
      reinterpret_cast<const void *const *>(arrayA),
      CUDA_R_16F,
      lda,
      &betaFp32,
      reinterpret_cast<void *const *>(arrayC),
      CUDA_R_16F,
      ldc,
      batchSize,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);

  cudaDeviceSynchronize();

  if (status == CUBLAS_STATUS_SUCCESS)
    return lut::ErrorCode::OK;
  else
    return lut::ErrorCode::Aborted;
}


}  // cuda
}  // op
}  // ly

std::shared_ptr<ly::op::cuda::Gemm> llynCreateCudaOpExtGemm() {
  return ly::op::cuda::CublasGemm::create();
}
