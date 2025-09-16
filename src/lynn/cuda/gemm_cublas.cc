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

#include "lynn/cuda/gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "lutil/shared_library.h"

#define LL_CHECK_CUBLAS_STATUS(x)                                                          \
  {                                                                                        \
    cublasStatus_t status = x;                                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                                 \
      LOG(ERROR) << "Error while calling: " << #x << ": " << cublasGetErrorString(status); \
      throw lut::AbortedError(cublasGetErrorString(status));                               \
    }                                                                                      \
  }

extern "C" {

typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasGemmBatchedExFunc_t)(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void *alpha,
    const void *const Aarray[],
    cudaDataType Atype,
    int lda,
    const void *const Barray[],
    cudaDataType Btype,
    int ldb,
    const void *beta,
    void *const Carray[],
    cudaDataType Ctype,
    int ldc,
    int batchCount,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);

typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasGemmExFunc_t)(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const void *alpha,
    const void *A,
    cudaDataType Atype,
    int lda,
    const void *B,
    cudaDataType Btype,
    int ldb,
    const void *beta,
    void *C,
    cudaDataType Ctype,
    int ldc,
    cublasComputeType_t computeType,
    cublasGemmAlgo_t algo);
}

typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasCreateFunc_t)(cublasHandle_t *handle);
typedef CUBLASAPI cublasStatus_t CUBLASWINAPI (*cublasDestroyFunc_t)(cublasHandle_t handle);

namespace ly {
namespace op {
namespace cuda {

const char *cublasGetErrorString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:
      return "Unknown cuBLAS error";
  }
}

class CublasGemm::Impl {
 public:
  cublasGemmBatchedExFunc_t _cublasGemmBatchedEx;
  cublasGemmExFunc_t _cublasGemmEx;
  cublasCreateFunc_t _cublasCreate;
  cublasDestroyFunc_t _cublasDestroy;
  cublasHandle_t _handle;

  static std::unique_ptr<Impl> create() {
    std::unique_ptr<Impl> impl = std::make_unique<Impl>();

    impl->_libCublas = lut::SharedLibrary::open("cublas");
    impl->_cublasGemmBatchedEx = impl->_libCublas->getFunc<cublasGemmBatchedExFunc_t>(
        "cublasGemmBatchedEx");
    impl->_cublasGemmEx = impl->_libCublas->getFunc<cublasGemmExFunc_t>("cublasGemmEx");
    impl->_cublasCreate = impl->_libCublas->getFunc<cublasCreateFunc_t>("cublasCreate_v2");
    impl->_cublasDestroy = impl->_libCublas->getFunc<cublasDestroyFunc_t>("cublasDestroy_v2");

    LL_CHECK_CUBLAS_STATUS(impl->_cublasCreate(&impl->_handle));
    return impl;
  }

  Impl()
      : _handle(nullptr) {
  }

  ~Impl() {
    if (_cublasDestroy && _handle) {
      cublasStatus_t status = _cublasDestroy(_handle);
      if (status != CUBLAS_STATUS_SUCCESS) {
        LOG(ERROR) << "Error while calling cublasDestroy(): " << cublasGetErrorString(status);
      }

      _handle = nullptr;
    }
  }

 private:
  std::unique_ptr<lut::SharedLibrary> _libCublas;
};

std::shared_ptr<Gemm> CublasGemm::create() {
  std::shared_ptr<CublasGemm> mm{new CublasGemm()};
  mm->_impl = Impl::create();

  return mm;
}

void CublasGemm::hgemm(
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

  LL_CHECK_CUBLAS_STATUS(_impl->_cublasGemmEx(
      _impl->_handle,
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
      CUBLAS_GEMM_DEFAULT));
}

void CublasGemm::hgemmArray(
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

  LL_CHECK_CUBLAS_STATUS(_impl->_cublasGemmBatchedEx(
      _impl->_handle,
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
      CUBLAS_GEMM_DEFAULT));
}

}  // namespace cuda
}  // namespace op
}  // namespace ly