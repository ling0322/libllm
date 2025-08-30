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

#pragma once

#include <cublas_v2.h>

#include "lynn/cuda/gemm.h"

#if defined(_WIN32)
#define EXTAPI __declspec(dllexport)
#else
#define EXTAPI
#endif

namespace libllm {
namespace op {
namespace cuda {

/// @brief Operators implemented by cuBLAS.
class CublasGemm : public Gemm {
 public:
  static Gemm *create();

  lut::ErrorCode hgemm(
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
      int ldc) override;

  lut::ErrorCode hgemmArray(
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
      int batchSize) override;

 private:
  auto_handle<cublasHandle_t> _handle;
  static void safeDestroyCublas(cublasHandle_t handle);
};

}  // namespace cuda
}  // namespace op
}  // namespace libllm

extern "C" {
EXTAPI libllm::op::cuda::Gemm *llmGemmExt_New();
EXTAPI void llmGemmExt_Delete(libllm::op::cuda::Gemm *gemm);
}  // extern "C"
