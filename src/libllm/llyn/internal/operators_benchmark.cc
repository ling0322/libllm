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

#include <chrono>
#include <functional>

#include "cblas.h"
#include "common/environment.h"
#include "common/test_helper.h"
#include "llyn/module.h"
#include "llyn/operators.h"
#include "pmpack/pmpack.h"
#include "util/strings.h"

using namespace nn;
using namespace std::literals;

Tensor callGEMM(Operators *F, TensorCRef A, TensorCRef B) {
  return F->matmul(A, B);
}

Tensor callGEMV(Operators *F, TensorCRef A, TensorCRef B) {
  return F->matmul(A, B);
}

Tensor callOpenblasGEMM(Operators *F, TensorCRef A, TensorCRef B) {
  Tensor C = F->createTensor({A.getShape(0), B.getShape(1)}, DType::kFloat);
  cblas_sgemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      A.getShape(0), B.getShape(1), A.getShape(0),
      1.0f,
      A.getData<float>(), A.getStride(0),
      B.getData<float>(), B.getStride(0),
      0.0f,
      C.getData<float>(), C.getStride(0));

  return C;
}

Tensor callOpenblasGEMV(Operators *F, TensorCRef A, TensorCRef B) {
  Tensor C = F->createTensor({A.getShape(1)}, DType::kFloat);
  int lda = A.getStride(0) > 1 ? A.getStride(0) : A.getStride(1);
  cblas_sgemv(
      CblasRowMajor, CblasNoTrans, A.getShape(0), A.getShape(1), 1.0f, A.getData<float>(),
      lda, B.getData<float>(), 1, 0.0f, C.getData<float>(), 1);

  return C;
}

enum GEMMType {
  GEMM_LLMRT,
  GEMM_OPENBLAS,
  GEMV_LLMRT,
  GEMV_OPENBLAS,
  GEMV_TRANS_LLMRT,
  GEMV_TRANS_OPENBLAS
};


void benchmarkGEMM(int n, GEMMType gemmType, int numRun = 1) {
  auto F = Operators::create(Device::createForCPU());

  Tensor A = F->rand({n, n}, DType::kFloat);
  Tensor B = gemmType == GEMM_LLMRT || gemmType == GEMM_OPENBLAS
      ? F->rand({n, n}, DType::kFloat)
      : F->rand({n, 1}, DType::kFloat);

  if (gemmType == GEMV_TRANS_LLMRT || gemmType == GEMV_TRANS_OPENBLAS) {
    A = A.transpose(0, 1);
  }

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < numRun; ++i) {
    Tensor C;
    switch (gemmType) {
      case GEMM_LLMRT:
        C = callGEMM(F.get(), A, B);
        break;
      case GEMM_OPENBLAS:
        C = callOpenblasGEMM(F.get(), A, B);
        break;
      case GEMV_LLMRT:
        C = callGEMV(F.get(), A, B);
        break;
      case GEMV_OPENBLAS:
        C = callOpenblasGEMV(F.get(), A, B);
        break;
      case GEMV_TRANS_LLMRT:
        C = callGEMV(F.get(), A, B);
        break;
      case GEMV_TRANS_OPENBLAS:
        C = callOpenblasGEMV(F.get(), A, B);
        break;
      default:
        NOT_IMPL();
    }
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  auto delta = t1 - t0;
  auto duration_ms = delta / 1ns / numRun / 1e6f;

  LOG(INFO) << str::sprintf("n = %d t = %.2f ms", n, duration_ms);
}

void benchmarkSGEMM() {
  LOG(INFO) << "openblas SGEMM:";
  benchmarkGEMM(256, GEMM_OPENBLAS, 400);
  benchmarkGEMM(512, GEMM_OPENBLAS, 200);
  benchmarkGEMM(1024, GEMM_OPENBLAS, 50);
  benchmarkGEMM(2048, GEMM_OPENBLAS, 5);
  benchmarkGEMM(4096, GEMM_OPENBLAS, 1);

  LOG(INFO) << "LLmRT SGEMM:";
  benchmarkGEMM(256, GEMM_LLMRT, 400);
  benchmarkGEMM(512, GEMM_LLMRT, 200);
  benchmarkGEMM(1024, GEMM_LLMRT, 50);
  benchmarkGEMM(2048, GEMM_LLMRT, 5);
  benchmarkGEMM(4096, GEMM_LLMRT, 1);

  LOG(INFO) << " ";
}

void benchmarkSGEMV() {
  LOG(INFO) << "openblas SGEMV:";
  benchmarkGEMM(256, GEMV_OPENBLAS, 400);
  benchmarkGEMM(512, GEMV_OPENBLAS, 200);
  benchmarkGEMM(1024, GEMV_OPENBLAS, 50);
  benchmarkGEMM(2048, GEMV_OPENBLAS, 5);
  benchmarkGEMM(4096, GEMV_OPENBLAS, 1);

  LOG(INFO) << "LLmRT SGEMV:";
  benchmarkGEMM(256, GEMV_LLMRT, 400);
  benchmarkGEMM(512, GEMV_LLMRT, 200);
  benchmarkGEMM(1024, GEMV_LLMRT, 50);
  benchmarkGEMM(2048, GEMV_LLMRT, 5);
  benchmarkGEMM(4096, GEMV_LLMRT, 1);

  LOG(INFO) << "openblas TransA SGEMV:";
  benchmarkGEMM(256, GEMV_TRANS_OPENBLAS, 400);
  benchmarkGEMM(512, GEMV_TRANS_OPENBLAS, 200);
  benchmarkGEMM(1024, GEMV_TRANS_OPENBLAS, 50);
  benchmarkGEMM(2048, GEMV_TRANS_OPENBLAS, 5);
  benchmarkGEMM(4096, GEMV_TRANS_OPENBLAS, 1);

  LOG(INFO) << "LLmRT TransA SGEMV:";
  benchmarkGEMM(256, GEMV_TRANS_LLMRT, 400);
  benchmarkGEMM(512, GEMV_TRANS_LLMRT, 200);
  benchmarkGEMM(1024, GEMV_TRANS_LLMRT, 50);
  benchmarkGEMM(2048, GEMV_TRANS_LLMRT, 5);
  benchmarkGEMM(4096, GEMV_TRANS_LLMRT, 1);

  LOG(INFO) << " ";
}

CATCH_TEST_CASE("benchmark for float32 GEMM", "[gemm][benchmark]") {
  openblas_set_num_threads(1);
  lymath_set_num_threads(1);

  LOG(INFO) << "Benchmark SGEMM with numThreads=1";
  benchmarkSGEMM();
  
  openblas_set_num_threads(8);
  lymath_set_num_threads(8);

  LOG(INFO) << "Benchmark SGEMM with numThreads=8";
  benchmarkSGEMM();
}

CATCH_TEST_CASE("benchmark for float32 GEMV", "[gemv][benchmark]") {
  openblas_set_num_threads(1);
  lymath_set_num_threads(1);

  LOG(INFO) << "Benchmark SGEMV with numThreads=1";
  benchmarkSGEMV();

  openblas_set_num_threads(8);
  lymath_set_num_threads(8);

  LOG(INFO) << "Benchmark SGEMV with numThreads=8";
  benchmarkSGEMV();
}
