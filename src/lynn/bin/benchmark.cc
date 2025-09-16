// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include "catch2/catch_amalgamated.hpp"
#include "lynn/cuda/cuda_operators.h"
#include "lynn/operator_benchmark.h"

using ly::op::cuda::CudaOperators;

void benchmarkMatmul(int matmulOption) {
  ly::OperatorBenchmark b;
  b = b.withWarmUpLoop(5).withLoop(10).withDType(ly::DType::kFloat16);
  b = b.withOperators(CudaOperators::create(matmulOption));

  int shapes[][3] = {
      {4096, 4096, 4096},
      {1, 4096, 4096},
      {16, 4096, 4096},
      {128, 4096, 4096},
      {256, 4096, 4096},
      {256, 256, 256},
      {1, 256, 256},
      {0}};

  for (int i = 0; shapes[i][0] != 0; ++i) {
    int m = shapes[i][0];
    int n = shapes[i][1];
    int k = shapes[i][2];

    b.benchmarkMatMul(m, n, k, false, false);
    b.benchmarkMatMul(m, n, k, false, true);
  }

  b.printResult();
}

CATCH_TEST_CASE("matmul cublas vs cutlass", "[matmul][cuda]") {
  benchmarkMatmul(CudaOperators::OPT_CUBLAS_GEMM);
  benchmarkMatmul(CudaOperators::OPT_CUTLASS_GEMM);
}

int main(int argc, char **argv) {
  ly::initOperators();

  int result = Catch::Session().run(argc, argv);

  ly::destroyOperators();
  return result;
}
