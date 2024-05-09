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

#include "catch2/catch_amalgamated.hpp"

#ifdef MKL_ENABLED
#include <mkl.h>
#endif

#include <chrono>
#include <functional>

#include "libllm/cpu/kernel/abstract.h"
#include "libllm/cpu/kernel/gemm.h"
#include "libllm/cpu/kernel/interface.h"
#include "libllm/lut/attributes.h"
#include "libllm/lut/log.h"
#include "libllm/lut/strings.h"
#include "libllm/lut/time.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

void benchmarkPack(Block<float> A, Block<float> Ap, int KC) {
  double t0 = lut::now();
  int kb = (A.numRows + KC - 1) / KC;
  int lastKc = A.numRows % KC;
  for (int i = 0; i < kb; ++i) {
    int kc = (i != kb - 1 || lastKc == 0) ? KC : lastKc;
    Block<float> Ai = A.sliceRow(i * KC, kc);
    Pack<float, Mode::OMP>(Ai, Ap, Ap.stride);
  }
  LOG(INFO) << lut::sprintf(
      "pack (%d, %d) stride=%d KC=%d T=%d: %f",
      A.numRows,
      A.numCols,
      A.stride,
      KC,
      A.transposed,
      lut::now() - t0);
}

double benchmarkSgemm(int M, int K, int N, int numLoops = 2) {
  std::vector<float> dA(M * K);
  std::vector<float> dB(K * N);
  std::vector<float> dC(M * N);

  double t0 = lut::now();
  for (int i = 0; i < numLoops; ++i)
    libllm::op::cpu::kernel::gemmFloat(
        false,
        true,
        M,
        N,
        K,
        dA.data(),
        K,
        dB.data(),
        K,
        dC.data(),
        N,
        Mode::OMP);

  double dt = (lut::now() - t0) / numLoops;
  return dt;
}

double benchmarkHgemm(int M, int K, int N, int numLoops = 2) {
  std::vector<Float16> dA(M * K);
  std::vector<Float16> dB(K * N);
  std::vector<Float16> dC(M * N);

  double t0 = lut::now();
  for (int i = 0; i < numLoops; ++i)
    libllm::op::cpu::kernel::gemmHalf(
        false,
        true,
        M,
        N,
        K,
        dA.data(),
        K,
        dB.data(),
        K,
        dC.data(),
        N,
        Mode::OMP);

  double dt = (lut::now() - t0) / numLoops;
  return dt;
}

#ifdef MKL_ENABLED
double benchmarkMklSgemm(int M, int K, int N, int numLoops = 2) {
  std::vector<float> dA(M * K);
  std::vector<float> dB(K * N);
  std::vector<float> dC(M * N);

  double t0 = lut::now();
  for (int i = 0; i < numLoops; ++i)
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasTrans,
        M,
        N,
        K,
        1.0f,
        dA.data(),
        K,
        dB.data(),
        K,
        0.0f,
        dC.data(),
        N);

  double dt = (lut::now() - t0) / numLoops;
  return dt;
}
#endif

CATCH_TEST_CASE("benchmark Pack", "[benchmark][cpu_kernel][pack]") {
  constexpr int ROW = 4096;
  constexpr int COL = 4096;
  constexpr int NR = 16;
  constexpr int NC = 512;

  std::vector<float> dA(ROW * COL);
  std::vector<float> dAp(ROW * NC);

  Block<float> A = Block<float>{dA.data(), ROW, ROW, COL, true};
  Block<float> Ap = Block<float>{dAp.data(), NR, ROW * NC / NR, NR, false};
  benchmarkPack(A, Ap, NC);
}

int gemmBenchmarkShapes[][4] = {
    {17, 4096, 27392, 2},
    {17, 13696, 4096, 2},
    {4096, 4096, 4096, 10},
    {1, 4096, 27392, 10},
    {1, 13696, 4096, 10},
    {0, 0, 0, 0}};

#if LUT_CPU_ARCH == LUT_AMD64

CATCH_TEST_CASE("benchmark SGEMM", "[benchmark][cpu_kernel][sgemm]") {
  int(*pshape)[4];

  for (pshape = &gemmBenchmarkShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];
    int numLoops = (*pshape)[3];

    double dLlm = benchmarkSgemm(m, k, n, numLoops);
#ifdef MKL_ENABLED
    double dMkl = benchmarkMklSgemm(m, k, n, numLoops);
    LOG(INFO) << lut::sprintf("SGEMM (M,K,N)=(%d,%d,%d): mkl=%f libllm=%f", m, k, n, dMkl, dLlm);
#else
    LOG(INFO) << lut::sprintf("SGEMM (M,K,N)=(%d,%d,%d): libllm=%f", m, k, n, dLlm);
#endif
  }
}

#endif  // LUT_CPU_ARCH == LUT_AMD64

#if LUT_CPU_ARCH == LUT_AARCH64

CATCH_TEST_CASE("benchmark HGEMM", "[benchmark][cpu][cpu_kernel][hgemm]") {
  int(*pshape)[4];

  for (pshape = &gemmBenchmarkShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];
    int numLoops = (*pshape)[3];

    double dLlm = benchmarkHgemm(m, k, n, numLoops);
    LOG(INFO) << lut::sprintf("HGEMM (M,K,N)=(%d,%d,%d): libllm=%f", m, k, n, dLlm);
  }
}

#endif  // LUT_CPU_ARCH == LUT_AMD64

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
