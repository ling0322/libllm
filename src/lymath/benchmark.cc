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

#include "third_party/catch2/catch_amalgamated.hpp"

#ifdef MKL_ENABLED
#include <mkl.h>
#endif

#include <chrono>
#include <functional>
#include "lyutil/log.h"
#include "lymath/lymath.h"
#include "lyutil/strings.h"
#include "lyutil/time.h"
#include "lymath/common.h"
#include "lymath/gemm_common.h"

using lymath::Block;
using lymath::Mode;
using lymath::Pack;
using lymath::PackedBlock;

void benchmarkPack(Block A, Block Ap, int KC) {
  double t0 = lut::now();
  int kb = (A.numRows + KC - 1) / KC;
  int lastKc = A.numRows % KC;
  for (int i = 0; i < kb; ++i) {
    int kc = (i != kb - 1 || lastKc == 0) ? KC : lastKc;
    Block Ai = A.sliceRow(i * KC, kc);
    Pack<Mode::OMP>(Ai, Ap, Ap.stride);
  }
  LOG(INFO) << lut::sprintf("pack (%d, %d) stride=%d KC=%d T=%d: %f",
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
    lymath_sgemm(
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
        N);

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

CATCH_TEST_CASE("benchmark Pack", "[benchmark][lymath][pack]") {
  constexpr int ROW = 4096;
  constexpr int COL = 4096;
  constexpr int NR = 16;
  constexpr int NC = 512;

  std::vector<float> dA(ROW * COL);
  std::vector<float> dAp(ROW * NC);

  Block A = Block { dA.data(), ROW, ROW, COL, true };
  Block Ap = Block { dAp.data(), NR, ROW * NC / NR, NR, false };
  benchmarkPack(A, Ap, NC);
}

int gemmBenchmarkShapes[][4] = {
  {17, 4096, 27392, 2},
  {17, 13696, 4096, 2},
  {1, 4096, 27392, 10},
  {1, 13696, 4096, 10},
  {0, 0, 0, 0}
};

CATCH_TEST_CASE("benchmark SGEMM", "[benchmark][lymath][sgemm]") {
  int (*pshape)[4];
  
  for (pshape = &gemmBenchmarkShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];
    int numLoops = (*pshape)[3];

    double dLymath = benchmarkSgemm(m, k, n, numLoops);
#ifdef MKL_ENABLED
    double dMkl = benchmarkMklSgemm(m, k, n, numLoops);
    LOG(INFO) << lut::sprintf(
        "MKL SGEMM (M,K,N)=(%d,%d,%d): mkl=%f lymath=%f", m, k, n, dMkl, dLymath);
#else
    LOG(INFO) << lut::sprintf("MKL SGEMM (M,K,N)=(%d,%d,%d): lymath=%f", m, k, n, dLymath);
#endif
  }
}
