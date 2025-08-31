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

#include "lynn/cpu/kernel/interface.h"

#include <math.h>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/half.h"
#include "lutil/log.h"
#include "lutil/random.h"
#include "lynn/cpu/kernel/test_common.h"
#include "lynn/cpu/kernel/util.h"

namespace ly {
namespace op {
namespace cpu {
namespace kernel {

template<typename T>
void refGemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const T *A,
    int lda,
    const T *B,
    int ldb,
    T *C,
    int ldc) {
  int stride0A = transA ? 1 : lda;
  int stride1A = transA ? lda : 1;
  int stride0B = transB ? 1 : ldb;
  int stride1B = transB ? ldb : 1;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = C[ldc * m + n];
      for (int k = 0; k < K; ++k) {
        T va = A[stride0A * m + k * stride1A];
        T vb = B[stride0B * k + n * stride1B];
        sum += va * vb;
      }
      C[ldc * m + n] = cvtf<T>(sum);
    }
  }
}

inline void callGemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    float *C,
    int ldc) {
  return gemmFloat(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, Mode::OMP);
}

inline void callGemm(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const Float16 *B,
    int ldb,
    Float16 *C,
    int ldc) {
  return gemmHalf(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, Mode::OMP);
}

int gemmTestShapes[][3] = {{1, 2048, 2048}, {256, 256, 256}, {2, 2, 2},       {50, 50, 1},
                           {1, 50, 50},     {50, 50, 2},     {2, 50, 50},     {513, 2, 513},
                           {200, 1, 300},   {1, 200, 300},   {300, 200, 1},   {2, 200, 300},
                           {300, 200, 2},   {16, 512, 16},   {16, 1024, 16},  {16, 16, 2048},
                           {2048, 16, 16},  {2048, 2048, 1}, {2, 2048, 2048}, {2048, 2048, 2},
                           {0, 0, 0}};

template<typename T>
void testGemm(bool transA, bool transB, int M, int N, int K) {
  std::vector<T> A(M * K);
  std::vector<T> B(K * N);

  lut::Random random(MagicNumber);

  fillRandom(&random, lut::makeSpan(A));
  fillRandom(&random, lut::makeSpan(B));

  std::vector<T> C(M * N);
  std::vector<T> refC(M * N);
  memset(C.data(), 0, C.size() * sizeof(T));
  memset(refC.data(), 0, refC.size() * sizeof(T));

  refGemm<T>(
      transA,
      transB,
      M,
      N,
      K,
      A.data(),
      transA ? M : K,
      B.data(),
      transB ? K : N,
      refC.data(),
      N);

  callGemm(
      transA,
      transB,
      M,
      N,
      K,
      A.data(),
      transA ? M : K,
      B.data(),
      transB ? K : N,
      C.data(),
      N);

  CATCH_REQUIRE(getMaxDiff<T>(C, refC) / getMeanAbs<T>(refC) < 0.05);
}

void testHalfToFloat(int n) {
  std::vector<float> y(n);
  std::vector<float> yr(n);
  std::vector<Float16> x(n);

  lut::Random random(MagicNumber);
  random.fill(lut::makeSpan(yr));
  std::transform(yr.begin(), yr.end(), x.begin(), [](float x) { return cvt_s2h(x); });

  convertHalfToFloat(n, x.data(), y.data(), Mode::OMP);
  CATCH_REQUIRE(isClose<float>(yr, y, 1e-4, 1e-3));
}

#ifdef LUT_ARCH_AMD64

CATCH_TEST_CASE("test lymath_half2float", "[cpu_kernel][interface][cvt]") {
  std::vector<int> ns{1, 50, 200, 800, 1600, 1601, 3200, 3201};
  for (int n : ns) {
    testHalfToFloat(n);
  }
}

#endif  // LUT_ARCH_AMD64

#ifdef LUT_ARCH_AARCH64
CATCH_TEST_CASE("test hgemm", "[cpu_kernel][interface][hgemm]") {
  int (*pshape)[3];

  for (pshape = &gemmTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];

    testGemm<Float16>(true, true, m, n, k);
    testGemm<Float16>(true, false, m, n, k);
    testGemm<Float16>(false, true, m, n, k);
    testGemm<Float16>(false, false, m, n, k);
  }
}

#endif  // LUT_ARCH_AARCH64

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace ly
