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

#include "catch2/catch_amalgamated.hpp"

#include <omp.h>
#include <math.h>
#include "libllm/cpu/kernel/unittest_common.h"
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/kernel/cvt_h.h"
#include "libllm/cpu/kernel/dequant_sq4.h"
#include "libllm/cpu/kernel/kernel_sq4.h"
#include "libllm/cpu/kernel/kernel_s.h"
#include "libllm/cpu/kernel/kernel_h.h"
#include "libllm/cpu/kernel/kernel_hq4.h"
#include "libllm/cpu/kernel/util.h"
#include "libllm/lut/half.h"
#include "libllm/lut/random.h"
#include "libllm/lut/log.h"

namespace libllm {
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

float callDotQInt4(int64_t n, const float *x, DataQInt4 y, int64_t offsetY) {
  return SQInt4DotFallbackKernel::apply(n, x, y, offsetY);
}

Float16 callDotQInt4(int64_t n, const Float16 *x, DataQInt4 y, int64_t offsetY) {
  return HQInt4DotFallbackKernel::apply(n, x, y, offsetY);
}

template<typename T>
void refGemmQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const T *A,
    int lda,
    const UInt4x2 *B,
    const Float16 *scaleB,
    const UInt4x2 *zeroB,
    T *C,
    int ldc) {
  CHECK(transA == false);
  std::fill_n(C, M * N, 0.0f);

  for (int j = 0; j < M; ++j) {
    if (transB) {
      for (int i = 0; i < N; ++i) {
        const T *Aj = A + j * lda;
        C[j * ldc + i] = callDotQInt4(K, Aj, {B, scaleB, zeroB}, i * K);
      }
    } else {
      NOT_IMPL();
    }
  }
}

inline void callGemmQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const Float16 *A,
    int lda,
    const UInt4x2 *dataB,
    const Float16 *scaleB,
    const UInt4x2 *zeroPointB,
    Float16 *C,
    int ldc) {
  return gemmHalfQInt4(transA, transB, M, N, K, A, lda, dataB, scaleB, zeroPointB, C, ldc);
}

inline void callGemmQInt4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    const float *A,
    int lda,
    const UInt4x2 *dataB,
    const Float16 *scaleB,
    const UInt4x2 *zeroPointB,
    float *C,
    int ldc) {
  return gemmFloatQInt4(transA, transB, M, N, K, A, lda, dataB, scaleB, zeroPointB, C, ldc);
}

template<typename T>
void testGemmQInt4(bool transB, int M, int N, int K) {
  std::vector<T> A(M * K);
  std::vector<UInt4x2> B(K * N / 2);
  std::vector<Float16> scaleB(K * N / GroupSizeQInt4);
  std::vector<UInt4x2> zeroB(K * N / GroupSizeQInt4 / 2);

  lut::Random random(MagicNumber);

  fillRandom(&random, lut::makeSpan(A));
  fillRandomQInt4(&random, lut::makeSpan(B), lut::makeSpan(scaleB), lut::makeSpan(zeroB));
  std::vector<T> C(M * N);
  std::vector<T> Cr(M * N);

  refGemmQInt4(
      false,
      transB,
      M,
      N,
      K,
      A.data(),
      K,
      reinterpret_cast<const UInt4x2 *>(B.data()),
      scaleB.data(),
      reinterpret_cast<const UInt4x2 *>(zeroB.data()),
      Cr.data(),
      N);

  callGemmQInt4(
      false,
      transB,
      M,
      N,
      K,
      A.data(),
      K,
      (const UInt4x2 *)B.data(),
      (const Float16 *)scaleB.data(),
      (const UInt4x2 *)zeroB.data(),
      C.data(),
      N);

  CATCH_REQUIRE(getMaxDiff<T>(C, Cr) / getMeanAbs<T>(Cr) < 0.05);
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
  return gemmFloat(transA, transB, M, N, K, A, lda, B, ldb, C, ldc);
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
  return gemmHalf(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, Mode::SingleThread);
}

int gemmTestShapes[][3] = {
  {   1, 2048, 2048},
  { 256,  256,  256},
  {   2,    2,    2},
  {  50,   50,    1},
  {   1,   50,   50},
  {  50,   50,    2},
  {   2,   50,   50},
  { 513,    2,  513},
  { 200,    1,  300},
  {   1,  200,  300},
  { 300,  200,    1},
  {   2,  200,  300},
  { 300,  200,    2},
  {  16,  512,   16},
  {  16, 1024,   16},
  {  16,   16, 2048},
  {2048,   16,   16},
  {2048, 2048,    1},
  {   2, 2048, 2048},
  {2048, 2048,    2},
  {   0,    0,    0}
};


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

CATCH_TEST_CASE("test sgemm", "[op][cpu][kernel][sgemm]") {
  int (*pshape)[3];
  
  for (pshape = &gemmTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];

    testGemm<float>(true, true, m, n, k);
    testGemm<float>(true, false, m, n, k);
    testGemm<float>(false, true, m, n, k);
    testGemm<float>(false, false, m, n, k);
  }
}

void testHalfToFloat(int n) {
  std::vector<float> y(n);
  std::vector<float> yr(n);
  std::vector<Float16> x(n);

  lut::Random random(MagicNumber);
  random.fill(lut::makeSpan(yr));
  std::transform(yr.begin(), yr.end(), x.begin(), [](float x) {
    return cvt_s2h(x);
  });

  convertHalfToFloat(n, x.data(), y.data());
  CATCH_REQUIRE(isClose<float>(yr, y, 1e-4, 1e-3));
}

#ifdef LUT_ARCH_AMD64

CATCH_TEST_CASE("test sqint4gemm", "[lymath][api][q4]") {
  testGemmQInt4<float>(true, 1, 32, 128);
  testGemmQInt4<float>(true, 1, 64, 4096);
  testGemmQInt4<float>(true, 64, 64, 256);
}

CATCH_TEST_CASE("test lymath_half2float", "[lymath][cvt]") {
  std::vector<int> ns{1, 50, 200, 800, 1600, 1601, 3200, 3201};
  for (int n : ns) {
    testHalfToFloat(n);
  }
}


#endif  // LUT_ARCH_AMD64

#ifdef LUT_ARCH_AARCH64
CATCH_TEST_CASE("test hgemm", "[op][cpu][kernel][hgemm]") {
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

CATCH_TEST_CASE("test gemmHalfQInt4", "[op][cpu][kernel][hqint4gemm]") {
  testGemmQInt4<Float16>(true, 1, 32, 128);
  testGemmQInt4<Float16>(true, 1, 1023, 2048);
  testGemmQInt4<Float16>(true, 64, 200, 4096);
}

#endif  // LUT_ARCH_AARCH64

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
