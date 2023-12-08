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

#include "../../third_party/catch2/catch_amalgamated.hpp"

#include <omp.h>
#include "lymath/common.h"
#include "lymath/lymath.h"
#include "lymath/q4kernel.h"
#include "lymath/q8kernel.h"
#include "lymath/skernel.h"
#include "lymath/util.h"
#include "lyutil/half.h"
#include "lyutil/random.h"
#include "lyutil/log.h"

using namespace lymath;

constexpr uint32_t MagicNumber = 0x55aa;

void refSgemm(
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
  int stride0A = transA ? 1 : lda;
  int stride1A = transA ? lda : 1;
  int stride0B = transB ? 1 : ldb;
  int stride1B = transB ? ldb : 1;

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      for (int k = 0; k < K; ++k) {
        float va = A[stride0A * m + k * stride1A];
        float vb = B[stride0B * k + n * stride1B];
        C[ldc * m + n] += va * vb;
      }
    }
  }
}

bool isClose(float l, float r) {
  constexpr float atol = 1e-5;
  constexpr float rtol = 1e-5;

  return fabs(l - r) <= atol + rtol * fabs(r);
}

bool isClose(lut::Span<const float> A, lut::Span<const float> B) {
  if (A.size() != B.size()) 
    return false;

  for (int i = 0; i < A.size(); ++i) {
    if (!isClose(A[i], B[i])) {
      printf("%d: %f vs %f\n", i, A[i], B[i]);
      return false;
    }
  }

  return true;
}

CATCH_TEST_CASE("test q4 dequantization", "[lymath][dequant][q4]") {
  constexpr int DIM = DequantMinElemPerThread + Q4GroupSize;

  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleXFp32(DIM / Q4GroupSize);
  std::vector<Fp16> scaleX(DIM / Q4GroupSize);
  std::vector<UInt8> zeroX(DIM / Q4GroupSize / 2);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);

  lut::Random random(MagicNumber);

  random.fillUInt8(lut::makeSpan(x));
  random.fillUInt8(lut::makeSpan(zeroX));
  random.fill(lut::makeSpan(scaleXFp32));
  std::transform(scaleXFp32.begin(), scaleXFp32.end(), scaleX.begin(), lut::cvtss_sh);

  DequantQ4FallbackKernel::apply(DIM, x.data(), scaleX.data(), zeroX.data(), yRef.data());
  DequantQ4Avx2Kernel::apply(DIM, x.data(), scaleX.data(), zeroX.data(), y.data());
  CATCH_REQUIRE(isClose(y, yRef));
}

CATCH_TEST_CASE("test q4 axpy kernels", "[lymath][axpy][q4]") {
  constexpr int DIM = 1024;

  float a = 0.1f;
  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleFp32(DIM / Q4GroupSize);
  std::vector<UInt8> zeros(DIM / Q4GroupSize / 2);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);
  std::vector<Fp16> scale(DIM / Q4GroupSize);

  lut::Random random(MagicNumber);
  random.fillUInt8(lut::makeSpan(x));
  random.fillUInt8(lut::makeSpan(zeros));
  random.fill(lut::makeSpan(scaleFp32));
  random.fill(lut::makeSpan(yRef));
  std::transform(scaleFp32.begin(), scaleFp32.end(), scale.begin(), lut::cvtss_sh);

  std::copy(yRef.begin(), yRef.end(), y.begin());

  AxpyQ4Avx2Kernel::apply(DIM, a, x.data(), scale.data(), zeros.data(), y.data());
  AxpyQ4FallbackKernel::apply(DIM, a, x.data(), scale.data(), zeros.data(), yRef.data());

  CATCH_REQUIRE(isClose(y, yRef));
}

CATCH_TEST_CASE("test q4 dot kernels", "[lymath][dot][q4]") {
  constexpr int DIM = 1024;

  std::vector<float> x(DIM);
  std::vector<Q4x2> y(DIM / 2);
  std::vector<float> scaleYFp32(DIM / Q4GroupSize);
  std::vector<UInt8> zeroY(DIM / Q4GroupSize / 2);
  std::vector<Fp16> scaleY(DIM / Q4GroupSize);

  lut::Random random(MagicNumber);
  random.fillUInt8(lut::makeSpan(y));
  random.fillUInt8(lut::makeSpan(zeroY));
  random.fill(lut::makeSpan(scaleYFp32));
  random.fill(lut::makeSpan(x));
  std::transform(scaleYFp32.begin(), scaleYFp32.end(), scaleY.begin(), lut::cvtss_sh);

  float a = DotQ4Avx2Kernel::apply(DIM, x.data(), y.data(), scaleY.data(), zeroY.data());
  float aRef = DotQ4FallbackKernel::apply(DIM, x.data(), y.data(), scaleY.data(), zeroY.data());

  CATCH_REQUIRE(isClose(a, aRef));
}

void testSgemm(bool transA, bool transB, int M, int N, int K) {
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);

  lut::Random random(MagicNumber);

  random.fill(lut::makeSpan(A));
  random.fill(lut::makeSpan(B));

  std::vector<float> C(M * N);
  std::vector<float> refC(M * N);

  refSgemm(
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

  lymath_sgemm(
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

  CATCH_REQUIRE(isClose(C, refC));
}

int gemmTestShapes[][3] = {
  {16, 5000, 16},
  {50, 50, 1},
  {1, 1, 1},
  {2, 2, 2},
  {50, 50, 1},
  {513, 2, 513},
  {200, 1, 300},
  {1, 200, 300},
  {200, 300, 1},
  {16, 16, 5000},
  {16, 512, 16},
  {16, 1024, 16},
  {5000, 16, 16},
  {0, 0, 0}
};

CATCH_TEST_CASE("test lymath_sgemm", "[lymath][sgemm]") {
  int (*pshape)[3];
  
  for (pshape = &gemmTestShapes[0]; **pshape != 0; ++pshape) {
    int m = (*pshape)[0];
    int k = (*pshape)[1];
    int n = (*pshape)[2];

    testSgemm(true, true, m, n, k);
    testSgemm(true, false, m, n, k);
    testSgemm(false, true, m, n, k);
    testSgemm(false, false, m, n, k);
  }
}
