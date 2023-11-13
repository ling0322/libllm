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

#include <omp.h>
#include "lymath/common.h"
#include "lymath/lymath.h"
#include "lymath/q4kernel.h"
#include "lymath/q4sym_dequant.h"
#include "lymath/q4sym_kernel.h"
#include "lymath/q8kernel.h"
#include "lymath/skernel.h"
#include "lymath/util.h"
#include "lyutil/random.h"
#include "lyutil/log.h"

using namespace lymath;

constexpr uint32_t MagicNumber = 0x55aa;

int main(int argc, char **argv) {
  lymath_init();
  int result = Catch::Session().run(argc, argv);
  lymath_destroy();

  return result;
}


void refGemmNqnInt4SymGroup32(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    PCFp32 A,
    int lda,
    PCQ4x2 B,
    PCFp16 scaleB,
    PFp32 C,
    int ldc) {
  CHECK(transA == false);
  std::fill_n(C, M * N, 0.0f);

  for (int j = 0; j < M; ++j) {
    if (transB) {
      for (int i = 0; i < N; ++i) {
        PCFp32 Aj = A + j * lda;
        PCQ4x2 Bi = reinterpret_cast<PCQ4x2>(B) + i * K / 2;
        PCFp16 si = scaleB + i * K / Q4GroupSize;
        C[j * ldc + i] = DotQ4SymFallbackKernel::apply(K, Aj, Bi, si);
      }
    } else {
      for (int i = 0; i < K; ++i) {
        const Fp32 Aji = A[j * lda + i];
        PCQ4x2 Bi = reinterpret_cast<PCQ4x2>(B) + i * N / 2;
        PCFp16 si = scaleB + i * N / Q4GroupSize;
        PFp32 Cj = C + j * ldc;
        AxpyQ4SymFallbackKernel::apply(N, Aji, Bi, si, Cj);
      }
    }
  }
}

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

bool isClose(ly::Span<const float> A, ly::Span<const float> B) {
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

void testGemmFp32QInt4Fp32(bool transB, int M, int N, int K) {
  std::vector<float> A(M * K);
  std::vector<uint8_t> B(K * N / 2);
  std::vector<float> scaleBFp32(K * N / Q4GroupSize);
  std::vector<Fp16> scaleB(K * N / Q4GroupSize);

  ly::Random random(MagicNumber);

  random.fill(ly::makeSpan(A));
  random.fillUInt8(ly::makeSpan(B));
  random.fill(ly::makeSpan(scaleBFp32));

  std::transform(scaleBFp32.begin(), scaleBFp32.end(), scaleB.begin(), cvtss_sh);

  std::vector<float> C(M * N);
  std::vector<float> refC(M * N);

  refGemmNqnInt4SymGroup32(
      false,
      transB,
      M,
      N,
      K,
      A.data(),
      K,
      B.data(),
      scaleB.data(),
      refC.data(),
      N);

  lymath_qgemm_nqn_q4sym_omp(
      false,
      transB,
      M,
      N,
      K,
      A.data(),
      K,
      (const lymath_q4x2_t *)B.data(),
      (const lymath_float16_t *)scaleB.data(),
      C.data(),
      N);

  CATCH_REQUIRE(isClose(C, refC));
}

CATCH_TEST_CASE("test q4sym dequantization", "[llyn][lymath][q4sym]") {
  constexpr int DIM = DequantMinElemPerThread + Q4GroupSize;

  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleXFp32(DIM / Q4GroupSize);
  std::vector<Fp16> scaleX(DIM / Q4GroupSize);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);

  ly::Random random(MagicNumber);

  random.fillUInt8(ly::makeSpan(x));
  random.fill(ly::makeSpan(scaleXFp32));
  std::transform(scaleXFp32.begin(), scaleXFp32.end(), scaleX.begin(), cvtss_sh);

  DequantQ4SymFallbackKnl::apply(DIM, x.data(), scaleX.data(), y.data());

  DequantQ4SymFallbackOMP().apply(DIM, x.data(), scaleX.data(), yRef.data());
  CATCH_REQUIRE(isClose(y, yRef));

  DequantQ4SymAvx2().apply(DIM, x.data(), scaleX.data(), yRef.data());
  CATCH_REQUIRE(isClose(y, yRef));

  DequantQ4SymAvx2OMP().apply(DIM, x.data(), scaleX.data(), yRef.data());
  CATCH_REQUIRE(isClose(y, yRef));
}

CATCH_TEST_CASE("test q4 dequantization", "[llyn][lymath][kernel][q4]") {
  constexpr int DIM = DequantMinElemPerThread + Q4GroupSize;

  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleXFp32(DIM / Q4GroupSize);
  std::vector<Fp16> scaleX(DIM / Q4GroupSize);
  std::vector<Int8> zeroPointX(DIM / Q4GroupSize);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);

  ly::Random random(MagicNumber);

  random.fillUInt8(ly::makeSpan(x));
  random.fill(ly::makeSpan(scaleXFp32));
  random.fillInt8(ly::makeSpan(zeroPointX), -112, 127);
  std::transform(scaleXFp32.begin(), scaleXFp32.end(), scaleX.begin(), cvtss_sh);

  DequantQ4FallbackKernel::apply(DIM, x.data(), scaleX.data(), zeroPointX.data(), y.data());
}

CATCH_TEST_CASE("test int4 dot kernels", "[llyn][lymath][kernel][int4]") {
  constexpr int DIM = 1024;

  std::vector<float> x(DIM);
  std::vector<uint8_t> y(DIM / 2);
  std::vector<float> yscaleFp32(DIM / Q4GroupSize);
  std::vector<Fp16> yscale(DIM / Q4GroupSize);

  ly::Random random(MagicNumber);

  random.fill(ly::makeSpan(x));
  random.fillUInt8(ly::makeSpan(y));
  random.fill(ly::makeSpan(yscaleFp32));
  std::transform(yscaleFp32.begin(), yscaleFp32.end(), yscale.begin(), cvtss_sh);

  float rs = DotQ4SymFallbackKernel::apply(DIM, x.data(), y.data(), yscale.data());
  float s = DotQ4SymAvx2Kernel::apply(DIM, x.data(), y.data(), yscale.data());

  CATCH_REQUIRE(fabs(rs - s) < 1e-5);
}

CATCH_TEST_CASE("test q4sym axpy kernels", "[llyn][lymath][kernel][q4sym]") {
  constexpr int DIM = 1024;

  float a = 0.1f;
  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleFp32(DIM / Q4GroupSize);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);
  std::vector<Fp16> scale(DIM / Q4GroupSize);

  ly::Random random(MagicNumber);
  random.fillUInt8(ly::makeSpan(x));
  random.fill(ly::makeSpan(scaleFp32));
  random.fill(ly::makeSpan(yRef));
  std::transform(scaleFp32.begin(), scaleFp32.end(), scale.begin(), cvtss_sh);

  std::copy(yRef.begin(), yRef.end(), y.begin());

  AxpyQ4SymAvx2Kernel::apply(DIM, a, x.data(), scale.data(), y.data());
  AxpyQ4SymFallbackKernel::apply(DIM, a, x.data(), scale.data(), yRef.data());

  CATCH_REQUIRE(isClose(y, yRef));
}


CATCH_TEST_CASE("test q4 axpy kernels", "[llyn][lymath][kernel][q4]") {
  constexpr int DIM = 1024;

  float a = 0.1f;
  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleFp32(DIM / Q4GroupSize);
  std::vector<Int8> zeroPointX(DIM / Q4GroupSize);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);
  std::vector<Fp16> scale(DIM / Q4GroupSize);

  ly::Random random(MagicNumber);
  random.fillUInt8(ly::makeSpan(x));
  random.fillInt8(ly::makeSpan(zeroPointX), -112, 127);
  random.fill(ly::makeSpan(scaleFp32));
  random.fill(ly::makeSpan(yRef));
  std::transform(scaleFp32.begin(), scaleFp32.end(), scale.begin(), cvtss_sh);

  std::copy(yRef.begin(), yRef.end(), y.begin());

  AxpyQ4Avx2Kernel::apply(DIM, a, x.data(), scale.data(), zeroPointX.data(), y.data());
  AxpyQ4FallbackKernel::apply(DIM, a, x.data(), scale.data(), zeroPointX.data(), yRef.data());

  CATCH_REQUIRE(isClose(y, yRef));
}

CATCH_TEST_CASE("test q4 dot kernels", "[llyn][lymath][kernel][q4]") {
  constexpr int DIM = 1024;

  std::vector<float> x(DIM);
  std::vector<Q4x2> y(DIM / 2);
  std::vector<float> scaleYFp32(DIM / Q4GroupSize);
  std::vector<Int8> zeroPointY(DIM / Q4GroupSize);
  std::vector<Fp16> scaleY(DIM / Q4GroupSize);

  ly::Random random(MagicNumber);
  random.fillUInt8(ly::makeSpan(y));
  random.fillInt8(ly::makeSpan(zeroPointY), -112, 127);
  random.fill(ly::makeSpan(scaleYFp32));
  random.fill(ly::makeSpan(x));
  std::transform(scaleYFp32.begin(), scaleYFp32.end(), scaleY.begin(), cvtss_sh);

  float a = DotQ4Avx2Kernel::apply(DIM, x.data(), y.data(), scaleY.data(), zeroPointY.data());
  float aRef = DotQ4FallbackKernel::apply(
      DIM, x.data(), y.data(), scaleY.data(), zeroPointY.data());

  CATCH_REQUIRE(isClose(a, aRef));
}

CATCH_TEST_CASE("test int8b dequant kernels", "[llyn][lymath][kernel][int8]") {
  constexpr int DIM = 512;

  float a = 0.1f;
  std::vector<uint8_t> qdata(DIM);
  std::vector<float> scaleZp{0.01, -200, 1.0, 0.0, 0.1, -100, 0.2, 10};
  std::vector<float> rdata;
  std::vector<float> rdataRef;

  ly::Random random(MagicNumber);
  random.fillUInt8(ly::makeSpan(qdata));

  rdata.resize(2);
  rdataRef.resize(2);
  DequantInt8BAvx2Kernel::apply(2, qdata.data(), scaleZp.data(), 100, rdata.data());
  DequantInt8BFallbackKernel::apply(2, qdata.data(), scaleZp.data(), 100, rdataRef.data());
  CATCH_REQUIRE(isClose(rdata, rdataRef));
  CATCH_REQUIRE(isClose(rdata[0], qdata[100] * scaleZp[0] + scaleZp[1]));
  CATCH_REQUIRE(isClose(rdata[1], qdata[101] * scaleZp[0] + scaleZp[1]));

  rdata.resize(256);
  rdataRef.resize(256);
  DequantInt8BAvx2Kernel::apply(256, qdata.data(), scaleZp.data(), 0, rdata.data());
  DequantInt8BFallbackKernel::apply(256, qdata.data(), scaleZp.data(), 0, rdataRef.data());
  CATCH_REQUIRE(isClose(rdata, rdataRef));

  rdata.resize(258);
  rdataRef.resize(258);
  DequantInt8BAvx2Kernel::apply(258, qdata.data(), scaleZp.data(), 127, rdata.data());
  DequantInt8BFallbackKernel::apply(258, qdata.data(), scaleZp.data(), 127, rdataRef.data());
  CATCH_REQUIRE(isClose(rdata, rdataRef));
  CATCH_REQUIRE(isClose(rdata[0], qdata[127] * scaleZp[0] + scaleZp[1]));
  CATCH_REQUIRE(isClose(rdata[257], qdata[257 + 127] * scaleZp[6] + scaleZp[7]));
}

CATCH_TEST_CASE("test lymath_gemm_fp32qint4fp32", "[llyn][lymath][api][int4]") {
  testGemmFp32QInt4Fp32(true, 1, 32, 32);
  testGemmFp32QInt4Fp32(true, 32, 32, 32);
  testGemmFp32QInt4Fp32(false, 1, 32, 32);
  testGemmFp32QInt4Fp32(false, 32, 32, 32);
  testGemmFp32QInt4Fp32(false, 1, 32, 4096);
}

void testSgemm(bool transA, bool transB, int M, int N, int K) {
  std::vector<float> A(M * K);
  std::vector<float> B(K * N);

  ly::Random random(MagicNumber);

  random.fill(ly::makeSpan(A));
  random.fill(ly::makeSpan(B));

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

CATCH_TEST_CASE("test lymath_sgemm", "[sgemm]") {
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
