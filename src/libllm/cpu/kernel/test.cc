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

#include <omp.h>
#include <math.h>
#include "libllm/cpu/kernel/common.h"
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/kernel/cvt_half.h"
#include "libllm/cpu/kernel/dequant_q4.h"
#include "libllm/cpu/kernel/kernel_q4.h"
#include "libllm/cpu/kernel/kernel_float.h"
#include "libllm/cpu/kernel/kernel_half.h"
#include "libllm/cpu/kernel/util.h"
#include "libllm/lut/half.h"
#include "libllm/lut/random.h"
#include "libllm/lut/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

constexpr uint32_t MagicNumber = 0x55aa;

bool isClose(float l, float r, float atol = 1e-5, float rtol = 1e-5) {
  bool close = fabs(l - r) <= atol + rtol * fabs(r);
  if (!close) {
    printf("%f vs %f\n", l, r);
  }
  return close;
}

bool isClose(Fp16 l, Fp16 r, float atol = 1e-5, float rtol = 1e-5) {
  float lf = cvt_h2s(l);
  float rf = cvt_h2s(r);
  bool close = fabs(lf - rf <= atol + rtol * fabs(rf));
  if (!close) {
    printf("%f vs %f\n", lf, rf);
  }
  return close;
}

template<typename T>
bool isClose(lut::Span<const T> A,
             lut::Span<const T> B,
             float atol = 1e-5,
             float rtol = 1e-5) {
  if (A.size() != B.size()) 
    return false;

  for (int i = 0; i < A.size(); ++i) {
    if (!isClose(A[i], B[i], atol, rtol)) {
      return false;
    }
  }

  return true;
}

float toFloat(float x) {
  return x;
}
float toFloat(Fp16 x) {
  return cvt_h2s(x);
}

template<typename T>
float getMSE(lut::Span<const T> A, lut::Span<const T> B) {
  CHECK(A.size() == B.size());

  double sum = 0;
  for (int i = 0; i < A.size(); ++i) {
    double d = toFloat(A[i]) - toFloat(B[i]);
    sum += d * d;
  }
  double mse = sum / A.size();
  return static_cast<float>(mse);
}

template<typename T>
float getVar(lut::Span<const T> A) {
  double sum = 0;
  for (int i = 0; i < A.size(); ++i) {
    sum += toFloat(A[i]);
  }

  double mean = sum / A.size();
  sum = 0;
  for (int i = 0; i < A.size(); ++i) {
    double d = toFloat(A[i]) - mean;
    sum += d * d;
  }
  double var = sum / A.size();
  return static_cast<float>(var);
}

template<typename T>
float getRSquared(lut::Span<const T> predict, lut::Span<const T> correct) {
  return 1.0f - getMSE<T>(predict, correct) / getVar<T>(predict);
}

void fillRandomHalf(lut::Random *r, lut::Span<Fp16> v) {
  std::vector<float> vf(v.size());
  r->fill(lut::makeSpan(vf), -1, 1);
  std::transform(vf.begin(), vf.end(), v.begin(), [](float v){
    return cvt_s2h(v);
  });
}

void refGemmQ4(
    bool transA,
    bool transB,
    int M,
    int N,
    int K,
    PCFp32 A,
    int lda,
    PCQ4x2 B,
    PCFp16 scaleB,
    PCUInt8 zeroB,
    PFp32 C,
    int ldc) {
  CHECK(transA == false);
  std::fill_n(C, M * N, 0.0f);

  for (int j = 0; j < M; ++j) {
    if (transB) {
      for (int i = 0; i < N; ++i) {
        PCFp32 Aj = A + j * lda;
        C[j * ldc + i] = DotQ4FallbackKernel::apply(K, Aj, {B, scaleB, zeroB}, i * K);
      }
    } else {
      NOT_IMPL();
    }
  }
}

void testGemmQ4(bool transB, int M, int N, int K) {
  std::vector<float> A(M * K);
  std::vector<uint8_t> B(K * N / 2);
  std::vector<float> scaleBFp32(K * N / GroupSizeQ4);
  std::vector<Fp16> scaleB(K * N / GroupSizeQ4);
  std::vector<UInt8> zeroB(K * N / GroupSizeQ4 / 2);

  lut::Random random(MagicNumber);

  random.fill(lut::makeSpan(A));
  random.fillUInt8(lut::makeSpan(B));
  random.fillUInt8(lut::makeSpan(zeroB));

  fillRandomHalf(&random, lut::makeSpan(scaleB));
  std::vector<float> C(M * N);
  std::vector<float> refC(M * N);

  refGemmQ4(
      false,
      transB,
      M,
      N,
      K,
      A.data(),
      K,
      B.data(),
      scaleB.data(),
      zeroB.data(),
      refC.data(),
      N);

  gemmQ4(
      false,
      transB,
      M,
      N,
      K,
      A.data(),
      K,
      (const Q4x2 *)B.data(),
      (const Fp16 *)scaleB.data(),
      (const uint8_t *)zeroB.data(),
      C.data(),
      N);

  CATCH_REQUIRE(isClose<float>(C, refC, 1e-3));
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


#ifdef LIBLLM_ARCH_X86_64

CATCH_TEST_CASE("test q4 dequantization", "[lymath][dequant][q4]") {
  constexpr int DIM = DequantMinElemPerThread * 2 + GroupSizeQ4;

  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleXFp32(DIM / GroupSizeQ4);
  std::vector<Fp16> scaleX(DIM / GroupSizeQ4);
  std::vector<UInt8> zeroX(DIM / GroupSizeQ4 / 2);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);

  lut::Random random(MagicNumber);

  random.fillUInt8(lut::makeSpan(x));
  random.fillUInt8(lut::makeSpan(zeroX));
  random.fill(lut::makeSpan(scaleXFp32));
  std::transform(scaleXFp32.begin(), scaleXFp32.end(), scaleX.begin(), cvt_s2h);

  DequantQ4FallbackKernel::apply(DIM, {x.data(), scaleX.data(), zeroX.data()}, 0, yRef.data());
  DequantQ4Avx2Kernel::apply(DIM, {x.data(), scaleX.data(), zeroX.data()}, 0, y.data());
  CATCH_REQUIRE(isClose<float>(y, yRef));

  random.fill(lut::makeSpan(y));
  random.fill(lut::makeSpan(yRef));
  DequantQ4FallbackKernel::apply(
      GroupSizeQ4, {x.data(), scaleX.data(), zeroX.data()}, DequantMinElemPerThread, yRef.data());
  DequantQ4Avx2Kernel::apply(
      GroupSizeQ4, {x.data(), scaleX.data(), zeroX.data()}, DequantMinElemPerThread, y.data());
  CATCH_REQUIRE(isClose<float>(lut::makeConstSpan(y).subspan(0, GroupSizeQ4),
                               lut::makeConstSpan(yRef).subspan(0, GroupSizeQ4)));

  // test api.
  random.fill(lut::makeSpan(y));
  DequantQ4FallbackKernel::apply(DIM, {x.data(), scaleX.data(), zeroX.data()}, 0, yRef.data());
  DequantQ4Avx2().apply(DIM, {x.data(), scaleX.data(), zeroX.data()}, 0, y.data());
  CATCH_REQUIRE(isClose<float>(y, yRef));

  random.fill(lut::makeSpan(y));
  DequantQ4Avx2OMP().apply(DIM, {x.data(), scaleX.data(), zeroX.data()}, 0, y.data());
  CATCH_REQUIRE(isClose<float>(y, yRef));
}

CATCH_TEST_CASE("test q4 dot kernels", "[lymath][dot][q4]") {
  constexpr int DIM = 1024;

  std::vector<float> x(DIM);
  std::vector<Q4x2> y(DIM / 2);
  std::vector<float> scaleYFp32(DIM / GroupSizeQ4);
  std::vector<UInt8> zeroY(DIM / GroupSizeQ4 / 2);
  std::vector<Fp16> scaleY(DIM / GroupSizeQ4);

  lut::Random random(MagicNumber);
  random.fillUInt8(lut::makeSpan(y));
  random.fillUInt8(lut::makeSpan(zeroY));
  random.fill(lut::makeSpan(scaleYFp32));
  random.fill(lut::makeSpan(x));
  std::transform(scaleYFp32.begin(), scaleYFp32.end(), scaleY.begin(), cvt_s2h);

  float a = DotQ4Avx2Kernel::apply(DIM, x.data(), {y.data(), scaleY.data(), zeroY.data()}, 0);
  float aRef = DotQ4FallbackKernel::apply(
      DIM, x.data(), {y.data(), scaleY.data(), zeroY.data()}, 0);

  CATCH_REQUIRE(isClose(a, aRef));
}

// to reproduce a zero-point index bug in q4 kernels.
CATCH_TEST_CASE("test q4 dot kernels apply row", "[lymath][dot][q4]") {
  constexpr int NUM_ROW = 32;
  constexpr int NUM_COL = 128;
  constexpr int NUMEL = NUM_COL * NUM_ROW;

  std::vector<float> x(NUM_COL);
  std::vector<float> y(NUM_ROW);
  std::vector<Q4x2> A(NUMEL / 2);
  std::vector<float> scaleAFp32(NUMEL / GroupSizeQ4);
  std::vector<UInt8> zeroA(NUMEL / GroupSizeQ4 / 2);
  std::vector<Fp16> scaleA(NUMEL / GroupSizeQ4);

  lut::Random random(MagicNumber);
  random.fillUInt8(lut::makeSpan(A));
  random.fillUInt8(lut::makeSpan(zeroA));
  random.fill(lut::makeSpan(scaleAFp32));
  random.fill(lut::makeSpan(x));
  std::transform(scaleAFp32.begin(), scaleAFp32.end(), scaleA.begin(), cvt_s2h);

  Q4GemvArgs gemvArgs;
  gemvArgs.A = {A.data(), scaleA.data(), zeroA.data()};
  gemvArgs.incX = 1;
  gemvArgs.incY = 1;
  gemvArgs.M = NUM_ROW;
  gemvArgs.N = NUM_COL;
  gemvArgs.transA = false;
  gemvArgs.x = x.data();
  gemvArgs.y = nullptr;

  float a0 = DotQ4Avx2Kernel::applyRow(gemvArgs, 0);
  float a1 = DotQ4Avx2Kernel::applyRow(gemvArgs, 1);

  std::vector<float> x2(NUM_COL * 2);
  std::copy(x.begin(), x.end(), x2.begin());
  std::copy(x.begin(), x.end(), x2.begin() + NUM_COL);


  float a = DotQ4Avx2Kernel::apply(NUM_COL * 2, x2.data(), {A.data(), scaleA.data(), zeroA.data()}, 0);
  CATCH_REQUIRE(isClose(a, a0 + a1));
}
#endif  // LIBLLM_ARCH_X86_64

CATCH_TEST_CASE("test lymath_q4gemm", "[lymath][api][q4]") {
  testGemmQ4(true, 1, 32, 128);
  testGemmQ4(true, 1, 64, 4096);
  testGemmQ4(true, 64, 64, 256);
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

  sgemm(
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

  CATCH_REQUIRE(isClose<float>(C, refC));
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


void testHalfToFloat(int n) {
  std::vector<float> y(n);
  std::vector<float> yr(n);
  std::vector<Fp16> x(n);

  lut::Random random(MagicNumber);
  random.fill(lut::makeSpan(yr));
  std::transform(yr.begin(), yr.end(), x.begin(), [](float x) {
    return cvt_s2h(x);
  });

  convertHalfToFloat(n, x.data(), y.data());
  CATCH_REQUIRE(isClose<float>(yr, y, 1e-4, 1e-3));
}

CATCH_TEST_CASE("test lymath_half2float", "[lymath][cvt]") {
  std::vector<int> ns{1, 50, 200, 800, 1600, 1601, 3200, 3201};
  for (int n : ns) {
    testHalfToFloat(n);
  }
}


template<class TAxpyHalfKernel>
void testAxpyHalfKernel(int n) {
  std::vector<Fp16> x(n);
  std::vector<Fp16> y(n);
  std::vector<Fp16> yr(n);

  lut::Random random(MagicNumber);
  fillRandomHalf(&random, lut::makeSpan<Fp16>(x));
  Fp16 a = x[0];

  TAxpyHalfKernel::apply(n, a, x.data(), y.data());
  AxpyHalfFallbackKernel::apply(n, a, x.data(), yr.data());

  CATCH_REQUIRE(isClose<Fp16>(yr, y));
}

template<class TDotHalfKernel>
void testDotHalfKernel(int n, float rtol = 1e-3) {
  std::vector<Fp16> x(n);
  std::vector<Fp16> y(n);

  lut::Random random(MagicNumber);
  fillRandomHalf(&random, lut::makeSpan<Fp16>(x));
  fillRandomHalf(&random, lut::makeSpan<Fp16>(y));

  Fp16 z = TDotHalfKernel::apply(n, x.data(), y.data());
  Fp16 zr = DotHalfFallbackKernel::apply(n, x.data(), y.data());

  CATCH_REQUIRE(isClose(z, zr, 0, rtol));
}

template<class TGemmHalfKernel, class TGemmHalfReferenceKernel>
struct GemmMicroKernelTester {
  void test(int k, float rtol = 0.01) {
    static_assert(TGemmHalfKernel::MR == TGemmHalfReferenceKernel::MR, "gemm kernal size mismatch");
    static_assert(TGemmHalfKernel::NR == TGemmHalfReferenceKernel::NR, "gemm kernal size mismatch");

    int MR = TGemmHalfKernel::MR;
    int NR = TGemmHalfReferenceKernel::NR;

    std::vector<Fp16> A(MR * k);
    std::vector<Fp16> B(NR * k);
    std::vector<Fp16> C(MR * NR);
    std::vector<Fp16> Cr(MR * NR);

    lut::Random random(MagicNumber);
    fillRandomHalf(&random, lut::makeSpan<Fp16>(A));
    fillRandomHalf(&random, lut::makeSpan<Fp16>(B));
    fillRandomHalf(&random, lut::makeSpan<Fp16>(C));
    std::copy(C.begin(), C.end(), Cr.begin());

    TGemmHalfKernel::apply(k, A.data(), B.data(), C.data(), NR);
    TGemmHalfReferenceKernel::apply(k, A.data(), B.data(), Cr.data(), NR);
    CATCH_REQUIRE(getRSquared<Fp16>(C, Cr) > 0.99995);
  }
};

#ifdef LIBLLM_ARCH_AARCH64

CATCH_TEST_CASE("test AxpyHalfAsimdhpKernel", "[libllm][cpu_kernel][axpy][half]") {
  testAxpyHalfKernel<AxpyHalfAsimdhpKernel>(1);
  testAxpyHalfKernel<AxpyHalfAsimdhpKernel>(8);
  testAxpyHalfKernel<AxpyHalfAsimdhpKernel>(16);
  testAxpyHalfKernel<AxpyHalfAsimdhpKernel>(17);
  testAxpyHalfKernel<AxpyHalfAsimdhpKernel>(128);
  testAxpyHalfKernel<AxpyHalfAsimdhpKernel>(2001);
}

CATCH_TEST_CASE("test DotHalfAsimdhpKernel", "[libllm][cpu_kernel][dot][half]") {
  testDotHalfKernel<DotHalfAsimdhpKernel>(1);
  testDotHalfKernel<DotHalfAsimdhpKernel>(8);
  testDotHalfKernel<DotHalfAsimdhpKernel>(16);
  testDotHalfKernel<DotHalfAsimdhpKernel>(17);
  testDotHalfKernel<DotHalfAsimdhpKernel>(128);
  testDotHalfKernel<DotHalfAsimdhpKernel>(160);
  testDotHalfKernel<DotHalfAsimdhpKernel>(1500);
  testDotHalfKernel<DotHalfAsimdhpKernel>(2001);
  testDotHalfKernel<DotHalfAsimdhpKernel>(20000);
}

CATCH_TEST_CASE("test GemmHalf12x16AsimdhpKernel", "[libllm][cpu_kernel][gemm_kernel][half]") {
  GemmMicroKernelTester<GemmHalf12x16FallbackKernel, GemmHalf12x16AsimdhpKernel> tester;
  tester.test(1);
  tester.test(8);
  tester.test(17);
  tester.test(64);
  tester.test(100);
  tester.test(256);
  tester.test(500);
}

#endif  // LIBLLM_ARCH_AARCH64

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
