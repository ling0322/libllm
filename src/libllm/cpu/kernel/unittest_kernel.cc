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
#include "libllm/cpu/kernel/unittest_common.h"
#include "libllm/cpu/kernel/interfaces.h"
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

template<typename T, class TKernel, class TRefKernel>
void testQInt4DequantKernel(int n) {
  CHECK(n % GroupSizeQInt4 == 0);
  int ng = n / GroupSizeQInt4;

  std::vector<UInt4x2> qdata(n / 2);
  std::vector<Float16> qscale(ng);
  std::vector<UInt4x2> qzero((ng + 1) / 2);
  std::vector<T> y(n);
  std::vector<T> yr(n);

  lut::Random random(MagicNumber);
  fillRandomQInt4(&random, lut::makeSpan(qdata), lut::makeSpan(qscale), lut::makeSpan(qzero));

  DataQInt4 d(qdata.data(), qscale.data(), qzero.data());
  TKernel::apply(n, d, 0, y.data());
  TRefKernel::apply(n, d, 0, yr.data());
  CATCH_REQUIRE(isClose<T>(y, yr));
}

template<class TKernel>
void testQInt4HalfDequantKernel(int n) {
  testQInt4DequantKernel<Float16, TKernel, DequantQInt4ToHalfFallbackKernel>(n);
}

template<class TGemmHalfKernel, class TGemmHalfReferenceKernel>
struct GemmMicroKernelTester {
  void test(int k, float rtol = 0.01) {
    static_assert(TGemmHalfKernel::MR == TGemmHalfReferenceKernel::MR, "gemm kernal size mismatch");
    static_assert(TGemmHalfKernel::NR == TGemmHalfReferenceKernel::NR, "gemm kernal size mismatch");

    int MR = TGemmHalfKernel::MR;
    int NR = TGemmHalfReferenceKernel::NR;

    std::vector<Float16> A(MR * k);
    std::vector<Float16> B(NR * k);
    std::vector<Float16> C(MR * NR);
    std::vector<Float16> Cr(MR * NR);

    lut::Random random(MagicNumber);
    fillRandom(&random, lut::makeSpan<Float16>(A));
    fillRandom(&random, lut::makeSpan<Float16>(B));
    fillRandom(&random, lut::makeSpan<Float16>(C));
    std::copy(C.begin(), C.end(), Cr.begin());

    TGemmHalfKernel::apply(k, A.data(), B.data(), C.data(), NR);
    TGemmHalfReferenceKernel::apply(k, A.data(), B.data(), Cr.data(), NR);

    CATCH_REQUIRE(getMaxDiff<Float16>(C, Cr) / getMeanAbs<Float16>(Cr) < 0.05);
  }
};

template<class TAxpyHalfKernel>
void testAxpyHalfKernel(int n) {
  std::vector<Float16> x(n);
  std::vector<float> y(n);
  std::vector<float> yr(n);

  lut::Random random(MagicNumber);
  fillRandom(&random, lut::makeSpan<Float16>(x));
  Float16 a = x[0];

  TAxpyHalfKernel::apply(n, a, x.data(), y.data());
  AxpyHalfFallbackKernel::apply(n, a, x.data(), yr.data());

  CATCH_REQUIRE(isClose<float>(yr, y));
}

template<class TDotHalfKernel>
void testDotHalfKernel(int n, float rtol = 5e-2) {
  std::vector<Float16> x(n);
  std::vector<Float16> y(n);

  lut::Random random(MagicNumber);
  fillRandom(&random, lut::makeSpan<Float16>(x));
  fillRandom(&random, lut::makeSpan<Float16>(y));

  Float16 z = TDotHalfKernel::apply(n, x.data(), y.data());
  Float16 zr = DotHalfFallbackKernel::apply(n, x.data(), y.data());

  CATCH_REQUIRE(isClose(z, zr, 0, rtol));
}

template<typename T, class TQInt4DotKernel, class TQInt4DotRefKernel>
void testQInt4DotKernel(int n, float rtol = 5e-2) {
  CHECK(n % GroupSizeQInt4 == 0);
  int ng = n / GroupSizeQInt4;

  std::vector<UInt4x2> qdata(n / 2);
  std::vector<Float16> qscale(ng);
  std::vector<UInt4x2> qzero((ng + 1) / 2);
  std::vector<T> x(n);

  lut::Random random(MagicNumber);
  fillRandomQInt4(&random, lut::makeSpan(qdata), lut::makeSpan(qscale), lut::makeSpan(qzero));
  fillRandom(&random, lut::makeSpan<Float16>(x));

  DataQInt4 y(qdata.data(), qscale.data(), qzero.data());
  T a = TQInt4DotKernel::apply(n, x.data(), y, 0);
  T ar = TQInt4DotRefKernel::apply(n, x.data(), y, 0);

  CATCH_REQUIRE(isClose(a, ar, 0, rtol));
}

template<class TKernel>
void testHQInt4DotKernel(int n) {
  testQInt4DotKernel<Float16, TKernel, HQInt4DotFallbackKernel>(n);
}

template<typename TSrc, typename TTgt, class TCvtKernel, class TRefCvtKernel>
void testCvtKernel(int n) {
  std::vector<TSrc> x(n);
  std::vector<TTgt> y(n);
  std::vector<TTgt> yr(n);

  lut::Random random(MagicNumber);
  fillRandom(&random, lut::makeSpan<TSrc>(x));
  fillRandom(&random, lut::makeSpan<TTgt>(y));

  TCvtKernel::apply(n, x.data(), y.data());
  TRefCvtKernel::apply(n, x.data(), yr.data());

  CATCH_REQUIRE(isClose<TTgt>(y, yr));
}

template<class TKernel>
void testCvtHalfToFloatKernel(int n) {
  testCvtKernel<Float16, float, TKernel, CvtHalfToFloatFallbackKernel>(n);
}

template<class TKernel>
void testCvtFloatToHalfKernel(int n) {
  testCvtKernel<float, Float16, TKernel, CvtFloatToHalfFallbackKernel>(n);
}

#ifdef LUT_ARCH_AMD64

CATCH_TEST_CASE("test q4 dequantization", "[lymath][dequant][q4]") {
  constexpr int DIM = DequantMinElemPerThread * 2 + GroupSizeQInt4;

  std::vector<uint8_t> x(DIM / 2);
  std::vector<float> scaleXFp32(DIM / GroupSizeQInt4);
  std::vector<Float16> scaleX(DIM / GroupSizeQInt4);
  std::vector<uint8_t> zeroX(DIM / GroupSizeQInt4 / 2);
  std::vector<float> y(DIM);
  std::vector<float> yRef(DIM);

  lut::Random random(MagicNumber);

  random.fillUInt8(lut::makeSpan(x));
  random.fillUInt8(lut::makeSpan(zeroX));
  random.fill(lut::makeSpan(scaleXFp32));
  std::transform(scaleXFp32.begin(), scaleXFp32.end(), scaleX.begin(), cvt_s2h);

  DequantQInt4FallbackKernel::apply(DIM, {
      (const UInt4x2 *)x.data(),
      scaleX.data(),
      (const UInt4x2 *)zeroX.data()},
      0,
      yRef.data());
  DequantQInt4Avx2Kernel::apply(DIM, {
      (const UInt4x2 *)x.data(),
      scaleX.data(),
      (const UInt4x2 *)zeroX.data()},
      0,
      y.data());
  CATCH_REQUIRE(isClose<float>(y, yRef));

  random.fill(lut::makeSpan(y));
  random.fill(lut::makeSpan(yRef));
  DequantQInt4FallbackKernel::apply(GroupSizeQInt4, {
      (const UInt4x2 *)x.data(),
      scaleX.data(),
      (const UInt4x2 *)zeroX.data()}, DequantMinElemPerThread, yRef.data());
  DequantQInt4Avx2Kernel::apply(GroupSizeQInt4, {
      (const UInt4x2 *)x.data(),
      scaleX.data(),
      (const UInt4x2 *)zeroX.data()}, DequantMinElemPerThread, y.data());
  CATCH_REQUIRE(isClose<float>(lut::makeConstSpan(y).subspan(0, GroupSizeQInt4),
                               lut::makeConstSpan(yRef).subspan(0, GroupSizeQInt4)));

  // test api.
  random.fill(lut::makeSpan(y));
  DequantQInt4FallbackKernel::apply(DIM, {
      (const UInt4x2 *)x.data(),
      scaleX.data(),
      (const UInt4x2 *)zeroX.data()}, 0, yRef.data());
  DequantQInt4Avx2().apply(DIM, {
      (const UInt4x2 *)x.data(),
      scaleX.data(),
      (const UInt4x2 *)zeroX.data()}, 0, y.data());
  CATCH_REQUIRE(isClose<float>(y, yRef));

  random.fill(lut::makeSpan(y));
  DequantQInt4Avx2OMP().apply(DIM, {
      (const UInt4x2 *)x.data(),
      scaleX.data(),
      (const UInt4x2 *)zeroX.data()}, 0, y.data());
  CATCH_REQUIRE(isClose<float>(y, yRef));
}

CATCH_TEST_CASE("test q4 dot kernels", "[lymath][dot][q4]") {
  constexpr int DIM = 1024;

  std::vector<float> x(DIM);
  std::vector<uint8_t> y(DIM / 2);
  std::vector<float> scaleYFp32(DIM / GroupSizeQInt4);
  std::vector<uint8_t> zeroY(DIM / GroupSizeQInt4 / 2);
  std::vector<Float16> scaleY(DIM / GroupSizeQInt4);

  lut::Random random(MagicNumber);
  random.fillUInt8(lut::makeSpan(y));
  random.fillUInt8(lut::makeSpan(zeroY));
  random.fill(lut::makeSpan(scaleYFp32));
  random.fill(lut::makeSpan(x));
  std::transform(scaleYFp32.begin(), scaleYFp32.end(), scaleY.begin(), cvt_s2h);

  float a = SQInt4DotAvx2Kernel::apply(DIM, x.data(), {
      (const UInt4x2 *)y.data(),
      scaleY.data(),
      (const UInt4x2 *)zeroY.data()}, 0);
  float aRef = SQInt4DotFallbackKernel::apply(DIM, x.data(), {
      (const UInt4x2 *)y.data(),
      scaleY.data(),
      (const UInt4x2 *)zeroY.data()}, 0);

  CATCH_REQUIRE(isClose(a, aRef));
}

// to reproduce a zero-point index bug in q4 kernels.
CATCH_TEST_CASE("test q4 dot kernels apply row", "[lymath][dot][q4]") {
  constexpr int NUM_ROW = 32;
  constexpr int NUM_COL = 128;
  constexpr int NUMEL = NUM_COL * NUM_ROW;

  std::vector<float> x(NUM_COL);
  std::vector<float> y(NUM_ROW);
  std::vector<uint8_t> A(NUMEL / 2);
  std::vector<float> scaleAFp32(NUMEL / GroupSizeQInt4);
  std::vector<uint8_t> zeroA(NUMEL / GroupSizeQInt4 / 2);
  std::vector<Float16> scaleA(NUMEL / GroupSizeQInt4);

  lut::Random random(MagicNumber);
  random.fillUInt8(lut::makeSpan(A));
  random.fillUInt8(lut::makeSpan(zeroA));
  random.fill(lut::makeSpan(scaleAFp32));
  random.fill(lut::makeSpan(x));
  std::transform(scaleAFp32.begin(), scaleAFp32.end(), scaleA.begin(), cvt_s2h);

  QInt4GemvArgs<float> gemvArgs;
  gemvArgs.A = {(const UInt4x2 *)A.data(), scaleA.data(), (const UInt4x2 *)zeroA.data()};
  gemvArgs.incX = 1;
  gemvArgs.incY = 1;
  gemvArgs.M = NUM_ROW;
  gemvArgs.N = NUM_COL;
  gemvArgs.transA = false;
  gemvArgs.x = x.data();
  gemvArgs.y = nullptr;

  float a0 = SQInt4DotAvx2Kernel::applyRow(gemvArgs, 0);
  float a1 = SQInt4DotAvx2Kernel::applyRow(gemvArgs, 1);

  std::vector<float> x2(NUM_COL * 2);
  std::copy(x.begin(), x.end(), x2.begin());
  std::copy(x.begin(), x.end(), x2.begin() + NUM_COL);


  float a = SQInt4DotAvx2Kernel::apply(NUM_COL * 2, x2.data(), {
      (const UInt4x2 *)A.data(),
      scaleA.data(),
      (const UInt4x2 *)zeroA.data()}, 0);
  CATCH_REQUIRE(isClose(a, a0 + a1));
}


#endif  // LUT_ARCH_AMD64

#ifdef LUT_ARCH_AARCH64

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

CATCH_TEST_CASE("test HQInt4DotAsimdhpKernel", "[libllm][cpu_kernel][dot][half]") {
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(GroupSizeQInt4);
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(2 * GroupSizeQInt4);
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(16 * GroupSizeQInt4);
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(17 * GroupSizeQInt4);
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(31 * GroupSizeQInt4);
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(32 * GroupSizeQInt4);
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(33 * GroupSizeQInt4);
  testHQInt4DotKernel<HQInt4DotAsimdhpKernel>(50 * GroupSizeQInt4);
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
  tester.test(2048);
}

CATCH_TEST_CASE("test dequantQInt4Half kernels", "[libllm][cpu_kernel][dequant][qint4][half]") {
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(GroupSizeQInt4);
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(2 * GroupSizeQInt4);
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(10 * GroupSizeQInt4);
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(11 * GroupSizeQInt4);
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(12 * GroupSizeQInt4);
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(50 * GroupSizeQInt4);
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(51 * GroupSizeQInt4);
  testQInt4HalfDequantKernel<DequantQInt4ToHalfAsimdhpKernel>(52 * GroupSizeQInt4);
}

CATCH_TEST_CASE("test CvtHalfToFloatAsimdhpKernel kernel", "[libllm][cpu_kernel][cvt][half]") {
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(1);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(7);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(8);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(9);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(63);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(64);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(65);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(127);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(128);
  testCvtHalfToFloatKernel<CvtHalfToFloatAsimdhpKernel>(129);
}

CATCH_TEST_CASE("test CvtFloatToHalfAsimdhpKernel kernel", "[libllm][cpu_kernel][cvt][half]") {
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(1);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(7);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(8);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(9);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(63);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(64);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(65);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(127);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(128);
  testCvtFloatToHalfKernel<CvtFloatToHalfAsimdhpKernel>(129);
}

#endif  // LUT_ARCH_AARCH64

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
