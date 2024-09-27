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

#include <math.h>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/cpu/kernel/abstract.h"
#include "libllm/cpu/kernel/test_common.h"
#include "libllm/cpu/kernel/util.h"
#include "lutil/half.h"
#include "lutil/log.h"
#include "lutil/random.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

CATCH_TEST_CASE("test qscvtAvx2Kernel", "[cpu_kernel][kernel][avx2]") {
  CvtKernelTester<QInt4x32, float, CpuMathBackend::AVX2> tester;
  tester.test(GroupSizeQInt4);
  tester.test(2 * GroupSizeQInt4);
  tester.test(10 * GroupSizeQInt4);
  tester.test(11 * GroupSizeQInt4);
  tester.test(12 * GroupSizeQInt4);
  tester.test(50 * GroupSizeQInt4);
  tester.test(51 * GroupSizeQInt4);
  tester.test(52 * GroupSizeQInt4);
  tester.test(51 * GroupSizeQInt4);
  tester.test(52 * GroupSizeQInt4, GroupSizeQInt4);
  tester.test(52 * GroupSizeQInt4, GroupSizeQInt4 * 2);
}

CATCH_TEST_CASE("test hscvtAvx2Kernel", "[cpu_kernel][kernel][avx2]") {
  CvtKernelTester<Float16, float, CpuMathBackend::AVX2> tester;
  tester.test(1);
  tester.test(7);
  tester.test(8);
  tester.test(9);
  tester.test(63);
  tester.test(64);
  tester.test(65);
  tester.test(127);
  tester.test(128);
  tester.test(129);
  tester.test(129, 1);
  tester.test(129, 16);
  tester.test(129, 17);
}

CATCH_TEST_CASE("test sgemm6x16Avx2Kernel", "[cpu_kernel][kernel][avx2]") {
  GemmMicroKernelTester<float, float, float, 6, 16, CpuMathBackend::AVX2> tester;
  tester.test(1);
  tester.test(8);
  tester.test(17);
  tester.test(64);
  tester.test(100);
  tester.test(256);
  tester.test(500);
  tester.test(2047);
  tester.test(2048);
}

CATCH_TEST_CASE("test sdotAvx2Kernel", "[cpu_kernel][kernel][avx2]") {
  DotKernelTester<float, float, float, CpuMathBackend::AVX2> tester;
  tester.test(1);
  tester.test(8);
  tester.test(16);
  tester.test(17);
  tester.test(128);
  tester.test(160);
  tester.test(1500);
  tester.test(2001);
  tester.test(2001, 50);
  tester.test(2001, 60);
  tester.test(20000);
}

CATCH_TEST_CASE("test sqdotAvx2Kernel", "[cpu_kernel][kernel][avx2]") {
  DotKernelTester<float, float, QInt4x32, CpuMathBackend::AVX2> tester;
  tester.test(GroupSizeQInt4);
  tester.test(2 * GroupSizeQInt4);
  tester.test(16 * GroupSizeQInt4);
  tester.test(17 * GroupSizeQInt4);
  tester.test(31 * GroupSizeQInt4);
  tester.test(32 * GroupSizeQInt4);
  tester.test(33 * GroupSizeQInt4);
  tester.test(50 * GroupSizeQInt4);
  tester.test(50 * GroupSizeQInt4, GroupSizeQInt4);
  tester.test(50 * GroupSizeQInt4, 2 * GroupSizeQInt4);
}

CATCH_TEST_CASE("test saxpyAvx2Kernel", "[cpu_kernel][kernel][avx2]") {
  AxpyKernelTester<float, float, float, CpuMathBackend::AVX2> tester;
  tester.test(1);
  tester.test(8);
  tester.test(16);
  tester.test(17);
  tester.test(128);
  tester.test(2001);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
