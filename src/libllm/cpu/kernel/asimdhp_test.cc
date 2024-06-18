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

#include "libllm/cpu/kernel/asimdhp.h"

#include <math.h>

#include "catch2/catch_amalgamated.hpp"
#include "libllm/cpu/kernel/abstract.h"
#include "libllm/cpu/kernel/test_common.h"
#include "libllm/cpu/kernel/util.h"
#include "libllm/lut/half.h"
#include "libllm/lut/log.h"
#include "libllm/lut/random.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

CATCH_TEST_CASE("test hgemm12x16AsimdhpKernel", "[cpu_kernel][kernel][asimdhp]") {
  GemmMicroKernelTester<Float16, Float16, Float16, 12, 16, CpuMathBackend::ASIMDHP> tester;
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

CATCH_TEST_CASE("test hsaxpyAsimdhpKernel", "[cpu_kernel][kernel][asimdhp]") {
  AxpyKernelTester<Float16, Float16, float, CpuMathBackend::ASIMDHP> tester;
  tester.test(1);
  tester.test(8);
  tester.test(16);
  tester.test(17);
  tester.test(128);
  tester.test(2001);
}

CATCH_TEST_CASE("test qhcvtAsimdhpKernel", "[cpu_kernel][kernel][asimdhp]") {
  CvtKernelTester<QInt4x32, Float16, CpuMathBackend::ASIMDHP> tester;
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

CATCH_TEST_CASE("test hscvtAsimdhpKernel", "[cpu_kernel][kernel][asimdhp]") {
  CvtKernelTester<Float16, float, CpuMathBackend::ASIMDHP> tester;
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

CATCH_TEST_CASE("test shcvtAsimdhpKernel", "[cpu_kernel][kernel][asimdhp]") {
  CvtKernelTester<float, Float16, CpuMathBackend::ASIMDHP> tester;
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

CATCH_TEST_CASE("test hdotAsimdhpKernel", "[cpu_kernel][kernel][asimdhp]") {
  DotKernelTester<Float16, Float16, Float16, CpuMathBackend::ASIMDHP> tester;
  tester.test(1);
  tester.test(8);
  tester.test(16);
  tester.test(17);
  tester.test(128);
  tester.test(160);
  tester.test(1500);
  tester.test(2001);
  tester.test(20000);
}

CATCH_TEST_CASE("test hqdotAsimdhpKernel", "[cpu_kernel][kernel][asimdhp]") {
  DotKernelTester<Float16, Float16, QInt4x32, CpuMathBackend::ASIMDHP> tester;
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

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
