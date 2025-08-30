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
#include "lutil/half.h"
#include "lutil/log.h"
#include "lutil/random.h"
#include "lynn/cpu/kernel/abstract.h"
#include "lynn/cpu/kernel/test_common.h"
#include "lynn/cpu/kernel/util.h"
#include "ruapu/ruapu.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

CATCH_TEST_CASE("test sgemm6x16Avx2Kernel", "[cpu_kernel][kernel][avx512]") {
  bool isaAvx512f = ruapu_supports("avx512f") > 0;
  if (!isaAvx512f) {
    CATCH_SKIP("skip sgemm6x16Avx2Kernel tesing since CPU not supported.");
  }

  GemmMicroKernelTester<float, float, float, 12, 32, CpuMathBackend::AVX512> tester;
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

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
