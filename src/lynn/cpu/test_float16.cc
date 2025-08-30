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
#include "lynn/context.h"
#include "lynn/cpu/fingerprint.h"
#include "lynn/functional.h"
#include "lynn/operator_tester.h"
#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cpu {

OperatorTester getOperatorTester() {
  return OperatorTester()
      .withOperators(getOperators(Device::kCpu))
      .withDevice(Device::getCpu())
      .withFloatType(DType::kFloat16);
}

CATCH_TEST_CASE("test CPU fp16 binary operators", "[op][cpu][float16]") {
  OperatorTester tester = getOperatorTester().withTol(5e-3);
  CATCH_REQUIRE(tester.testBinaryOp(OperatorTester::OperatorType::Add));
  CATCH_REQUIRE(tester.testBinaryOp(OperatorTester::OperatorType::Mul));
}

CATCH_TEST_CASE("test CPU fp16 copy operators", "[op][cpu][float16]") {
  OperatorTester tester = getOperatorTester();
  CATCH_REQUIRE(tester.testCopy({2, 10, 50}, true));
  CATCH_REQUIRE(tester.testCopy5D());
}

CATCH_TEST_CASE("test CPU fp16 matmul operators", "[op][cpu][float16]") {
  OperatorTester tester = getOperatorTester().withTol(5e-2);
  CATCH_REQUIRE(tester.testMatmulSlice({10, 20}, {40, 30}));
  CATCH_REQUIRE(tester.testMatmulSlice({5, 10, 20}, {40, 30}));
  CATCH_REQUIRE(tester.testMatmulSlice({5, 10, 5, 20}, {10, 40, 30}));
  CATCH_REQUIRE(tester.testMatmulQInt4({5, 10, 50}, {50, 128}, false));
  CATCH_REQUIRE(tester.testMatmulQInt4({1, 1, 128}, {50, 128}, true));
}

CATCH_TEST_CASE("test CPU fp16 rmsNorm operator", "[op][cpu][float16]") {
  OperatorTester tester = getOperatorTester().withTol(5e-2);
  CATCH_REQUIRE(tester.testRmsNorm({2, 5, 10}));
}

CATCH_TEST_CASE("test CPU fp16 activation operators", "[op][cpu][float16]") {
  OperatorTester tester = getOperatorTester().withTol(5e-2);
  CATCH_REQUIRE(tester.testUnaryOp(OperatorTester::OperatorType::Softmax, {2, 5, 150}));
  CATCH_REQUIRE(tester.testUnaryOp(OperatorTester::OperatorType::Swiglu, {2, 5, 150}));
}

CATCH_TEST_CASE("test CPU fp16 tensor operators", "[op][cpu][float16]") {
  OperatorTester tester = getOperatorTester().withTol(5e-2);
  CATCH_REQUIRE(tester.testCausalMask());
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
