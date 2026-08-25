// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cmath>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toCuda(const Tensor &a) {
  return F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

using UnaryFn = Tensor (*)(Tensor);

struct Case {
  const char *name;
  UnaryFn fn;
};

/// Every element-wise function, so a new one added to the enum without a CUDA arm shows up here
/// rather than at a call site.
const std::vector<Case> &allUnary() {
  static const std::vector<Case> cases = {
      {"neg", F::neg},
      {"abs", F::abs},
      {"exp", F::exp},
      {"square", F::square},
      {"sigmoid", F::sigmoid},
      {"tanh", F::tanh},
      {"relu", F::relu},
      {"gelu", F::gelu},
      {"silu", F::silu},
  };
  return cases;
}

}  // namespace

CATCH_TEST_CASE("test CUDA unary operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Values spread either side of zero, including magnitudes where the saturating functions have
  // flattened. exp(8) is still inside the half range; much beyond that it would overflow.
  Tensor a = Tensor::create<float>(
      {2, 5},
      {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.5f, -2.5f, 8.0f, -8.0f, 0.125f});
  Tensor x = toCuda(a);

  for (const Case &c : allUnary()) {
    CATCH_INFO("op = " << c.name);
    CATCH_REQUIRE(F::allClose(toCpu(c.fn(x)), c.fn(a), 5e-3, 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA unary operators (positive domain)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // sqrt and rsqrt need a non-negative input, so they get their own values.
  Tensor a = Tensor::create<float>({5}, {0.25f, 1.0f, 2.0f, 9.0f, 0.0625f});
  Tensor x = toCuda(a);

  CATCH_REQUIRE(F::allClose(toCpu(F::sqrt(x)), F::sqrt(a), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::rsqrt(x)), F::rsqrt(a), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA unary operators (shapes and strides)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The kernel has a packed arm and an accessor arm per rank; walk both, and a size that makes
  // every thread go round the grid-stride loop more than once.
  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {1},
           {17},
           {4, 5},
           {2, 3, 4},
           {2, 3, 4, 5},
           {256, 1024}}) {
    Tensor a = F::rand(shape, DType::kFloat);
    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(F::allClose(toCpu(F::neg(toCuda(a))), F::neg(a), 5e-3, 5e-3));
    CATCH_REQUIRE(F::allClose(toCpu(F::silu(toCuda(a))), F::silu(a), 5e-3, 5e-3));
  }

  // a strided view has to be read through its strides, not as a flat buffer.
  Tensor a = F::rand({2, 3, 4}, DType::kFloat);
  Tensor strided = toCuda(a).transpose(0, 2);
  CATCH_REQUIRE(!strided.isContiguous());
  CATCH_REQUIRE(F::allClose(toCpu(F::neg(strided)), F::neg(a.transpose(0, 2)), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA unary operators (float tensors)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The kernels are instantiated for float as well as half; moving without a cast selects it.
  Tensor a = F::rand({3, 8}, DType::kFloat);
  Tensor x = F::to(Device::getCuda(), a);
  CATCH_REQUIRE(x.getDType() == DType::kFloat);

  for (const Case &c : allUnary()) {
    Tensor actual = c.fn(x);
    CATCH_INFO("op = " << c.name);
    CATCH_REQUIRE(actual.getDType() == DType::kFloat);
    CATCH_REQUIRE(F::allClose(F::to(Device::getCpu(), actual), c.fn(a), 1e-4, 1e-5));
  }
}

CATCH_TEST_CASE("test CUDA unary operators (known values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Fixed points that separate the activations from one another: at x = 0 relu, gelu and silu all
  // give 0 while sigmoid gives 0.5, and a large negative input drives all three to 0.
  Tensor a = Tensor::create<float>({3}, {0.0f, -10.0f, 1.0f});
  Tensor x = toCuda(a);

  Tensor reluOut = toCpu(F::relu(x));
  const float *relu = reluOut.getInternalData()->getData<float>(reluOut.getInternalOffset());
  CATCH_REQUIRE(relu[0] == 0.0f);
  CATCH_REQUIRE(relu[1] == 0.0f);
  CATCH_REQUIRE(std::fabs(relu[2] - 1.0f) < 1e-2f);

  Tensor sigmoidOut = toCpu(F::sigmoid(x));
  const float *sigmoid =
      sigmoidOut.getInternalData()->getData<float>(sigmoidOut.getInternalOffset());
  CATCH_REQUIRE(std::fabs(sigmoid[0] - 0.5f) < 1e-2f);
  CATCH_REQUIRE(sigmoid[1] < 1e-2f);

  // gelu and silu both pass through the origin, unlike sigmoid.
  Tensor geluOut = toCpu(F::gelu(x));
  Tensor siluOut = toCpu(F::silu(x));
  CATCH_REQUIRE(geluOut.getInternalData()->getData<float>(geluOut.getInternalOffset())[0] == 0.0f);
  CATCH_REQUIRE(siluOut.getInternalData()->getData<float>(siluOut.getInternalOffset())[0] == 0.0f);
}

CATCH_TEST_CASE("test CUDA div (element-wise)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b = Tensor::create<float>({2, 3}, {2.0f, 4.0f, 4.0f, 8.0f, 10.0f, 3.0f});
  CATCH_REQUIRE(F::allClose(toCpu(F::div(toCuda(a), toCuda(b))), F::div(a, b), 5e-3, 5e-3));

  // the divisor broadcasts over the leading dimensions, which takes the strided kernel.
  Tensor row = Tensor::create<float>({3}, {1.0f, 2.0f, 4.0f});
  CATCH_REQUIRE(F::allClose(toCpu(F::div(toCuda(a), toCuda(row))), F::div(a, row), 5e-3, 5e-3));

  // dividing by itself is 1 everywhere, which a kernel that dropped the divisor would not give.
  Tensor ones = toCpu(F::div(toCuda(a), toCuda(a)));
  Tensor expected = F::tensor({2, 3}, DType::kFloat, Device::getCpu());
  F::fill(expected, 1.0f);
  CATCH_REQUIRE(F::allClose(ones, expected, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA min", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // An all-positive row would still look right if min returned the initial +inf only when every
  // element is larger, so include a row that is entirely negative.
  Tensor a = Tensor::create<float>({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});
  CATCH_REQUIRE(F::allClose(toCpu(F::min(toCuda(a))), F::min(a), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::max(toCuda(a))), F::max(a), 5e-3, 5e-3));

  // widths either side of the block size, where the reduction loops.
  for (int width : {1, 255, 256, 257, 1000}) {
    Tensor b = F::rand({2, 3, width}, DType::kFloat);
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(F::allClose(toCpu(F::min(toCuda(b))), F::min(b), 5e-3, 5e-3));
  }
}

}  // namespace fl
