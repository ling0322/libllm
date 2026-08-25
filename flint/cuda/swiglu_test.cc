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

}  // namespace

CATCH_TEST_CASE("test CUDA swiglu", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {150, 152}) {
    Tensor a = F::rand({2, 5, lastDim}, DType::kFloat);
    CATCH_REQUIRE(F::allClose(toCpu(F::swiglu(toCuda(a))), F::swiglu(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA swiglu (strided)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor b = F::rand({2, 3, 152}, DType::kFloat);
  Tensor y = F::swiglu(toCuda(b).transpose(0, 1));
  CATCH_REQUIRE(F::allClose(toCpu(y), F::swiglu(b.transpose(0, 1)), 5e-3));

  Tensor c = F::rand({2, 152, 3}, DType::kFloat);
  Tensor z = F::swiglu(toCuda(c).transpose(1, 2));
  CATCH_REQUIRE(F::allClose(toCpu(z), F::swiglu(c.transpose(1, 2)), 5e-3));
}

CATCH_TEST_CASE("test CUDA swiglu (packed 2D batch)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A packed batch arrives as [tokens, 2 * hidden]; the operator wraps it in a leading axis and
  // unwraps the result, so the output must still be 2D.
  for (int width : {4, 6, 512, 600}) {
    Tensor a = F::rand({5, width}, DType::kFloat);
    Tensor x = F::swiglu(toCuda(a));

    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{5, width / 2});
    CATCH_REQUIRE(F::allClose(toCpu(x), F::swiglu(a), 5e-3));
  }

  // a single token is the decode-step shape.
  Tensor one = F::rand({1, 16}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::swiglu(toCuda(one))), F::swiglu(one), 5e-3));
}

CATCH_TEST_CASE("test CUDA swiglu (output widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The output half is split across 256-thread blocks in x, and an odd output width turns off
  // the half2 path. Walk both sides of the block boundary with each parity.
  for (int outputWidth : {1, 2, 3, 255, 256, 257, 512, 1000}) {
    Tensor a = F::rand({2, 3, outputWidth * 2}, DType::kFloat);
    Tensor x = F::swiglu(toCuda(a));

    CATCH_INFO("outputWidth = " << outputWidth);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{2, 3, outputWidth});
    CATCH_REQUIRE(F::allClose(toCpu(x), F::swiglu(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA swiglu (gate saturation)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // silu(gate) * value at the extremes: a large negative gate drives the output to zero and a
  // large positive one leaves the value essentially untouched. A zero gate contributes nothing.
  Tensor a = Tensor::create<float>({1, 8}, {-20.0f, 0.0f, 20.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f});
  Tensor x = toCpu(F::swiglu(toCuda(a)));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());

  CATCH_REQUIRE(std::fabs(data[0]) < 1e-2f);          // silu(-20) * 3 ~= 0
  CATCH_REQUIRE(std::fabs(data[1]) < 1e-3f);          // silu(0) * 5 == 0
  CATCH_REQUIRE(std::fabs(data[2] - 20.0f * 7.0f) < 1.0f);  // silu(20) ~= 20
  CATCH_REQUIRE(F::allClose(x, F::swiglu(a), 5e-3));
}

}  // namespace fl
