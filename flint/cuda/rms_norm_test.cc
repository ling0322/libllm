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

CATCH_TEST_CASE("test CUDA rmsNorm", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {10, 11}) {
    Tensor a = F::rand({2, 5, lastDim}, DType::kFloat);
    Tensor w = F::rand({lastDim}, DType::kFloat);
    Tensor x = F::rmsNorm(toCuda(a), toCuda(w), 1e-5);
    CATCH_REQUIRE(F::allClose(toCpu(x), F::rmsNorm(a, w, 1e-5), 5e-3));
  }

  // strided input.
  Tensor a = F::rand({2, 3, 11}, DType::kFloat);
  Tensor w = F::rand({11}, DType::kFloat);
  Tensor x = F::rmsNorm(toCuda(a).transpose(0, 1), toCuda(w), 1e-5);
  CATCH_REQUIRE(F::allClose(toCpu(x), F::rmsNorm(a.transpose(0, 1), w, 1e-5), 5e-3));
}

CATCH_TEST_CASE("test CUDA rmsNorm (packed 2D batch)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A packed batch is [tokens, hidden]; the operator adds a leading axis and strips it again, so
  // the result must come back 2D.
  for (int hidden : {8, 11, 512}) {
    Tensor a = F::rand({4, hidden}, DType::kFloat);
    Tensor w = F::rand({hidden}, DType::kFloat);
    Tensor x = F::rmsNorm(toCuda(a), toCuda(w), 1e-5);

    CATCH_INFO("hidden = " << hidden);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{4, hidden});
    CATCH_REQUIRE(F::allClose(toCpu(x), F::rmsNorm(a, w, 1e-5), 5e-3));
  }

  // one token, the decode-step shape.
  Tensor one = F::rand({1, 64}, DType::kFloat);
  Tensor oneW = F::rand({64}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(
      toCpu(F::rmsNorm(toCuda(one), toCuda(oneW), 1e-5)),
      F::rmsNorm(one, oneW, 1e-5),
      5e-3));
}

CATCH_TEST_CASE("test CUDA rmsNorm (hidden sizes)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // One 256-thread block reduces a whole row, so widths on both sides of the block size take a
  // different number of loop iterations, and odd widths disable the half2 path.
  for (int hidden : {1, 2, 3, 255, 256, 257, 512, 2048, 4096}) {
    Tensor a = F::rand({2, 3, hidden}, DType::kFloat);
    Tensor w = F::rand({hidden}, DType::kFloat);

    CATCH_INFO("hidden = " << hidden);
    CATCH_REQUIRE(F::allClose(
        toCpu(F::rmsNorm(toCuda(a), toCuda(w), 1e-5)),
        F::rmsNorm(a, w, 1e-5),
        1e-2));
  }
}

CATCH_TEST_CASE("test CUDA rmsNorm (strided weight)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A non-contiguous weight is enough on its own to force the strided kernel, even when the
  // input is contiguous.
  Tensor a = F::rand({2, 3, 6}, DType::kFloat);
  Tensor wSource = F::rand({6, 4}, DType::kFloat);
  Tensor w = wSource.transpose(0, 1).subtensor(0);
  Tensor wDevice = toCuda(wSource).transpose(0, 1).subtensor(0);
  CATCH_REQUIRE(!wDevice.isContiguous());

  Tensor x = F::rmsNorm(toCuda(a), wDevice, 1e-5);
  CATCH_REQUIRE(F::allClose(toCpu(x), F::rmsNorm(a, F::contiguous(w), 1e-5), 5e-3));
}

CATCH_TEST_CASE("test CUDA rmsNorm (eps dominates a zero row)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // An all-zero row has zero mean square, so eps is the only thing keeping the reciprocal square
  // root finite. The output stays zero rather than becoming NaN.
  Tensor a = F::zeros({2, 8}, DType::kFloat);
  Tensor w = F::rand({8}, DType::kFloat);

  Tensor x = toCpu(F::rmsNorm(toCuda(a), toCuda(w), 1e-5));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  for (int i = 0; i < 16; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(!std::isnan(data[i]));
    CATCH_REQUIRE(data[i] == 0.0f);
  }
}

}  // namespace fl
