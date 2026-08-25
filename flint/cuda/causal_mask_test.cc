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

Tensor toCpu(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

}  // namespace

CATCH_TEST_CASE("test CUDA causalMask", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int Dim = 129;
  Tensor xr = F::softmax(F::causalMask(Dim));
  Tensor x = F::softmax(F::causalMask(Dim, Device::getCuda()));

  CATCH_REQUIRE(F::allClose(toCpu(x), xr, 1e-3, 1e-4));
}

CATCH_TEST_CASE("test CUDA causalMask (structure)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Comparing through softmax hides which side of the diagonal is masked, so check the raw mask:
  // zero on and below the diagonal, -inf strictly above it.
  constexpr int Dim = 5;
  Tensor mask = toCpu(F::causalMask(Dim, Device::getCuda()));
  CATCH_REQUIRE(mask.getShape() == std::vector<int>{Dim, Dim});

  const float *data = mask.getInternalData()->getData<float>(mask.getInternalOffset());
  for (int y = 0; y < Dim; ++y) {
    for (int x = 0; x < Dim; ++x) {
      CATCH_INFO("y = " << y << ", x = " << x);
      if (x > y) {
        CATCH_REQUIRE(std::isinf(data[y * Dim + x]));
        CATCH_REQUIRE(data[y * Dim + x] < 0.0f);
      } else {
        CATCH_REQUIRE(data[y * Dim + x] == 0.0f);
      }
    }
  }
}

CATCH_TEST_CASE("test CUDA causalMask (sizes)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Rows are filled by a grid 256 threads wide, so sizes either side of that boundary decide
  // whether the far end of a row is written at all. Size 1 is the single unmasked element.
  for (int size : {1, 2, 255, 256, 257}) {
    Tensor mask = toCpu(F::causalMask(size, Device::getCuda()));
    CATCH_INFO("size = " << size);
    CATCH_REQUIRE(mask.getShape() == std::vector<int>{size, size});

    const float *data = mask.getInternalData()->getData<float>(mask.getInternalOffset());
    // the last row is entirely visible and the first row sees only itself.
    CATCH_REQUIRE(data[0] == 0.0f);
    for (int x = 0; x < size; ++x) {
      CATCH_INFO("x = " << x);
      CATCH_REQUIRE(data[(size - 1) * size + x] == 0.0f);
      if (x > 0) CATCH_REQUIRE(std::isinf(data[x]));
    }
  }
}

}  // namespace fl
