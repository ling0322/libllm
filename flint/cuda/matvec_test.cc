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

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/cuda/matvec.h"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {

CATCH_TEST_CASE("test gemv", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor w = F::rand({8000, 4096}, DType::kFloat, Device::kCpu);
  Tensor x = F::rand({1, 4096}, DType::kFloat, Device::kCpu);

  w = F::cast(F::to(Device::getCuda(), w), DType::kFloat16);
  x = F::cast(F::to(Device::getCuda(), x), DType::kFloat16);

  // the layout Linear::forward feeds to matmul: a transposed weight, so getStride(0) == 1.
  Tensor wT = w.transpose(0, 1);

  Tensor xr = F::matmul(x, F::contiguous(wT));
  Tensor xv = op::cuda::gemvHalf(x.subtensor(0), wT);

  xr = F::to(Device::getCpu(), F::cast(xr, DType::kFloat));
  xv = F::to(Device::getCpu(), F::cast(xv, DType::kFloat));

  CATCH_REQUIRE(F::allClose(xr, xv, 5e-3f));
}

CATCH_TEST_CASE("test gemv (shapes)", "[fl][op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Each output row is produced by one warp, and rows are grouped four to a block, so an output
  // count that is not a multiple of four leaves a partly idle block whose extra rows must not be
  // written. n is stepped in units of 8 because each thread loads eight halves at a time.
  auto runCase = [](int n, int d) {
    Tensor w = F::rand({d, n}, DType::kFloat, Device::kCpu);
    Tensor x = F::rand({1, n}, DType::kFloat, Device::kCpu);

    w = F::cast(F::to(Device::getCuda(), w), DType::kFloat16);
    x = F::cast(F::to(Device::getCuda(), x), DType::kFloat16);

    Tensor wT = w.transpose(0, 1);
    Tensor expected = F::matmul(x, F::contiguous(wT));
    Tensor actual = op::cuda::gemvHalf(x.subtensor(0), wT);

    CATCH_INFO("n = " << n << ", d = " << d);
    CATCH_REQUIRE(actual.getShape() == std::vector<int>{1, d});
    return F::allClose(
        F::to(Device::getCpu(), F::cast(expected, DType::kFloat)),
        F::to(Device::getCpu(), F::cast(actual, DType::kFloat)),
        5e-3f);
  };

  // output counts either side of the four-rows-per-block grouping.
  CATCH_REQUIRE(runCase(64, 1));
  CATCH_REQUIRE(runCase(64, 3));
  CATCH_REQUIRE(runCase(64, 4));
  CATCH_REQUIRE(runCase(64, 5));
  CATCH_REQUIRE(runCase(64, 10));

  // the shortest row the eight-wide load allows, and one that needs several warp iterations.
  CATCH_REQUIRE(runCase(8, 16));
  CATCH_REQUIRE(runCase(256, 16));
  CATCH_REQUIRE(runCase(2048, 7));
}

}  // namespace fl
