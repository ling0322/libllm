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

}  // namespace fl
