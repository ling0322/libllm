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

CATCH_TEST_CASE("test CUDA lookup", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor embd = F::rand({10, 32}, DType::kFloat);
  Tensor ids = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor xr = F::lookup(embd, ids);

  Tensor x = F::lookup(toCuda(embd), F::to(Device::getCuda(), ids));

  CATCH_REQUIRE(F::allClose(toCpu(x), xr));

  // packed indices are 1D and give one embedding row per index.
  Tensor packedIds = Tensor::create<LongType>({3}, {1, 2, 3});
  Tensor packedRef = F::lookup(embd, packedIds);
  Tensor packed = F::lookup(toCuda(embd), F::to(Device::getCuda(), packedIds));

  CATCH_REQUIRE(packed.getShape() == std::vector<int>{3, 32});
  CATCH_REQUIRE(F::allClose(toCpu(packed), packedRef));
}

}  // namespace fl
