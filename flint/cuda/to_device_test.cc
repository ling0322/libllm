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

CATCH_TEST_CASE("test CUDA to and cast", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({100, 200}, DType::kFloat);

  Tensor roundTrip = F::to(Device::getCpu(), F::to(Device::getCuda(), a));
  CATCH_REQUIRE(F::allClose(roundTrip, a));

  CATCH_REQUIRE(F::allClose(toCpu(toCuda(a)), a));
}

CATCH_TEST_CASE("test CUDA to (rank and dtype)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // the copy is a flat memcpy, so what varies across ranks is only the shape that comes back.
  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {1},
           {7},
           {1, 1},
           {3, 5},
           {2, 3, 4},
           {2, 3, 4, 5}}) {
    Tensor a = F::rand(shape, DType::kFloat);
    Tensor roundTrip = F::to(Device::getCpu(), F::to(Device::getCuda(), a));
    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(roundTrip.getShape() == shape);
    CATCH_REQUIRE(F::allClose(roundTrip, a));
  }

  // long tensors travel the same path but are never cast, so values that do not survive a float
  // round trip must come back exactly.
  Tensor ids = Tensor::create<LongType>({2, 3}, {-1, 0, 1, 2, 3, LongType{1} << 40});
  Tensor idsRoundTrip = F::to(Device::getCpu(), F::to(Device::getCuda(), ids));
  const LongType *data = idsRoundTrip.getInternalData()->getData<LongType>(
      idsRoundTrip.getInternalOffset());
  CATCH_REQUIRE(data[0] == -1);
  CATCH_REQUIRE(data[5] == (LongType{1} << 40));
}

CATCH_TEST_CASE("test CUDA cast is a no-op for the same dtype", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::to(Device::getCuda(), F::rand({4, 8}, DType::kFloat));
  Tensor same = F::cast(a, DType::kFloat);

  // the same dtype short-circuits, so no copy is made and the storage is shared.
  CATCH_REQUIRE(same.getInternalData() == a.getInternalData());
}

CATCH_TEST_CASE("test CUDA cast saturates outside the half range", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // 65504 is the largest finite half. Round-to-nearest keeps everything below the 65520 midpoint
  // finite and sends the rest to infinity; far below the smallest subnormal the value flushes to
  // zero. allClose cannot express either end, so read the values back directly.
  Tensor a = Tensor::create<float>({7}, {65504.0f, 65519.0f, 65521.0f, 1e5f, -1e5f, 1e-8f, 0.0f});
  Tensor x = toCpu(toCuda(a));
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());

  CATCH_REQUIRE(data[0] == 65504.0f);
  CATCH_REQUIRE(data[1] == 65504.0f);
  CATCH_REQUIRE(std::isinf(data[2]));
  CATCH_REQUIRE(std::isinf(data[3]));
  CATCH_REQUIRE(std::isinf(data[4]));
  CATCH_REQUIRE(data[4] < 0.0f);
  CATCH_REQUIRE(data[5] == 0.0f);
  CATCH_REQUIRE(data[6] == 0.0f);
}

}  // namespace fl
