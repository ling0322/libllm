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

#include <algorithm>
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

bool equalLong(Tensor a, Tensor b) {
  a.throwIfInvalidShape(b.getShape(), "equalLong");

  const LongType *pa = a.getInternalData()->getData<LongType>(a.getInternalOffset());
  const LongType *pb = b.getInternalData()->getData<LongType>(b.getInternalOffset());
  return std::equal(pa, pa + a.getNumEl(), pb);
}

}  // namespace

CATCH_TEST_CASE("test CUDA arange", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor arange = F::to(Device::getCpu(), F::arange(0, 10, 2, Device::getCuda()));
  CATCH_REQUIRE(equalLong(arange, Tensor::create<LongType>({5}, {0, 2, 4, 6, 8})));
}

CATCH_TEST_CASE("test CUDA arange (step and bounds)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The length is a truncating division, so a step that does not divide the span evenly stops
  // short of the end rather than overshooting it.
  Tensor uneven = F::to(Device::getCpu(), F::arange(0, 10, 3, Device::getCuda()));
  CATCH_REQUIRE(uneven.getShape() == std::vector<int>{3});
  CATCH_REQUIRE(equalLong(uneven, Tensor::create<LongType>({3}, {0, 3, 6})));

  // a step of 1 is the common case, and a non-zero start offsets every element.
  Tensor unit = F::to(Device::getCpu(), F::arange(5, 9, 1, Device::getCuda()));
  CATCH_REQUIRE(equalLong(unit, Tensor::create<LongType>({4}, {5, 6, 7, 8})));

  // a negative start walks up through zero.
  Tensor negative = F::to(Device::getCpu(), F::arange(-3, 3, 2, Device::getCuda()));
  CATCH_REQUIRE(equalLong(negative, Tensor::create<LongType>({3}, {-3, -1, 1})));

  // a single element.
  Tensor one = F::to(Device::getCpu(), F::arange(7, 8, 1, Device::getCuda()));
  CATCH_REQUIRE(one.getShape() == std::vector<int>{1});
  CATCH_REQUIRE(equalLong(one, Tensor::create<LongType>({1}, {7})));
}

CATCH_TEST_CASE("test CUDA arange (descending)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A negative step makes both the span and the step negative, so the length stays positive and
  // the values count down.
  Tensor down = F::to(Device::getCpu(), F::arange(10, 0, -2, Device::getCuda()));
  CATCH_REQUIRE(down.getShape() == std::vector<int>{5});
  CATCH_REQUIRE(equalLong(down, Tensor::create<LongType>({5}, {10, 8, 6, 4, 2})));
}

CATCH_TEST_CASE("test CUDA arange (crosses a block boundary)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // More elements than one 256-thread block covers, so the tail past the boundary would be left
  // uninitialised by a kernel that forgot its bounds check.
  constexpr int Count = 300;
  Tensor x = F::to(Device::getCpu(), F::arange(0, Count, 1, Device::getCuda()));
  CATCH_REQUIRE(x.getShape() == std::vector<int>{Count});

  std::vector<LongType> expected(Count);
  for (int i = 0; i < Count; ++i) expected[i] = i;
  CATCH_REQUIRE(equalLong(x, Tensor::create<LongType>({Count}, expected)));
}

}  // namespace fl
