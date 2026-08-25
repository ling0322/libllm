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

CATCH_TEST_CASE("test CUDA binary operators", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({5}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toCuda(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toCuda(b);

  CATCH_REQUIRE(F::allClose(toCpu(F::add(xt, y)), F::add(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sub(xt, y)), F::sub(at, b), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::mul(xt, 0.1f)), F::mul(at, 0.1f), 1e-3, 1e-4));
}

CATCH_TEST_CASE("test CUDA binary operators (contiguous fast path)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Equal shapes on both sides means no broadcast and no strides, which is the separate
  // element-at-a-time kernel rather than the accessor-based one the strided test covers.
  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {1},
           {17},
           {4, 5},
           {2, 3, 4},
           {2, 3, 4, 5}}) {
    Tensor a = F::rand(shape, DType::kFloat);
    Tensor b = F::rand(shape, DType::kFloat);
    Tensor x = toCuda(a);
    Tensor y = toCuda(b);

    // allClose scales the error by the size of the result, and subtracting two independent
    // uniform values cancels down to near zero often enough that the relative test alone is not
    // meaningful. An absolute tolerance above half's round-off covers that without hiding a
    // kernel that returns a wholly wrong value.
    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(F::allClose(toCpu(F::add(x, y)), F::add(a, b), 5e-3, 5e-3));
    CATCH_REQUIRE(F::allClose(toCpu(F::sub(x, y)), F::sub(a, b), 5e-3, 5e-3));
    CATCH_REQUIRE(F::allClose(toCpu(F::mul(x, y)), F::mul(a, b), 5e-3, 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA binary operators (broadcast shapes)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 3, 4}, DType::kFloat);

  // A operand with fewer dims gets leading axes of stride 0; a size-1 axis is stretched in
  // place. Both make the right-hand side non-contiguous and force the generic kernel.
  for (std::vector<int> shape :
       std::vector<std::vector<int>>{{4}, {1, 4}, {3, 4}, {1, 1, 4}, {1, 3, 4}, {2, 1, 4}}) {
    Tensor b = F::rand(shape, DType::kFloat);
    CATCH_INFO("rhs rank = " << shape.size());
    CATCH_REQUIRE(F::allClose(toCpu(F::add(toCuda(a), toCuda(b))), F::add(a, b), 5e-3, 5e-3));
    CATCH_REQUIRE(F::allClose(toCpu(F::mul(toCuda(a), toCuda(b))), F::mul(a, b), 5e-3, 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA binary operators (4D strided)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // 4 is the highest rank the generic kernel is instantiated for, so it is the one that would
  // fall off the end of the dispatch chain.
  Tensor a = F::rand({2, 3, 4, 5}, DType::kFloat);
  Tensor b = F::rand({5}, DType::kFloat);

  Tensor at = a.transpose(1, 3);
  Tensor xt = toCuda(a).transpose(1, 3);
  CATCH_REQUIRE(!xt.isContiguous());

  // after the transpose the last axis is 3 long, so broadcast a matching operand instead.
  Tensor c = F::rand({3}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::add(xt, toCuda(c))), F::add(at, c), 5e-3, 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::sub(xt, toCuda(c))), F::sub(at, c), 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA binary operators (crosses the grid-stride loop)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // More elements than the capped grid can cover in one pass, so each thread loops.
  Tensor a = F::rand({256, 1024}, DType::kFloat);
  Tensor b = F::rand({1024}, DType::kFloat);

  CATCH_REQUIRE(F::allClose(toCpu(F::add(toCuda(a), toCuda(b))), F::add(a, b), 5e-3, 5e-3));
}

}  // namespace fl
