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

CATCH_TEST_CASE("test CUDA matmul", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](std::initializer_list<int> shapeA, std::initializer_list<int> shapeB) {
    Tensor a = F::rand(shapeA, DType::kFloat);
    Tensor b = F::rand(shapeB, DType::kFloat);
    Tensor xr = F::matmul(a, b.slice(-1, {8, 32}).transpose(-1, -2));

    Tensor y = toCuda(b).slice(-1, {8, 32}).transpose(-1, -2);
    Tensor x = F::matmul(toCuda(a), y);

    return F::allClose(toCpu(x), xr, 5e-2);
  };

  CATCH_REQUIRE(runCase({10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 24}, {40, 64}));
  CATCH_REQUIRE(runCase({5, 10, 5, 24}, {10, 40, 64}));
}

CATCH_TEST_CASE("test CUDA matmul (2D shapes)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](int m, int k, int n) {
    Tensor a = F::rand({m, k}, DType::kFloat);
    Tensor b = F::rand({k, n}, DType::kFloat);

    Tensor x = F::matmul(toCuda(a), toCuda(b));
    CATCH_INFO("m = " << m << ", k = " << k << ", n = " << n);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{m, n});
    return F::allClose(toCpu(x), F::matmul(a, b), 5e-2);
  };

  // A single row or column collapses the output to a vector shape, and k == 1 makes the
  // accumulation a single product. A one-column B is the case that looks like a transposed
  // weight to the gemv dispatch (its leading stride is also 1) without being one.
  CATCH_REQUIRE(runCase(1, 8, 1));
  CATCH_REQUIRE(runCase(1, 8, 16));
  CATCH_REQUIRE(runCase(16, 8, 1));
  CATCH_REQUIRE(runCase(4, 1, 4));
  CATCH_REQUIRE(runCase(1, 1, 1));
  // a single row against a k that the vector kernel's eight-wide load cannot divide.
  CATCH_REQUIRE(runCase(1, 12, 16));
  // shapes that are not multiples of a tile.
  CATCH_REQUIRE(runCase(17, 33, 65));
}

CATCH_TEST_CASE("test CUDA matmul (dispatches a row to gemv)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single-row left operand against a transposed weight is the layout Linear::forward feeds
  // during decode, and it is routed to the vector kernel rather than the general GEMM.
  constexpr int K = 64;
  constexpr int N = 40;
  Tensor x = F::rand({1, K}, DType::kFloat);
  Tensor w = F::rand({N, K}, DType::kFloat);

  Tensor wT = toCuda(w).transpose(0, 1);
  CATCH_REQUIRE(wT.getStride(0) == 1);

  Tensor actual = F::matmul(toCuda(x), wT);
  CATCH_REQUIRE(actual.getShape() == std::vector<int>{1, N});
  CATCH_REQUIRE(F::allClose(toCpu(actual), F::matmul(x, w.transpose(0, 1)), 5e-2));
}

CATCH_TEST_CASE("test CUDA matmul (batched)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Both operands batched, with one and two batch axes: these are the two instantiations of the
  // pointer-array gather that feeds the batched GEMM.
  Tensor a3 = F::rand({4, 6, 8}, DType::kFloat);
  Tensor b3 = F::rand({4, 8, 5}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(
      toCpu(F::matmul(toCuda(a3), toCuda(b3))),
      F::matmul(a3, b3),
      5e-2));

  Tensor a4 = F::rand({2, 3, 6, 8}, DType::kFloat);
  Tensor b4 = F::rand({2, 3, 8, 5}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(
      toCpu(F::matmul(toCuda(a4), toCuda(b4))),
      F::matmul(a4, b4),
      5e-2));

  // a right operand with fewer batch axes is broadcast across the batch.
  Tensor b2 = F::rand({8, 5}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(
      toCpu(F::matmul(toCuda(a3), toCuda(b2))),
      F::matmul(a3, b2),
      5e-2));

  // a single-element batch still has to take the batched path rather than degenerating.
  Tensor a1 = F::rand({1, 6, 8}, DType::kFloat);
  Tensor b1 = F::rand({1, 8, 5}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(
      toCpu(F::matmul(toCuda(a1), toCuda(b1))),
      F::matmul(a1, b1),
      5e-2));
}

CATCH_TEST_CASE("test CUDA matmul (strided operands)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A non-contiguous left operand cannot be flattened into a plain GEMM, so it goes through the
  // batched path instead; the answer must not depend on which one was chosen.
  Tensor a = F::rand({4, 6, 8}, DType::kFloat);
  Tensor b = F::rand({8, 5}, DType::kFloat);

  Tensor strided = toCuda(a).slice(1, {1, 5});
  CATCH_REQUIRE(!strided.isContiguous());

  CATCH_REQUIRE(F::allClose(
      toCpu(F::matmul(strided, toCuda(b))),
      F::matmul(a.slice(1, {1, 5}), b),
      5e-2));
}

}  // namespace fl
