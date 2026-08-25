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

Tensor toCuda(const Tensor &a) {
  return F::cast(F::to(Device::getCuda(), a), DType::kFloat16);
}

Tensor toCpu(const Tensor &a) {
  return F::to(Device::getCpu(), F::cast(a, DType::kFloat));
}

bool equalLong(Tensor a, Tensor b) {
  a.throwIfInvalidShape(b.getShape(), "equalLong");

  const LongType *pa = a.getInternalData()->getData<LongType>(a.getInternalOffset());
  const LongType *pb = b.getInternalData()->getData<LongType>(b.getInternalOffset());
  return std::equal(pa, pa + a.getNumEl(), pb);
}

}  // namespace

CATCH_TEST_CASE("test CUDA copy", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  auto runCase = [](std::initializer_list<int> shape, bool transpose) {
    Tensor a = F::rand(shape, DType::kFloat);

    Tensor x = toCuda(a);
    if (transpose) x = x.transpose(1, 0);
    Tensor dest = F::tensorLike(x);
    F::copy(x, dest);

    dest = toCpu(dest);
    if (transpose) dest = dest.transpose(1, 0);
    return F::allClose(a, dest);
  };

  CATCH_REQUIRE(runCase({10, 50}, true));
  CATCH_REQUIRE(runCase({2, 10, 50}, false));
  CATCH_REQUIRE(runCase({2, 10, 50}, true));
  CATCH_REQUIRE(runCase({2, 3, 10, 50}, true));
}

CATCH_TEST_CASE("test CUDA copy (long)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<LongType>({2, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 0});

  Tensor x = F::to(Device::getCuda(), a);
  Tensor dest = F::tensorLike(x);
  F::copy(x, dest);
  dest = F::to(Device::getCpu(), dest);

  CATCH_REQUIRE(equalLong(dest, a));
}

CATCH_TEST_CASE("test CUDA copy (expanded 5D)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({10, 2, 5, 20}, DType::kFloat);
  Tensor xr = F::contiguous(a.unsqueeze(1).expand({10, 4, 2, 5, 20}));

  Tensor x = toCuda(a).unsqueeze(1).expand({10, 4, 2, 5, 20});
  Tensor dest = F::tensorLike(x);
  F::copy(x, dest);

  CATCH_REQUIRE(F::allClose(toCpu(dest), xr));
}

CATCH_TEST_CASE("test CUDA copy (into a strided destination)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A destination that is a window into a bigger buffer is what F::cat produces internally: the
  // kernel has to honour the destination strides and leave everything outside the window alone.
  constexpr int Rows = 4;
  constexpr int Cols = 10;
  Tensor dest = F::zeros({Rows, Cols}, DType::kFloat16, Device::getCuda());
  Tensor window = dest.slice(1, {2, 6});
  CATCH_REQUIRE(!window.isContiguous());

  Tensor src = F::rand({Rows, 4}, DType::kFloat);
  F::copy(toCuda(src), window);

  Tensor host = toCpu(dest);
  CATCH_REQUIRE(F::allClose(F::contiguous(host.slice(1, {2, 6})), src, 5e-3));
  // the columns either side of the window are still zero.
  CATCH_REQUIRE(F::allClose(
      F::contiguous(host.slice(1, {0, 2})),
      F::zeros({Rows, 2}, DType::kFloat)));
  CATCH_REQUIRE(F::allClose(
      F::contiguous(host.slice(1, {6, 10})),
      F::zeros({Rows, 4}, DType::kFloat)));
}

CATCH_TEST_CASE("test CUDA copy (single element and single row)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Degenerate shapes: one element, and a shape whose leading dims are all 1. Both keep the
  // grid at a single partially-filled block.
  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {1, 1},
           {1, 5},
           {5, 1},
           {1, 1, 1},
           {1, 1, 1, 1}}) {
    Tensor a = F::rand(shape, DType::kFloat);
    Tensor x = toCuda(a);
    Tensor dest = F::tensorLike(x);
    F::copy(x, dest);

    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(F::allClose(toCpu(dest), a, 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA copy (crosses the grid-stride loop)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Larger than the grid the launcher caps at, so every thread runs the loop body more than once
  // and a kernel that assumed one element per thread would drop the tail.
  Tensor a = F::rand({512, 1024}, DType::kFloat);
  Tensor x = toCuda(a).transpose(1, 0);
  Tensor dest = F::tensorLike(x);
  F::copy(x, dest);

  CATCH_REQUIRE(F::allClose(toCpu(dest).transpose(1, 0), a, 5e-3));
}

CATCH_TEST_CASE("test CUDA cat", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 10, 16}, DType::kFloat);
  Tensor b = F::rand({2, 2, 16}, DType::kFloat);

  Tensor x = F::cat(toCuda(a), toCuda(b), 1);

  CATCH_REQUIRE(F::allClose(toCpu(x), F::cat(a, b, 1), 5e-3));
}

CATCH_TEST_CASE("test CUDA cat (every axis)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Concatenating on the first axis keeps both halves contiguous; on the last axis neither half
  // is, which is the case that needs the strided copy kernel.
  Tensor a = F::rand({3, 4, 6}, DType::kFloat);
  Tensor b = F::rand({3, 4, 6}, DType::kFloat);

  for (int dim : {0, 1, 2}) {
    Tensor x = F::cat(toCuda(a), toCuda(b), dim);
    CATCH_INFO("dim = " << dim);
    CATCH_REQUIRE(F::allClose(toCpu(x), F::cat(a, b, dim), 5e-3));
  }

  // negative axes address the same dimensions from the back.
  CATCH_REQUIRE(F::allClose(toCpu(F::cat(toCuda(a), toCuda(b), -1)), F::cat(a, b, 2), 5e-3));
}

CATCH_TEST_CASE("test CUDA cat (uneven and single-element parts)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A one-token append is the shape the decode loop produces every step.
  Tensor a = F::rand({2, 7, 16}, DType::kFloat);
  Tensor b = F::rand({2, 1, 16}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::cat(toCuda(a), toCuda(b), 1)), F::cat(a, b, 1), 5e-3));

  // 2D, and a left operand that is itself only one row.
  Tensor c = F::rand({1, 8}, DType::kFloat);
  Tensor d = F::rand({3, 8}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::cat(toCuda(c), toCuda(d), 0)), F::cat(c, d, 0), 5e-3));
}

}  // namespace fl
