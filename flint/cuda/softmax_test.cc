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
#include <limits>
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

CATCH_TEST_CASE("test CUDA softmax", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  for (int lastDim : {150, 151}) {
    Tensor a = F::rand({2, 5, lastDim}, DType::kFloat);
    CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toCuda(a))), F::softmax(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA softmax (strided)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 3, 5}, DType::kFloat);
  Tensor x = F::softmax(toCuda(a).transpose(1, 2));
  CATCH_REQUIRE(F::allClose(toCpu(x), F::softmax(a.transpose(1, 2)), 5e-3));
}

CATCH_TEST_CASE("test CUDA softmax (all ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Contiguous inputs of every rank go down the fused kernel, which flattens everything but the
  // last axis into rows.
  for (std::vector<int> shape : std::vector<std::vector<int>>{
           {8},
           {9},
           {3, 8},
           {2, 3, 8},
           {2, 3, 4, 8}}) {
    Tensor a = F::rand(shape, DType::kFloat);
    CATCH_INFO("shape rank = " << shape.size());
    CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toCuda(a))), F::softmax(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA softmax (strided ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Each rank has its own entry point that reshapes into the strided 3D kernel; rank 1 and 2 in
  // particular are only reachable when the input is not contiguous.
  Tensor a2 = F::rand({4, 6}, DType::kFloat);
  Tensor a3 = F::rand({2, 3, 5}, DType::kFloat);
  Tensor a4 = F::rand({2, 3, 4, 5}, DType::kFloat);

  // rank 1: one row of a transposed matrix.
  Tensor h1 = a2.transpose(0, 1).subtensor(0);
  Tensor d1 = toCuda(a2).transpose(0, 1).subtensor(0);
  CATCH_REQUIRE(!d1.isContiguous());
  CATCH_REQUIRE(F::allClose(toCpu(F::softmax(d1)), F::softmax(F::contiguous(h1)), 5e-3));

  // rank 2.
  Tensor h2 = a2.transpose(0, 1);
  Tensor d2 = toCuda(a2).transpose(0, 1);
  CATCH_REQUIRE(F::allClose(toCpu(F::softmax(d2)), F::softmax(h2), 5e-3));

  // rank 3.
  CATCH_REQUIRE(F::allClose(
      toCpu(F::softmax(toCuda(a3).transpose(0, 2))),
      F::softmax(a3.transpose(0, 2)),
      5e-3));

  // rank 4 swaps the last two axes, the way attention scores are permuted. The rank-4 entry
  // point flattens the two leading axes with a view, so a permutation that separated those two
  // in memory would not be expressible and the operator rejects it instead of guessing.
  CATCH_REQUIRE(F::allClose(
      toCpu(F::softmax(toCuda(a4).transpose(2, 3))),
      F::softmax(F::contiguous(a4.transpose(2, 3))),
      5e-3));
}

CATCH_TEST_CASE("test CUDA softmax (row widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A row is reduced by a whole 256-thread block, so widths below, at, and well above the block
  // size take different numbers of loop iterations. Odd widths also disable the half2 path.
  for (int width : {1, 2, 3, 255, 256, 257, 511, 512, 1024, 2048, 4096}) {
    Tensor a = F::rand({2, width}, DType::kFloat);
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toCuda(a))), F::softmax(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA softmax (degenerate rows)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single-element row normalises to exactly 1 no matter what the logit was.
  Tensor single = Tensor::create<float>({3, 1}, {-100.0f, 0.0f, 100.0f});
  CATCH_REQUIRE(F::allClose(
      toCpu(F::softmax(toCuda(single))),
      Tensor::create<float>({3, 1}, {1.0f, 1.0f, 1.0f}),
      5e-3));

  // Equal logits give a uniform distribution, which is where a missing max-subtraction still
  // looks right but a broken sum does not.
  Tensor flat = F::zeros({2, 8}, DType::kFloat);
  Tensor uniform = F::tensor({2, 8}, DType::kFloat);
  F::fill(uniform, 0.125f);
  CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toCuda(flat))), uniform, 5e-3));
}

CATCH_TEST_CASE("test CUDA softmax (extreme values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = Tensor::create<float>(
      {1, 1, 4},
      {-999.0f, -998.0f, -997.0f, -std::numeric_limits<float>::infinity()});

  CATCH_REQUIRE(F::allClose(toCpu(F::softmax(toCuda(a))), F::softmax(a)));
}

CATCH_TEST_CASE("test CUDA softmax (large logits do not overflow)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Without subtracting the row max, exp() of these overflows to infinity and the row comes back
  // as NaN. The result is the same distribution as the equivalent small logits.
  Tensor big = Tensor::create<float>({1, 4}, {1000.0f, 1001.0f, 1002.0f, 1003.0f});
  Tensor small = Tensor::create<float>({1, 4}, {0.0f, 1.0f, 2.0f, 3.0f});

  Tensor x = toCpu(F::softmax(toCuda(big)));
  CATCH_REQUIRE(F::allClose(x, F::softmax(small), 5e-3));

  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  for (int i = 0; i < 4; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(!std::isnan(data[i]));
  }
}

}  // namespace fl
