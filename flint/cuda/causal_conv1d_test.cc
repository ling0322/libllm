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

#include <numeric>
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

Tensor offsetsOf(const std::vector<int> &lengths) {
  std::vector<IntType> values(lengths.size() + 1, 0);
  for (size_t i = 0; i < lengths.size(); ++i) values[i + 1] = values[i] + lengths[i];
  return Tensor::create<IntType>({static_cast<int>(values.size())}, values);
}

/// Runs the packed batch on both backends and reports whether they agree. The CPU result is the
/// reference: it is checked against hand-computed values in cpu/causal_conv1d_test.cc.
bool runCase(const std::vector<int> &lengths, int numChannels, int kernelSize) {
  Tensor cuSeqlens = offsetsOf(lengths);
  int totalTokens = std::accumulate(lengths.begin(), lengths.end(), 0);

  Tensor input = F::rand({totalTokens, numChannels}, DType::kFloat);
  Tensor weight = F::rand({numChannels, kernelSize}, DType::kFloat);
  Tensor expected = F::causalConv1d(input, weight, cuSeqlens);

  Tensor actual = F::causalConv1d(
      toCuda(input),
      toCuda(weight),
      F::to(Device::getCuda(), cuSeqlens));

  if (actual.getShape() != std::vector<int>{totalTokens, numChannels}) return false;
  return F::allClose(toCpu(actual), expected, 5e-3, 5e-3);
}

}  // namespace

CATCH_TEST_CASE("test CUDA causalConv1d", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A kernel of 4 over a few thousand channels is the shape the linear-attention layers use.
  CATCH_REQUIRE(runCase({16}, 8, 4));
  CATCH_REQUIRE(runCase({64}, 2048, 4));
  CATCH_REQUIRE(runCase({300}, 512, 4));
}

CATCH_TEST_CASE("test CUDA causalConv1d (packed batches)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Sequences of different lengths in one batch: the grid is sized by the longest, so the short
  // ones have to stop at their own end rather than running into the next.
  CATCH_REQUIRE(runCase({4, 1, 9}, 16, 4));
  CATCH_REQUIRE(runCase({1, 1, 1}, 8, 4));
  CATCH_REQUIRE(runCase({257, 3, 128}, 64, 4));

  // a single token is the decode-step shape.
  CATCH_REQUIRE(runCase({1}, 128, 4));
}

CATCH_TEST_CASE("test CUDA causalConv1d (kernel sizes)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A kernel of one is a per-channel scale; the widest the operator accepts is eight.
  for (int kernelSize : {1, 2, 3, 4, 8}) {
    CATCH_INFO("kernelSize = " << kernelSize);
    CATCH_REQUIRE(runCase({12, 5}, 32, kernelSize));
  }

  // A kernel longer than the sequence leaves every window hanging off the front.
  CATCH_REQUIRE(runCase({2}, 16, 8));
}

}  // namespace fl
