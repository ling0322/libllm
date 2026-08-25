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

CATCH_TEST_CASE("test CUDA repetitionPenalty", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 16}, DType::kFloat);
  Tensor history = Tensor::create<LongType>({2, 4}, {1, 0, 1, 3, 0, 0, 0, 1});

  Tensor x = toCuda(a);
  F::repetitionPenalty(x, F::to(Device::getCuda(), history), 1.5);
  F::repetitionPenalty(a, history, 1.5);

  CATCH_REQUIRE(F::allClose(toCpu(x), a, 1e-3));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (packed 1D logits)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A single sequence arrives as 1D logits with a 1D history; the operator wraps both in a
  // leading axis, and the penalty has to land on the same positions as the 2D form.
  Tensor a = F::rand({16}, DType::kFloat);
  Tensor history = Tensor::create<LongType>({3}, {2, 5, 11});

  Tensor x = toCuda(a);
  F::repetitionPenalty(x, F::to(Device::getCuda(), history), 1.5);
  F::repetitionPenalty(a, history, 1.5);

  CATCH_REQUIRE(F::allClose(toCpu(x), a, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (sign and known values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // A positive logit is divided by the weight and a negative one multiplied, so the penalty
  // always moves a score towards -inf. Zero and untouched positions stay exactly as they were.
  Tensor a = Tensor::create<float>({1, 5}, {2.0f, -2.0f, 0.0f, 4.0f, -4.0f});
  Tensor history = Tensor::create<LongType>({1, 3}, {0, 1, 2});

  Tensor x = toCuda(a);
  F::repetitionPenalty(x, F::to(Device::getCuda(), history), 2.0f);

  Tensor host = toCpu(x);
  const float *data = host.getInternalData()->getData<float>(host.getInternalOffset());
  CATCH_REQUIRE(std::fabs(data[0] - 1.0f) < 1e-2f);   // 2 / 2
  CATCH_REQUIRE(std::fabs(data[1] + 4.0f) < 1e-2f);   // -2 * 2
  CATCH_REQUIRE(data[2] == 0.0f);                     // zero is left alone
  CATCH_REQUIRE(std::fabs(data[3] - 4.0f) < 1e-2f);   // not in the history
  CATCH_REQUIRE(std::fabs(data[4] + 4.0f) < 1e-2f);   // not in the history
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (weight of one is a no-op)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 16}, DType::kFloat);
  Tensor history = Tensor::create<LongType>({2, 3}, {1, 2, 3, 4, 5, 6});

  Tensor x = toCuda(a);
  Tensor before = toCpu(x);
  F::repetitionPenalty(x, F::to(Device::getCuda(), history), 1.0f);

  CATCH_REQUIRE(F::allClose(toCpu(x), before, 5e-3, 5e-3));
}

CATCH_TEST_CASE("test CUDA repetitionPenalty (history lengths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The kernel always launches a fixed 64 threads and returns early past the end of the
  // history, so a short history must not penalise positions it does not name. 63 is the longest
  // history the operator accepts.
  for (int length : {1, 2, 63}) {
    Tensor a = F::rand({2, 64}, DType::kFloat);
    std::vector<LongType> ids(2 * length);
    for (int i = 0; i < 2 * length; ++i) ids[i] = i % 64;
    Tensor history = Tensor::create<LongType>({2, length}, ids);

    Tensor x = toCuda(a);
    F::repetitionPenalty(x, F::to(Device::getCuda(), history), 1.5);
    F::repetitionPenalty(a, history, 1.5);

    // The CUDA side is half and the reference is float, so the comparison has to leave room for
    // half's own round-off. A penalty that failed to apply would be off by the weight itself,
    // which is far outside this.
    CATCH_INFO("history length = " << length);
    CATCH_REQUIRE(F::allClose(toCpu(x), a, 5e-3, 5e-3));
  }
}

}  // namespace fl
