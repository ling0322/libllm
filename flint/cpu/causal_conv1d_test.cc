// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

Tensor offsets(std::vector<IntType> values) {
  return Tensor::create<IntType>({static_cast<int>(values.size())}, values);
}

}  // namespace

CATCH_TEST_CASE("test CPU causalConv1d", "[core][nn][operators]") {
  // One channel, kernel 3, a single sequence: the first two outputs are the ones whose window
  // hangs off the front and has to be zero-padded.
  Tensor input = Tensor::create<float>({4, 1}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor weight = Tensor::create<float>({1, 3}, {0.5f, 2.0f, 10.0f});

  Tensor out = F::causalConv1d(input, weight, offsets({0, 4}));
  CATCH_REQUIRE(out.getShape() == std::vector<int>{4, 1});

  // out[t] = 0.5*x[t-2] + 2*x[t-1] + 10*x[t], with x[<0] taken as zero.
  CATCH_REQUIRE(F::allClose(
      out,
      Tensor::create<float>({4, 1}, {10.0f, 22.0f, 34.5f, 47.0f})));
}

CATCH_TEST_CASE("test CPU causalConv1d (per channel)", "[core][nn][operators]") {
  // Each channel has its own filter and must not read another channel's values.
  Tensor input = Tensor::create<float>({3, 2}, {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f});
  Tensor weight = Tensor::create<float>({2, 2}, {1.0f, 1.0f, 0.0f, 1.0f});

  Tensor out = F::causalConv1d(input, weight, offsets({0, 3}));
  // channel 0 sums the current and previous position; channel 1 keeps only the current one.
  CATCH_REQUIRE(F::allClose(
      out,
      Tensor::create<float>({3, 2}, {1.0f, 10.0f, 3.0f, 20.0f, 5.0f, 30.0f})));
}

CATCH_TEST_CASE("test CPU causalConv1d (sequence boundaries)", "[core][nn][operators]") {
  // Two sequences packed back to back. The second one starts its window fresh: if the kernel
  // read across the boundary, its first output would pick up the tail of the first sequence.
  Tensor input = Tensor::create<float>({5, 1}, {1.0f, 2.0f, 100.0f, 200.0f, 300.0f});
  Tensor weight = Tensor::create<float>({1, 2}, {1.0f, 1.0f});

  Tensor out = F::causalConv1d(input, weight, offsets({0, 2, 5}));
  CATCH_REQUIRE(F::allClose(
      out,
      Tensor::create<float>({5, 1}, {1.0f, 3.0f, 100.0f, 300.0f, 500.0f})));

  // A one-token sequence has no history at all, so it is just its own value scaled.
  Tensor single = Tensor::create<float>({2, 1}, {7.0f, 9.0f});
  Tensor scale = Tensor::create<float>({1, 2}, {1000.0f, 2.0f});
  CATCH_REQUIRE(F::allClose(
      F::causalConv1d(single, scale, offsets({0, 1, 2})),
      Tensor::create<float>({2, 1}, {14.0f, 18.0f})));
}

CATCH_TEST_CASE("test CPU causalConv1d (kernel of one)", "[core][nn][operators]") {
  // A kernel of one is a per-channel scale, which is the degenerate case of the window.
  Tensor input = Tensor::create<float>({3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor weight = Tensor::create<float>({2, 1}, {2.0f, 10.0f});

  CATCH_REQUIRE(F::allClose(
      F::causalConv1d(input, weight, offsets({0, 3})),
      Tensor::create<float>({3, 2}, {2.0f, 20.0f, 6.0f, 40.0f, 10.0f, 60.0f})));
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
