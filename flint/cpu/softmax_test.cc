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

#include <limits>

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

CATCH_TEST_CASE("test softmax", "[core][nn][operators]") {
  Tensor input = Tensor::create<float>({3}, {0.1f, 0.2f, 0.3f});
  Tensor output = Tensor::create<float>({3}, {0.3006f, 0.3322f, 0.3672f});
  CATCH_REQUIRE(F::allClose(F::softmax(input), output));

  constexpr float inf = std::numeric_limits<float>::infinity();
  input = Tensor::create<float>({3}, {0.1f, 0.2f, -inf});
  output = Tensor::create<float>({3}, {0.4750f, 0.5250f, 0.0f});
  CATCH_REQUIRE(F::allClose(F::softmax(input), output));
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
