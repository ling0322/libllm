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
#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

CATCH_TEST_CASE("test CPU batched sampling parameters", "[fl][op][cpu][sampling]") {
  Tensor logits = Tensor::create<float>(
      {3, 4},
      {std::numeric_limits<float>::quiet_NaN(), 4.0f, 2.0f, 1.0f,
       0.0f, 1.0f, 5.0f, 2.0f,
       0.0f, 1.0f, 2.0f, 6.0f});
  Tensor temperatures = Tensor::create<float>({3}, {0.0f, 1.0f, 1.0f});
  Tensor topKs = Tensor::create<IntType>({3}, {0, 1, 0});
  Tensor topPs = Tensor::create<float>({3}, {1.0f, 1.0f, 0.1f});

  Tensor sampled = F::sample(logits, temperatures, topKs, topPs);
  CATCH_REQUIRE(sampled.getShape() == std::vector<int>{3});
  const LongType *data = sampled.getInternalData()->getData<LongType>(
      sampled.getInternalOffset());
  CATCH_REQUIRE(data[0] == 1);
  CATCH_REQUIRE(data[1] == 2);
  CATCH_REQUIRE(data[2] == 3);
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
