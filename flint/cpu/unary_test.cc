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
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

/// The values each function is checked at: zero, both signs, and magnitudes far enough out that a
/// saturating function has flattened.
const std::vector<float> &probes() {
  static const std::vector<float> values = {
      0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.5f, -2.5f, 8.0f, -8.0f, 0.125f};
  return values;
}

Tensor probeTensor() {
  return Tensor::create<float>({static_cast<int>(probes().size())}, probes());
}

void checkAgainst(Tensor actual, float (*reference)(float), const char *name) {
  const float *data = actual.getInternalData()->getData<float>(actual.getInternalOffset());
  for (size_t i = 0; i < probes().size(); ++i) {
    CATCH_INFO(name << " at x = " << probes()[i]);
    CATCH_REQUIRE(std::fabs(data[i] - reference(probes()[i])) < 1e-5f);
  }
}

}  // namespace

CATCH_TEST_CASE("test CPU unary operators", "[core][nn][operators]") {
  Tensor x = probeTensor();

  checkAgainst(F::neg(x), [](float v) { return -v; }, "neg");
  checkAgainst(F::abs(x), [](float v) { return std::fabs(v); }, "abs");
  checkAgainst(F::exp(x), [](float v) { return std::exp(v); }, "exp");
  checkAgainst(F::square(x), [](float v) { return v * v; }, "square");
  checkAgainst(F::tanh(x), [](float v) { return std::tanh(v); }, "tanh");
  checkAgainst(F::relu(x), [](float v) { return v > 0.0f ? v : 0.0f; }, "relu");
  checkAgainst(
      F::sigmoid(x),
      [](float v) { return 1.0f / (1.0f + std::exp(-v)); },
      "sigmoid");
  checkAgainst(
      F::silu(x),
      [](float v) { return v / (1.0f + std::exp(-v)); },
      "silu");
  checkAgainst(
      F::gelu(x),
      [](float v) { return v * 0.5f * (1.0f + std::erf(v * 0.70710678118654752f)); },
      "gelu");
}

CATCH_TEST_CASE("test CPU unary operators (positive domain)", "[core][nn][operators]") {
  // sqrt and rsqrt are only defined for non-negative and positive inputs, so they get their own
  // probes rather than the shared set.
  std::vector<float> values = {0.25f, 1.0f, 2.0f, 9.0f, 1e-4f};
  Tensor x = Tensor::create<float>({static_cast<int>(values.size())}, values);

  Tensor rootTensor = F::sqrt(x);
  Tensor invRootTensor = F::rsqrt(x);
  const float *root = rootTensor.getInternalData()->getData<float>(rootTensor.getInternalOffset());
  const float *invRoot =
      invRootTensor.getInternalData()->getData<float>(invRootTensor.getInternalOffset());

  for (size_t i = 0; i < values.size(); ++i) {
    CATCH_INFO("x = " << values[i]);
    CATCH_REQUIRE(std::fabs(root[i] - std::sqrt(values[i])) < 1e-5f);
    CATCH_REQUIRE(std::fabs(invRoot[i] - 1.0f / std::sqrt(values[i])) < 1e-3f);
  }

  // sqrt(0) is 0 rather than NaN. Read the value directly: F::elem has no CPU implementation.
  Tensor zero = F::sqrt(Tensor::create<float>({1}, {0.0f}));
  CATCH_REQUIRE(zero.getInternalData()->getData<float>(zero.getInternalOffset())[0] == 0.0f);
}

CATCH_TEST_CASE("test CPU unary operators (shapes)", "[core][nn][operators]") {
  // The kernel walks the tensor a row at a time, so a shape has to survive unchanged and every
  // element has to be visited, including in a strided view.
  Tensor a = F::rand({2, 3, 4}, DType::kFloat);
  Tensor negated = F::neg(a);
  CATCH_REQUIRE(negated.getShape() == std::vector<int>{2, 3, 4});
  CATCH_REQUIRE(F::allClose(F::add(a, negated), F::zeros({2, 3, 4}, DType::kFloat)));

  Tensor strided = a.transpose(0, 2);
  Tensor stridedNeg = F::neg(strided);
  CATCH_REQUIRE(stridedNeg.getShape() == std::vector<int>{4, 3, 2});
  CATCH_REQUIRE(F::allClose(F::add(strided, stridedNeg), F::zeros({4, 3, 2}, DType::kFloat)));

  // applying a function twice is not the same as applying it once, so a no-op kernel fails here.
  CATCH_REQUIRE(F::allClose(F::neg(negated), a));
}

CATCH_TEST_CASE("test CPU div and min", "[core][nn][operators]") {
  Tensor a = Tensor::create<float>({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  Tensor b = Tensor::create<float>({2, 3}, {2.0f, 4.0f, 4.0f, 8.0f, 10.0f, 3.0f});

  CATCH_REQUIRE(F::allClose(
      F::div(a, b),
      Tensor::create<float>({2, 3}, {0.5f, 0.5f, 0.75f, 0.5f, 0.5f, 2.0f})));

  // the divisor is broadcast over the leading dimensions the way mul does it.
  Tensor row = Tensor::create<float>({3}, {1.0f, 2.0f, 4.0f});
  CATCH_REQUIRE(F::allClose(
      F::div(a, row),
      Tensor::create<float>({2, 3}, {1.0f, 1.0f, 0.75f, 4.0f, 2.5f, 1.5f})));

  // min drops the last dimension the way max does, and must not report the initial +inf.
  Tensor c = Tensor::create<float>({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});
  CATCH_REQUIRE(F::allClose(F::min(c), Tensor::create<float>({2}, {1.0f, -4.0f})));
  CATCH_REQUIRE(F::allClose(F::max(c), Tensor::create<float>({2}, {4.0f, -1.0f})));

  // a single-element row is both the min and the max.
  Tensor one = Tensor::create<float>({2, 1}, {7.0f, -7.0f});
  CATCH_REQUIRE(F::allClose(F::min(one), Tensor::create<float>({2}, {7.0f, -7.0f})));
}

CATCH_TEST_CASE("test CPU arange", "[core][nn][operators]") {
  Tensor x = F::arange(0, 10, 2);
  CATCH_REQUIRE(x.getShape() == std::vector<int>{5});
  CATCH_REQUIRE(x.getDType() == DType::kLong);

  const LongType *data = x.getInternalData()->getData<LongType>(x.getInternalOffset());
  for (int i = 0; i < 5; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(data[i] == 2 * i);
  }

  // a step that does not divide the span evenly stops short rather than overshooting, and a
  // negative step counts down.
  CATCH_REQUIRE(F::arange(0, 10, 3).getShape() == std::vector<int>{3});
  Tensor down = F::arange(10, 0, -2);
  CATCH_REQUIRE(down.getShape() == std::vector<int>{5});
  CATCH_REQUIRE(
      down.getInternalData()->getData<LongType>(down.getInternalOffset())[0] == 10);
}

CATCH_TEST_CASE("test CPU randn", "[core][nn][operators]") {
  // An odd element count is the case the pair-at-a-time Gaussian fill has to pad for.
  for (int count : {1, 2, 3, 4096}) {
    Tensor x = F::randn({count});
    CATCH_INFO("count = " << count);
    CATCH_REQUIRE(x.getShape() == std::vector<int>{count});
    CATCH_REQUIRE(x.getDType() == DType::kFloat);
  }

  Tensor x = F::randn({8192});
  const float *data = x.getInternalData()->getData<float>(x.getInternalOffset());
  double sum = 0.0;
  double sumSquare = 0.0;
  for (int i = 0; i < x.getNumEl(); ++i) {
    CATCH_REQUIRE(!std::isnan(data[i]));
    sum += data[i];
    sumSquare += static_cast<double>(data[i]) * data[i];
  }

  double mean = sum / x.getNumEl();
  double stddev = std::sqrt(sumSquare / x.getNumEl() - mean * mean);
  CATCH_REQUIRE(std::fabs(mean) < 0.1);
  CATCH_REQUIRE(std::fabs(stddev - 1.0) < 0.1);
}

CATCH_TEST_CASE("test CPU elem and scalar div", "[core][nn][operators]") {
  CATCH_REQUIRE(F::elem(Tensor::create<float>({1}, {1.5f})) == 1.5f);
  CATCH_REQUIRE(F::elem(Tensor::create<float>({1}, {0.0f})) == 0.0f);
  CATCH_REQUIRE(F::elem(Tensor::create<float>({1}, {-2.5f})) == -2.5f);

  Tensor a = Tensor::create<float>({2, 2}, {1.0f, 2.0f, 4.0f, 8.0f});
  CATCH_REQUIRE(F::allClose(
      F::div(a, 2.0f),
      Tensor::create<float>({2, 2}, {0.5f, 1.0f, 2.0f, 4.0f})));
  // dividing by one leaves the tensor alone.
  CATCH_REQUIRE(F::allClose(F::div(a, 1.0f), a));
}

CATCH_TEST_CASE("test CPU mod", "[core][nn][operators]") {
  Tensor ids = Tensor::create<LongType>({2, 4}, {0, 3, 4, 5, 7, 8, 99, 100});
  Tensor x = F::mod(ids, 4);

  const LongType *data = x.getInternalData()->getData<LongType>(x.getInternalOffset());
  const LongType expected[] = {0, 3, 0, 1, 3, 0, 3, 0};
  for (int i = 0; i < 8; ++i) {
    CATCH_INFO("i = " << i);
    CATCH_REQUIRE(data[i] == expected[i]);
  }
}

CATCH_TEST_CASE("test CPU eq and all", "[core][nn][operators]") {
  // Tensor::create is not instantiated for UInt8, so build these through the allocator and
  // write the bytes in.
  auto makeUInt8 = [](std::vector<uint8_t> values) {
    Tensor x = F::tensor({static_cast<int>(values.size())}, DType::kUInt8);
    UInt8 *data = x.getInternalData()->getData<UInt8>(x.getInternalOffset());
    for (size_t i = 0; i < values.size(); ++i) data[i].v = values[i];
    return x;
  };

  Tensor a = makeUInt8({1, 2, 3, 4});
  Tensor same = makeUInt8({1, 2, 3, 4});
  Tensor different = makeUInt8({1, 2, 9, 4});

  CATCH_REQUIRE(F::all(F::eq(a, same)));
  CATCH_REQUIRE(!F::all(F::eq(a, different)));

  // eq answers per element, so the one mismatch has to be the only false.
  Tensor mask = F::eq(a, different);
  CATCH_REQUIRE(mask.getDType() == DType::kBool);
  const BoolType *data = mask.getInternalData()->getData<BoolType>(mask.getInternalOffset());
  CATCH_REQUIRE(data[0]);
  CATCH_REQUIRE(data[1]);
  CATCH_REQUIRE(!data[2]);
  CATCH_REQUIRE(data[3]);
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
