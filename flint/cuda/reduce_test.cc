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

CATCH_TEST_CASE("test CUDA reductions", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  Tensor a = F::rand({2, 5, 150}, DType::kFloat);
  CATCH_REQUIRE(F::allClose(toCpu(F::sum(toCuda(a))), F::sum(a), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::max(toCuda(a))), F::max(a), 5e-3));
}

CATCH_TEST_CASE("test CUDA reductions (all ranks)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Ranks 1 and 2 are reshaped to 3D on the way in and back again on the way out, so the shape
  // that comes out has to match what the CPU reduction produces for the same input.
  Tensor a1 = F::rand({8}, DType::kFloat);
  Tensor s1 = toCpu(F::sum(toCuda(a1)));
  CATCH_REQUIRE(s1.getShape() == F::sum(a1).getShape());
  CATCH_REQUIRE(F::allClose(s1, F::sum(a1), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::max(toCuda(a1))), F::max(a1), 5e-3));

  Tensor a2 = F::rand({3, 8}, DType::kFloat);
  Tensor s2 = toCpu(F::sum(toCuda(a2)));
  CATCH_REQUIRE(s2.getShape() == F::sum(a2).getShape());
  CATCH_REQUIRE(F::allClose(s2, F::sum(a2), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::max(toCuda(a2))), F::max(a2), 5e-3));

  Tensor a3 = F::rand({2, 3, 8}, DType::kFloat);
  Tensor s3 = toCpu(F::sum(toCuda(a3)));
  CATCH_REQUIRE(s3.getShape() == F::sum(a3).getShape());
  CATCH_REQUIRE(F::allClose(s3, F::sum(a3), 5e-3));
}

CATCH_TEST_CASE("test CUDA reductions (row widths)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // One 256-thread block reduces a whole row, so a width below the block size leaves most
  // threads with nothing to contribute and a width above it makes every thread loop. Both have
  // to reach the same answer as the sequential CPU reduction.
  for (int width : {1, 2, 3, 255, 256, 257, 1000, 4096}) {
    Tensor a = F::rand({2, 3, width}, DType::kFloat);
    CATCH_INFO("width = " << width);
    CATCH_REQUIRE(F::allClose(toCpu(F::sum(toCuda(a))), F::sum(a), 1e-2));
    CATCH_REQUIRE(F::allClose(toCpu(F::max(toCuda(a))), F::max(a), 5e-3));
  }
}

CATCH_TEST_CASE("test CUDA reductions (strided rows)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // The reduction reads through an accessor, so a row whose elements are not adjacent has to
  // work as well as a packed one.
  Tensor a = F::rand({2, 3, 5}, DType::kFloat);
  Tensor strided = toCuda(a).transpose(0, 2);
  CATCH_REQUIRE(!strided.isContiguous());

  CATCH_REQUIRE(F::allClose(toCpu(F::sum(strided)), F::sum(a.transpose(0, 2)), 5e-3));
  CATCH_REQUIRE(F::allClose(toCpu(F::max(strided)), F::max(a.transpose(0, 2)), 5e-3));
}

CATCH_TEST_CASE("test CUDA reductions (known values)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // Fixed inputs, so a reduction that silently dropped or double-counted an element shows up as
  // a wrong total rather than as noise inside a tolerance.
  Tensor a = Tensor::create<float>({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});
  Tensor x = toCuda(a);

  CATCH_REQUIRE(F::allClose(
      toCpu(F::sum(x)),
      Tensor::create<float>({2}, {10.0f, -10.0f}),
      5e-3));
  // max over an all-negative row must not fall back to the zero initial value.
  CATCH_REQUIRE(F::allClose(
      toCpu(F::max(x)),
      Tensor::create<float>({2}, {4.0f, -1.0f}),
      5e-3));

  // summing zeros stays zero rather than accumulating the initial value once per thread.
  Tensor zeros = F::zeros({2, 300}, DType::kFloat16, Device::getCuda());
  CATCH_REQUIRE(F::allClose(toCpu(F::sum(zeros)), F::zeros({2}, DType::kFloat)));
}

}  // namespace fl
