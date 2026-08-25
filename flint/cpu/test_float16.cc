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

#include "catch2/catch_amalgamated.hpp"
#include "flint/functional.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

// the CPU fp16 kernels are checked against the fp32 kernels on the same device.
Tensor toFp16(const Tensor &a) {
  return F::cast(a, DType::kFloat16);
}

Tensor toFp32(const Tensor &a) {
  return F::cast(a, DType::kFloat);
}

}  // namespace

CATCH_TEST_CASE("test CPU fp16 binary operators", "[op][cpu][float16]") {
  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor b = F::rand({5}, DType::kFloat);
  Tensor at = a.transpose(2, 1).slice(1, {1, 9});
  Tensor xt = toFp16(a).transpose(2, 1).slice(1, {1, 9});
  Tensor y = toFp16(b);

  CATCH_REQUIRE(F::allClose(toFp32(F::add(xt, y)), F::add(at, b), 5e-3));
  CATCH_REQUIRE(F::allClose(toFp32(F::mul(xt, y)), F::mul(at, b), 5e-3));
}

CATCH_TEST_CASE("test CPU fp16 copy operators", "[op][cpu][float16]") {
  Tensor a = F::rand({2, 10, 50}, DType::kFloat);
  Tensor x = toFp16(a).transpose(1, 0);
  Tensor dest = F::tensorLike(x);
  F::copy(x, dest);
  CATCH_REQUIRE(F::allClose(toFp32(dest).transpose(1, 0), a));

  Tensor b = F::rand({10, 2, 5, 20}, DType::kFloat);
  Tensor expanded = toFp16(b).unsqueeze(1).expand({10, 4, 2, 5, 20});
  Tensor dest5d = F::tensorLike(expanded);
  F::copy(expanded, dest5d);
  CATCH_REQUIRE(
      F::allClose(toFp32(dest5d), F::contiguous(b.unsqueeze(1).expand({10, 4, 2, 5, 20}))));
}

CATCH_TEST_CASE("test CPU fp16 matmul operators", "[op][cpu][float16]") {
  auto runCase = [](std::initializer_list<int> shapeA, std::initializer_list<int> shapeB) {
    Tensor a = F::rand(shapeA, DType::kFloat);
    Tensor b = F::rand(shapeB, DType::kFloat);
    Tensor xr = F::matmul(a, b.slice(-1, {8, 32}).transpose(-1, -2));

    Tensor y = toFp16(b).slice(-1, {8, 32}).transpose(-1, -2);
    Tensor x = F::matmul(toFp16(a), y);

    return F::allClose(toFp32(x), xr, 5e-2);
  };

  CATCH_REQUIRE(runCase({10, 20}, {40, 30}));
  CATCH_REQUIRE(runCase({5, 10, 20}, {40, 30}));
  CATCH_REQUIRE(runCase({5, 10, 5, 20}, {10, 40, 30}));
}

CATCH_TEST_CASE("test CPU fp16 rmsNorm operator", "[op][cpu][float16]") {
  Tensor a = F::rand({2, 5, 10}, DType::kFloat);
  Tensor w = F::rand({10}, DType::kFloat);
  Tensor x = F::rmsNorm(toFp16(a), toFp16(w), 1e-5);

  CATCH_REQUIRE(F::allClose(toFp32(x), F::rmsNorm(a, w, 1e-5), 5e-2));
}

CATCH_TEST_CASE("test CPU fp16 activation operators", "[op][cpu][float16]") {
  Tensor a = F::rand({2, 5, 150}, DType::kFloat);

  CATCH_REQUIRE(F::allClose(toFp32(F::softmax(toFp16(a))), F::softmax(a), 5e-2));
  CATCH_REQUIRE(F::allClose(toFp32(F::swiglu(toFp16(a))), F::swiglu(a), 5e-2));
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
