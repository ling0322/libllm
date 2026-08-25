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

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/functional.h"
#include "flint/operators.h"

namespace fl {
namespace {

Tensor toCpuFloat(Tensor tensor) {
  return F::to(Device::getCpu(), F::cast(tensor, DType::kFloat));
}

Tensor baselineRotaryEmbedding(Tensor input, Tensor roPE) {
  Tensor cos = roPE.subtensor(0);
  Tensor sin = roPE.subtensor(1);
  cos = cos.expand({cos.getShape(0), input.getShape(1), cos.getShape(2)});
  sin = sin.expand({sin.getShape(0), input.getShape(1), sin.getShape(2)});

  int halfDim = input.getShape(-1) / 2;
  Tensor rotated = F::tensorLike(input);
  Tensor first = input.slice(-1, {0, halfDim});
  Tensor second = F::mul(input.slice(-1, {halfDim, None}), -1.0f);
  F::copy(first, rotated.slice(-1, {halfDim, None}));
  F::copy(second, rotated.slice(-1, {0, halfDim}));

  return F::add(
      F::mul(input, F::contiguous(cos)),
      F::mul(rotated, F::contiguous(sin)));
}

bool runCase(int numQueryHeads, int numKeyHeads, int headDim) {
  std::vector<LongType> positionValues = {7, 2, 31};
  int numTokens = static_cast<int>(positionValues.size());
  Device device = Device::getCuda();

  Tensor positions = F::to(
      device,
      Tensor::create<LongType>({numTokens}, positionValues));
  Tensor query = F::rand({numTokens, numQueryHeads, headDim}, DType::kFloat16, device);
  Tensor key = F::rand({numTokens, numKeyHeads, headDim}, DType::kFloat16, device);
  Tensor cache = F::rand({64, 2 * headDim}, DType::kFloat16, device);

  Tensor gathered = F::lookup(cache, positions);
  gathered = gathered.view({numTokens, 2, 1, headDim}).transpose(0, 1);
  Tensor expectedQuery = baselineRotaryEmbedding(query, gathered);
  Tensor expectedKey = baselineRotaryEmbedding(key, gathered);

  Tensor actualQuery = F::contiguous(query);
  Tensor actualKey = F::contiguous(key);
  F::rotaryEmbedding(positions, actualQuery, actualKey, cache);

    return F::allClose(toCpuFloat(actualQuery), toCpuFloat(expectedQuery), 5e-3f) &&
      F::allClose(toCpuFloat(actualKey), toCpuFloat(expectedKey), 5e-3f);
}

/// Rotates only the first `rotaryDim` of each head, leaving the tail alone, which is what a
/// partial rotary factor asks for.
bool runPartialCase(int numQueryHeads, int numKeyHeads, int headDim, int rotaryDim) {
  std::vector<LongType> positionValues = {7, 2, 31};
  int numTokens = static_cast<int>(positionValues.size());
  Device device = Device::getCuda();

  Tensor positions = F::to(device, Tensor::create<LongType>({numTokens}, positionValues));
  Tensor query = F::rand({numTokens, numQueryHeads, headDim}, DType::kFloat16, device);
  Tensor key = F::rand({numTokens, numKeyHeads, headDim}, DType::kFloat16, device);

  // The cache is only as wide as the rotated part; that is what tells the operator how much of
  // each head to touch.
  Tensor cache = F::rand({64, 2 * rotaryDim}, DType::kFloat16, device);

  Tensor gathered = F::lookup(cache, positions);
  gathered = gathered.view({numTokens, 2, 1, rotaryDim}).transpose(0, 1);

  // The reference rotates the front of each head and copies the tail through unchanged.
  Tensor expectedQuery = F::contiguous(query);
  Tensor expectedKey = F::contiguous(key);
  Tensor rotatedQuery = baselineRotaryEmbedding(
      F::contiguous(query.slice(-1, {0, rotaryDim})),
      gathered);
  Tensor rotatedKey = baselineRotaryEmbedding(
      F::contiguous(key.slice(-1, {0, rotaryDim})),
      gathered);
  Tensor queryFront = expectedQuery.slice(-1, {0, rotaryDim});
  Tensor keyFront = expectedKey.slice(-1, {0, rotaryDim});
  F::copy(rotatedQuery, queryFront);
  F::copy(rotatedKey, keyFront);

  Tensor actualQuery = F::contiguous(query);
  Tensor actualKey = F::contiguous(key);
  F::rotaryEmbedding(positions, actualQuery, actualKey, cache);

  return F::allClose(toCpuFloat(actualQuery), toCpuFloat(expectedQuery), 5e-3f) &&
         F::allClose(toCpuFloat(actualKey), toCpuFloat(expectedKey), 5e-3f);
}

}  // namespace

CATCH_TEST_CASE("test CUDA rotaryEmbedding", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  CATCH_REQUIRE(runCase(4, 2, 64));
  CATCH_REQUIRE(runCase(24, 8, 128));
  CATCH_REQUIRE(runCase(8, 8, 256));
}

CATCH_TEST_CASE("test CUDA rotaryEmbedding (partial rotary factor)", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  // headDim 256 rotating 64 of it is the 0.25 partial rotary factor Qwen3.5 uses.
  CATCH_REQUIRE(runPartialCase(24, 4, 256, 64));
  CATCH_REQUIRE(runPartialCase(8, 2, 128, 32));
  CATCH_REQUIRE(runPartialCase(4, 4, 64, 32));

  // rotating the whole head is the same thing the non-partial path does.
  CATCH_REQUIRE(runPartialCase(4, 2, 64, 64));
}

}  // namespace fl
