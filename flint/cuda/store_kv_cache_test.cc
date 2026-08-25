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

Tensor toCudaHalf(const Tensor &tensor) {
  return F::cast(F::to(Device::getCuda(), tensor), DType::kFloat16);
}

Tensor toCpuFloat(const Tensor &tensor) {
  return F::to(Device::getCpu(), F::cast(F::contiguous(tensor), DType::kFloat));
}

Tensor makeInput(
    int numTokens,
    int numHeads,
    int headDim,
    int prefix,
    int suffix,
    float bias,
    Tensor *reference) {
  int elementsPerToken = numHeads * headDim;
  int rowSize = prefix + elementsPerToken + suffix;
  std::vector<float> values(numTokens * rowSize);
  for (int token = 0; token < numTokens; ++token) {
    for (int i = 0; i < rowSize; ++i) {
      values[token * rowSize + i] = bias + token + static_cast<float>(i) / rowSize;
    }
  }

  Tensor cpu = Tensor::create<float>({numTokens, rowSize}, values);
  *reference = cpu.slice(1, {prefix, prefix + elementsPerToken})
                   .view({numTokens, numHeads, headDim});
  return toCudaHalf(cpu)
      .slice(1, {prefix, prefix + elementsPerToken})
      .view({numTokens, numHeads, headDim});
}

Tensor makeCache(
    int numBlocks,
    int blockSize,
    int numHeads,
    int headDim,
    int prefix,
    int suffix) {
  int elementsPerToken = numHeads * headDim;
  Tensor storage = F::zeros(
      {numBlocks, blockSize, prefix + elementsPerToken + suffix},
      DType::kFloat16,
      Device::getCuda());
  return storage.slice(2, {prefix, prefix + elementsPerToken})
      .view({numBlocks, blockSize, numHeads, headDim});
}

void checkStored(
    const Tensor &cache,
    const Tensor &reference,
    lut::Span<const IntType> slots,
    int blockSize) {
  Tensor cpuCache = toCpuFloat(cache);
  for (int token = 0; token < static_cast<int>(slots.size()); ++token) {
    int blockId = slots[token] / blockSize;
    int offset = slots[token] % blockSize;
    CATCH_INFO("token = " << token);
    CATCH_REQUIRE(
        F::allClose(
            cpuCache.subtensor(blockId).subtensor(offset),
            reference.subtensor(token),
            5e-3));
  }
}

}  // namespace

CATCH_TEST_CASE("test CUDA storeKVCache follows slot mapping", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int BlockSize = 16;
  constexpr int NumBlocks = 6;
  constexpr int NumHeads = 2;
  constexpr int HeadDim = 8;
  std::vector<IntType> slots = {
      5 * BlockSize,
      1 * BlockSize + 15,
      3 * BlockSize + 7,
      1 * BlockSize};

  Tensor k = F::rand({static_cast<int>(slots.size()), NumHeads, HeadDim}, DType::kFloat);
  Tensor v = F::rand({static_cast<int>(slots.size()), NumHeads, HeadDim}, DType::kFloat);
  Tensor keyCache = makeCache(NumBlocks, BlockSize, NumHeads, HeadDim, 0, 0);
  Tensor valueCache = makeCache(NumBlocks, BlockSize, NumHeads, HeadDim, 0, 0);
  Tensor slotMapping = Tensor::create<IntType>({static_cast<int>(slots.size())}, slots);

  F::storeKVCache(
      toCudaHalf(k),
      toCudaHalf(v),
      keyCache,
      valueCache,
      F::to(Device::getCuda(), slotMapping));

  checkStored(keyCache, k, lut::makeConstSpan(slots), BlockSize);
  checkStored(valueCache, v, lut::makeConstSpan(slots), BlockSize);

  Tensor untouched = toCpuFloat(keyCache).subtensor(0);
  CATCH_REQUIRE(F::allClose(untouched, F::zeros(untouched.getShape(), DType::kFloat)));
}

CATCH_TEST_CASE("test CUDA storeKVCache handles strided model views", "[op][cuda]") {
  if (!isOperatorsAvailable(Device::kCuda)) CATCH_SKIP("cuda device not available");

  constexpr int NumTokens = 3;
  constexpr int NumHeads = 8;
  constexpr int HeadDim = 128;
  constexpr int BlockSize = 256;
  constexpr int NumBlocks = 3;
  std::vector<IntType> slots = {
      2 * BlockSize + 255,
      0,
      1 * BlockSize + 127};

  Tensor expectedK;
  Tensor expectedV;
  Tensor k = makeInput(NumTokens, NumHeads, HeadDim, 3, 5, 10.0f, &expectedK);
  Tensor v = makeInput(NumTokens, NumHeads, HeadDim, 7, 2, 20.0f, &expectedV);
  Tensor keyCache = makeCache(NumBlocks, BlockSize, NumHeads, HeadDim, 1, 3);
  Tensor valueCache = makeCache(NumBlocks, BlockSize, NumHeads, HeadDim, 5, 2);
  Tensor slotMapping = Tensor::create<IntType>({NumTokens}, slots);

  CATCH_REQUIRE(k.getStride(0) != v.getStride(0));
  CATCH_REQUIRE(keyCache.getStride(1) != valueCache.getStride(1));

  F::storeKVCache(
      k,
      v,
      keyCache,
      valueCache,
      F::to(Device::getCuda(), slotMapping));

  checkStored(keyCache, expectedK, lut::makeConstSpan(slots), BlockSize);
  checkStored(valueCache, expectedV, lut::makeConstSpan(slots), BlockSize);

  Tensor untouched = toCpuFloat(valueCache).subtensor(2).subtensor(0);
  CATCH_REQUIRE(F::allClose(untouched, F::zeros(untouched.getShape(), DType::kFloat)));
}

}  // namespace fl