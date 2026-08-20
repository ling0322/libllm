// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "libllm/kv_cache.h"

#include <vector>

#include "catch2/catch_amalgamated.hpp"
#include "flint/device.h"
#include "flint/dtype.h"

namespace libllm {

namespace {

constexpr int NumLayers = 3;
constexpr int NumKeyValueHeads = 2;
constexpr int HeadDim = 4;
constexpr int MaxContextLength = 100;
constexpr int BlockSize = 8;
constexpr int NumBlocks = 6;

KVCacheSpec createSpec() {
  return KVCacheSpec(NumLayers, NumKeyValueHeads, HeadDim, MaxContextLength, fl::DType::kFloat);
}

KVCacheManager createManager() {
  return KVCacheManager(createSpec(), BlockSize, NumBlocks, fl::Device::getCpu());
}

}  // namespace

CATCH_TEST_CASE("kv cache manager allocates storage for each layer", "[libllm][kv_cache]") {
  KVCacheManager manager = createManager();

  CATCH_REQUIRE(manager.getBlockSize() == BlockSize);
  CATCH_REQUIRE(manager.getNumBlocks() == NumBlocks);
  CATCH_REQUIRE(manager.getNumFreeBlocks() == NumBlocks);

  for (int layer = 0; layer < NumLayers; ++layer) {
    for (const fl::Tensor &cache : {manager.getKeyCache(layer), manager.getValueCache(layer)}) {
      CATCH_REQUIRE(cache.getDim() == 4);
      CATCH_REQUIRE(cache.getShape(0) == NumBlocks);
      CATCH_REQUIRE(cache.getShape(1) == BlockSize);
      CATCH_REQUIRE(cache.getShape(2) == NumKeyValueHeads);
      CATCH_REQUIRE(cache.getShape(3) == HeadDim);
      CATCH_REQUIRE(cache.getDType() == fl::DType::kFloat);
    }
  }
}

CATCH_TEST_CASE("kv cache manager gives each layer its own storage", "[libllm][kv_cache]") {
  KVCacheManager manager = createManager();

  CATCH_REQUIRE(
      manager.getKeyCache(0).getInternalData() != manager.getKeyCache(1).getInternalData());
  CATCH_REQUIRE(
      manager.getKeyCache(0).getInternalData() != manager.getValueCache(0).getInternalData());
}

CATCH_TEST_CASE("kv cache manager allocates and frees blocks", "[libllm][kv_cache]") {
  KVCacheManager manager = createManager();

  std::vector<int> first = manager.allocateBlocks(2);
  CATCH_REQUIRE(first == std::vector<int>({0, 1}));
  CATCH_REQUIRE(manager.getNumFreeBlocks() == NumBlocks - 2);

  std::vector<int> second = manager.allocateBlocks(3);
  CATCH_REQUIRE(second == std::vector<int>({2, 3, 4}));
  CATCH_REQUIRE(manager.getNumFreeBlocks() == NumBlocks - 5);

  manager.freeBlocks(first);
  CATCH_REQUIRE(manager.getNumFreeBlocks() == NumBlocks - 3);

  manager.freeBlocks(second);
  CATCH_REQUIRE(manager.getNumFreeBlocks() == NumBlocks);
}

CATCH_TEST_CASE("kv cache manager reuses freed blocks", "[libllm][kv_cache]") {
  KVCacheManager manager = createManager();

  std::vector<int> blockIds = manager.allocateBlocks(NumBlocks);
  CATCH_REQUIRE(manager.getNumFreeBlocks() == 0);

  manager.freeBlocks(std::vector<int>({blockIds[1]}));
  CATCH_REQUIRE(manager.allocateBlocks(1) == std::vector<int>({blockIds[1]}));
}

CATCH_TEST_CASE("kv cache manager rejects allocation without enough blocks", "[libllm][kv_cache]") {
  KVCacheManager manager = createManager();

  CATCH_REQUIRE(manager.allocateBlocks(NumBlocks + 1).empty());
  CATCH_REQUIRE(manager.getNumFreeBlocks() == NumBlocks);

  manager.allocateBlocks(NumBlocks - 1);
  CATCH_REQUIRE(manager.allocateBlocks(2).empty());
  CATCH_REQUIRE(manager.getNumFreeBlocks() == 1);
}

CATCH_TEST_CASE("kv cache manager converts token count to block count", "[libllm][kv_cache]") {
  KVCacheManager manager = createManager();

  CATCH_REQUIRE(manager.getNumBlocksForTokens(0) == 0);
  CATCH_REQUIRE(manager.getNumBlocksForTokens(1) == 1);
  CATCH_REQUIRE(manager.getNumBlocksForTokens(BlockSize) == 1);
  CATCH_REQUIRE(manager.getNumBlocksForTokens(BlockSize + 1) == 2);

  // ceil(100 / 8)
  CATCH_REQUIRE(manager.getMaxNumBlocksPerRequest() == 13);

  CATCH_REQUIRE(manager.allocateBlocksForTokens(BlockSize + 1) == std::vector<int>({0, 1}));
}

CATCH_TEST_CASE("kv cache manager rejects invalid block config", "[libllm][kv_cache]") {
  CATCH_REQUIRE_THROWS(KVCacheManager(createSpec(), 0, NumBlocks, fl::Device::getCpu()));
  CATCH_REQUIRE_THROWS(KVCacheManager(createSpec(), BlockSize, 0, fl::Device::getCpu()));
}

CATCH_TEST_CASE("kv cache block size in bytes covers every layer", "[libllm][kv_cache]") {
  // key and value of each layer.
  int64_t expected = 2LL * NumLayers * BlockSize * NumKeyValueHeads * HeadDim * sizeof(float);

  CATCH_REQUIRE(getKVCacheBytesPerBlock(createSpec(), BlockSize) == expected);
  CATCH_REQUIRE(
      getKVCacheBytesPerBlock(createSpec(), 2 * BlockSize) ==
      2 * getKVCacheBytesPerBlock(createSpec(), BlockSize));
}

}  // namespace libllm
