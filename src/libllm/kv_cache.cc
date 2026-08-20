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

#include "libllm/kv_cache.h"

#include <algorithm>
#include <memory>

#include "flint/functional.h"
#include "flint/memory.h"
#include "libllm/model_for_generation.h"
#include "lutil/error.h"
#include "lutil/log.h"
#include "lutil/strings.h"

namespace libllm {

namespace F = fl::F;

KVCacheSpec::KVCacheSpec(
    int numLayers,
    int numKeyValueHeads,
    int headDim,
    int maxContextLength,
    fl::DType dtype)
    : _numLayers(numLayers),
      _numKeyValueHeads(numKeyValueHeads),
      _headDim(headDim),
      _maxContextLength(maxContextLength),
      _dtype(dtype) {
}

int KVCacheSpec::getNumLayers() const {
  return _numLayers;
}

int KVCacheSpec::getNumKeyValueHeads() const {
  return _numKeyValueHeads;
}

int KVCacheSpec::getHeadDim() const {
  return _headDim;
}

int KVCacheSpec::getMaxContextLength() const {
  return _maxContextLength;
}

fl::DType KVCacheSpec::getDType() const {
  return _dtype;
}

int64_t getKVCacheBytesPerBlock(const KVCacheSpec &spec, int blockSize) {
  CHECK(blockSize > 0);

  int64_t numElements = 2LL * spec.getNumLayers() * blockSize * spec.getNumKeyValueHeads() *
                        spec.getHeadDim();
  return spec.getDType().getTotalSize(numElements);
}

KVCache KVCache::clone() const {
  KVCache cache;
  cache._dict = _dict;
  cache._intDict = _intDict;

  return cache;
}

fl::Tensor KVCache::getTensor(const std::string &name) const {
  auto it = _dict.find(name);
  if (it == _dict.end()) {
    throw lut::AbortedError(lut::sprintf("tensor \"%s\" not found in kv cache.", name));
  }

  return it->second;
}

void KVCache::putTensor(const std::string &name, fl::Tensor tensor) {
  _dict[name] = tensor;
}

bool KVCache::hasTensor(const std::string &name) const {
  return _dict.find(name) != _dict.end();
}

int KVCache::getValue(const std::string &name) const {
  auto it = _intDict.find(name);
  if (it == _intDict.end()) {
    throw lut::AbortedError(lut::sprintf("value \"%s\" not found in kv cache.", name));
  }

  return it->second;
}

void KVCache::putValue(const std::string &name, int value) {
  _intDict[name] = value;
}

bool KVCache::hasValue(const std::string &name) const {
  return _intDict.find(name) != _intDict.end();
}

int64_t KVCacheManager::estimateMemoryBudget(
    const ModelForGeneration &model,
    const EngineConfig &config) {
  float memoryUtilization = config.kvCacheMemoryUtilization;
  CHECK(memoryUtilization > 0.0f && memoryUtilization <= 1.0f);
  CHECK(config.maxNumBatchedTokens > 0);

  fl::Device device = model.getDevice();
  int maxContextLength = model.getKVCacheSpec().getMaxContextLength();

  // the peak of a full-size batch tells how much memory the cache has to leave for activation.
  fl::MemorySnapshot::resetPeakStats(device);
  model.profileRun(std::min(config.maxNumBatchedTokens, maxContextLength));

  fl::MemorySnapshot snapshot = fl::MemorySnapshot::capture(device);
  if (snapshot.getTotalMemory() <= 0) {
    throw lut::AbortedError("device does not report its memory usage");
  }

  LOG(INFO) << "model memory profile: weights="
            << lut::formatNumber(snapshot.getAllocatedMemory()) << "B, activation="
            << lut::formatNumber(
                   snapshot.getPeakAllocatedMemory() - snapshot.getAllocatedMemory())
            << "B, peak=" << lut::formatNumber(snapshot.getPeakAllocatedMemory()) << "B";

  int64_t budget = static_cast<int64_t>(snapshot.getTotalMemory() * memoryUtilization) -
                   snapshot.getPeakAllocatedMemory();

  // another process may hold memory this process never gets to see.
  return std::min(budget, snapshot.getFreeMemory());
}

std::shared_ptr<KVCacheManager> KVCacheManager::create(
    const ModelForGeneration &model,
    const EngineConfig &config) {
  KVCacheSpec spec = model.getKVCacheSpec();
  int64_t memoryBudget = estimateMemoryBudget(model, config);

  int blockSize = config.kvCacheBlockSize;
  int64_t bytesPerBlock = getKVCacheBytesPerBlock(spec, blockSize);
  int64_t numBlocks = memoryBudget > 0 ? memoryBudget / bytesPerBlock : 0;
  if (numBlocks <= 0) {
    throw lut::AbortedError("not enough memory to store a single block of the kv cache");
  }

  LOG(INFO) << "kv cache: allocated=" << lut::formatNumber(numBlocks * bytesPerBlock)
            << "B, blocks=" << numBlocks << ", block size=" << blockSize
            << " tokens, capacity=" << lut::formatNumber(numBlocks * blockSize)
            << " tokens";

  return std::make_shared<KVCacheManager>(
      spec,
      blockSize,
      static_cast<int>(numBlocks),
      model.getDevice());
}

KVCacheManager::KVCacheManager(
    const KVCacheSpec &spec,
    int blockSize,
    int numBlocks,
    fl::Device device)
    : _spec(spec),
      _blockSize(blockSize),
      _numBlocks(numBlocks) {
  if (blockSize <= 0 || numBlocks <= 0) {
    throw lut::AbortedError("invalid block_size or num_blocks for kv cache");
  }

  std::vector<int> shape{blockSize, spec.getNumKeyValueHeads(), spec.getHeadDim()};
  shape.insert(shape.begin(), numBlocks);

  for (int i = 0; i < spec.getNumLayers(); ++i) {
    _keyCache.emplace_back(F::tensor(shape, spec.getDType(), device));
    _valueCache.emplace_back(F::tensor(shape, spec.getDType(), device));
  }

  // descending order, so the smallest block id is allocated first.
  _freeBlocks.reserve(numBlocks);
  for (int i = numBlocks - 1; i >= 0; --i) {
    _freeBlocks.push_back(i);
  }
}

fl::Tensor KVCacheManager::getKeyCache(int layer) const {
  CHECK(layer >= 0 && layer < static_cast<int>(_keyCache.size()));
  return _keyCache[layer];
}

fl::Tensor KVCacheManager::getValueCache(int layer) const {
  CHECK(layer >= 0 && layer < static_cast<int>(_valueCache.size()));
  return _valueCache[layer];
}

int KVCacheManager::getBlockSize() const {
  return _blockSize;
}

int KVCacheManager::getNumBlocks() const {
  return _numBlocks;
}

int KVCacheManager::getNumFreeBlocks() const {
  return static_cast<int>(_freeBlocks.size());
}

int KVCacheManager::getNumBlocksForTokens(int numTokens) const {
  CHECK(numTokens >= 0);
  return (numTokens + _blockSize - 1) / _blockSize;
}

int KVCacheManager::getMaxNumBlocksPerRequest() const {
  return getNumBlocksForTokens(_spec.getMaxContextLength());
}

std::vector<int> KVCacheManager::allocateBlocks(int numBlocks) {
  CHECK(numBlocks >= 0);
  if (numBlocks > getNumFreeBlocks()) {
    return {};
  }

  std::vector<int> blockIds;
  blockIds.reserve(numBlocks);
  for (int i = 0; i < numBlocks; ++i) {
    blockIds.push_back(_freeBlocks.back());
    _freeBlocks.pop_back();
  }

  return blockIds;
}

std::vector<int> KVCacheManager::allocateBlocksForTokens(int numTokens) {
  return allocateBlocks(getNumBlocksForTokens(numTokens));
}

void KVCacheManager::freeBlocks(lut::Span<const int> blockIds) {
  for (int blockId : blockIds) {
    CHECK(blockId >= 0 && blockId < _numBlocks);
    _freeBlocks.push_back(blockId);
  }
}

}  // namespace libllm
