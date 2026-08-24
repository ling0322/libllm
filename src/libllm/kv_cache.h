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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/tensor.h"
#include "libllm/engine_config.h"
#include "lutil/span.h"

namespace libllm {

class ModelForGeneration;

/// @brief Describes the model-specific layout requirements of its KV cache. Allocation properties
/// such as block size and number of blocks are determined by the KV cache manager.
class KVCacheSpec {
 public:
  /// @brief Construct a KV cache specification.
  /// @param numLayers Number of attention layers that produce KV cache entries.
  /// @param numKeyValueHeads Number of key-value heads in each attention layer.
  /// @param headDim Dimension of each key-value head.
  /// @param maxContextLength Maximum number of tokens supported by the model.
  /// @param dtype Data type used to store key and value tensors.
  KVCacheSpec(
      int numLayers,
      int numKeyValueHeads,
      int headDim,
      int maxContextLength,
      fl::DType dtype);

  /// @brief Get the number of attention layers that produce KV cache entries.
  /// @return Number of attention layers.
  int getNumLayers() const;

  /// @brief Get the number of key-value heads in each attention layer.
  /// @return Number of key-value heads.
  int getNumKeyValueHeads() const;

  /// @brief Get the dimension of each key-value head.
  /// @return Key-value head dimension.
  int getHeadDim() const;

  /// @brief Get the maximum number of tokens supported by the model.
  /// @return Maximum context length.
  int getMaxContextLength() const;

  /// @brief Get the data type used to store key and value tensors.
  /// @return KV cache data type.
  fl::DType getDType() const;

 private:
  int _numLayers;
  int _numKeyValueHeads;
  int _headDim;
  int _maxContextLength;
  fl::DType _dtype;
};

/// @brief Get the bytes one block occupies, counting the key and the value of every layer.
/// @param spec The KV cache layout required by the model.
/// @param blockSize Number of tokens stored in each block.
/// @return The number of bytes.
int64_t getKVCacheBytesPerBlock(const KVCacheSpec &spec, int blockSize);

/// @brief Owns the paged KV cache storage described by a KVCacheSpec and allocates blocks from it.
/// A block id refers to the same token range in every layer, but each layer addresses it in its own
/// key and value tensor.
class KVCacheManager {
 public:
  /// @brief Create a manager that fills the device memory left over once the model weights and the
  /// activation of a full-size batch are accounted for. Profiles the activation by forwarding a
  /// dummy batch through `model`. Requires a device that reports its memory usage.
  /// @param model The model that owns the KV cache.
  /// @param config The engine configuration.
  /// @return The created manager.
  static std::shared_ptr<KVCacheManager> create(
      const ModelForGeneration &model,
      const EngineConfig &config);

  /// @brief Create a manager and allocate the KV cache storage of all layers.
  /// @param spec The KV cache layout required by the model.
  /// @param blockSize Number of tokens stored in each block.
  /// @param numBlocks Total number of blocks to allocate for each layer.
  /// @param device Device to store the KV cache.
  KVCacheManager(const KVCacheSpec &spec, int blockSize, int numBlocks, fl::Device device);

  KVCacheManager(const KVCacheManager &) = delete;
  KVCacheManager &operator=(const KVCacheManager &) = delete;

  /// @brief Get the key cache of a layer.
  /// @param layer The layer index.
  /// @return <dtype>(numBlocks, blockSize, numKeyValueHeads, headDim): the key cache.
  fl::Tensor getKeyCache(int layer) const;

  /// @brief Get the value cache of a layer.
  /// @param layer The layer index.
  /// @return <dtype>(numBlocks, blockSize, numKeyValueHeads, headDim): the value cache.
  fl::Tensor getValueCache(int layer) const;

  /// @brief Get the number of tokens stored in each block.
  /// @return The block size.
  int getBlockSize() const;

  /// @brief Get the total number of blocks owned by this manager.
  /// @return The total number of blocks.
  int getNumBlocks() const;

  /// @brief Get the number of blocks that are currently not allocated.
  /// @return The number of free blocks.
  int getNumFreeBlocks() const;

  /// @brief Get the number of blocks needed to store `numTokens` tokens.
  /// @param numTokens The number of tokens.
  /// @return The number of blocks.
  int getNumBlocksForTokens(int numTokens) const;

  /// @brief Get the number of blocks needed by a request that reaches the maximum context length.
  /// @return The maximum number of blocks per request.
  int getMaxNumBlocksPerRequest() const;

  /// @brief Allocate blocks.
  /// @param numBlocks The number of blocks to allocate.
  /// @return The ids of the allocated blocks, or an empty vector if there is not enough free
  /// blocks. Nothing is allocated in the latter case.
  std::vector<int> allocateBlocks(int numBlocks);

  /// @brief Allocate the blocks needed to store `numTokens` tokens.
  /// @param numTokens The number of tokens.
  /// @return The ids of the allocated blocks, or an empty vector if there is not enough free
  /// blocks.
  std::vector<int> allocateBlocksForTokens(int numTokens);

  /// @brief Return blocks to this manager.
  /// @param blockIds The ids of the blocks to free.
  void freeBlocks(lut::Span<const int> blockIds);

 private:
  /// @brief Get the bytes the KV cache is allowed to take, which is what the engine config leaves
  /// over once the model weights and the activation of one forward pass are taken out. Measures
  /// the latter by forwarding a full-size batch through `model`.
  /// @param model The model that owns the KV cache.
  /// @param config The engine configuration.
  /// @return The budget in bytes. Not positive when nothing is left over.
  static int64_t estimateMemoryBudget(
      const ModelForGeneration &model,
      const EngineConfig &config);

  KVCacheSpec _spec;
  int _blockSize;
  int _numBlocks;

  std::vector<fl::Tensor> _keyCache;
  std::vector<fl::Tensor> _valueCache;
  std::vector<int> _freeBlocks;
};

}  // namespace libllm
