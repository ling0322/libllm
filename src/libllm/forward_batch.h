// The MIT License (MIT)
//
// Copyright (c) 2023-2026 Xiaoyang Chen
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
#include <vector>

#include "lutil/span.h"
#include "flint/device.h"
#include "flint/tensor.h"
#include "libllm/kv_cache.h"

namespace libllm {

// Describes how a packed (ragged/varlen) 2D activation tensor <float>(totalTokens, hidden)
// decomposes into per-sequence chunks for one forward() call, and carries the rotary position id
// of every packed token.
//
// `cuSeqlensQ`/`cuSeqlensK` follow the FlashAttention convention: exclusive-prefix-sum token
// offsets of length numSequences()+1, e.g. {0, 3, 7} means sequence 0 occupies tokens [0, 3) and
// sequence 1 occupies tokens [3, 7).
class ForwardBatch {
 public:
  // A single contiguous sequence of length `qLen`, whose KV-cache already holds `pastLen` past
  // tokens. Describes the layout only, the caller feeds the activations separately.
  static ForwardBatch single(int qLen, int pastLen);

  // Same, but carrying the tokens to forward. This is what a scheduler hands to a model.
  static ForwardBatch single(lut::Span<const fl::LongType> tokenIds, int pastLen);

  // A packed multi-sequence batch. `tokenIds` and `positionIds` both have length
  // cuSeqlensQ.back(), in packed query order.
  static ForwardBatch packed(
      std::vector<fl::LongType> tokenIds,
      std::vector<int> cuSeqlensQ,
      std::vector<int> cuSeqlensK,
      std::vector<fl::LongType> positionIds);

  int numSequences() const;
  int totalQLen() const;
  int totalKLen() const;
  int maxQLen() const;
  int maxKLen() const;

  // the packed query tokens, empty when this batch only describes a layout.
  lut::Span<const fl::LongType> tokenIds() const;

  lut::Span<const int> cuSeqlensQ() const;
  lut::Span<const int> cuSeqlensK() const;
  lut::Span<const fl::LongType> positionIds() const;

  // the paged KV cache storage this batch reads and writes. Empty until attached.
  void setKVCacheManager(std::weak_ptr<KVCacheManager> manager);
  std::weak_ptr<KVCacheManager> kvCacheManager() const;

  // the blocks each sequence owns, in token order, one entry per sequence. The scheduler assigns
  // them; a batch without them can only describe a layout, not address a cache.
  void setBlockIds(std::vector<std::vector<int>> blockIds);
  bool hasBlockIds() const;

  // Materialize all model-independent device tensors once. Repeated calls for the same device are
  // no-ops; preparing one batch for different device types is invalid.
  void prepare(fl::Device device);

  fl::Tensor tokenIdsTensor() const;          // <long>(totalQLen)
  fl::Tensor positionIdsTensor() const;       // <long>(totalQLen)
  fl::Tensor lastQueryIndicesTensor() const;  // <long>(numSequences)

  fl::Tensor cuSeqlensQTensor() const;  // <int>(numSequences + 1)
  fl::Tensor seqlensKTensor() const;    // <int>(numSequences)
  fl::Tensor blockTableTensor() const;  // <int>(numSequences, maxNumBlocks)

  // the pool slot each packed query token is stored at, as blockId * blockSize + offset.
  fl::Tensor slotMappingTensor() const;  // <int>(totalQLen)

 private:
  std::vector<fl::LongType> _tokenIds;
  std::vector<int> _cuSeqlensQ;
  std::vector<int> _cuSeqlensK;
  std::vector<fl::LongType> _positionIds;
  std::vector<std::vector<int>> _blockIds;
  std::weak_ptr<KVCacheManager> _kvCacheManager;

  fl::Device::Type _preparedDeviceType = fl::Device::kUnknown;
  fl::Tensor _tokenIdsTensor;
  fl::Tensor _positionIdsTensor;
  fl::Tensor _lastQueryIndicesTensor;
  fl::Tensor _cuSeqlensQTensor;
  fl::Tensor _seqlensKTensor;
  fl::Tensor _blockTableTensor;
  fl::Tensor _slotMappingTensor;

  // the block size of the attached manager, which every paged index depends on.
  int getBlockSize() const;

  ForwardBatch() = default;
};

}  // namespace libllm
