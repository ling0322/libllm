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
class PackedBatch {
 public:
  // A single contiguous sequence of length `qLen`, whose KV-cache already holds `pastLen` past
  // tokens. Describes the layout only, the caller feeds the activations separately.
  static PackedBatch single(int qLen, int pastLen);

  // Same, but carrying the tokens to forward. This is what a scheduler hands to a model.
  static PackedBatch single(lut::Span<const fl::LongType> tokenIds, int pastLen);

  // A packed multi-sequence batch. `positionIds` has length cuSeqlensQ.back() (one rotary
  // position per packed query token, in packed order).
  static PackedBatch packed(
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

  // materialize the packed query tokens as a device <long>(totalQLen) tensor.
  fl::Tensor tokenIdsTensor(fl::Device device) const;

  // materialize the per-token rotary position ids as a device <long>(totalQLen) tensor.
  fl::Tensor positionIdsTensor(fl::Device device) const;

  // materialize cuSeqlensQ/cuSeqlensK as device <long>(numSequences+1) tensors. Backends that need
  // int32 device pointers (e.g. the CUDA varlen FlashAttention kernel's cu_seqlens_q/k fields)
  // cast down from this at the point of use.
  fl::Tensor cuSeqlensQTensor(fl::Device device) const;
  fl::Tensor cuSeqlensKTensor(fl::Device device) const;

 private:
  std::vector<fl::LongType> _tokenIds;
  std::vector<int> _cuSeqlensQ;
  std::vector<int> _cuSeqlensK;
  std::vector<fl::LongType> _positionIds;
  std::weak_ptr<KVCacheManager> _kvCacheManager;

  PackedBatch() = default;
};

}  // namespace libllm
