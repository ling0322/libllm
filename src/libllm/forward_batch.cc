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

#include "libllm/forward_batch.h"

#include <algorithm>
#include <utility>

#include "lutil/log.h"
#include "flint/functional.h"

namespace libllm {

ForwardBatch ForwardBatch::single(int qLen, int pastLen) {
  CHECK(qLen > 0 && pastLen >= 0);

  ForwardBatch batch;
  batch._cuSeqlensQ = {0, qLen};
  batch._cuSeqlensK = {0, pastLen + qLen};

  batch._positionIds.reserve(qLen);
  for (int i = 0; i < qLen; ++i) {
    batch._positionIds.push_back(pastLen + i);
  }

  return batch;
}

ForwardBatch ForwardBatch::single(lut::Span<const fl::LongType> tokenIds, int pastLen) {
  CHECK(!tokenIds.empty());

  ForwardBatch batch = single(static_cast<int>(tokenIds.size()), pastLen);
  batch._tokenIds.assign(tokenIds.begin(), tokenIds.end());

  return batch;
}

ForwardBatch ForwardBatch::packed(
    std::vector<int> cuSeqlensQ,
    std::vector<int> cuSeqlensK,
    std::vector<fl::LongType> positionIds) {
  CHECK(cuSeqlensQ.size() == cuSeqlensK.size() && cuSeqlensQ.size() >= 2);
  CHECK(cuSeqlensQ.back() == static_cast<int>(positionIds.size()));

  ForwardBatch batch;
  batch._cuSeqlensQ = std::move(cuSeqlensQ);
  batch._cuSeqlensK = std::move(cuSeqlensK);
  batch._positionIds = std::move(positionIds);

  return batch;
}

int ForwardBatch::numSequences() const {
  return static_cast<int>(_cuSeqlensQ.size()) - 1;
}

void ForwardBatch::setKVCacheManager(std::weak_ptr<KVCacheManager> manager) {
  _kvCacheManager = std::move(manager);
}

std::weak_ptr<KVCacheManager> ForwardBatch::kvCacheManager() const {
  return _kvCacheManager;
}

int ForwardBatch::totalQLen() const {
  return _cuSeqlensQ.back();
}

int ForwardBatch::totalKLen() const {
  return _cuSeqlensK.back();
}

int ForwardBatch::maxQLen() const {
  int maxLen = 0;
  for (int i = 0; i < numSequences(); ++i) {
    maxLen = std::max(maxLen, _cuSeqlensQ[i + 1] - _cuSeqlensQ[i]);
  }

  return maxLen;
}

int ForwardBatch::maxKLen() const {
  int maxLen = 0;
  for (int i = 0; i < numSequences(); ++i) {
    maxLen = std::max(maxLen, _cuSeqlensK[i + 1] - _cuSeqlensK[i]);
  }

  return maxLen;
}

lut::Span<const fl::LongType> ForwardBatch::tokenIds() const {
  return lut::makeConstSpan(_tokenIds);
}

lut::Span<const int> ForwardBatch::cuSeqlensQ() const {
  return lut::makeConstSpan(_cuSeqlensQ);
}

lut::Span<const int> ForwardBatch::cuSeqlensK() const {
  return lut::makeConstSpan(_cuSeqlensK);
}

lut::Span<const fl::LongType> ForwardBatch::positionIds() const {
  return lut::makeConstSpan(_positionIds);
}

fl::Tensor ForwardBatch::tokenIdsTensor(fl::Device device) const {
  CHECK(static_cast<int>(_tokenIds.size()) == totalQLen()) << "batch carries no tokens";

  fl::Tensor ids = fl::Tensor::create<fl::LongType>(
      {static_cast<int>(_tokenIds.size())},
      lut::makeConstSpan(_tokenIds));

  return fl::F::to(device, ids);
}

fl::Tensor ForwardBatch::positionIdsTensor(fl::Device device) const {
  fl::Tensor ids = fl::Tensor::create<fl::LongType>(
      {static_cast<int>(_positionIds.size())},
      lut::makeConstSpan(_positionIds));

  return fl::F::to(device, ids);
}

namespace {

fl::Tensor toIntTensor(lut::Span<const int> values) {
  return fl::Tensor::create<fl::IntType>({static_cast<int>(values.size())}, values);
}

}  // namespace

void ForwardBatch::setBlockIds(std::vector<std::vector<int>> blockIds) {
  CHECK(static_cast<int>(blockIds.size()) == numSequences());
  _blockIds = std::move(blockIds);
}

bool ForwardBatch::hasBlockIds() const {
  return !_blockIds.empty();
}

int ForwardBatch::getBlockSize() const {
  std::shared_ptr<KVCacheManager> manager = _kvCacheManager.lock();
  CHECK(manager) << "batch has no KV cache manager attached";

  return manager->getBlockSize();
}

fl::Tensor ForwardBatch::cuSeqlensQTensor(fl::Device device) const {
  return fl::F::to(device, toIntTensor(lut::makeConstSpan(_cuSeqlensQ)));
}

fl::Tensor ForwardBatch::seqlensKTensor(fl::Device device) const {
  std::vector<int> seqlensK(numSequences());
  for (int i = 0; i < numSequences(); ++i) {
    seqlensK[i] = _cuSeqlensK[i + 1] - _cuSeqlensK[i];
  }

  return fl::F::to(device, toIntTensor(lut::makeConstSpan(seqlensK)));
}

fl::Tensor ForwardBatch::blockTableTensor(fl::Device device) const {
  CHECK(hasBlockIds()) << "batch has no blocks assigned";

  int maxNumBlocks = 0;
  for (const std::vector<int> &blocks : _blockIds) {
    maxNumBlocks = std::max(maxNumBlocks, static_cast<int>(blocks.size()));
  }
  CHECK(maxNumBlocks > 0);

  // Rows are padded to the longest one; the kernel never reads past a sequence's own length.
  std::vector<int> table(static_cast<size_t>(numSequences()) * maxNumBlocks, 0);
  for (int i = 0; i < numSequences(); ++i) {
    std::copy(_blockIds[i].begin(), _blockIds[i].end(), table.begin() + i * maxNumBlocks);
  }

  fl::Tensor blockTable = fl::Tensor::create<fl::IntType>(
      {numSequences(), maxNumBlocks},
      lut::makeConstSpan(table));

  return fl::F::to(device, blockTable);
}

fl::Tensor ForwardBatch::slotMappingTensor(fl::Device device) const {
  CHECK(hasBlockIds()) << "batch has no blocks assigned";
  CHECK(static_cast<int>(_positionIds.size()) == totalQLen());

  int blockSize = getBlockSize();
  std::vector<int> slots(totalQLen());
  for (int i = 0; i < numSequences(); ++i) {
    const std::vector<int> &blocks = _blockIds[i];
    for (int t = _cuSeqlensQ[i]; t < _cuSeqlensQ[i + 1]; ++t) {
      // A token's rotary position is also its index in the sequence, so it names the slot.
      int position = static_cast<int>(_positionIds[t]);
      int block = position / blockSize;
      CHECK(block < static_cast<int>(blocks.size())) << "sequence outgrew its blocks";
      slots[t] = blocks[block] * blockSize + position % blockSize;
    }
  }

  return fl::F::to(device, toIntTensor(lut::makeConstSpan(slots)));
}

}  // namespace libllm
