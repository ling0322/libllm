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

#include "libllm/packed_batch.h"

#include <algorithm>
#include <utility>

#include "lutil/log.h"
#include "flint/functional.h"

namespace libllm {

PackedBatch PackedBatch::single(int qLen, int pastLen) {
  CHECK(qLen > 0 && pastLen >= 0);

  PackedBatch batch;
  batch._cuSeqlensQ = {0, qLen};
  batch._cuSeqlensK = {0, pastLen + qLen};

  batch._positionIds.reserve(qLen);
  for (int i = 0; i < qLen; ++i) {
    batch._positionIds.push_back(pastLen + i);
  }

  return batch;
}

PackedBatch PackedBatch::packed(
    std::vector<int> cuSeqlensQ,
    std::vector<int> cuSeqlensK,
    std::vector<fl::LongType> positionIds) {
  CHECK(cuSeqlensQ.size() == cuSeqlensK.size() && cuSeqlensQ.size() >= 2);
  CHECK(cuSeqlensQ.back() == static_cast<int>(positionIds.size()));

  PackedBatch batch;
  batch._cuSeqlensQ = std::move(cuSeqlensQ);
  batch._cuSeqlensK = std::move(cuSeqlensK);
  batch._positionIds = std::move(positionIds);

  return batch;
}

int PackedBatch::numSequences() const {
  return static_cast<int>(_cuSeqlensQ.size()) - 1;
}

void PackedBatch::setKVCacheManager(std::weak_ptr<KVCacheManager> manager) {
  _kvCacheManager = std::move(manager);
}

std::weak_ptr<KVCacheManager> PackedBatch::kvCacheManager() const {
  return _kvCacheManager;
}

int PackedBatch::totalQLen() const {
  return _cuSeqlensQ.back();
}

int PackedBatch::totalKLen() const {
  return _cuSeqlensK.back();
}

int PackedBatch::maxQLen() const {
  int maxLen = 0;
  for (int i = 0; i < numSequences(); ++i) {
    maxLen = std::max(maxLen, _cuSeqlensQ[i + 1] - _cuSeqlensQ[i]);
  }

  return maxLen;
}

int PackedBatch::maxKLen() const {
  int maxLen = 0;
  for (int i = 0; i < numSequences(); ++i) {
    maxLen = std::max(maxLen, _cuSeqlensK[i + 1] - _cuSeqlensK[i]);
  }

  return maxLen;
}

lut::Span<const int> PackedBatch::cuSeqlensQ() const {
  return lut::makeConstSpan(_cuSeqlensQ);
}

lut::Span<const int> PackedBatch::cuSeqlensK() const {
  return lut::makeConstSpan(_cuSeqlensK);
}

lut::Span<const fl::LongType> PackedBatch::positionIds() const {
  return lut::makeConstSpan(_positionIds);
}

fl::Tensor PackedBatch::positionIdsTensor(fl::Device device) const {
  fl::Tensor ids = fl::Tensor::create<fl::LongType>(
      {static_cast<int>(_positionIds.size())},
      lut::makeConstSpan(_positionIds));

  return fl::F::to(device, ids);
}

namespace {

fl::Tensor toLongTensor(lut::Span<const int> values) {
  std::vector<fl::LongType> longValues(values.begin(), values.end());
  return fl::Tensor::create<fl::LongType>(
      {static_cast<int>(longValues.size())},
      lut::makeConstSpan(longValues));
}

}  // namespace

fl::Tensor PackedBatch::cuSeqlensQTensor(fl::Device device) const {
  return fl::F::to(device, toLongTensor(lut::makeConstSpan(_cuSeqlensQ)));
}

fl::Tensor PackedBatch::cuSeqlensKTensor(fl::Device device) const {
  return fl::F::to(device, toLongTensor(lut::makeConstSpan(_cuSeqlensK)));
}

}  // namespace libllm
