// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "libllm/mp.h"

#include <algorithm>
#include <functional>

#include "lutil/range.h"

namespace libllm {

lut::Range MP::splitRange(lut::Range range, int chunkIdx, int numChunks) {
  int numel = (range.getEnd() - range.getBegin()) / range.getStep();
  int partitionSize = numel / numChunks;
  int remain = numel % numChunks;

  int begin = chunkIdx * partitionSize;
  begin += std::min(chunkIdx, remain);
  int end = begin + partitionSize;
  if (chunkIdx < remain) ++end;

  begin = begin * range.getStep() + range.getBegin();
  end = std::min(end * range.getStep() + range.getBegin(), range.getEnd());

  return lut::Range(begin, end, range.getStep());
}

MP::Partition::Partition(
    lut::Range range,
    int partitionIdx,
    int numPartitions,
    int attachedThreadIdx)
    : _range(range),
      _partitionIdx(partitionIdx),
      _numPartitions(numPartitions),
      _attachedThreadIdx(attachedThreadIdx) {
}

MP::Partition::Partition(lut::Range range)
    : Partition(range, 0, 1, -1) {
}

lut::Range MP::Partition::getRange() const {
  return _range;
}

int MP::Partition::getPartitionIdx() const {
  return _partitionIdx;
}

int MP::Partition::getNumPartitions() const {
  return _numPartitions;
}

int MP::Partition::getAttachedThreadIdx() const {
  return _attachedThreadIdx;
}

}  // namespace libllm
