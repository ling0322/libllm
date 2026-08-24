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

#include "libllm/sampling_batch.h"

#include <utility>

#include "flint/functional.h"
#include "lutil/error.h"
#include "lutil/span.h"

namespace libllm {

SamplingBatch::SamplingBatch(
    std::vector<fl::LongType> sequenceIndices,
    std::vector<float> temperatures,
    std::vector<fl::IntType> topKs,
    std::vector<float> topPs)
    : _sequenceIndices(std::move(sequenceIndices)),
      _temperatures(std::move(temperatures)),
      _topKs(std::move(topKs)),
      _topPs(std::move(topPs)) {
  CHECK(_temperatures.size() == _sequenceIndices.size());
  CHECK(_topKs.size() == _sequenceIndices.size());
  CHECK(_topPs.size() == _sequenceIndices.size());
}

bool SamplingBatch::empty() const {
  return _sequenceIndices.empty();
}

int SamplingBatch::size() const {
  return static_cast<int>(_sequenceIndices.size());
}

const std::vector<fl::LongType> &SamplingBatch::sequenceIndices() const {
  return _sequenceIndices;
}

void SamplingBatch::prepare(fl::Device device) {
  if (_preparedDeviceType != fl::Device::kUnknown) {
    CHECK(_preparedDeviceType == device.getType()) << "sampling batch prepared for another device";
    return;
  }
  if (empty()) {
    _preparedDeviceType = device.getType();
    return;
  }

  _sequenceIndicesTensor = fl::F::to(
      device,
      fl::Tensor::create<fl::LongType>({size()}, lut::makeConstSpan(_sequenceIndices)));
  _temperaturesTensor = fl::F::to(
      device,
      fl::Tensor::create<float>({size()}, lut::makeConstSpan(_temperatures)));
  _topKsTensor = fl::F::to(
      device,
      fl::Tensor::create<fl::IntType>({size()}, lut::makeConstSpan(_topKs)));
  _topPsTensor = fl::F::to(
      device,
      fl::Tensor::create<float>({size()}, lut::makeConstSpan(_topPs)));
  _preparedDeviceType = device.getType();
}

fl::Tensor SamplingBatch::sample(fl::Tensor logits) const {
  CHECK(_preparedDeviceType != fl::Device::kUnknown) << "sampling batch is not prepared";
  CHECK(!empty());
  CHECK(logits.getDim() == 2);

  fl::Tensor sampledLogits = fl::F::lookup(logits, _sequenceIndicesTensor);
  return fl::F::sample(sampledLogits, _temperaturesTensor, _topKsTensor, _topPsTensor);
}

}  // namespace libllm
