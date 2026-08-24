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

#pragma once

#include <vector>

#include "flint/device.h"
#include "flint/tensor.h"

namespace libllm {

/// Per-sequence sampling metadata for one model forward.
class SamplingBatch {
 public:
  SamplingBatch(
      std::vector<fl::LongType> sequenceIndices,
      std::vector<float> temperatures,
      std::vector<fl::IntType> topKs,
      std::vector<float> topPs);

  bool empty() const;
  int size() const;
  const std::vector<fl::LongType> &sequenceIndices() const;

  void prepare(fl::Device device);
  fl::Tensor sample(fl::Tensor logits) const;

 private:
  std::vector<fl::LongType> _sequenceIndices;
  std::vector<float> _temperatures;
  std::vector<fl::IntType> _topKs;
  std::vector<float> _topPs;

  fl::Device::Type _preparedDeviceType = fl::Device::kUnknown;
  fl::Tensor _sequenceIndicesTensor;
  fl::Tensor _temperaturesTensor;
  fl::Tensor _topKsTensor;
  fl::Tensor _topPsTensor;
};

}  // namespace libllm
