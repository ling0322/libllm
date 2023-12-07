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

#include "llm/common/generator.h"

#include <math.h>
#include <algorithm>
#include "ly/ly.h"
#include "lyutil/log.h"

using ly::DType;
using ly::Tensor;

namespace F = ly::functional;

namespace libllm {


Sampler::Sampler(int topK, float topP) : _topK(topK), _topP(topP) {}

std::vector<int> Sampler::getTopP(const ly::Tensor &distribution, lut::Span<const int> topK) {
  CHECK(distribution.getDim() == 1 && distribution.getDType() == DType::kFloat);
  float sumP = 0.0f;

  std::vector<int> topP;
  const float *d = distribution.getData<float>();
  for (int label : topK) {
    float p = d[label];
    topP.push_back(label);

    sumP += p;
    if (sumP >= _topP) {
      break;
    }
  }

  return topP;
}

std::vector<int> Sampler::getTopK(const Tensor &distribution) {
  CHECK(_topK <= distribution.getShape(0) && distribution.getStride(0) == 1);
  if (_topBuffer.size() != distribution.getShape(0))
    _topBuffer.resize(distribution.getShape(0));

  const float *d = distribution.getData<float>();
  for (int32_t i = 0; i < distribution.getShape(0); ++i) {
    _topBuffer[i] = std::make_pair(i, d[i]);
  }

  std::partial_sort(_topBuffer.begin(), _topBuffer.begin() + _topK, _topBuffer.end(), 
                    [](const std::pair<int32_t, float> &a, const std::pair<int32_t, float> &b) { 
                      return a.second > b.second; 
                    });

  std::vector<int> topK;
  LOG(DEBUG) << "Sampler TopK (K=" << _topK << ")";
  for (int i = 0; i < _topK; ++i) {
    topK.push_back(_topBuffer[i].first);
    LOG(DEBUG) << i << ": " <<_topBuffer[i].first << ", " << _topBuffer[i].second;
  }

  return topK;
}

int Sampler::sampleTopP(const ly::Tensor &distribution, lut::Span<const int> topP) {
  CHECK(distribution.getDim() == 1 && distribution.getDType() == DType::kFloat);
  std::vector<float> probAcc;

  float sumP = 0.0f;
  const float *probData = distribution.getData<float>();
  for (int label : topP) {
    float p = probData[label];
    sumP += p;
    probAcc.push_back(sumP);
  }

  float r = _random.nextFloat() * sumP;
  for (int i = 0; i < topP.size(); ++i) {
    if (r < probAcc[i]) {
      return topP[i];
    }
  }
  return topP.back();
}

int Sampler::sample(const Tensor &distribution) {
  CHECK(distribution.getDim() == 1 && distribution.getDType() == DType::kFloat);

  std::vector<int> topK = getTopK(distribution);  // topK is sorted by its prob in x
  std::vector<int> topP = getTopP(distribution, topK);

  return sampleTopP(distribution, topP);
}

}  // namespace libllm
