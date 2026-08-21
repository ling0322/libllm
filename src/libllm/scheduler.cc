// The MIT License (MIT)
//
// Copyright (c) 2023-2024 Xiaoyang Chen
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

#include "libllm/scheduler.h"

#include <string.h>

#include <algorithm>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "libllm/kv_cache.h"
#include "libllm/forward_batch.h"
#include "libllm/request.h"
#include "flint/functional.h"

namespace libllm {

GenerationConfig::GenerationConfig()
    : topK(50),
      topP(1.0f),
      temperature(1.0f) {
}

// -----------------------------------------------------------------------------------------------+
// class Sampler                                                                                  |
// -----------------------------------------------------------------------------------------------+

Sampler::Sampler(int topK, float topP)
    : _topK(topK),
      _topP(topP) {
}

std::vector<int> Sampler::getTopP(const fl::Tensor &distribution, lut::Span<const int> topK) {
  CHECK(distribution.getDim() == 1 && distribution.getDType() == fl::DType::kFloat);
  float sumP = 0.0f;

  std::vector<int> topP;
  const float *d = distribution.getInternalData()->getData<float>(distribution.getInternalOffset());
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

std::vector<int> Sampler::getTopK(const fl::Tensor &distribution) {
  CHECK(_topK <= distribution.getShape(0) && distribution.getStride(0) == 1);
  if (_topBuffer.size() != distribution.getShape(0)) _topBuffer.resize(distribution.getShape(0));

  const float *d = distribution.getInternalData()->getData<float>(distribution.getInternalOffset());
  for (int32_t i = 0; i < distribution.getShape(0); ++i) {
    _topBuffer[i] = std::make_pair(i, d[i]);
  }

  std::partial_sort(
      _topBuffer.begin(),
      _topBuffer.begin() + _topK,
      _topBuffer.end(),
      [](const std::pair<int32_t, float> &a, const std::pair<int32_t, float> &b) {
        return a.second > b.second;
      });

  std::vector<int> topK;
  LOG(DEBUG) << "Sampler TopK (K=" << _topK << ")";
  for (int i = 0; i < _topK; ++i) {
    topK.push_back(_topBuffer[i].first);
    LOG(DEBUG) << i << ": " << _topBuffer[i].first << ", " << _topBuffer[i].second;
  }

  return topK;
}

int Sampler::sampleTopP(const fl::Tensor &distribution, lut::Span<const int> topP) {
  CHECK(distribution.getDim() == 1 && distribution.getDType() == fl::DType::kFloat);
  std::vector<float> probAcc;

  float sumP = 0.0f;
  const float *probData = distribution.getInternalData()->getData<float>(
      distribution.getInternalOffset());
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

int Sampler::sample(const fl::Tensor &distribution) {
  if (distribution.getDevice().getType() == fl::Device::kCuda) {
    fl::Tensor sampled = fl::F::sample(distribution, _topK, _topP);
    sampled = fl::F::to(fl::Device::getCpu(), sampled);
    CHECK(sampled.getNumEl() == 1 && sampled.getDType() == fl::DType::kLong);
    return static_cast<int>(*sampled.getInternalData()->getData<fl::LongType>(
        sampled.getInternalOffset()));
  }
  CHECK(distribution.getDim() == 1 && distribution.getDType() == fl::DType::kFloat);

  std::vector<int> topK = getTopK(distribution);  // topK is sorted by its prob in x
  std::vector<int> topP = getTopP(distribution, topK);

  return sampleTopP(distribution, topP);
}

// -----------------------------------------------------------------------------------------------+
// class Scheduler                                                                                |
// -----------------------------------------------------------------------------------------------+

Scheduler::Scheduler(const GenerationConfig &config, std::shared_ptr<ModelForGeneration> model)
    : _model(model),
      _currentToken(-1),
      _sampler(config.topK, config.topP),
      _temperature(config.temperature) {
}

std::shared_ptr<Scheduler> Scheduler::create(
    const GenerationConfig &config,
    std::shared_ptr<ModelForGeneration> model) {
  return std::shared_ptr<Scheduler>{new Scheduler(config, model)};
}

bool Scheduler::generate() {
  if (_model->isStopToken(_currentToken)) return false;

  lut::Span<const fl::LongType> tokenIds = _request->getTokenIds();
  int contextLength = _request->getContextLength();
  int numTokensToCompute = static_cast<int>(tokenIds.size()) - contextLength;
  CHECK(numTokensToCompute > 0);

  reserveBlocks(contextLength + numTokensToCompute);

  ForwardBatch batch = ForwardBatch::single(
      tokenIds.subspan(contextLength, numTokensToCompute),
      contextLength);
  batch.setKVCacheManager(_model->getKVCacheManager());
  lut::Span<const int> blockIds = _request->getBlockIds();
  batch.setBlockIds({std::vector<int>(blockIds.begin(), blockIds.end())});

  _currentToken = searchToken(_model->forward(batch));

  _request->advanceComputedTokens(numTokensToCompute);
  _request->setContextLength(contextLength + numTokensToCompute);

  LOG(DEBUG) << lut::sprintf(
      "%d -> \"%s\"",
      _currentToken,
      _model->getVocab()->getTokenString(_currentToken));
  if (_model->isStopToken(_currentToken)) {
    _request->finish();
    releaseBlocks();
    return false;
  }

  _request->appendToken(_currentToken);
  return true;
}

void Scheduler::reserveBlocks(int numTokens) {
  std::shared_ptr<KVCacheManager> manager = _model->getKVCacheManager().lock();
  CHECK(manager) << "model has no KV cache manager";

  int numBlocksHeld = static_cast<int>(_request->getBlockIds().size());
  int numBlocksNeeded = manager->getNumBlocksForTokens(numTokens);
  if (numBlocksNeeded <= numBlocksHeld) return;

  std::vector<int> blockIds = manager->allocateBlocks(numBlocksNeeded - numBlocksHeld);
  if (blockIds.empty()) {
    throw lut::AbortedError("KV cache is full");
  }

  _request->addBlockIds(blockIds);
}

void Scheduler::releaseBlocks() {
  if (_request->getBlockIds().empty()) return;

  std::shared_ptr<KVCacheManager> manager = _model->getKVCacheManager().lock();
  if (manager) manager->freeBlocks(_request->getBlockIds());
  _request->clearBlockIds();
}

Scheduler::~Scheduler() {
  // A caller that stops early still owes the manager its blocks.
  if (_request) releaseBlocks();
}

void Scheduler::setRequest(std::shared_ptr<Request> request) {
  CHECK(request);
  _request = request;
}

std::string Scheduler::getToken() {
  if (_currentToken < 0) return "";

  const Vocab *vocab = _model->getVocab();
  const char *token = vocab->getTokenPiece(_currentToken).c_str();
  return token;
}

std::string Scheduler::getTokenName() {
  if (_currentToken < 0) return "";

  const Vocab *vocab = _model->getVocab();
  const char *token = vocab->getTokenString(_currentToken).c_str();
  return token;
}

int Scheduler::searchToken(const fl::Tensor &logits) {
  CHECK(logits.getDim() == 2 && logits.getShape(0) == 1);

  fl::Tensor x = logits.subtensor(0);
  if (_temperature != 1.0f) {
    x = fl::F::mul(x, 1.0f / _temperature);
  }

  x = fl::F::softmax(x);
  if (x.getDevice().getType() == fl::Device::kCuda) {
    return _sampler.sample(x);
  }
  if (x.getDType() == fl::DType::kFloat16) {
    x = fl::F::cast(x, fl::DType::kFloat);
  }

  return _sampler.sample(x);
}

}  // namespace libllm
