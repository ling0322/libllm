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

#include "libllm/generator.h"

#include <string.h>

#include <algorithm>

#include "libllm/functional.h"
#include "libllm/whisper.h"
#include "lut/error.h"
#include "lut/strings.h"

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

std::vector<int> Sampler::getTopP(const Tensor &distribution, lut::Span<const int> topK) {
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
  if (_topBuffer.size() != distribution.getShape(0)) _topBuffer.resize(distribution.getShape(0));

  const float *d = distribution.getData<float>();
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

int Sampler::sampleTopP(const Tensor &distribution, lut::Span<const int> topP) {
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

// -----------------------------------------------------------------------------------------------+
// class BaseGenerator                                                                            |
// -----------------------------------------------------------------------------------------------+

BaseGenerator::BaseGenerator(std::shared_ptr<ModelForGeneration> model)
    : _model(model),
      _currentToken(-1) {
}

bool BaseGenerator::generate() {
  if (_model->isStopToken(_currentToken)) return false;

  if (_currentToken >= 0) {
    _currentToken = searchToken(_model->decode(_past, _currentToken));
  } else {
    _currentToken = searchToken(_model->prefill(_past, _prompt));
  }

  LOG(DEBUG) << lut::sprintf(
      "%d -> \"%s\"",
      _currentToken,
      _model->getVocab()->getTokenString(_currentToken));
  if (_model->isStopToken(_currentToken)) return false;

  return true;
}

void BaseGenerator::setPrompt(const Prompt &prompt) {
  _prompt = prompt;
}

std::string BaseGenerator::getToken() {
  if (_currentToken < 0) return "";

  const Vocab *vocab = _model->getVocab();
  const char *token = vocab->getTokenPiece(_currentToken).c_str();
  return token;
}

std::string BaseGenerator::getTokenName() {
  if (_currentToken < 0) return "";

  const Vocab *vocab = _model->getVocab();
  const char *token = vocab->getTokenString(_currentToken).c_str();
  return token;
}

// -----------------------------------------------------------------------------------------------+
// class SamplingGenerator                                                                        |
// -----------------------------------------------------------------------------------------------+

SamplingGenerator::SamplingGenerator(
    const GenerationConfig &config,
    std::shared_ptr<ModelForGeneration> model)
    : BaseGenerator(model),
      _sampler(config.topK, config.topP),
      _temperature(config.temperature) {
}

std::shared_ptr<SamplingGenerator> SamplingGenerator::newGenerator(
    const GenerationConfig &config,
    std::shared_ptr<ModelForGeneration> model) {
  std::shared_ptr<SamplingGenerator> generator{new SamplingGenerator(config, model)};
  return generator;
}

int SamplingGenerator::searchToken(const Tensor &logits) {
  CHECK(logits.getDim() == 3 && logits.getShape(0) == 1 && logits.getShape(1) == 1);

  Tensor x = logits.subtensor(0).subtensor(0);
  if (_temperature != 1.0f) {
    x = F::mul(x, 1.0f / _temperature);
  }

  x = F::softmax(x);
  if (x.getDType() == DType::kFloat16) {
    x = F::cast(x, DType::kFloat);
  }
  if (x.getDevice().getType() == Device::kCuda) {
    x = F::to(Device::kCpu, x);
  }

  return _sampler.sample(x);
}

// -----------------------------------------------------------------------------------------------+
// class WhisperGreedyGenerator                                                                   |
// -----------------------------------------------------------------------------------------------+

WhisperGreedyGenerator::WhisperGreedyGenerator(
    const GenerationConfig &config,
    std::shared_ptr<ModelForGeneration> model)
    : BaseGenerator(model),
      _temperature(config.temperature) {
}

std::shared_ptr<WhisperGreedyGenerator> WhisperGreedyGenerator::newGenerator(
    const GenerationConfig &config,
    std::shared_ptr<ModelForGeneration> model) {
  std::shared_ptr<WhisperGreedyGenerator> generator{new WhisperGreedyGenerator(config, model)};
  std::string modelName = model->getName();
  if (modelName.find("whisper") == std::string::npos) {
    throw lut::AbortedError("use WhisperGreedyGenerator for a non-whipser model");
  }

  generator->_whisperLogitsProcessor = whisper::WhisperLogitsProcessor::newProcessor(
      model->getVocab());

  return generator;
}

void WhisperGreedyGenerator::setPrompt(const Prompt &prompt) {
  CHECK(!prompt.empty());
  const PromptBlock &lastBlock = prompt.getBlocks().back();
  if (lastBlock.blockType != PromptBlock::ControlToken ||
      (lastBlock.text != "<|startoftranscript|>" && lastBlock.text != "<|transcript|>" &&
       lastBlock.text != "<|translate|>" && lastBlock.text != "<|notimestamps|>")) {
    throw lut::AbortedError(
        "last token of prompt for whisper should be one of <|startoftranscript|>, <|transcript|>, "
        "<|translate|> or <|notimestamps|>");
  }

  _prompt = prompt;
}

int WhisperGreedyGenerator::searchToken(const Tensor &logits) {
  CHECK(logits.getDim() == 3 && logits.getShape(0) == 1 && logits.getShape(1) == 1);

  Tensor x = logits.subtensor(0).subtensor(0);
  if (_temperature != 1.0f) {
    x = F::mul(x, 1.0f / _temperature);
  }

  _whisperLogitsProcessor->processLogits(x);

  x = F::softmax(x);
  if (x.getDType() == DType::kFloat16) {
    x = F::cast(x, DType::kFloat);
  }
  if (x.getDevice().getType() == Device::kCuda) {
    x = F::to(Device::kCpu, x);
  }

  // repetition penalty
  for (int i = 0; i < 20; ++i) CHECK(x.getDim() == 1 && x.getStride(0) == 1);
  const float *data = x.getData<float>();
  const float *best = std::max_element(data, data + x.getShape(0));

  int tokenId = static_cast<int>(best - data);
  _whisperLogitsProcessor->notifyToken(tokenId);
  _history.push_back(tokenId);
  return tokenId;
}

}  // namespace libllm
