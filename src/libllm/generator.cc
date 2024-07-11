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

#include "libllm/functional.h"
#include "libllm/lut/strings.h"

namespace libllm {

GenerationConfig::GenerationConfig()
    : topK(50),
      topP(1.0f),
      temperature(1.0f) {
}

Generator::Generator(const GenerationConfig &config, std::shared_ptr<ModelForGeneration> model)
    : _sampler(config.topK, config.topP),
      _model(model),
      _currentToken(-1),
      _temperature(config.temperature) {
}

std::shared_ptr<Generator> Generator::newGenerator(
    const GenerationConfig &config,
    std::shared_ptr<ModelForGeneration> model) {
  std::shared_ptr<Generator> generator{new Generator(config, model)};
  Device device = model->getDevice();

  if (!config.supressedTokens.empty()) {
    std::vector<float> supressTensor(model->getOutputDim());
    std::fill(supressTensor.begin(), supressTensor.end(), 0.0f);
    for (int tokenId : config.supressedTokens) {
      supressTensor[tokenId] = -std::numeric_limits<float>::infinity();
    }

    Tensor supress = Tensor::create({model->getOutputDim()}, lut::makeConstSpan(supressTensor));
    supress = F::to(model->getDevice(), supress);
    supress = F::cast(supress, F::getDefaultFloatType(device));
    generator->_supress = supress;
  }

  return generator;
}

void Generator::forwardPrompt(const Prompt &prompt) {
  _currentToken = sampleToken(_model->prefill(_past, prompt));
}

const char *Generator::nextToken() {
  if (stopped()) return nullptr;

  const Vocab *vocab = _model->getVocab();
  const char *token = vocab->getTokenPiece(_currentToken).c_str();
  LOG(INFO) << lut::sprintf("%d -> \"%s\"", _currentToken, vocab->getTokenString(_currentToken));

  _currentToken = sampleToken(_model->decode(_past, _currentToken));
  return token;
}

bool Generator::stopped() const {
  return _model->isStopToken(_currentToken) || _currentToken < 0;
}

int Generator::sampleToken(const Tensor &logits) {
  CHECK(logits.getDim() == 3 && logits.getShape(0) == 1 && logits.getShape(1) == 1);

  Tensor x = logits.subtensor(0).subtensor(0);
  if (_temperature != 1.0f) {
    x = F::mul(x, 1.0f / _temperature);
  }
  if (!_supress.empty()) {
    x = F::add(x, _supress);
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

}  // namespace libllm
