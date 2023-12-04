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

#include "lyutil/strings.h"

using llyn::Tensor;

namespace F = llyn::functional;

namespace libllm {

GenerationConfig::GenerationConfig() :
    topK(50),
    topP(1.0f),
    temperature(1.0f) {}

Generator::Generator(
    GenerationConfig config,
    std::shared_ptr<ModelForGeneration> model,
    std::shared_ptr<lytok::Tokenizer> tokenizer) :
        _config(config),
        _sampler(config.topK, config.topP),
        _tokenizer(tokenizer),
        _model(model),
        _currentToken(-1) {}

void Generator::setPrompt(const std::string &query) {
  Tensor inputs = _model->buildInput(*_tokenizer, query);
  inputs = F::to(_model->getDevice(), inputs);

  Tensor hiddenState = _model->forward(_past, inputs);

  CHECK(hiddenState.getDim() == 3);
  Tensor x = hiddenState.slice(1, {-1, llyn::None});
  Tensor logits = _model->forwardHidden(x);
  _currentToken = sampleToken(logits);
}

const char *Generator::nextToken() {
  if (stopped()) return nullptr;
  
  const lytok::Vocab *vocab = _tokenizer->getVocab();
  const char *token = vocab->getTokenPiece(_currentToken).c_str();
  LOG(DEBUG) << ly::sprintf("%d -> piece='%s' string='%s'",
                            _currentToken,
                            vocab->getTokenPiece(_currentToken),
                            vocab->getTokenString(_currentToken));

  std::array<llyn::LongType, 1> inputData{_currentToken};
  Tensor inputs = Tensor::create<llyn::LongType>({1, 1}, inputData);
  inputs = F::to(_model->getDevice(), inputs);

  Tensor x = _model->forward(_past, inputs);
  Tensor logits = _model->forwardHidden(x);
  _currentToken = sampleToken(logits);

  return token;
}

bool Generator::stopped() const {
  return _currentToken == _model->getEosId() || _currentToken < 0;
}

int Generator::sampleToken(const llyn::Tensor &logits) {
  CHECK(logits.getDim() == 3 && logits.getShape(0) == 1 && logits.getShape(1) == 1);

  Tensor x = logits.subtensor(0).subtensor(0);
  if (_config.temperature != 1.0f) {
    x = F::mul(x, 1.0f / _config.temperature);
  }
  x = F::softmax(x);
  if (x.getDevice().getType() == llyn::Device::kCuda) {
    x = F::cast(x, llyn::DType::kFloat);
    x = F::to(llyn::Device::kCpu, x);
  }


  return _sampler.sample(x);
}

}  // namespace libllm
