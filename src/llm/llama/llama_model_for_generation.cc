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

#include "llm/llama/llama_model_for_generation.h"

#include "llm/common/constants.h"

using lut::IniConfig;
using lut::IniSection;
using ly::Tensor;

namespace F = ly::functional;

namespace libllm {
namespace llama {

const char *LlamaModelForGeneration::_modelName = "llama";

LlamaModelForGeneration::LlamaModelForGeneration() : _bosId(0), _eosId(0) {}

std::shared_ptr<LlamaModelForGeneration> LlamaModelForGeneration::create(
    const ly::Context &ctx,
    const IniConfig &config) {
  std::shared_ptr<LlamaModelForGeneration> model{new LlamaModelForGeneration()};

  // create model
  LlamaConfig llamaConfig = LlamaConfig::loadConfig(config);
  model->_model = LlamaModel::create(ctx, llamaConfig);

  // initialize parameters
  const IniSection &modelSection = config.getSection(ModelSection);
  lut::Path modelPath = modelSection.getPath(ModelFileField);

  ly::StateMap stateMap;
  stateMap.read(modelPath.string());
  model->_model->initParameters(stateMap);

  // get EOS token
  const IniSection &llamaSection = config.getSection(Llama2Section);
  model->_eosId = llamaSection.getInt("eos_token_id");
  model->_bosId = llamaSection.getInt("bos_token_id");

  return model;
}

Tensor LlamaModelForGeneration::forward(ly::StateMap &past, Tensor input) const {
  Tensor x = _model->forward(past, input);
  return x;
}

Tensor LlamaModelForGeneration::forwardHidden(Tensor hidden) const {
  return _model->forwardHidden(hidden);
}

Tensor LlamaModelForGeneration::buildInput(const std::vector<ly::LongType> &prompt) const {
  std::vector<ly::LongType> inputData{};
  inputData.insert(inputData.end(), prompt.begin(), prompt.end());

  int len = inputData.size();
  Tensor inputs = Tensor::create<ly::LongType>({1, len}, inputData);
  return inputs;
}

int LlamaModelForGeneration::getEosId() const {
  return _eosId;
}

const char *LlamaModelForGeneration::getName() const {
  return _modelName;
}

ly::Device LlamaModelForGeneration::getDevice() const {
  return _model->getCtx().getDevice();
}

}  // namespace llama
}  // namespace libllm
