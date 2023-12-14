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

#include "llm/chatglm/chatglm_model_for_generation.h"

#include "lyutil/strings.h"
#include "llm/common/constants.h"

using ly::Tensor;
namespace F = ly::functional;


namespace libllm {
namespace chatglm {


std::shared_ptr<ChatGlmModelForGeneration> ChatGlmModelForGeneration::create(
    const ly::Context &ctx,
    const lut::IniConfig &config) {
  std::shared_ptr<ChatGlmModelForGeneration> model{new ChatGlmModelForGeneration()};

  ChatGlmConfig ChatGlmConfig = ChatGlmConfig::loadConfig(config);
  model->_model = ChatGlmModel::create(ctx, ChatGlmConfig);
  model->_config = ChatGlmConfig;
  model->_modelName = config.getSection(ModelSection).getString(ModelTypeField);

  // initialize parameters.
  ly::StateMap stateMap;
  lut::Path modelPath = config.getSection(ModelSection).getPath(ModelFileField);
  stateMap.read(modelPath.string());

  model->_model->initParameters(stateMap);
  return model;
}

ly::Tensor ChatGlmModelForGeneration::buildInput(const std::vector<ly::LongType> &prompt) const {
  std::vector<ly::LongType> inputData{_config.symbolGMask, _config.symbolSOP};
  inputData.insert(inputData.end(), prompt.begin(), prompt.end());

  int len = inputData.size();
  Tensor inputs = Tensor::create<ly::LongType>({1, len}, inputData);
  return inputs;
}

Tensor ChatGlmModelForGeneration::forward(ly::StateMap &past, Tensor input) const {
  Tensor x = _model->forward(past, input);
  return x;
}

Tensor ChatGlmModelForGeneration::forwardHidden(Tensor hidden) const {
  return _model->forwardHidden(hidden);
}

int ChatGlmModelForGeneration::getEosId() const {
  return _config.symbolEOS;
}

const char *ChatGlmModelForGeneration::getName() const {
  return _modelName.c_str();
}

ly::Device ChatGlmModelForGeneration::getDevice() const {
  return _model->getCtx().getDevice();
}

}  // namespace chatglm
}  // namespace libllm
