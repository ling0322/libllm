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

#include "llm/chatglm2/chatglm2_model_for_generation.h"

#include "lyutil/strings.h"
#include "llm/common/constants.h"

using ly::Tensor;
namespace F = ly::functional;


namespace libllm {
namespace chatglm2 {

const char *ChatGLM2ModelForGeneration::_modelName = "chatglm2";

std::shared_ptr<ChatGLM2ModelForGeneration> ChatGLM2ModelForGeneration::create(
    const ly::Context &ctx,
    const lut::IniConfig &config) {
  std::shared_ptr<ChatGLM2ModelForGeneration> model{new ChatGLM2ModelForGeneration()};

  ChatGLM2Config chatglm2Config = ChatGLM2Config::loadConfig(config);
  model->_model = ChatGLM2Model::create(ctx, chatglm2Config);
  model->_config = chatglm2Config;

  // initialize parameters.
  ly::StateMap stateMap;
  lut::Path modelPath = config.getSection(ModelSection).getPath(ModelFileField);
  stateMap.read(modelPath.string());

  model->_model->initParameters(stateMap);
  return model;
}

ly::Tensor ChatGLM2ModelForGeneration::buildInput(
    const lytok::Tokenizer &tokenizer,
    const std::string &query) const {
  std::vector<int> tokenIds = tokenizer.encode(query);
  std::vector<ly::LongType> inputData{_config.symbolGMask, _config.symbolSOP};
  const lytok::Vocab *vocab = tokenizer.getVocab();
  for (int tokenId : tokenIds) {
    LOG(DEBUG) << lut::sprintf("'%s' -> %d", vocab->getTokenString(tokenId), tokenId);
    inputData.push_back(tokenId);
  }

  int len = inputData.size();
  Tensor inputs = Tensor::create<ly::LongType>({1, len}, inputData);
  return inputs;
}

Tensor ChatGLM2ModelForGeneration::forward(ly::StateMap &past, Tensor input) const {
  Tensor x = _model->forward(past, input);
  return x;
}

Tensor ChatGLM2ModelForGeneration::forwardHidden(Tensor hidden) const {
  return _model->forwardHidden(hidden);
}

int ChatGLM2ModelForGeneration::getEosId() const {
  return _config.symbolEOS;
}

const char *ChatGLM2ModelForGeneration::getName() const {
  return _modelName;
}

ly::Device ChatGLM2ModelForGeneration::getDevice() const {
  return _model->getCtx().getDevice();
}

}  // namespace chatglm2
}  // namespace libllm
