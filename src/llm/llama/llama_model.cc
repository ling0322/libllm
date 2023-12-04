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

#include "llm/llama/llama_model.h"

#include "lyutil/strings.h"

using llyn::Context;
using llyn::Tensor;
using llyn::StateMap;
using llyn::nn::Embedding;
using llyn::nn::RMSNorm;

namespace F = llyn::functional;

namespace libllm {
namespace llama {

constexpr char LlamaModel::Llama[];
constexpr char LlamaModel::RoPE[];

std::shared_ptr<LlamaModel> LlamaModel::create(const Context &rootCtx, LlamaConfig config) {
  std::shared_ptr<LlamaModel> model{new LlamaModel()};
  Context ctx = rootCtx.withName(Llama);
  model->setCtx(ctx);
  
  model->_config = config;
  model->_embedding = Embedding::create(ctx.withName("embd"), config.hiddenSize, config.vocabSize);
  model->_norm = RMSNorm::create(ctx.withName("norm"), config.hiddenSize, config.normEps);
  for (int i = 0; i < config.numLayers; ++i) {
    model->_layers.emplace_back(
        DecodeLayer::create(ctx.withName(ly::sprintf("block%d", i)), config));
  }
  return model;
}

void LlamaModel::initParameters(const StateMap &stateDict) {
  _embedding->initParameters(stateDict);
  _norm->initParameters(stateDict);

  for (int i = 0; i < _config.numLayers; ++i) {
    _layers[i]->initParameters(stateDict);
  }

  _wOutput = stateDict.getTensor(getCtx().name("out_weight"));
  _wOutput.throwIfInvalidShape({_config.vocabSize, _config.hiddenSize});
  _wOutput = moveAndCastFloat(_wOutput, getCtx());
}

llyn::Tensor LlamaModel::forward(StateMap &past, Tensor input) const {
  Tensor x = _embedding->forward(input);

  for (int i = 0; i < _config.numLayers; ++i) {
    x = _layers[i]->forward(past, x);
  }

  x = _norm->forward(x);
  return x;
}

llyn::Tensor LlamaModel::forwardHidden(llyn::Tensor hidden) const {
  return F::matmul(hidden, _wOutput.transpose(0, 1));
}

}  // namespace llama
}  // namespace libllm
