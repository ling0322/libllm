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

#include "llm/chatglm/chatglm_model.h"

#include "ly/ly.h"
#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "lytok/lytok.h"

using ly::Context;
using ly::Tensor;
using ly::StateMap;
using ly::nn::Embedding;
using ly::nn::RMSNorm;

namespace F = ly::functional;

namespace libllm {
namespace chatglm {

constexpr char ChatGlmModel::ChatGlm2[];
constexpr char ChatGlmModel::Embd[];
constexpr char ChatGlmModel::RoPE[];
constexpr char ChatGlmModel::Block[];
constexpr char ChatGlmModel::FinalNorm[];
constexpr char ChatGlmModel::OutputWeight[];

ChatGlmModel::ChatGlmModel() {}

std::unique_ptr<ChatGlmModel> ChatGlmModel::create(const Context &rootCtx, ChatGlmConfig config) {
  std::unique_ptr<ChatGlmModel> model{new ChatGlmModel()};
  Context ctx = rootCtx.withName(ChatGlm2);
  model->setCtx(ctx);

  model->_config = config;
  model->_embedding = Embedding::create(ctx.withName(Embd), config.hiddenSize, config.vocabSize);
  model->_finalNorm = RMSNorm::create(ctx.withName(FinalNorm), config.hiddenSize, config.normEps);
  for (int i = 0; i < config.numLayers; ++i) {
    model->_blocks.emplace_back(
        GLMBlock::create(ctx.withName(lut::sprintf("%s%d", Block, i)), config));
  }

  if (config.kvChannels % 4 != 0) {
    throw lut::AbortedError("invalid kv_channels");
  }

  return model;
}

void ChatGlmModel::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  _embedding->initParameters(stateDict);
  _finalNorm->initParameters(stateDict);

  _rope = stateDict.getTensor(ctx.name(RoPE));
  _rope.throwIfInvalidShape({_config.seqLength, _config.kvChannels / 4, 2});
  _rope = _rope.view({_config.seqLength, 1, _config.kvChannels / 2});

  _output = stateDict.getTensor(ctx.name(OutputWeight));
  _output.throwIfInvalidShape({_config.vocabSize, _config.hiddenSize});

  for (int i = 0; i < _config.numLayers; ++i) {
    _blocks[i]->initParameters(stateDict);
  }

  _rope = moveAndCastFloat(_rope, ctx);
  _output = moveAndCastFloat(_output, ctx);
}

ly::Tensor ChatGlmModel::forwardHidden(ly::Tensor hiddenState) const {
  return F::matmul(hiddenState, _output.transpose(0, 1));
}

Tensor ChatGlmModel::forward(StateMap &past, Tensor input) const {
  Tensor x = _embedding->forward(input);
  for (int i = 0; i < _config.numLayers; ++i) {
    x = _blocks[i]->forward(past, x, _rope);
  }
  x = _finalNorm->forward(x);

  return x;
}

}  // namespace chatglm
}
