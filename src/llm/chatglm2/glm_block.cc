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

#include "llm/chatglm2/glm_block.h"

#include "lyutil/time.h"
#include "llyn/llyn.h"

using llyn::Context;
using llyn::Tensor;
using llyn::StateMap;
using llyn::nn::RMSNorm;

namespace F = llyn::functional;

namespace libllm {
namespace chatglm2 {
    
std::unique_ptr<GLMBlock> GLMBlock::create(const Context &ctx, ChatGLM2Config config) {
  std::unique_ptr<GLMBlock> layer{new GLMBlock()};

  int hiddenSize = config.hiddenSize;
  float normEps = config.normEps;

  layer->_ctx = ctx;
  layer->_inputNorm = RMSNorm::create(ctx.withName("norm"), hiddenSize, normEps);
  layer->_attnNorm = RMSNorm::create(ctx.withName("attn_norm"), hiddenSize, normEps);
  layer->_attn = SelfAttention::create(ctx.withName("attn"), config);
  layer->_mlp = MLP::create(ctx.withName("mlp"), config);

  return layer;
}

void GLMBlock::initParameters(const StateMap &stateMap) {
  _attn->initParameters(stateMap);
  _inputNorm->initParameters(stateMap);
  _attnNorm->initParameters(stateMap);
  _mlp->initParameters(stateMap);
}

Tensor GLMBlock::forward(StateMap &past, Tensor input, Tensor roPE) const {
  Tensor residual = input;

  // norm+attention
  Tensor x = _inputNorm->forward(input);
  x = _attn->forward(past, x, roPE);

  // residual
  x = F::add(x, residual);
  residual = x;

  // norm+mlp
  x = _attnNorm->forward(x);
  x = _mlp->forward(x);

  // residual
  x = F::add(x, residual);

  return x;
}

}  // namespace chatglm2
}  // namespace libllm
