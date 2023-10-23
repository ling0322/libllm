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

#include "llm/llama/decoder_layer.h"

#include "llyn/functional.h"

using llyn::Context;
using llyn::StateMap;
using llyn::Tensor;
using llyn::nn::RMSNorm;

namespace F = llyn::functional;

namespace libllm {
namespace llama {

std::shared_ptr<DecodeLayer> DecodeLayer::create(const Context &ctx, const LlamaConfig &config) {
  std::shared_ptr<DecodeLayer> layer{new DecodeLayer()};
  layer->_ctx = ctx;

  layer->_attn = Attention::create(ctx.withName("attn"), config);
  layer->_mlp = MLP::create(ctx.withName("mlp"), config);
  layer->_inputNorm = RMSNorm::create(
      ctx.withName("input_norm"),
      config.hiddenSize,
      config.normEps);
  layer->_postAttnNorm = RMSNorm::create(
      ctx.withName("post_attn_norm"),
      config.hiddenSize,
      config.normEps);
  
  return layer;
}

void DecodeLayer::initParameters(const llyn::StateMap &stateDict) {
  _attn->initParameters(stateDict);
  _mlp->initParameters(stateDict);
  _inputNorm->initParameters(stateDict);
  _postAttnNorm->initParameters(stateDict);
}

Tensor DecodeLayer::forward(StateMap &past, Tensor input) const {
  Tensor residual = input;
  
  // norm + attn
  Tensor x = _inputNorm->forward(input);
  x = _attn->forward(past, x);
  x = F::add(x, residual);

  // norm + mlp
  residual = x;
  x = _postAttnNorm->forward(x);
  x = _mlp->forward(x);
  x = F::add(x, residual);

  return x;
}

}  // namespace llama
}  // namespace libllm
