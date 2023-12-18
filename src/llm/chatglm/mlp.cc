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

#include "llm/chatglm/mlp.h"

#include "ly/ly.h"
#include "lyutil/error.h"

namespace libllm {
namespace chatglm {

namespace F = ly::functional;

using ly::Context;
using ly::Tensor;

std::unique_ptr<MLP> MLP::create(const Context &ctx, ChatGlmConfig config) {
  std::unique_ptr<MLP> layer{new MLP()};
  layer->setCtx(ctx);

  layer->_ffnHiddenSize = config.ffnHiddenSize;
  layer->_hiddenSize = config.hiddenSize;
  return layer;
}

void MLP::initParameters(const ly::StateMap &stateDict) {
  const Context &ctx = getCtx();

  _dense1Weight = stateDict.getTensor(ctx.name("dense1_weight"));
  _dense2Weight = stateDict.getTensor(ctx.name("dense2_weight"));

  _dense1Weight.throwIfInvalidShape({_ffnHiddenSize * 2, _hiddenSize});
  _dense2Weight.throwIfInvalidShape({_hiddenSize, _ffnHiddenSize});

  _dense1Weight = moveAndCastFloat(_dense1Weight, ctx);
  _dense2Weight = moveAndCastFloat(_dense2Weight, ctx);
}

void MLP::initParameters(lut::Random *generator, ly::DType weightType) {
  _dense1Weight = F::rand(
      {_ffnHiddenSize * 2, _hiddenSize}, weightType, ly::Device::getCpu(), generator, -0.1, 0.1);
  _dense1Weight = moveAndCastFloat(_dense1Weight, getCtx());

  _dense2Weight = F::rand(
      {_hiddenSize, _ffnHiddenSize}, weightType, ly::Device::getCpu(), generator, -0.1, 0.1);
  _dense2Weight = moveAndCastFloat(_dense2Weight, getCtx());
}

ly::Tensor MLP::forward(const ly::Tensor &input) const {
  CHECK(!_dense1Weight.empty());

  Tensor x = F::matmul(input, _dense1Weight.transpose(0, 1));
  x = F::swiglu(x);
  x = F::matmul(x, _dense2Weight.transpose(0, 1));

  return x;
}

}  // namespace chatglm
}  // namespace libllm
