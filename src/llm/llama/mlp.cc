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

#include "llm/llama/mlp.h"

#include "ly/functional.h"

using ly::Context;
using ly::Tensor;
using ly::StateMap;

namespace F = ly::functional;

namespace libllm {
namespace llama {

MLP::MLP() : _hiddenSize(0), _intermediateSize(0) {}

std::shared_ptr<MLP> MLP::create(const Context &ctx, const LlamaConfig &config) {
  std::shared_ptr<MLP> mlp{new MLP()};
  mlp->setCtx(ctx);

  mlp->_hiddenSize = config.hiddenSize;
  mlp->_intermediateSize = config.intermediateSize;
  
  return mlp;
}

void MLP::initParameters(const StateMap &stateDict) {
  _wGateUpProj = stateDict.getTensor(getCtx().name("gate_up_proj"));
  _wDownProj = stateDict.getTensor(getCtx().name("down_proj"));

  _wGateUpProj.throwIfInvalidShape({_intermediateSize * 2, _hiddenSize});
  _wDownProj.throwIfInvalidShape({_hiddenSize, _intermediateSize});

  _wGateUpProj = moveAndCastFloat(_wGateUpProj, getCtx());
  _wDownProj = moveAndCastFloat(_wDownProj, getCtx());
}

Tensor MLP::forward(Tensor input) const {
  CHECK(!_wGateUpProj.empty());

  Tensor x = F::matmul(input, _wGateUpProj.transpose(0, 1));
  x = F::swiglu(x);
  x = F::matmul(x, _wDownProj.transpose(0, 1));

  return x;
}

}  // namespace llama
}  // namespace libllm

