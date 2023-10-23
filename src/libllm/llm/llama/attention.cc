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

#include "llm/llama/attention.h"

#include "lyutil/error.h"
#include "llm/llama/llama_model.h"

using llyn::Context;
using llyn::Tensor;
using llyn::StateMap;

namespace F = llyn::functional;

namespace libllm {
namespace llama {

Attention::Attention() : 
    _hiddenSize(0),
    _numHead(0),
    _headDim(0),
    _maxCtxLen(0) {}

std::shared_ptr<Attention> Attention::create(const Context &ctx, const LlamaConfig &config) {
  std::shared_ptr<Attention> layer{new Attention()};

  if (config.hiddenSize % config.numHeads != 0)
    throw ly::AbortedError("invalid hidden_size and num_heads");
  
  layer->_ctx = ctx;
  layer->_hiddenSize = config.hiddenSize;
  layer->_numHead = config.numHeads;
  layer->_headDim = config.hiddenSize / config.numHeads;
  layer->_maxCtxLen = config.maxContextLength;

  layer->_namePastK = ctx.name("k");
  layer->_namePastV = ctx.name("v");
  layer->_namePastLen = ctx.name("len");

  return layer;
}

void Attention::initParameters(const llyn::StateMap &stateDict) {
  _qkvProj = stateDict.getTensor(_ctx.name("qkv_proj"));
  _outProj = stateDict.getTensor(_ctx.name("out_proj"));
  _roPE = stateDict.getTensor(Context::joinName(LlamaModel::Llama, LlamaModel::RoPE));

  _qkvProj.throwIfInvalidShape({_hiddenSize * 3, _hiddenSize});
  _outProj.throwIfInvalidShape({_hiddenSize, _hiddenSize});
  _roPE.throwIfInvalidShape({_maxCtxLen, _headDim});
}

int Attention::getCtxLength(const llyn::StateMap &past) const {
  if (past.hasValue<int>(_namePastLen)) {
    return past.getValue<int>(_namePastLen);
  } else {
    return 0;
  }
}

llyn::Tensor Attention::forward(llyn::StateMap &past, llyn::Tensor input) const {
  CHECK(input.getDim() == 3);

  Tensor qkv = F::matmul(input, _qkvProj.transpose(0, 1));
  
  Tensor q = qkv.slice(-1, {0, _hiddenSize});
  Tensor k = qkv.slice(-1, {_hiddenSize, 2 * _hiddenSize});
  Tensor v = qkv.slice(-1, {2 * _hiddenSize, 3 * _hiddenSize});

  int N = qkv.getShape(0);
  int qLen = qkv.getShape(1);
  int kvLen = qLen + getCtxLength(past);

  past.putValue<int>(_namePastLen, kvLen);

  q = q.view({N, qLen, _numHead, _headDim});
  k = k.view({N, qLen, _numHead, _headDim});
  v = v.view({N, qLen, _numHead, _headDim});
  Tensor roPE = _roPE.slice({kvLen - qLen, kvLen});

  q = F::applRotaryPosEmbd(q, roPE);
  k = F::applRotaryPosEmbd(k, roPE);

  // concat past for k and v.
  if (past.hasTensor(_namePastK) && past.hasTensor(_namePastV)) {
    k = F::cat(past.getTensor(_namePastK), k, 1);
    v = F::cat(past.getTensor(_namePastV), v, 1);
    
    CHECK(k.getShape(1) == v.getShape(1) && k.getShape(1) == kvLen);
  }

  // update past.
  past.putTensor(_namePastK, k);
  past.putTensor(_namePastV, v);

  // apply attention.
  // TODO: streaming mode support.
  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);
  Tensor x = qLen == 1 ? F::attention(q, k, v)
                       : F::attention(q, k, v, F::causalMask(q.getShape(2)));

  x = F::contiguous(x.transpose(1, 2)).view({N, qLen, _hiddenSize});
  x = F::matmul(x, _outProj.transpose(0, 1));

  return x;
}

}  // namespace llama
}  // namespace libllm
