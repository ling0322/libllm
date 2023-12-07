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

#include "llm/chatglm2/self_attention.h"

#include "llyn/llyn.h"
#include "lyutil/error.h"
#include "lyutil/time.h"

using llyn::StateMap;
using llyn::Tensor;
using llyn::nn::Linear;

namespace F = llyn::functional;

namespace libllm {
namespace chatglm2 {

std::unique_ptr<SelfAttention> SelfAttention::create(
    const llyn::Context &ctx,
    ChatGLM2Config config) {
  std::unique_ptr<SelfAttention> layer{new SelfAttention()};
  layer->setCtx(ctx);

  layer->_kvProjDim = config.hiddenSizePerAttentionHead * config.multiQueryGroupNum;
  layer->_qProjDim = config.hiddenSize;
  layer->_hiddenSizePerHead = config.hiddenSizePerAttentionHead;
  layer->_namePastK = ctx.name("k");
  layer->_namePastV = ctx.name("v");
  layer->_namePastLength = ctx.name("len");

  if (config.hiddenSize % config.hiddenSizePerAttentionHead != 0) {
    throw ly::AbortedError("invalid hidden_size and hidden_size_per_head");
  }

  int qkvProjOutDim = layer->_qProjDim + 2 * layer->_kvProjDim;
  layer->_qkvProj = Linear::create(ctx.withName("qkv_proj"), config.hiddenSize, qkvProjOutDim);
  return layer;
}

void SelfAttention::initParameters(const llyn::StateMap &stateDict) {
  _qkvProj->initParameters(stateDict);

  int dModel = _qProjDim;
  _denseWeight = stateDict.getTensor(getCtx().name("dense_weight"));
  _denseWeight.throwIfInvalidShape({dModel, dModel});
  _denseWeight = moveAndCastFloat(_denseWeight, getCtx());
}

int SelfAttention::getCtxLength(llyn::StateMap *past) const {
  if (past && past->hasValue<int>(_namePastLength)) {
    return past->getValue<int>(_namePastLength);
  } else {
    return 0;
  }
}

Tensor SelfAttention::forward(StateMap &past, Tensor input, Tensor roPE) const {
  Tensor qkvProj = _qkvProj->forward(input);
  
  CHECK(qkvProj.getDim() == 3 && qkvProj.getShape(-1) == _kvProjDim * 2 + _qProjDim);
  Tensor qProj = qkvProj.slice(-1, {0, _qProjDim});
  Tensor kProj = qkvProj.slice(-1, {_qProjDim, _qProjDim + _kvProjDim});
  Tensor vProj = qkvProj.slice(-1, {_qProjDim + _kvProjDim, _qProjDim + 2 * _kvProjDim});

  int N = input.getShape(0);  // batch size
  int qL = input.getShape(1);  // sequence length
  int qNH = _qProjDim / _hiddenSizePerHead;  // query num-heads
  int kvNH = _kvProjDim / _hiddenSizePerHead;  // key and value num-heads
  int D = _hiddenSizePerHead; 

  Tensor q = qProj.view({N, qL, qNH, D});
  Tensor k = kProj.view({N, qL, kvNH, D});
  Tensor v = vProj.view({N, qL, kvNH, D});

  // apply roPE to [..., :_hiddenSizePerHead / 2] of QKV 
  Tensor qe = q.slice(-1, {0, D / 2});
  Tensor ke = k.slice(-1, {0, D / 2});
  Tensor ve = v.slice(-1, {0, D / 2});

  // fetch and update past length.
  // TODO: check kvL length oveflow.
  int kvL = qL;
  if (past.hasValue<int>(_namePastLength)) {
    kvL += past.getValue<int>(_namePastLength);
  }
  past.putValue<int>(_namePastLength, kvL);

  // apply rope.
  Tensor qkRoPE = roPE.slice({kvL - qL, kvL});
  F::copy(F::applyRotaryPosEmb(qe, qkRoPE), qe);
  F::copy(F::applyRotaryPosEmb(ke, qkRoPE), ke);

  // fetch and update past k.
  if (past.hasTensor(_namePastK) && past.hasTensor(_namePastV)) {
    const Tensor &pastK = past.getTensor(_namePastK);
    const Tensor &pastV = past.getTensor(_namePastV);

    k = F::cat(pastK, k, 1);
    v = F::cat(pastV, v, 1);
    
    CHECK(k.getShape(1) == v.getShape(1) && k.getShape(1) == kvL);
  }

  // update kv_cache in past.
  past.putTensor(_namePastK, k);
  past.putTensor(_namePastV, v);

  // expand KV
  CHECK(qNH % kvNH == 0);
  std::initializer_list<int> expandShape = {N, kvL, kvNH, qNH / kvNH, D};
  std::initializer_list<int> qShape = {N, kvL, qNH, D};
  k = F::contiguous(k.unsqueeze(3).expand(expandShape)).view(qShape);
  v = F::contiguous(v.unsqueeze(3).expand(expandShape)).view(qShape);

  // apply attention.
  // TODO: streaming mode support.
  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);
  Tensor x = qL == 1 ? F::attention(q, k, v)
                     : F::attention(q, k, v, F::causalMask(q.getShape(2), getCtx().getDevice()));

  x = F::contiguous(x.transpose(1, 2)).view({N, qL, qNH * D});
  x = F::matmul(x, _denseWeight.transpose(0, 1));

  return x;
}

}  // namespace chatglm2
}  // namespace libllm
