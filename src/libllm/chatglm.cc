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

#include "libllm/chatglm.h"

#include "libllm/lut/strings.h"
#include "libllm/constants.h"

namespace libllm {
namespace chatglm {

// -----------------------------------------------------------------------------------------------+
// class ChatGlmConfig                                                                            |
// -----------------------------------------------------------------------------------------------+

ChatGlmConfig::ChatGlmConfig()
    : hiddenSize(0),
      vocabSize(0),
      kvChannels(0),
      seqLength(0),
      hiddenSizePerAttentionHead(0),
      multiQueryGroupNum(0),
      normEps(0.0f),
      numLayers(0),
      symbolGMask(0),
      symbolSOP(0),
      symbolEOS(0) {}

ChatGlmConfig ChatGlmConfig::loadConfig(const lut::IniConfig &ini) {
  const lut::IniSection &section = ini.getSection(kSection);

  ChatGlmConfig config;
  config.hiddenSize = section.getInt("hidden_size");
  config.vocabSize = section.getInt("vocab_size");
  config.kvChannels = section.getInt("kv_channels");
  config.seqLength = section.getInt("seq_length");
  config.hiddenSizePerAttentionHead = section.getInt("hidden_size_per_attention_head");
  config.multiQueryGroupNum = section.getInt("multi_query_group_num");
  config.normEps = section.getFloat("norm_eps");
  config.ffnHiddenSize = section.getInt("ffn_hidden_size");
  config.numLayers = section.getInt("num_layers");
  config.symbolGMask = section.getInt("symbol_gmask");
  config.symbolSOP = section.getInt("symbol_sop");
  config.symbolEOS = section.getInt("symbol_eos");

  return config;
}

// -----------------------------------------------------------------------------------------------+
// class MLP                                                                                      |
// -----------------------------------------------------------------------------------------------+

std::unique_ptr<MLP> MLP::create(const Context &ctx, ChatGlmConfig config) {
  std::unique_ptr<MLP> layer{new MLP()};
  layer->setCtx(ctx);

  layer->_ffnHiddenSize = config.ffnHiddenSize;
  layer->_hiddenSize = config.hiddenSize;
  return layer;
}

void MLP::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  _dense1Weight = stateDict.getTensor(ctx.name("dense1_weight"));
  _dense2Weight = stateDict.getTensor(ctx.name("dense2_weight"));

  _dense1Weight.throwIfInvalidShape({_ffnHiddenSize * 2, _hiddenSize});
  _dense2Weight.throwIfInvalidShape({_hiddenSize, _ffnHiddenSize});

  _dense1Weight = moveAndCastFloat(_dense1Weight, ctx);
  _dense2Weight = moveAndCastFloat(_dense2Weight, ctx);
}

void MLP::initParameters(lut::Random *generator, DType weightType) {
  Device dCpu = Device::getCpu();

  float xs = sqrtf(6.0f / (_ffnHiddenSize * 2 + _hiddenSize));
  _dense1Weight = F::rand({_ffnHiddenSize * 2, _hiddenSize}, weightType, dCpu, generator, -xs, xs);
  _dense1Weight = moveAndCastFloat(_dense1Weight, getCtx());

  xs = sqrtf(6.0f / (_ffnHiddenSize + _hiddenSize));
  _dense2Weight = F::rand({_hiddenSize, _ffnHiddenSize}, weightType, dCpu, generator, -xs, xs);
  _dense2Weight = moveAndCastFloat(_dense2Weight, getCtx());
}

Tensor MLP::forward(const Tensor &input) const {
  CHECK(!_dense1Weight.empty());

  Tensor x = F::matmul(input, _dense1Weight.transpose(0, 1));
  x = F::swiglu(x);
  x = F::matmul(x, _dense2Weight.transpose(0, 1));

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class SelfAttention                                                                            |
// -----------------------------------------------------------------------------------------------+

std::unique_ptr<SelfAttention> SelfAttention::create(
    const Context &ctx,
    ChatGlmConfig config) {
  std::unique_ptr<SelfAttention> layer{new SelfAttention()};
  layer->setCtx(ctx);

  layer->_kvProjDim = config.hiddenSizePerAttentionHead * config.multiQueryGroupNum;
  layer->_qProjDim = config.hiddenSize;
  layer->_hiddenSizePerHead = config.hiddenSizePerAttentionHead;
  layer->_namePastK = ctx.name("k");
  layer->_namePastV = ctx.name("v");
  layer->_namePastLength = ctx.name("len");

  if (config.hiddenSize % config.hiddenSizePerAttentionHead != 0) {
    throw lut::AbortedError("invalid hidden_size and hidden_size_per_head");
  }

  int qkvProjOutDim = layer->_qProjDim + 2 * layer->_kvProjDim;
  layer->_qkvProj = Linear::create(ctx.withName("qkv_proj"), config.hiddenSize, qkvProjOutDim);
  return layer;
}

void SelfAttention::initParameters(const StateMap &stateDict) {
  _qkvProj->initParameters(stateDict);

  int dModel = _qProjDim;
  _denseWeight = stateDict.getTensor(getCtx().name("dense_weight"));
  _denseWeight.throwIfInvalidShape({dModel, dModel});
  _denseWeight = moveAndCastFloat(_denseWeight, getCtx());
}

void SelfAttention::initParameters(lut::Random *g, DType weightType) {
  _qkvProj->initParameters(g, weightType);

  float xs = sqrtf(6.0f / (_qProjDim + _qProjDim));
  _denseWeight = F::rand({_qProjDim, _qProjDim}, weightType, Device::getCpu(), g, -xs, xs);
  _denseWeight = moveAndCastFloat(_denseWeight, getCtx());
}

int SelfAttention::getCtxLength(StateMap *past) const {
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

// -----------------------------------------------------------------------------------------------+
// class GLMBlock                                                                                 |
// -----------------------------------------------------------------------------------------------+

std::unique_ptr<GLMBlock> GLMBlock::create(const Context &ctx, ChatGlmConfig config) {
  std::unique_ptr<GLMBlock> layer{new GLMBlock()};
  layer->setCtx(ctx);

  int hiddenSize = config.hiddenSize;
  float normEps = config.normEps;

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


void GLMBlock::initParameters(lut::Random *generator, DType weightType) {
  _attn->initParameters(generator, weightType);
  _inputNorm->initParameters(generator, weightType);
  _attnNorm->initParameters(generator, weightType);
  _mlp->initParameters(generator, weightType);
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

// -----------------------------------------------------------------------------------------------+
// class ChatGlmModel                                                                             |
// -----------------------------------------------------------------------------------------------+

ChatGlmModel::ChatGlmModel() {}

std::unique_ptr<ChatGlmModel> ChatGlmModel::create(const Context &ctx, ChatGlmConfig c) {
  std::unique_ptr<ChatGlmModel> model{new ChatGlmModel()};
  model->setCtx(ctx);

  model->_config = c;
  model->_embedding = Embedding::create(ctx.withName("embd"), c.hiddenSize, c.vocabSize);
  model->_finalNorm = RMSNorm::create(ctx.withName("final_norm"), c.hiddenSize, c.normEps);
  for (int i = 0; i < c.numLayers; ++i) {
    model->_blocks.emplace_back(
        GLMBlock::create(ctx.withName(lut::sprintf("%s%d", "block", i)), c));
  }

  if (c.kvChannels % 4 != 0) {
    throw lut::AbortedError("invalid kv_channels");
  }

  return model;
}

void ChatGlmModel::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  _embedding->initParameters(stateDict);
  _finalNorm->initParameters(stateDict);

  _rope = stateDict.getTensor(ctx.name("rope"));
  _rope.throwIfInvalidShape({_config.seqLength, _config.kvChannels / 4, 2});
  _rope = _rope.view({_config.seqLength, 1, _config.kvChannels / 2});

  _output = stateDict.getTensor(ctx.name("output_weight"));
  _output.throwIfInvalidShape({_config.vocabSize, _config.hiddenSize});

  for (int i = 0; i < _config.numLayers; ++i) {
    _blocks[i]->initParameters(stateDict);
  }

  _rope = moveAndCastFloat(_rope, ctx);
  _output = moveAndCastFloat(_output, ctx);
}

void ChatGlmModel::initParameters(lut::Random *generator, DType weightType) {
  Context ctx = getCtx();

  _embedding->initParameters(generator, weightType);
  _finalNorm->initParameters(generator, weightType);

  _rope = F::rand({_config.seqLength, 1, _config.kvChannels / 2},
                  DType::kFloat,  // roPE must be float
                  Device::getCpu(),
                  generator);
  _rope = moveAndCastFloat(_rope, ctx);

  float xs = sqrtf(6.0f / (_config.vocabSize + _config.hiddenSize));
  _output = F::rand({_config.vocabSize, _config.hiddenSize},
                     weightType,
                     Device::getCpu(),
                     generator,
                     -xs,
                     xs);
  _output = moveAndCastFloat(_output, ctx);

  for (int i = 0; i < _config.numLayers; ++i) {
    _blocks[i]->initParameters(generator, weightType);
  }
}

Tensor ChatGlmModel::forwardHidden(Tensor hiddenState) const {
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

// -----------------------------------------------------------------------------------------------+
// class ChatGlmModelForGeneration                                                                |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<ChatGlmModelForGeneration> ChatGlmModelForGeneration::create(
    const Context &ctx,
    const lut::IniConfig &config) {
  std::shared_ptr<ChatGlmModelForGeneration> model{new ChatGlmModelForGeneration()};

  ChatGlmConfig ChatGlmConfig = ChatGlmConfig::loadConfig(config);
  model->_model = ChatGlmModel::create(ctx, ChatGlmConfig);
  model->_config = ChatGlmConfig;
  model->_modelName = config.getSection(ModelSection).getString(ModelTypeField);

  // initialize parameters.
  StateMap stateMap;
  lut::Path modelPath = config.getSection(ModelSection).getPath(ModelFileField);
  stateMap.read(modelPath.string());

  model->_model->initParameters(stateMap);
  return model;
}

Tensor ChatGlmModelForGeneration::buildInput(const std::vector<LongType> &prompt) const {
  std::vector<LongType> inputData{_config.symbolGMask, _config.symbolSOP};
  inputData.insert(inputData.end(), prompt.begin(), prompt.end());

  int len = inputData.size();
  Tensor inputs = Tensor::create<LongType>({1, len}, inputData);
  return inputs;
}

Tensor ChatGlmModelForGeneration::forward(StateMap &past, Tensor input) const {
  Tensor x = _model->forward(past, input);
  return x;
}

Tensor ChatGlmModelForGeneration::forwardHidden(Tensor hidden) const {
  return _model->forwardHidden(hidden);
}

bool ChatGlmModelForGeneration::isStopToken(int tokenId) const {
  return tokenId == _config.symbolEOS;
}

const char *ChatGlmModelForGeneration::getName() const {
  return _modelName.c_str();
}

Device ChatGlmModelForGeneration::getDevice() const {
  return _model->getCtx().getDevice();
}

}  // namespace chatglm
}  // namespace libllm
