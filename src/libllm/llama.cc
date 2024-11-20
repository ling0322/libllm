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

#include "libllm/llama.h"

#include <math.h>

#include <memory>

#include "libllm/constants.h"
#include "libllm/functional.h"
#include "libllm/model_for_generation.h"
#include "libllm/module.h"
#include "lutil/error.h"
#include "lutil/ini_config.h"
#include "lutil/strings.h"

namespace libllm {
namespace llama {

constexpr char LlamaModel::RoPECtxKey[];

// -----------------------------------------------------------------------------------------------+
// class LlamaConfig                                                                              |
// -----------------------------------------------------------------------------------------------+

LlamaConfig::LlamaConfig()
    : hiddenSize(0),
      numHeads(0),
      numKeyValueHeads(0),
      intermediateSize(0),
      normEps(0.0f),
      numLayers(0),
      vocabSize(0),
      maxContextLength(0),
      qkvProjBias(false) {
}

LlamaConfig LlamaConfig::loadConfig(const lut::IniSection &section) {
  LlamaConfig config;

  config.hiddenSize = section.getInt("hidden_size");
  config.numHeads = section.getInt("num_heads");
  config.intermediateSize = section.getInt("intermediate_size");
  config.normEps = section.getFloat("norm_eps");
  config.numLayers = section.getInt("num_layers");
  config.vocabSize = section.getInt("vocab_size");
  config.maxContextLength = section.getInt("max_ctx_length");

  if (section.hasKey("num_key_value_heads")) {
    config.numKeyValueHeads = section.getInt("num_key_value_heads");
  } else {
    config.numKeyValueHeads = config.numHeads;
  }

  if (section.hasKey("qkv_proj_bias")) {
    config.qkvProjBias = section.getBool("qkv_proj_bias");
  }

  return config;
}

// -----------------------------------------------------------------------------------------------+
// class MLP                                                                                      |
// -----------------------------------------------------------------------------------------------+

MLP::MLP()
    : _hiddenSize(0),
      _intermediateSize(0) {
}

std::shared_ptr<MLP> MLP::create(const Context &ctx, const LlamaConfig &config) {
  std::shared_ptr<MLP> mlp{new MLP()};
  mlp->setCtx(ctx);

  mlp->_hiddenSize = config.hiddenSize;
  mlp->_intermediateSize = config.intermediateSize;

  int d = config.hiddenSize;
  int di = config.intermediateSize;
  mlp->_gateUpProj = Linear::create(ctx.withName("gate_up_proj"), d, di * 2, false);
  mlp->_downProj = Linear::create(ctx.withName("down_proj"), di, d, false);

  return mlp;
}

void MLP::initParameters(const StateMap &stateDict) {
  _gateUpProj->initParameters(stateDict);
  _downProj->initParameters(stateDict);
}

void MLP::initParameters(lut::Random *generator, DType weightType) {
  _gateUpProj->initParameters(generator, weightType);
  _downProj->initParameters(generator, weightType);
}

Tensor MLP::forward(Tensor input) const {
  Tensor x = _gateUpProj->forward(input);
  x = F::swiglu(x);
  x = _downProj->forward(x);

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class Attention                                                                                |
// -----------------------------------------------------------------------------------------------+

Attention::Attention()
    : _hiddenSize(0),
      _numHead(0),
      _numKeyValueHead(0),
      _headDim(0),
      _maxCtxLen(0),
      _hasProjBias(false) {
}

std::shared_ptr<Attention> Attention::create(const Context &ctx, const LlamaConfig &config) {
  std::shared_ptr<Attention> layer{new Attention()};
  layer->setCtx(ctx);

  if (config.hiddenSize % config.numHeads != 0)
    throw lut::AbortedError("invalid hidden_size and num_heads");

  int headDim = config.hiddenSize / config.numHeads;
  layer->_hiddenSize = config.hiddenSize;
  layer->_numHead = config.numHeads;
  layer->_numKeyValueHead = config.numKeyValueHeads;
  layer->_headDim = headDim;
  layer->_maxCtxLen = config.maxContextLength;
  layer->_hasProjBias = config.qkvProjBias;

  layer->_namePastK = ctx.name("k");
  layer->_namePastV = ctx.name("v");
  layer->_namePastLen = ctx.name("len");

  int qkvProjDim = headDim * config.numKeyValueHeads * 2 + config.hiddenSize;
  int d = config.hiddenSize;
  layer->_qkvProj = Linear::create(ctx.withName("qkv_proj"), d, qkvProjDim, config.qkvProjBias);
  layer->_outProj = Linear::create(ctx.withName("out_proj"), d, d, false);

  return layer;
}

void Attention::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  _qkvProj->initParameters(stateDict);
  _outProj->initParameters(stateDict);

  _roPE = stateDict.getTensor(ctx.get(LlamaModel::RoPECtxKey));
  _roPE = _roPE.view({2, _maxCtxLen, 1, _headDim});
  _roPE = moveAndCastFloat(_roPE, ctx);
}

void Attention::initParameters(lut::Random *generator, DType weightType) {
  _qkvProj->initParameters(generator, weightType);
  _outProj->initParameters(generator, weightType);

  Device dCpu = Device::getCpu();
  float r = 0.2f;
  _roPE = F::rand({2, _maxCtxLen, 1, _headDim}, DType::kFloat, dCpu, generator, -r, r);
  _roPE = moveAndCastFloat(_roPE, getCtx());
}

int Attention::getCtxLength(const StateMap &past) const {
  if (past.hasValue<int>(_namePastLen)) {
    return past.getValue<int>(_namePastLen);
  } else {
    return 0;
  }
}

Tensor Attention::rotateHalf(Tensor x) const {
  Tensor rotated = F::tensorLike(x);
  int lastDim = x.getDim() - 1;
  int halfShape = x.getShape(lastDim) / 2;

  Tensor x1 = x.slice(lastDim, {0, halfShape});
  Tensor x2 = x.slice(lastDim, {halfShape, None});
  x2 = F::mul(x2, -1.0f);

  F::copy(x1, rotated.slice(lastDim, {halfShape, None}));
  F::copy(x2, rotated.slice(lastDim, {0, halfShape}));

  return rotated;
}

Tensor Attention::applyRoPE(Tensor input, Tensor roPE) const {
  Tensor cos = roPE.subtensor(0);
  Tensor sin = roPE.subtensor(1);

  cos = cos.expand({cos.getShape(0), input.getShape(2), cos.getShape(2)});
  sin = sin.expand({sin.getShape(0), input.getShape(2), sin.getShape(2)});

  return F::add(F::mul(input, F::contiguous(cos)), F::mul(rotateHalf(input), F::contiguous(sin)));
}

Tensor Attention::forward(StateMap &past, Tensor input) const {
  CHECK(input.getDim() == 3);
  Tensor qkv = _qkvProj->forward(input);

  int kvHiddenSize = _headDim * _numKeyValueHead;

  Tensor q = qkv.slice(-1, {0, _hiddenSize});
  Tensor k = qkv.slice(-1, {_hiddenSize, _hiddenSize + kvHiddenSize});
  Tensor v = qkv.slice(-1, {_hiddenSize + kvHiddenSize, _hiddenSize + 2 * kvHiddenSize});

  int N = qkv.getShape(0);
  int qLen = qkv.getShape(1);
  int kvLen = qLen + getCtxLength(past);

  past.putValue<int>(_namePastLen, kvLen);

  q = q.view({N, qLen, _numHead, _headDim});
  k = k.view({N, qLen, _numKeyValueHead, _headDim});
  v = v.view({N, qLen, _numKeyValueHead, _headDim});
  Tensor roPE = _roPE.slice(1, {kvLen - qLen, kvLen});

  q = applyRoPE(q, roPE);
  k = applyRoPE(k, roPE);

  // concat past for k and v.
  if (past.hasTensor(_namePastK) && past.hasTensor(_namePastV)) {
    k = F::cat(past.getTensor(_namePastK), k, 1);
    v = F::cat(past.getTensor(_namePastV), v, 1);

    CHECK(k.getShape(1) == v.getShape(1) && k.getShape(1) == kvLen);
  }

  // update past.
  past.putTensor(_namePastK, k);
  past.putTensor(_namePastV, v);

  // apply GQA
  if (_numKeyValueHead != _numHead) {
    CHECK(_numHead % _numKeyValueHead == 0);

    int groupSize = _numHead / _numKeyValueHead;
    std::initializer_list<int> expandShape = {N, kvLen, _numKeyValueHead, groupSize, _headDim};
    std::initializer_list<int> qShape = {N, kvLen, _numHead, _headDim};

    k = F::contiguous(k.unsqueeze(3).expand(expandShape)).view(qShape);
    v = F::contiguous(v.unsqueeze(3).expand(expandShape)).view(qShape);
  }

  // apply attention.
  // TODO: streaming mode support.
  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);
  Tensor x = qLen == 1 ? F::attention(q, k, v)
                       : F::attention(q, k, v, F::causalMask(q.getShape(2), getCtx().getDevice()));

  x = F::contiguous(x.transpose(1, 2)).view({N, qLen, _hiddenSize});
  x = _outProj->forward(x);

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class DecodeLayer                                                                              |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<DecodeLayer> DecodeLayer::create(const Context &ctx, const LlamaConfig &config) {
  std::shared_ptr<DecodeLayer> layer{new DecodeLayer()};
  layer->setCtx(ctx);

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

void DecodeLayer::initParameters(const StateMap &stateDict) {
  _attn->initParameters(stateDict);
  _mlp->initParameters(stateDict);
  _inputNorm->initParameters(stateDict);
  _postAttnNorm->initParameters(stateDict);
}

void DecodeLayer::initParameters(lut::Random *generator, DType weightType) {
  _attn->initParameters(generator, weightType);
  _mlp->initParameters(generator, weightType);
  _inputNorm->initParameters(generator, weightType);
  _postAttnNorm->initParameters(generator, weightType);
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

// -----------------------------------------------------------------------------------------------+
// class LlamaModel                                                                               |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<LlamaModel> LlamaModel::create(const Context &fromCtx, LlamaConfig config) {
  std::shared_ptr<LlamaModel> model{new LlamaModel()};

  Context ctx = fromCtx;
  ctx.set(RoPECtxKey, ctx.name("rope"));
  model->setCtx(ctx);

  int dh = config.hiddenSize;
  model->_config = config;
  model->_embedding = Embedding::create(ctx.withName("embd"), dh, config.vocabSize);
  model->_norm = RMSNorm::create(ctx.withName("norm"), dh, config.normEps);
  for (int i = 0; i < config.numLayers; ++i) {
    model->_layers.emplace_back(
        DecodeLayer::create(ctx.withName(lut::sprintf("block%d", i)), config));
  }

  model->_outProj = Linear::create(ctx.withName("out_proj"), dh, config.vocabSize, false);
  return model;
}

void LlamaModel::initParameters(const StateMap &stateDict) {
  _embedding->initParameters(stateDict);
  _norm->initParameters(stateDict);
  _outProj->initParameters(stateDict);

  for (int i = 0; i < _config.numLayers; ++i) {
    _layers[i]->initParameters(stateDict);
  }
}

void LlamaModel::initParameters(lut::Random *generator, DType weightType) {
  Device dCpu = Device::getCpu();

  _embedding->initParameters(generator, weightType);
  _norm->initParameters(generator, weightType);
  _outProj->initParameters(generator, weightType);

  for (int i = 0; i < _config.numLayers; ++i) {
    _layers[i]->initParameters(generator, weightType);
  }
}

Tensor LlamaModel::forward(StateMap &past, Tensor input) const {
  Tensor x = _embedding->forward(input);

  for (int i = 0; i < _config.numLayers; ++i) {
    x = _layers[i]->forward(past, x);
  }

  x = _norm->forward(x);
  return x;
}

Tensor LlamaModel::forwardLmHead(Tensor hidden) const {
  Tensor logits = _outProj->forward(hidden);
  return logits;
}

int LlamaModel::getOutputDim() const {
  return _config.vocabSize;
}

// -----------------------------------------------------------------------------------------------+
// class LlamaModelForGeneration                                                                  |
// -----------------------------------------------------------------------------------------------+

LlamaModelForGeneration::LlamaModelForGeneration()
    : _eotId(0) {
}

std::shared_ptr<LlamaModelForGeneration> LlamaModelForGeneration::fromPackage(
    const Context &ctx,
    lut::ZipFile *package) {
  std::shared_ptr<lut::Reader> reader = package->open(ModelConfig);
  std::shared_ptr<lut::IniConfig> ini = lut::IniConfig::fromStream(reader.get());

  std::string modelFile = ini->getSection(ModelSection).getString(ModelFileField);
  std::string modelType = ini->getSection(ModelSection).getString(ModelTypeField);

  const lut::IniSection &llamaIni = ini->getSection(modelType);

  std::shared_ptr<LlamaModelForGeneration> model{new LlamaModelForGeneration()};
  LlamaConfig llamaConfig = LlamaConfig::loadConfig(llamaIni);

  StateMap stateMap;

  stateMap.read(package->open(modelFile).get());
  model->_model = LlamaModel::create(ctx, llamaConfig);
  model->_model->initParameters(stateMap);
  model->_eotId = llamaIni.getInt("eot_token_id");
  model->_modelName = modelType;

  model->initTokenizer(package);
  return model;
}

Tensor LlamaModelForGeneration::prefill(StateMap &past, const Prompt &prompt) const {
  Tensor x = _model->forward(past, buildInput(prompt));
  CHECK(x.getDim() == 3);

  x = x.slice(1, {-1, None});
  x = _model->forwardLmHead(x);

  return x;
}

Tensor LlamaModelForGeneration::decode(StateMap &past, LongType inputToken) const {
  std::array<LongType, 1> inputData{inputToken};
  Tensor inputs = Tensor::create<LongType>({1, 1}, inputData);
  inputs = F::to(getDevice(), inputs);

  Tensor x = _model->forward(past, inputs);
  x = _model->forwardLmHead(x);

  return x;
}

Tensor LlamaModelForGeneration::buildInput(const Prompt &prompt) const {
  std::vector<LongType> inputData{};
  for (const PromptBlock &block : prompt.getBlocks()) {
    if (block.blockType == PromptBlock::ControlToken || block.blockType == PromptBlock::Text) {
      encodePromptBlock(block, inputData);
    } else {
      throw lut::AbortedError(lut::sprintf(
          "unexpected prompt type %s for model %s",
          PromptBlock::typeToString(block.blockType),
          _modelName));
    }
  }

  int len = inputData.size();
  Tensor inputs = Tensor::create<LongType>({1, len}, inputData);
  inputs = F::to(_model->getCtx().getDevice(), inputs);
  return inputs;
}

bool LlamaModelForGeneration::isStopToken(int tokenId) const {
  return tokenId == _eotId;
}

const char *LlamaModelForGeneration::getName() const {
  return _modelName.c_str();
}

Device LlamaModelForGeneration::getDevice() const {
  return _model->getCtx().getDevice();
}

int LlamaModelForGeneration::getOutputDim() const {
  return _model->getOutputDim();
}

Prompt LlamaModelForGeneration::buildPrompt(lut::Span<const Message> history) const {
  CHECK(!history.empty()) << "history is empty";

  Prompt prompt;
  prompt.appendControlToken("<|begin_of_text|>");
  for (const Message &message : history.subspan(0, history.size() - 1)) {
    prompt.appendControlToken("<|start_header_id|>");
    prompt.appendText(message.role);
    prompt.appendControlToken("<|end_header_id|>");
    prompt.appendText("\n\n" + message.content);
    prompt.appendControlToken("<|eot_id|>");
  }

  const Message &message = history.back();
  if (message.role == "user") {
    prompt.appendControlToken("<|start_header_id|>");
    prompt.appendText(message.role);
    prompt.appendControlToken("<|end_header_id|>");
    prompt.appendText("\n\n" + message.content);
    prompt.appendControlToken("<|eot_id|>");
    prompt.appendControlToken("<|start_header_id|>");
    prompt.appendText("assistant");
    prompt.appendControlToken("<|end_header_id|>");
    prompt.appendText("\n\n");
  } else if (message.role == "assistant") {
    prompt.appendControlToken("<|start_header_id|>");
    prompt.appendText(message.role);
    prompt.appendControlToken("<|end_header_id|>");
    prompt.appendText("\n\n" + message.content);
  } else {
    throw lut::AbortedError(
        "invalid messages: role of last message should be either user or assistant");
  }

  return prompt;
}

}  // namespace llama
}  // namespace libllm
