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
#include "libllm/layers.h"
#include "libllm/model_for_generation.h"
#include "lutil/error.h"
#include "lutil/ini_config.h"
#include "lutil/strings.h"
#include "flint/functional.h"

namespace libllm {
namespace llama {

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

std::shared_ptr<MLP> MLP::build(const LlamaConfig &config, const VarBuilder &vb) {
  std::shared_ptr<MLP> mlp{new MLP()};

  int d = config.hiddenSize;
  int di = config.intermediateSize;
  mlp->_gateUpProj = Linear::build(d, di * 2, false, vb.withName("gate_up_proj"));
  mlp->_downProj = Linear::build(di, d, false, vb.withName("down_proj"));

  return mlp;
}

fl::Tensor MLP::forward(fl::Tensor input) const {
  fl::Tensor x = _gateUpProj->forward(input);
  x = fl::F::swiglu(x);
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
      _headDim(0) {
}

std::shared_ptr<Attention> Attention::build(
    const LlamaConfig &config,
    const VarBuilder &vb,
    fl::Tensor roPE,
    int layerIndex) {
  std::shared_ptr<Attention> layer{new Attention()};

  if (config.hiddenSize % config.numHeads != 0)
    throw lut::AbortedError("invalid hidden_size and num_heads");

  int headDim = config.hiddenSize / config.numHeads;
  layer->_hiddenSize = config.hiddenSize;
  layer->_numHead = config.numHeads;
  layer->_numKeyValueHead = config.numKeyValueHeads;
  layer->_headDim = headDim;
  layer->_layerIndex = layerIndex;
  layer->_roPE = roPE;

  int qkvProjDim = headDim * config.numKeyValueHeads * 2 + config.hiddenSize;
  int d = config.hiddenSize;
  layer->_qkvProj = Linear::build(
      d,
      qkvProjDim,
      config.qkvProjBias,
      vb.withName("qkv_proj"));
  layer->_outProj = Linear::build(d, d, false, vb.withName("out_proj"));

  return layer;
}

fl::Tensor Attention::rotateHalf(fl::Tensor x) const {
  fl::Tensor rotated = fl::F::tensorLike(x);
  int lastDim = x.getDim() - 1;
  int halfShape = x.getShape(lastDim) / 2;

  fl::Tensor x1 = x.slice(lastDim, {0, halfShape});
  fl::Tensor x2 = x.slice(lastDim, {halfShape, fl::None});
  x2 = fl::F::mul(x2, -1.0f);

  fl::F::copy(x1, rotated.slice(lastDim, {halfShape, fl::None}));
  fl::F::copy(x2, rotated.slice(lastDim, {0, halfShape}));

  return rotated;
}

fl::Tensor Attention::applyRoPE(fl::Tensor input, fl::Tensor roPE) const {
  fl::Tensor cos = roPE.subtensor(0);
  fl::Tensor sin = roPE.subtensor(1);

  cos = cos.expand({cos.getShape(0), input.getShape(1), cos.getShape(2)});
  sin = sin.expand({sin.getShape(0), input.getShape(1), sin.getShape(2)});

  return fl::F::add(
      fl::F::mul(input, fl::F::contiguous(cos)),
      fl::F::mul(rotateHalf(input), fl::F::contiguous(sin)));
}

fl::Tensor Attention::forward(fl::Tensor input, const ForwardBatch &batch) const {
  CHECK(input.getDim() == 2);
  fl::Tensor qkv = _qkvProj->forward(input);

  int kvHiddenSize = _headDim * _numKeyValueHead;

  fl::Tensor q = qkv.slice(-1, {0, _hiddenSize});
  fl::Tensor k = qkv.slice(-1, {_hiddenSize, _hiddenSize + kvHiddenSize});
  fl::Tensor v = qkv.slice(-1, {_hiddenSize + kvHiddenSize, _hiddenSize + 2 * kvHiddenSize});

  int qLen = batch.totalQLen();
  int kvLen = batch.totalKLen();
  CHECK(qkv.getShape(0) == qLen);

  q = q.view({qLen, _numHead, _headDim});
  k = k.view({qLen, _numKeyValueHead, _headDim});
  v = v.view({qLen, _numKeyValueHead, _headDim});

  // A single sequence occupies one contiguous rotary range; packed batches take their positions
  // from batch.positionIds() instead, which lands with multi-sequence packing.
  CHECK(batch.numSequences() == 1);
  fl::Tensor roPE = _roPE.slice(1, {kvLen - qLen, kvLen});

  q = applyRoPE(q, roPE);
  k = applyRoPE(k, roPE);

  std::shared_ptr<KVCacheManager> manager = batch.kvCacheManager().lock();
  CHECK(manager) << "batch has no KV cache manager";

  fl::Device device = input.getDevice();
  fl::Tensor keyCache = manager->getKeyCache(_layerIndex);
  fl::Tensor valueCache = manager->getValueCache(_layerIndex);

  // The keys of this batch have to be in the pool before attention reads them: a causal query
  // attends to its own key.
  fl::F::storeKVCache(k, v, keyCache, valueCache, batch.slotMappingTensor(device));
  fl::Tensor x = fl::F::pagedAttention(
      q,
      keyCache,
      valueCache,
      batch.blockTableTensor(device),
      batch.cuSeqlensQTensor(device),
      batch.seqlensKTensor(device),
      batch.maxQLen(),
      batch.maxKLen(),
      true);

  x = fl::F::contiguous(x).view({qLen, _hiddenSize});
  x = _outProj->forward(x);

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class DecodeLayer                                                                              |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<DecodeLayer> DecodeLayer::build(
    const LlamaConfig &config,
    const VarBuilder &vb,
    fl::Tensor roPE,
    int layerIndex) {
  std::shared_ptr<DecodeLayer> layer{new DecodeLayer()};

  layer->_attn = Attention::build(config, vb.withName("attn"), roPE, layerIndex);
  layer->_mlp = MLP::build(config, vb.withName("mlp"));
  layer->_inputNorm = RMSNorm::build(
      config.hiddenSize,
      config.normEps,
      vb.withName("input_norm"));
  layer->_postAttnNorm = RMSNorm::build(
      config.hiddenSize,
      config.normEps,
      vb.withName("post_attn_norm"));

  return layer;
}

fl::Tensor DecodeLayer::forward(fl::Tensor input, const ForwardBatch &batch) const {
  fl::Tensor residual = input;

  // norm + attn
  fl::Tensor x = _inputNorm->forward(input);

  x = _attn->forward(x, batch);
  x = fl::F::add(x, residual);

  // norm + mlp
  residual = x;
  x = _postAttnNorm->forward(x);
  x = _mlp->forward(x);

  x = fl::F::add(x, residual);
  return x;
}


// -----------------------------------------------------------------------------------------------+
// class LlamaModel                                                                               |
// -----------------------------------------------------------------------------------------------+

std::shared_ptr<LlamaModel> LlamaModel::build(const LlamaConfig &config, const VarBuilder &vb) {
  std::shared_ptr<LlamaModel> model{new LlamaModel()};

  int dh = config.hiddenSize;
  int headDim = config.hiddenSize / config.numHeads;
  model->_config = config;
  model->_embedding = Embedding::build(dh, config.vocabSize, vb.withName("embd"));
  model->_norm = RMSNorm::build(dh, config.normEps, vb.withName("norm"));

  fl::Tensor roPE = vb.get("rope", {2, 1, config.maxContextLength, headDim});
  roPE = roPE.view({2, config.maxContextLength, 1, headDim});

  for (int i = 0; i < config.numLayers; ++i) {
    model->_layers.emplace_back(
        DecodeLayer::build(config, vb.withName(lut::sprintf("block%d", i)), roPE, i));
  }

  model->_outProj = Linear::build(dh, config.vocabSize, false, vb.withName("out_proj"));
  return model;
}

fl::Tensor LlamaModel::forward(fl::Tensor input, const ForwardBatch &batch) const {
  fl::Tensor x = _embedding->forward(input);

  for (int i = 0; i < _config.numLayers; ++i) {
    x = _layers[i]->forward(x, batch);
  }

  x = _norm->forward(x);
  return x;
}

fl::Tensor LlamaModel::forwardLmHead(fl::Tensor hidden) const {
  fl::Tensor logits = _outProj->forward(hidden);
  return logits;
}

int LlamaModel::getOutputDim() const {
  return _config.vocabSize;
}

const LlamaConfig &LlamaModel::getConfig() const {
  return _config;
}


// -----------------------------------------------------------------------------------------------+
// class LlamaModelForGeneration                                                                  |
// -----------------------------------------------------------------------------------------------+

LlamaModelForGeneration::LlamaModelForGeneration()
    : _floatType(fl::DType::kUnknown),
      _eotId(0) {
}

std::shared_ptr<LlamaModelForGeneration> LlamaModelForGeneration::fromPackage(
    const fl::Device &device,
    lut::ZipFile *package) {
  std::shared_ptr<lut::Reader> reader = package->open(ModelConfig);
  std::shared_ptr<lut::IniConfig> ini = lut::IniConfig::fromStream(reader.get());

  std::string modelFile = ini->getSection(ModelSection).getString(ModelFileField);
  std::string modelType = ini->getSection(ModelSection).getString(ModelTypeField);

  const lut::IniSection &llamaIni = ini->getSection(modelType);

  std::shared_ptr<LlamaModelForGeneration> model{new LlamaModelForGeneration()};
  LlamaConfig llamaConfig = LlamaConfig::loadConfig(llamaIni);

  VarBuilder vb = VarBuilder::fromStream(
      package->open(modelFile).get(),
      device,
      fl::F::getDefaultFloatType(device));
  model->_model = LlamaModel::build(llamaConfig, vb.withName(modelType));
  model->_config = llamaConfig;
  model->_device = device;
  model->_floatType = vb.getFloatDType();
  model->_eotId = llamaIni.getInt("eot_token_id");
  model->_modelName = modelType;

  model->initTokenizer(package);
  return model;
}

fl::Tensor LlamaModelForGeneration::forward(const ForwardBatch &batch) const {
  fl::Tensor x = _model->forward(batch.tokenIdsTensor(_device), batch);
  CHECK(x.getDim() == 2);

  x = x.slice(0, {-1, fl::None});
  return _model->forwardLmHead(x);
}

fl::Tensor LlamaModelForGeneration::buildInput(lut::Span<const fl::LongType> tokenIds) const {
  int len = static_cast<int>(tokenIds.size());
  fl::Tensor inputs = fl::Tensor::create<fl::LongType>({len}, tokenIds);
  inputs = fl::F::to(_device, inputs);
  return inputs;
}

bool LlamaModelForGeneration::isStopToken(int tokenId) const {
  return tokenId == _eotId;
}

const char *LlamaModelForGeneration::getName() const {
  return _modelName.c_str();
}

fl::Device LlamaModelForGeneration::getDevice() const {
  return _device;
}

int LlamaModelForGeneration::getOutputDim() const {
  return _model->getOutputDim();
}

KVCacheSpec LlamaModelForGeneration::getKVCacheSpec() const {
  return KVCacheSpec(
      _config.numLayers,
      _config.numKeyValueHeads,
      _config.hiddenSize / _config.numHeads,
      _config.maxContextLength,
      _floatType);
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
