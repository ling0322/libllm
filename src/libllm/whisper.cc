// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "libllm/whisper.h"

#include <limits>

#include "libllm/constants.h"
#include "libllm/functional.h"
#include "libllm/lut/error.h"
#include "libllm/lut/strings.h"

namespace libllm {
namespace whisper {

WhisperConfig::WhisperConfig()
    : hiddenSize(0),
      encoderNumHeads(0),
      encoderFfnDim(0),
      encoderNumLayers(0),
      decoderNumLayers(0),
      decoderFfnDim(0),
      vocabSize(0),
      maxTgtLength(0) {
}

WhisperConfig WhisperConfig::loadConfig(const lut::IniSection &section) {
  WhisperConfig config;

  config.hiddenSize = section.getInt("hidden_size");
  config.encoderNumHeads = section.getInt("encoder_num_heads");
  config.encoderFfnDim = section.getInt("encoder_ffn_dim");
  config.encoderNumLayers = section.getInt("encoder_num_layers");
  config.decoderNumLayers = section.getInt("decoder_num_layers");
  config.decoderFfnDim = section.getInt("decoder_ffn_dim");
  config.vocabSize = section.getInt("vocab_size");
  config.maxTgtLength = section.getInt("max_tgt_length");
  return config;
}

// -----------------------------------------------------------------------------------------------+
// class EncoderAttention                                                                         |
// -----------------------------------------------------------------------------------------------+

EncoderAttention::EncoderAttention()
    : _numHeads(0),
      _hiddenSize(0) {
}

EncoderAttention::~EncoderAttention() {
}

std::shared_ptr<EncoderAttention> EncoderAttention::fromConfig(
    const Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<EncoderAttention> model{new EncoderAttention()};
  model->setCtx(ctx);

  if (config.hiddenSize % config.encoderNumHeads != 0) {
    throw lut::AbortedError("invalid hiddenSize and numHeads");
  }

  model->_qkvProj = Linear::create(
      ctx.withName("qkv_proj"),
      config.hiddenSize,
      config.hiddenSize * 3);
  model->_outProj = Linear::create(ctx.withName("out_proj"), config.hiddenSize, config.hiddenSize);
  model->_hiddenSize = config.hiddenSize;
  model->_numHeads = config.encoderNumHeads;
  return model;
}

void EncoderAttention::initParameters(const StateMap &stateDict) {
  _qkvProj->initParameters(stateDict);
  _outProj->initParameters(stateDict);
}

void EncoderAttention::initParameters(lut::Random *generator, DType weightType) {
  _qkvProj->initParameters(generator, weightType);
  _outProj->initParameters(generator, weightType);
}

Tensor EncoderAttention::forward(Tensor inputs) {
  CHECK(inputs.getDim() == 3);
  Tensor qkv = _qkvProj->forward(inputs);

  Tensor q = qkv.slice(-1, {0, _hiddenSize});
  Tensor k = qkv.slice(-1, {_hiddenSize, _hiddenSize * 2});
  Tensor v = qkv.slice(-1, {_hiddenSize * 2, _hiddenSize * 3});

  int bsz = inputs.getShape(0);
  int len = inputs.getShape(1);
  int headDim = _hiddenSize / _numHeads;
  q = q.view({bsz, len, _numHeads, headDim});
  k = k.view({bsz, len, _numHeads, headDim});
  v = v.view({bsz, len, _numHeads, headDim});

  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);
  Tensor x = F::attention(q, k, v);

  x = F::contiguous(x.transpose(1, 2)).view({bsz, len, _hiddenSize});
  x = _outProj->forward(x);

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class EncoderLayer                                                                             |
// -----------------------------------------------------------------------------------------------+

EncoderLayer::EncoderLayer() {
}

EncoderLayer::~EncoderLayer() {
}

std::shared_ptr<EncoderLayer> EncoderLayer::fromConfig(const Context &ctx, WhisperConfig config) {
  std::shared_ptr<EncoderLayer> model{new EncoderLayer()};
  model->setCtx(ctx);

  model->_norm1 = LayerNorm::create(ctx.withName("norm1"), config.hiddenSize);
  model->_norm2 = LayerNorm::create(ctx.withName("norm2"), config.hiddenSize);
  model->_attn = EncoderAttention::fromConfig(ctx.withName("attn"), config);
  model->_fc1 = Linear::create(ctx.withName("fc1"), config.hiddenSize, config.encoderFfnDim);
  model->_fc2 = Linear::create(ctx.withName("fc2"), config.encoderFfnDim, config.hiddenSize);
  return model;
}

void EncoderLayer::initParameters(const StateMap &stateDict) {
  _norm1->initParameters(stateDict);
  _norm2->initParameters(stateDict);
  _attn->initParameters(stateDict);
  _fc1->initParameters(stateDict);
  _fc2->initParameters(stateDict);
}

void EncoderLayer::initParameters(lut::Random *generator, DType weightType) {
  _norm1->initParameters(generator, weightType);
  _norm2->initParameters(generator, weightType);
  _attn->initParameters(generator, weightType);
  _fc1->initParameters(generator, weightType);
  _fc2->initParameters(generator, weightType);
}

Tensor EncoderLayer::forward(Tensor inputs) {
  Tensor residual = inputs;

  Tensor x = _norm1->forward(inputs);

  x = _attn->forward(x);
  x = F::add(x, residual);

  residual = x;
  x = _norm2->forward(x);

  x = _fc1->forward(x);
  x = F::gelu(x);

  x = _fc2->forward(x);
  x = F::add(x, residual);

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class EncoderModel                                                                             |
// -----------------------------------------------------------------------------------------------+

EncoderModel::EncoderModel()
    : _hiddenSize(0) {
}

EncoderModel::~EncoderModel() {
}

std::shared_ptr<EncoderModel> EncoderModel::fromConfig(const Context &ctx, WhisperConfig config) {
  std::shared_ptr<EncoderModel> model{new EncoderModel()};
  model->setCtx(ctx);

  model->_conv1 = Conv1D::create(ctx.withName("conv1"), FeatDim, config.hiddenSize, 3);
  model->_conv2 = Conv1D::create(ctx.withName("conv2"), config.hiddenSize, config.hiddenSize, 3, 2);
  model->_hiddenSize = config.hiddenSize;
  for (int i = 0; i < config.encoderNumLayers; ++i) {
    model->_layers.emplace_back(
        EncoderLayer::fromConfig(ctx.withName(lut::sprintf("layer%d", i)), config));
  }
  model->_norm = LayerNorm::create(ctx.withName("norm"), config.hiddenSize);
  return model;
}

void EncoderModel::initParameters(const StateMap &stateDict) {
  Context ctx = getCtx();

  _conv1->initParameters(stateDict);
  _conv2->initParameters(stateDict);

  _posEmbd = stateDict.getTensor(ctx.name("pos_embd"));
  _posEmbd = moveAndCastFloat(_posEmbd, ctx);

  for (std::shared_ptr<EncoderLayer> &layer : _layers) {
    layer->initParameters(stateDict);
  }

  _norm->initParameters(stateDict);
}

void EncoderModel::initParameters(lut::Random *generator, DType weightType) {
  _conv1->initParameters(generator, weightType);
  _conv2->initParameters(generator, weightType);

  float r = 0.2f;
  Device dCpu = Device::getCpu();
  _posEmbd = F::rand({NumFrames, _hiddenSize}, DType::kFloat, dCpu, generator, -r, r);
  _posEmbd = moveAndCastFloat(_posEmbd, getCtx());

  for (std::shared_ptr<EncoderLayer> &layer : _layers) {
    layer->initParameters(generator, weightType);
  }

  _norm->initParameters(generator, weightType);
}

Tensor EncoderModel::forward(Tensor wave) {
  CHECK(wave.getDim() == 1 && wave.getShape(-1) <= InputSamples);

  // pad wave.
  if (wave.getShape(-1) < InputSamples) {
    Tensor pad = F::zeros({InputSamples}, wave.getDType(), wave.getDevice());
    F::copy(wave, pad.slice({0, wave.getShape(-1)}));
    wave = pad;
  }

  Tensor features = F::logMelSpectrogram(wave);
  features = moveAndCastFloat(features, getCtx());

  CHECK(features.getDim() == 2);
  features = features.unsqueeze(0);

  Tensor x = _conv1->forward(features);
  x = F::gelu(x);

  x = _conv2->forward(x);
  x = F::gelu(x);
  x = F::add(x, _posEmbd);

  for (const std::shared_ptr<EncoderLayer> &layer : _layers) {
    x = layer->forward(x);
  }

  x = _norm->forward(x);

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class DecoderInitModel                                                                         |
// -----------------------------------------------------------------------------------------------+

DecoderInitModel::DecoderInitModel()
    : _dModel(0) {
}

DecoderInitModel::~DecoderInitModel() {
}

std::shared_ptr<DecoderInitModel> DecoderInitModel::fromConfig(
    const Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<DecoderInitModel> model{new DecoderInitModel()};
  model->setCtx(ctx);

  int dModel = config.hiddenSize;
  for (int i = 0; i < config.encoderNumLayers; ++i) {
    Context ctxLayer = ctx.withName(lut::sprintf("layer%d", i)).withName(DecoderLayer::CrossAttn);
    model->_kvProjs.emplace_back(Linear::create(ctxLayer.withName("kv_proj"), dModel, dModel * 2));
  }
  model->_dModel = dModel;
  return model;
}

void DecoderInitModel::initParameters(const StateMap &stateDict) {
  for (std::shared_ptr<Linear> &layer : _kvProjs) {
    layer->initParameters(stateDict);
  }
}

void DecoderInitModel::initParameters(lut::Random *generator, DType weightType) {
  for (std::shared_ptr<Linear> &layer : _kvProjs) {
    layer->initParameters(generator, weightType);
  }
}

void DecoderInitModel::forward(StateMap &past, Tensor encoderHidden) {
  CHECK(encoderHidden.getDim() == 3);

  for (int i = 0; i < _kvProjs.size(); ++i) {
    Context ctxLayer = getCtx().withName(lut::sprintf("layer%d", i));
    Context ctxAttn = ctxLayer.withName(DecoderLayer::CrossAttn);

    Tensor x = _kvProjs[i]->forward(encoderHidden);
    Tensor cacheK = x.slice(2, {0, _dModel});
    Tensor cacheV = x.slice(2, {_dModel, 2 * _dModel});

    past.putTensor(ctxAttn.name("k"), cacheK);
    past.putTensor(ctxAttn.name("v"), cacheV);
  }
}

// -----------------------------------------------------------------------------------------------+
// class Attention                                                                                |
// -----------------------------------------------------------------------------------------------+

Attention::Attention()
    : _numHeads(0),
      _hiddenSize(0) {
}

Attention::~Attention() {
}

std::shared_ptr<Attention> Attention::selfAttn(const Context &ctx, WhisperConfig config) {
  std::shared_ptr<Attention> model{new Attention()};
  model->setCtx(ctx);
  model->initCommon(config);

  model->_proj = Linear::create(ctx.withName("qkv_proj"), config.hiddenSize, config.hiddenSize * 3);
  model->_selfAttn = true;
  return model;
}

std::shared_ptr<Attention> Attention::crossAttn(const Context &ctx, WhisperConfig config) {
  std::shared_ptr<Attention> model{new Attention()};
  model->setCtx(ctx);
  model->initCommon(config);

  model->_proj = Linear::create(ctx.withName("q_proj"), config.hiddenSize, config.hiddenSize);
  model->_selfAttn = false;
  return model;
}

int Attention::getCtxLength(const StateMap &past) const {
  if (past.hasValue<int>(_namePastLen)) {
    return past.getValue<int>(_namePastLen);
  } else {
    return 0;
  }
}

void Attention::initCommon(WhisperConfig config) {
  if (config.hiddenSize % config.encoderNumHeads != 0) {
    throw lut::AbortedError("invalid hiddenSize and numHeads");
  }

  _outProj = Linear::create(getCtx().withName("out_proj"), config.hiddenSize, config.hiddenSize);
  _hiddenSize = config.hiddenSize;
  _numHeads = config.encoderNumHeads;

  _namePastK = getCtx().name("k");
  _namePastV = getCtx().name("v");
  _namePastLen = getCtx().name("len");
}

void Attention::initParameters(const StateMap &stateDict) {
  _proj->initParameters(stateDict);
  _outProj->initParameters(stateDict);
}

void Attention::initParameters(lut::Random *generator, DType weightType) {
  _proj->initParameters(generator, weightType);
  _outProj->initParameters(generator, weightType);
}

std::pair<Tensor, Tensor> Attention::getPresentKV(StateMap &past, Tensor k, Tensor v) {
  Tensor pastK, pastV;

  int pastLen = getCtxLength(past);
  int presentLen = pastLen + k.getShape(1);

  int cacheLen = 0;
  if (pastLen > 0) {
    pastK = past.getTensor(_namePastK);
    pastV = past.getTensor(_namePastV);
    cacheLen = pastK.getShape(1);
    CHECK(pastK.getDim() == 3 && pastV.getDim() == 3 && pastK.getShape(1) == pastV.getShape(1));
  }

  if (cacheLen < presentLen) {
    LOG(DEBUG) << lut::sprintf(
        "update kv cache cacheLen=%d pastLen=%d presentLen=%d",
        cacheLen,
        pastLen,
        presentLen);

    // to reduce memory allocation, we extend the kv cache block by block.
    int nextNumBlocks = (presentLen + PastBlockSize - 1) / PastBlockSize;
    int nextLen = PastBlockSize * nextNumBlocks;

    int d0, d2;
    if (pastLen) {
      d0 = pastK.getShape(0);
      d2 = pastK.getShape(2);
    } else {
      d0 = k.getShape(0);
      d2 = k.getShape(2);
    }
    Tensor nextK = F::zeros({d0, nextLen, d2}, k.getDType(), k.getDevice());
    Tensor nextV = F::zeros({d0, nextLen, d2}, v.getDType(), v.getDevice());

    if (pastLen) {
      F::copy(pastK.slice(1, {0, pastLen}), nextK.slice(1, {0, pastLen}));
      F::copy(pastV.slice(1, {0, pastLen}), nextV.slice(1, {0, pastLen}));
    }

    past.putTensor(_namePastK, nextK);
    past.putTensor(_namePastV, nextV);

    pastK = nextK;
    pastV = nextV;
  }

  F::copy(k, pastK.slice(1, {pastLen, presentLen}));
  F::copy(v, pastV.slice(1, {pastLen, presentLen}));

  Tensor presentK = pastK.slice(1, {0, presentLen});
  Tensor presentV = pastV.slice(1, {0, presentLen});
  past.putValue<int>(_namePastLen, presentLen);

  return std::make_pair(presentK, presentV);
}

Tensor Attention::forward(StateMap &past, Tensor inputs) {
  CHECK(inputs.getDim() == 3);

  Tensor q, k, v;
  if (_selfAttn) {
    Tensor qkv = _proj->forward(inputs);
    q = qkv.slice(-1, {0, _hiddenSize});
    k = qkv.slice(-1, {_hiddenSize, _hiddenSize * 2});
    v = qkv.slice(-1, {_hiddenSize * 2, _hiddenSize * 3});

    std::tie(k, v) = getPresentKV(past, k, v);

  } else {
    q = _proj->forward(inputs);

    // initialized in the DecoderInitModel.
    k = past.getTensor(_namePastK);
    v = past.getTensor(_namePastV);
  }

  int bsz = inputs.getShape(0);
  int len = inputs.getShape(1);
  int headDim = _hiddenSize / _numHeads;
  q = q.view({bsz, len, _numHeads, headDim});
  k = k.view({bsz, k.getShape(1), _numHeads, headDim});
  v = v.view({bsz, v.getShape(1), _numHeads, headDim});

  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);

  Tensor x;
  if (_selfAttn && inputs.getShape(1) == 1) {
    x = F::attention(q, k, v, F::causalMask(q.getShape(2), getCtx().getDevice()));
  } else {
    x = F::attention(q, k, v);
  }

  x = F::contiguous(x.transpose(1, 2)).view({bsz, len, _hiddenSize});
  x = _outProj->forward(x);

  return x;
}

// -----------------------------------------------------------------------------------------------+
// class DecoderLayer                                                                             |
// -----------------------------------------------------------------------------------------------+

constexpr char DecoderLayer::CrossAttn[];
constexpr char DecoderLayer::SelfAttn[];

DecoderLayer::DecoderLayer() {
}

DecoderLayer::~DecoderLayer() {
}

std::shared_ptr<DecoderLayer> DecoderLayer::fromConfig(const Context &ctx, WhisperConfig config) {
  std::shared_ptr<DecoderLayer> model{new DecoderLayer()};
  model->setCtx(ctx);

  model->_norm1 = LayerNorm::create(ctx.withName("norm1"), config.hiddenSize);
  model->_norm2 = LayerNorm::create(ctx.withName("norm2"), config.hiddenSize);
  model->_norm3 = LayerNorm::create(ctx.withName("norm3"), config.hiddenSize);
  model->_selfAttn = Attention::selfAttn(ctx.withName(SelfAttn), config);
  model->_crossAttn = Attention::crossAttn(ctx.withName(CrossAttn), config);
  model->_fc1 = Linear::create(ctx.withName("fc1"), config.hiddenSize, config.decoderFfnDim);
  model->_fc2 = Linear::create(ctx.withName("fc2"), config.decoderFfnDim, config.hiddenSize);
  return model;
}

void DecoderLayer::initParameters(const StateMap &stateDict) {
  _norm1->initParameters(stateDict);
  _norm2->initParameters(stateDict);
  _norm3->initParameters(stateDict);
  _selfAttn->initParameters(stateDict);
  _crossAttn->initParameters(stateDict);
  _fc1->initParameters(stateDict);
  _fc2->initParameters(stateDict);
}

void DecoderLayer::initParameters(lut::Random *generator, DType weightType) {
  _norm1->initParameters(generator, weightType);
  _norm2->initParameters(generator, weightType);
  _norm3->initParameters(generator, weightType);
  _selfAttn->initParameters(generator, weightType);
  _crossAttn->initParameters(generator, weightType);
  _fc1->initParameters(generator, weightType);
  _fc2->initParameters(generator, weightType);
}

Tensor DecoderLayer::forward(StateMap &past, Tensor inputs) {
  Tensor residual = inputs;

  Tensor x = _norm1->forward(inputs);
  x = _selfAttn->forward(past, x);
  x = F::add(x, residual);

  residual = x;
  x = _norm2->forward(x);
  x = _crossAttn->forward(past, x);
  x = F::add(x, residual);

  residual = x;
  x = _norm3->forward(x);
  x = _fc1->forward(x);
  x = F::gelu(x);

  x = _fc2->forward(x);
  x = F::add(x, residual);
  return x;
}

// -----------------------------------------------------------------------------------------------+
// class DecoderModel                                                                             |
// -----------------------------------------------------------------------------------------------+

DecoderModel::DecoderModel()
    : _dModel(0),
      _maxTgtLength(0),
      _outputDim(0) {
}

DecoderModel::~DecoderModel() {
}

std::shared_ptr<DecoderModel> DecoderModel::fromConfig(const Context &ctx, WhisperConfig config) {
  std::shared_ptr<DecoderModel> model{new DecoderModel()};
  model->setCtx(ctx);

  model->_embd = Embedding::create(ctx.withName("embd"), config.hiddenSize, config.vocabSize);
  for (int i = 0; i < config.decoderNumLayers; ++i) {
    model->_layers.emplace_back(
        DecoderLayer::fromConfig(ctx.withName(lut::sprintf("layer%d", i)), config));
  }
  model->_norm = LayerNorm::create(ctx.withName("norm"), config.hiddenSize);
  model->_outProj = Linear::create(
      ctx.withName("out_proj"),
      config.hiddenSize,
      config.vocabSize,
      false);
  model->_maxTgtLength = config.maxTgtLength;
  model->_dModel = config.hiddenSize;
  model->_namePastLen = ctx.name("len");
  model->_outputDim = config.vocabSize;
  return model;
}

void DecoderModel::initParameters(const StateMap &stateDict) {
  Context ctx = getCtx();

  _embd->initParameters(stateDict);
  _norm->initParameters(stateDict);
  _outProj->initParameters(stateDict);

  _posEmbd = stateDict.getTensor(ctx.name("pos_embd"));
  _posEmbd.throwIfInvalidShape({_maxTgtLength, _dModel}, ctx.name("pos_embd"));
  _posEmbd = moveAndCastFloat(_posEmbd, ctx);

  for (std::shared_ptr<DecoderLayer> &layer : _layers) {
    layer->initParameters(stateDict);
  }
}

void DecoderModel::initParameters(lut::Random *generator, DType weightType) {
  _embd->initParameters(generator, weightType);
  _norm->initParameters(generator, weightType);
  _outProj->initParameters(generator, weightType);

  float r = 0.2f;
  Device dCpu = Device::getCpu();
  _posEmbd = F::rand({_maxTgtLength, _dModel}, DType::kFloat, dCpu, generator, -r, r);
  _posEmbd = moveAndCastFloat(_posEmbd, getCtx());

  for (std::shared_ptr<DecoderLayer> &layer : _layers) {
    layer->initParameters(generator, weightType);
  }
}

int DecoderModel::getCtxLength(const StateMap &past) const {
  if (past.hasValue<int>(_namePastLen)) {
    return past.getValue<int>(_namePastLen);
  } else {
    return 0;
  }
}

Tensor DecoderModel::forward(StateMap &past, Tensor inputs) {
  Tensor x = _embd->forward(inputs);

  // positional embedding.
  int pastLen = getCtxLength(past);
  int presentLen = pastLen + inputs.getShape(1);
  x = F::add(x, _posEmbd.slice({pastLen, presentLen}));
  past.putValue<int>(_namePastLen, presentLen);

  for (const std::shared_ptr<DecoderLayer> &layer : _layers) {
    x = layer->forward(past, x);
  }

  x = _norm->forward(x);
  return x;
}

Tensor DecoderModel::forwardLmHead(Tensor inputs) {
  return _outProj->forward(inputs);
}

int DecoderModel::getOutputDim() const {
  return _outputDim;
}

// -----------------------------------------------------------------------------------------------+
// class WhisperLogitsProcessor                                                                   |
// -----------------------------------------------------------------------------------------------+

WhisperLogitsProcessor::WhisperLogitsProcessor()
    : _lastTimeToken(-1),
      _beginTimeToken(-1),
      _endTimeToken(-1),
      _eotToken(-1) {
}

std::shared_ptr<WhisperLogitsProcessor> WhisperLogitsProcessor::newProcessor(const Vocab *vocab) {
  std::shared_ptr<WhisperLogitsProcessor> processor{new WhisperLogitsProcessor()};
  processor->_lastTimeToken = -1;
  processor->_beginTimeToken = vocab->findControlToken("<|0.00|>");
  processor->_endTimeToken = vocab->findControlToken("<|30.00|>");
  processor->_eotToken = vocab->findControlToken("<|endoftext|>");
  processor->_transcribeToken = vocab->findControlToken("<|transcribe|>");
  processor->_translateToken = vocab->findControlToken("<|translate|>");
  processor->_noTimestampToken = vocab->findControlToken("<|notimestamps|>");

  return processor;
}

void WhisperLogitsProcessor::notifyToken(int tokenId) {
  _history.push_back(tokenId);
  if (tokenId >= _beginTimeToken && tokenId <= _endTimeToken) {
    _lastTimeToken = tokenId;
  }
}

void WhisperLogitsProcessor::processLogits(Tensor logits) {
  bool lastWasTimestamp = _history.size() >= 1 && _history.back() >= _beginTimeToken;
  bool lastWasTranscribe = _history.size() >= 1 && _history.back() == _transcribeToken;
  bool penultimateWasTimestamp = _history.size() < 2 ||
                                 _history[_history.size() - 2] >= _beginTimeToken ||
                                 _history[_history.size() - 2] == _transcribeToken ||
                                 _history[_history.size() - 2] == _translateToken;

  if (lastWasTranscribe) {
    F::fill(logits.slice(-1, {_noTimestampToken, _noTimestampToken + 1}), -Inf);
  }

  if (lastWasTimestamp) {
    if (penultimateWasTimestamp) {
      F::fill(logits.slice(-1, {_beginTimeToken, _endTimeToken + 1}), -Inf);
    } else {
      F::fill(logits.slice(-1, {0, _eotToken}), -Inf);
    }
  }

  if (_lastTimeToken > 0) {
    F::fill(logits.slice(-1, {_beginTimeToken, _lastTimeToken + 1}), -Inf);
  }

  Tensor probs = F::softmax(logits);
  Tensor maxText = F::max(probs.slice(-1, {0, _eotToken + 1}));
  Tensor sumTimestamp = F::sum(probs.slice(-1, {_beginTimeToken, _endTimeToken + 1}));

  maxText = F::cast(F::to(Device::getCpu(), maxText), DType::kFloat);
  sumTimestamp = F::cast(F::to(Device::getCpu(), sumTimestamp), DType::kFloat);

  float maxTextVal = *maxText.getData<float>();
  float sumTimestampVal = *sumTimestamp.getData<float>();
  if (sumTimestampVal >= maxTextVal) {
    F::fill(logits.slice(-1, {0, _eotToken}), -Inf);
  }
}

// -----------------------------------------------------------------------------------------------+
// class WhisperModelForGeneration                                                                |
// -----------------------------------------------------------------------------------------------+

WhisperModelForGeneration::WhisperModelForGeneration()
    : _eotId(0) {
}

std::shared_ptr<WhisperModelForGeneration> WhisperModelForGeneration::fromPackage(
    const Context &ctx,
    lut::ZipFile *package) {
  std::shared_ptr<lut::Reader> reader = package->open(ModelConfig);
  std::shared_ptr<lut::IniConfig> ini = lut::IniConfig::fromStream(reader.get());

  std::string modelFile = ini->getSection(ModelSection).getString(ModelFileField);
  std::string modelType = ini->getSection(ModelSection).getString(ModelTypeField);

  const lut::IniSection &llamaIni = ini->getSection(modelType);

  std::shared_ptr<WhisperModelForGeneration> model{new WhisperModelForGeneration()};
  WhisperConfig llamaConfig = WhisperConfig::loadConfig(llamaIni);

  StateMap stateMap;

  stateMap.read(package->open(modelFile).get());
  model->_encoder = EncoderModel::fromConfig(ctx.withName("encoder"), llamaConfig);
  model->_decoderInit = DecoderInitModel::fromConfig(ctx.withName("decoder"), llamaConfig);
  model->_decoder = DecoderModel::fromConfig(ctx.withName("decoder"), llamaConfig);

  model->_encoder->initParameters(stateMap);
  model->_decoderInit->initParameters(stateMap);
  model->_decoder->initParameters(stateMap);
  model->_eotId = llamaIni.getInt("eot_token_id");
  model->_modelName = modelType;

  model->initTokenizer(package);
  return model;
}

Tensor WhisperModelForGeneration::buildDecoderInput(lut::Span<const PromptBlock> prompt) const {
  std::vector<LongType> inputData{};
  for (const PromptBlock &block : prompt) {
    if (block.blockType == PromptBlock::ControlToken || block.blockType == PromptBlock::Text) {
      encodePromptBlock(block, inputData);
    } else {
      throw lut::AbortedError("in whisper prompt, only one audio input is supported");
    }
  }

  int len = inputData.size();
  Tensor inputs = Tensor::create<LongType>({1, len}, inputData);
  inputs = F::to(_decoder->getCtx().getDevice(), inputs);
  return inputs;
}

Tensor WhisperModelForGeneration::prefill(StateMap &past, const Prompt &prompt) const {
  if (prompt.empty()) throw lut::AbortedError("prompt is empty");

  const PromptBlock &audioBlock = prompt.getBlocks()[0];
  if (audioBlock.blockType != PromptBlock::Wave) {
    throw lut::AbortedError("in whisper model, the first element in prompt should be the audio");
  }
  if (prompt.getBlocks().size() <= 1) throw lut::AbortedError("decoder prompt is empty");

  Tensor wave = Wave::read(audioBlock.data, audioBlock.waveFormat);
  Tensor encoderHidden = _encoder->forward(wave);

  _decoderInit->forward(past, encoderHidden);

  Tensor inputs = buildDecoderInput(prompt.getBlocks().subspan(1));
  Tensor x = _decoder->forward(past, inputs);

  x = x.slice(1, {-1, None});
  x = _decoder->forwardLmHead(x);
  return x;
}

Tensor WhisperModelForGeneration::decode(StateMap &past, LongType inputToken) const {
  std::array<LongType, 1> inputData{inputToken};
  Tensor inputs = Tensor::create<LongType>({1, 1}, inputData);
  inputs = F::to(getDevice(), inputs);

  Tensor x = _decoder->forward(past, inputs);
  x = _decoder->forwardLmHead(x);
  return x;
}

bool WhisperModelForGeneration::isStopToken(int tokenId) const {
  return tokenId == _eotId;
}

const char *WhisperModelForGeneration::getName() const {
  return _modelName.c_str();
}

Device WhisperModelForGeneration::getDevice() const {
  return _decoder->getCtx().getDevice();
}

int WhisperModelForGeneration::getOutputDim() const {
  return _decoder->getOutputDim();
}

}  // namespace whisper
}  // namespace libllm
