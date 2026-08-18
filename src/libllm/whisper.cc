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
#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/functional.h"

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
    const fl::Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<EncoderAttention> model{new EncoderAttention()};
  model->setCtx(ctx);

  if (config.hiddenSize % config.encoderNumHeads != 0) {
    throw lut::AbortedError("invalid hiddenSize and numHeads");
  }

  model->_qkvProj = fl::Linear::create(
      ctx.withName("qkv_proj"),
      config.hiddenSize,
      config.hiddenSize * 3);
  model->_outProj = fl::Linear::create(
      ctx.withName("out_proj"),
      config.hiddenSize,
      config.hiddenSize);
  model->_hiddenSize = config.hiddenSize;
  model->_numHeads = config.encoderNumHeads;
  return model;
}

void EncoderAttention::initParameters(const fl::StateMap &stateDict) {
  _qkvProj->initParameters(stateDict);
  _outProj->initParameters(stateDict);
}

fl::Tensor EncoderAttention::forward(fl::Tensor inputs) {
  CHECK(inputs.getDim() == 3);
  fl::Tensor qkv = _qkvProj->forward(inputs);

  fl::Tensor q = qkv.slice(-1, {0, _hiddenSize});
  fl::Tensor k = qkv.slice(-1, {_hiddenSize, _hiddenSize * 2});
  fl::Tensor v = qkv.slice(-1, {_hiddenSize * 2, _hiddenSize * 3});

  int bsz = inputs.getShape(0);
  int len = inputs.getShape(1);
  int headDim = _hiddenSize / _numHeads;
  q = q.view({bsz, len, _numHeads, headDim});
  k = k.view({bsz, len, _numHeads, headDim});
  v = v.view({bsz, len, _numHeads, headDim});

  q = q.transpose(1, 2);
  k = k.transpose(1, 2);
  v = v.transpose(1, 2);
  fl::Tensor x = fl::F::attention(q, k, v, fl::Tensor());

  x = fl::F::contiguous(x.transpose(1, 2)).view({bsz, len, _hiddenSize});
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

std::shared_ptr<EncoderLayer> EncoderLayer::fromConfig(
    const fl::Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<EncoderLayer> model{new EncoderLayer()};
  model->setCtx(ctx);

  model->_norm1 = fl::LayerNorm::create(ctx.withName("norm1"), config.hiddenSize);
  model->_norm2 = fl::LayerNorm::create(ctx.withName("norm2"), config.hiddenSize);
  model->_attn = EncoderAttention::fromConfig(ctx.withName("attn"), config);
  model->_fc1 = fl::Linear::create(ctx.withName("fc1"), config.hiddenSize, config.encoderFfnDim);
  model->_fc2 = fl::Linear::create(ctx.withName("fc2"), config.encoderFfnDim, config.hiddenSize);
  return model;
}

void EncoderLayer::initParameters(const fl::StateMap &stateDict) {
  _norm1->initParameters(stateDict);
  _norm2->initParameters(stateDict);
  _attn->initParameters(stateDict);
  _fc1->initParameters(stateDict);
  _fc2->initParameters(stateDict);
}

fl::Tensor EncoderLayer::forward(fl::Tensor inputs) {
  fl::Tensor residual = inputs;

  fl::Tensor x = _norm1->forward(inputs);

  x = _attn->forward(x);
  x = fl::F::add(x, residual);

  residual = x;
  x = _norm2->forward(x);

  x = _fc1->forward(x);
  x = fl::F::gelu(x);

  x = _fc2->forward(x);
  x = fl::F::add(x, residual);

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

std::shared_ptr<EncoderModel> EncoderModel::fromConfig(
    const fl::Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<EncoderModel> model{new EncoderModel()};
  model->setCtx(ctx);

  model->_conv1 = fl::Conv1D::create(ctx.withName("conv1"), FeatDim, config.hiddenSize, 3);
  model->_conv2 = fl::Conv1D::create(
      ctx.withName("conv2"),
      config.hiddenSize,
      config.hiddenSize,
      3,
      2);
  model->_hiddenSize = config.hiddenSize;
  for (int i = 0; i < config.encoderNumLayers; ++i) {
    model->_layers.emplace_back(
        EncoderLayer::fromConfig(ctx.withName(lut::sprintf("layer%d", i)), config));
  }
  model->_norm = fl::LayerNorm::create(ctx.withName("norm"), config.hiddenSize);
  return model;
}

void EncoderModel::initParameters(const fl::StateMap &stateDict) {
  fl::Context ctx = getCtx();

  _conv1->initParameters(stateDict);
  _conv2->initParameters(stateDict);

  _posEmbd = stateDict.getTensor(ctx.name("pos_embd"));
  _posEmbd = moveAndCastFloat(_posEmbd, ctx);

  for (std::shared_ptr<EncoderLayer> &layer : _layers) {
    layer->initParameters(stateDict);
  }

  _norm->initParameters(stateDict);
}

fl::Tensor EncoderModel::forward(fl::Tensor wave) {
  CHECK(wave.getDim() == 1 && wave.getShape(-1) <= InputSamples);

  // pad wave.
  if (wave.getShape(-1) < InputSamples) {
    fl::Tensor pad = fl::F::zeros({InputSamples}, wave.getDType(), wave.getDevice());
    fl::F::copy(wave, pad.slice({0, wave.getShape(-1)}));
    wave = pad;
  }

  fl::Tensor features = fl::F::logMelSpectrogram(wave);
  features = moveAndCastFloat(features, getCtx());

  CHECK(features.getDim() == 2);
  features = features.unsqueeze(0);

  fl::Tensor x = _conv1->forward(features);
  x = fl::F::gelu(x);

  x = _conv2->forward(x);
  x = fl::F::gelu(x);
  x = fl::F::add(x, _posEmbd);

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
    const fl::Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<DecoderInitModel> model{new DecoderInitModel()};
  model->setCtx(ctx);

  int dModel = config.hiddenSize;
  for (int i = 0; i < config.encoderNumLayers; ++i) {
    fl::Context ctxLayer = ctx.withName(lut::sprintf("layer%d", i))
                               .withName(DecoderLayer::CrossAttn);
    model->_kvProjs.emplace_back(
        fl::Linear::create(ctxLayer.withName("kv_proj"), dModel, dModel * 2));
  }
  model->_dModel = dModel;
  return model;
}

void DecoderInitModel::initParameters(const fl::StateMap &stateDict) {
  for (std::shared_ptr<fl::Linear> &layer : _kvProjs) {
    layer->initParameters(stateDict);
  }
}

void DecoderInitModel::forward(fl::StateMap &past, fl::Tensor encoderHidden) {
  CHECK(encoderHidden.getDim() == 3);

  for (int i = 0; i < _kvProjs.size(); ++i) {
    fl::Context ctxLayer = getCtx().withName(lut::sprintf("layer%d", i));
    fl::Context ctxAttn = ctxLayer.withName(DecoderLayer::CrossAttn);

    fl::Tensor x = _kvProjs[i]->forward(encoderHidden);
    fl::Tensor cacheK = x.slice(2, {0, _dModel});
    fl::Tensor cacheV = x.slice(2, {_dModel, 2 * _dModel});

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

std::shared_ptr<Attention> Attention::selfAttn(const fl::Context &ctx, WhisperConfig config) {
  std::shared_ptr<Attention> model{new Attention()};
  model->setCtx(ctx);
  model->initCommon(config);

  model->_proj = fl::Linear::create(
      ctx.withName("qkv_proj"),
      config.hiddenSize,
      config.hiddenSize * 3);
  model->_selfAttn = true;
  return model;
}

std::shared_ptr<Attention> Attention::crossAttn(const fl::Context &ctx, WhisperConfig config) {
  std::shared_ptr<Attention> model{new Attention()};
  model->setCtx(ctx);
  model->initCommon(config);

  model->_proj = fl::Linear::create(ctx.withName("q_proj"), config.hiddenSize, config.hiddenSize);
  model->_selfAttn = false;
  return model;
}

int Attention::getCtxLength(const fl::StateMap &past) const {
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

  _outProj = fl::Linear::create(
      getCtx().withName("out_proj"),
      config.hiddenSize,
      config.hiddenSize);
  _hiddenSize = config.hiddenSize;
  _numHeads = config.encoderNumHeads;

  _namePastK = getCtx().name("k");
  _namePastV = getCtx().name("v");
  _namePastLen = getCtx().name("len");
}

void Attention::initParameters(const fl::StateMap &stateDict) {
  _proj->initParameters(stateDict);
  _outProj->initParameters(stateDict);
}

std::pair<fl::Tensor, fl::Tensor> Attention::getPresentKV(
    fl::StateMap &past,
    fl::Tensor k,
    fl::Tensor v) {
  fl::Tensor pastK, pastV;

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
    fl::Tensor nextK = fl::F::zeros({d0, nextLen, d2}, k.getDType(), k.getDevice());
    fl::Tensor nextV = fl::F::zeros({d0, nextLen, d2}, v.getDType(), v.getDevice());

    if (pastLen) {
      fl::F::copy(pastK.slice(1, {0, pastLen}), nextK.slice(1, {0, pastLen}));
      fl::F::copy(pastV.slice(1, {0, pastLen}), nextV.slice(1, {0, pastLen}));
    }

    past.putTensor(_namePastK, nextK);
    past.putTensor(_namePastV, nextV);

    pastK = nextK;
    pastV = nextV;
  }

  fl::F::copy(k, pastK.slice(1, {pastLen, presentLen}));
  fl::F::copy(v, pastV.slice(1, {pastLen, presentLen}));

  fl::Tensor presentK = pastK.slice(1, {0, presentLen});
  fl::Tensor presentV = pastV.slice(1, {0, presentLen});
  past.putValue<int>(_namePastLen, presentLen);

  return std::make_pair(presentK, presentV);
}

fl::Tensor Attention::forward(fl::StateMap &past, fl::Tensor inputs) {
  CHECK(inputs.getDim() == 3);

  fl::Tensor q, k, v;
  if (_selfAttn) {
    fl::Tensor qkv = _proj->forward(inputs);
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

  fl::Tensor x;
  if (_selfAttn && inputs.getShape(1) > 1) {
    x = fl::F::attention(q, k, v, fl::F::causalMask(q.getShape(2), getCtx().getDevice()));
  } else {
    x = fl::F::attention(q, k, v, fl::Tensor());
  }

  x = fl::F::contiguous(x.transpose(1, 2)).view({bsz, len, _hiddenSize});
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

std::shared_ptr<DecoderLayer> DecoderLayer::fromConfig(
    const fl::Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<DecoderLayer> model{new DecoderLayer()};
  model->setCtx(ctx);

  model->_norm1 = fl::LayerNorm::create(ctx.withName("norm1"), config.hiddenSize);
  model->_norm2 = fl::LayerNorm::create(ctx.withName("norm2"), config.hiddenSize);
  model->_norm3 = fl::LayerNorm::create(ctx.withName("norm3"), config.hiddenSize);
  model->_selfAttn = Attention::selfAttn(ctx.withName(SelfAttn), config);
  model->_crossAttn = Attention::crossAttn(ctx.withName(CrossAttn), config);
  model->_fc1 = fl::Linear::create(ctx.withName("fc1"), config.hiddenSize, config.decoderFfnDim);
  model->_fc2 = fl::Linear::create(ctx.withName("fc2"), config.decoderFfnDim, config.hiddenSize);
  return model;
}

void DecoderLayer::initParameters(const fl::StateMap &stateDict) {
  _norm1->initParameters(stateDict);
  _norm2->initParameters(stateDict);
  _norm3->initParameters(stateDict);
  _selfAttn->initParameters(stateDict);
  _crossAttn->initParameters(stateDict);
  _fc1->initParameters(stateDict);
  _fc2->initParameters(stateDict);
}

fl::Tensor DecoderLayer::forward(fl::StateMap &past, fl::Tensor inputs) {
  fl::Tensor residual = inputs;

  fl::Tensor x = _norm1->forward(inputs);
  x = _selfAttn->forward(past, x);
  x = fl::F::add(x, residual);

  residual = x;
  x = _norm2->forward(x);
  x = _crossAttn->forward(past, x);
  x = fl::F::add(x, residual);

  residual = x;
  x = _norm3->forward(x);
  x = _fc1->forward(x);
  x = fl::F::gelu(x);

  x = _fc2->forward(x);
  x = fl::F::add(x, residual);
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

std::shared_ptr<DecoderModel> DecoderModel::fromConfig(
    const fl::Context &ctx,
    WhisperConfig config) {
  std::shared_ptr<DecoderModel> model{new DecoderModel()};
  model->setCtx(ctx);

  model->_embd = fl::Embedding::create(ctx.withName("embd"), config.hiddenSize, config.vocabSize);
  for (int i = 0; i < config.decoderNumLayers; ++i) {
    model->_layers.emplace_back(
        DecoderLayer::fromConfig(ctx.withName(lut::sprintf("layer%d", i)), config));
  }
  model->_norm = fl::LayerNorm::create(ctx.withName("norm"), config.hiddenSize);
  model->_outProj = fl::Linear::create(
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

void DecoderModel::initParameters(const fl::StateMap &stateDict) {
  fl::Context ctx = getCtx();

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

int DecoderModel::getCtxLength(const fl::StateMap &past) const {
  if (past.hasValue<int>(_namePastLen)) {
    return past.getValue<int>(_namePastLen);
  } else {
    return 0;
  }
}

fl::Tensor DecoderModel::forward(fl::StateMap &past, fl::Tensor inputs) {
  fl::Tensor x = _embd->forward(inputs);

  // positional embedding.
  int pastLen = getCtxLength(past);
  int presentLen = pastLen + inputs.getShape(1);
  x = fl::F::add(x, _posEmbd.slice({pastLen, presentLen}));
  past.putValue<int>(_namePastLen, presentLen);

  for (const std::shared_ptr<DecoderLayer> &layer : _layers) {
    x = layer->forward(past, x);
  }

  x = _norm->forward(x);
  return x;
}

fl::Tensor DecoderModel::forwardLmHead(fl::Tensor inputs) {
  return _outProj->forward(inputs);
}

int DecoderModel::getOutputDim() const {
  return _outputDim;
}

// -----------------------------------------------------------------------------------------------+
// class WhisperModel                                                                             |
// -----------------------------------------------------------------------------------------------+

WhisperModel::WhisperModel() {
}

std::shared_ptr<WhisperModel> WhisperModel::fromPackage(
    const fl::Context &ctx,
    lut::ZipFile *package) {
  std::shared_ptr<lut::Reader> reader = package->open(ModelForGeneration::ModelConfig);
  std::shared_ptr<lut::IniConfig> ini = lut::IniConfig::fromStream(reader.get());

  std::string modelFile = ini->getSection(ModelSection).getString(ModelFileField);
  std::string modelType = ini->getSection(ModelSection).getString(ModelTypeField);

  const lut::IniSection &llamaIni = ini->getSection(modelType);

  std::shared_ptr<WhisperModel> model{new WhisperModel()};
  WhisperConfig llamaConfig = WhisperConfig::loadConfig(llamaIni);

  fl::StateMap stateMap;

  stateMap.read(package->open(modelFile).get());
  model->_encoder = EncoderModel::fromConfig(ctx.withName("encoder"), llamaConfig);
  model->_decoderInit = DecoderInitModel::fromConfig(ctx.withName("decoder"), llamaConfig);
  model->_decoder = DecoderModel::fromConfig(ctx.withName("decoder"), llamaConfig);

  model->_encoder->initParameters(stateMap);
  model->_decoderInit->initParameters(stateMap);
  model->_decoder->initParameters(stateMap);
  model->_modelName = modelType;
  model->_tokenizer = Tokenizer::fromPackage(package);
  return model;
}

void WhisperModel::prefillAudio(fl::StateMap &past, fl::Tensor wave) const {
  fl::Tensor x = _encoder->forward(wave);
  _decoderInit->forward(past, x);
}

fl::Tensor WhisperModel::prefillPrompt(fl::StateMap &past, fl::Tensor inputs) const {
  fl::Tensor x = _decoder->forward(past, inputs);

  x = x.slice(1, {-1, fl::None});
  x = _decoder->forwardLmHead(x);
  return x;
}

fl::Tensor WhisperModel::decode(fl::StateMap &past, fl::LongType inputToken) const {
  std::array<fl::LongType, 1> inputData{inputToken};
  fl::Tensor inputs = fl::Tensor::create<fl::LongType>({1, 1}, inputData);
  inputs = fl::F::to(getDevice(), inputs);

  fl::Tensor x = _decoder->forward(past, inputs);
  x = _decoder->forwardLmHead(x);
  return x;
}

const char *WhisperModel::getName() const {
  return _modelName.c_str();
}

fl::Device WhisperModel::getDevice() const {
  return _decoder->getCtx().getDevice();
}

int WhisperModel::getOutputDim() const {
  return _decoder->getOutputDim();
}

const Vocab *WhisperModel::getVocab() const {
  return _tokenizer->getVocab();
}

}  // namespace whisper
}  // namespace libllm
