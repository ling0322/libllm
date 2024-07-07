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
      encoderNumLayers(0) {
}

WhisperConfig WhisperConfig::loadConfig(const lut::IniSection &section) {
  WhisperConfig config;

  config.hiddenSize = section.getInt("hidden_size");
  config.encoderNumHeads = section.getInt("encoder_num_heads");
  config.encoderFfnDim = section.getInt("encoder_ffn_dim");
  config.encoderNumLayers = section.getInt("encoder_num_layers");
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
// class EncoderLayer |
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
  // pad wave.
  if (wave.getShape(-1) < InputSamples) {
    Tensor pad = F::zeros({InputSamples}, wave.getDType(), wave.getDevice());
    F::print(pad);
    F::copy(wave, pad.slice({0, wave.getShape(-1)}));
    wave = pad;
  }

  Tensor features = F::logMelSpectrogram(wave);

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
  F::print(x);
  exit(22);

  return x;
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
  model->_model = EncoderModel::fromConfig(ctx.withName("encoder"), llamaConfig);
  model->_model->initParameters(stateMap);
  model->_eotId = llamaIni.getInt("eot_token_id");
  model->_modelName = modelType;

  model->initTokenizer(package);
  return model;
}

Tensor WhisperModelForGeneration::prefill(StateMap &past, const Prompt &prompt) const {
  bool hasAudio = false;
  for (const PromptBlock &block : prompt.getBlocks()) {
    if (block.blockType == PromptBlock::Wave) {
      if (hasAudio) {
        throw lut::AbortedError("In whisper model, only one audio input in prompt is supported.");
      }
      Tensor wave = Wave::read(block.data, block.waveFormat);
      _model->forward(wave);
      hasAudio = true;
    } else {
      throw lut::AbortedError(lut::sprintf(
          "unexpected prompt type %s for model %s",
          PromptBlock::typeToString(block.blockType),
          _modelName));
    }
  }

  return Tensor();
}

Tensor WhisperModelForGeneration::decode(StateMap &past, LongType inputToken) const {
  return Tensor();
}

bool WhisperModelForGeneration::isStopToken(int tokenId) const {
  return tokenId == _eotId;
}

const char *WhisperModelForGeneration::getName() const {
  return _modelName.c_str();
}

Device WhisperModelForGeneration::getDevice() const {
  return _model->getCtx().getDevice();
}

}  // namespace whisper
}  // namespace libllm
