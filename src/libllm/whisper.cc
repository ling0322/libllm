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
    : hiddenSize(0) {
}

WhisperConfig WhisperConfig::loadConfig(const lut::IniSection &section) {
  WhisperConfig config;

  config.hiddenSize = section.getInt("hidden_size");
  return config;
}

EncoderModel::~EncoderModel() {
}

EncoderModel::EncoderModel() {
}

void EncoderModel::initParameters(const StateMap &stateDict) {
}

void EncoderModel::initParameters(lut::Random *generator, DType weightType) {
}

void EncoderModel::forward(StateMap &past, Tensor wave) {
  Tensor features = F::logMelSpectrogram(wave);
  Tensor x = _conv1->forward(features);

  F::print(x);
}

std::shared_ptr<EncoderModel> EncoderModel::fromConfig(const Context &ctx, WhisperConfig config) {
  std::shared_ptr<EncoderModel> model{new EncoderModel()};
  model->setCtx(ctx);

  model->_conv1 = Conv1D::create(ctx.withName("conv1"), FeatDim, config.hiddenSize, 3);
  return model;
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
  model->_model = EncoderModel::fromConfig(ctx, llamaConfig);
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
      _model->forward(past, wave);
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
