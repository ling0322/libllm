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

#pragma once

#include <memory>

#include "libllm/lut/ini_config.h"
#include "libllm/model_for_generation.h"
#include "libllm/module.h"

namespace libllm {
namespace whisper {

struct WhisperConfig {
  int hiddenSize;

  WhisperConfig();

  static WhisperConfig loadConfig(const lut::IniSection &section);
};

class EncoderModel : public Module {
 public:
  static std::shared_ptr<EncoderModel> fromConfig(const Context &ctx, WhisperConfig config);
  ~EncoderModel();

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  /// @brief Forward the wave through the whisper encoder model and update the key-value cache in
  /// `past`.
  /// @param past the kv_cache to update.
  /// @param wave the input wave.
  void forward(StateMap &past, Tensor wave);

 private:
  static constexpr int FeatDim = 80;
  std::shared_ptr<Conv1D> _conv1;

  EncoderModel();
};

class WhisperModelForGeneration : public ModelForGeneration {
 public:
  static std::shared_ptr<WhisperModelForGeneration> fromPackage(
      const Context &ctx,
      lut::ZipFile *package);

  Tensor prefill(StateMap &past, const Prompt &prompt) const override;
  Tensor decode(StateMap &past, LongType inputToken) const override;

  bool isStopToken(int tokenId) const override;
  const char *getName() const override;
  Device getDevice() const override;

 protected:
  std::shared_ptr<EncoderModel> _model;
  std::string _modelName;
  int _eotId;

  WhisperModelForGeneration();
  void init(const Context &ctx, const lut::IniConfig &config);
  Tensor buildInput(const Prompt &prompt) const;
};

}  // namespace whisper
}  // namespace libllm
