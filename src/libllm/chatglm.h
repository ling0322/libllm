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

#pragma once

#include <math.h>

#include <memory>

#include "libllm/functional.h"
#include "libllm/lut/error.h"
#include "libllm/lut/ini_config.h"
#include "libllm/model_for_generation.h"
#include "libllm/module.h"

namespace libllm {
namespace chatglm {

struct ChatGlmConfig {
  // config section in ini
  static constexpr char kSection[] = "chatglm";

  int hiddenSize;
  int vocabSize;
  int kvChannels;
  int seqLength;
  int hiddenSizePerAttentionHead;
  int multiQueryGroupNum;
  float normEps;
  int ffnHiddenSize;
  int numLayers;

  int symbolGMask;
  int symbolSOP;
  int symbolEOS;

  ChatGlmConfig();
  static ChatGlmConfig loadConfig(const lut::IniConfig &ini);
};

class SelfAttention : public Module {
 public:
  static std::unique_ptr<SelfAttention> create(const Context &ctx, ChatGlmConfig config);

  // implement interface Module
  void initParameters(const StateMap &state_dict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  Tensor forward(StateMap &past, Tensor input, Tensor roPE) const;

 private:
  std::shared_ptr<Linear> _qkvProj;
  std::shared_ptr<Linear> _outProj;

  int _kvProjDim;
  int _qProjDim;
  int _hiddenSizePerHead;

  std::string _namePastK;
  std::string _namePastV;
  std::string _namePastLength;

  SelfAttention() = default;

  int getCtxLength(StateMap *past) const;
};

class MLP : public Module {
 public:
  static std::unique_ptr<MLP> create(const Context &ctx, ChatGlmConfig config);

  // implement interface Module
  void initParameters(const StateMap &state_dict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  Tensor forward(const Tensor &input) const;

 private:
  std::shared_ptr<Linear> _dense1;
  std::shared_ptr<Linear> _dense2;

  int _hiddenSize;
  int _ffnHiddenSize;

  MLP() = default;
};

class GLMBlock : public Module {
 public:
  static std::unique_ptr<GLMBlock> create(const Context &ctx, ChatGlmConfig config);

  // implement interface Module
  void initParameters(const StateMap &state_dict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  Tensor forward(StateMap &past, Tensor input, Tensor roPE) const;

 private:
  std::unique_ptr<RMSNorm> _inputNorm;
  std::unique_ptr<RMSNorm> _attnNorm;
  std::unique_ptr<SelfAttention> _attn;
  std::unique_ptr<MLP> _mlp;

  GLMBlock() = default;
};

// The ChatGLM2 model.
class ChatGlmModel : public Module {
 public:
  // create ChatGLM2 Model.
  static std::unique_ptr<ChatGlmModel> create(const Context &ctx, ChatGlmConfig config);

  // implement interface Module
  void initParameters(const StateMap &state_dict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  Tensor forward(StateMap &past, Tensor input) const;
  Tensor forwardHidden(Tensor hiddenState) const;

 private:
  ChatGlmConfig _config;

  std::unique_ptr<Embedding> _embedding;
  std::vector<std::unique_ptr<GLMBlock>> _blocks;
  std::unique_ptr<RMSNorm> _finalNorm;
  Tensor _rope;

  std::shared_ptr<Linear> _outProj;

  ChatGlmModel();
};

class ChatGlmModelForGeneration : public ModelForGeneration {
 public:
  static std::shared_ptr<ChatGlmModelForGeneration> fromConfig(
      const Context &ctx,
      const lut::IniConfig &config);

  // implements interface ModelForGeneration
  void initParameters(const StateMap &state_dict) override;

  Tensor forward(StateMap &past, Tensor input) const override;
  Tensor forwardHidden(Tensor hidden) const override;
  Tensor buildInput(const std::vector<LongType> &prompt) const override;
  bool isStopToken(int tokenId) const override;
  const char *getName() const override;
  Device getDevice() const override;

 private:
  std::string _modelName;

  std::shared_ptr<ChatGlmModel> _model;
  ChatGlmConfig _config;

  ChatGlmModelForGeneration() = default;
};

}  // namespace chatglm
}  // namespace libllm
