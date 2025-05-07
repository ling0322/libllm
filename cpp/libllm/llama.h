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

#include <memory>

#include "libllm/constants.h"
#include "libllm/functional.h"
#include "libllm/model_for_generation.h"
#include "libllm/module.h"
#include "lutil/error.h"
#include "lutil/ini_config.h"

namespace libllm {
namespace llama {

struct LlamaConfig {
  int hiddenSize;
  int numHeads;
  int numKeyValueHeads;
  int intermediateSize;
  float normEps;
  int numLayers;
  int vocabSize;
  int maxContextLength;
  bool qkvProjBias;

  LlamaConfig();

  static LlamaConfig loadConfig(const lut::IniSection &section);
};

class MLP : public Module {
 public:
  static std::shared_ptr<MLP> create(const Context &ctx, const LlamaConfig &config);

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;
  Tensor forward(Tensor input) const;

 private:
  std::shared_ptr<Linear> _gateUpProj;
  std::shared_ptr<Linear> _downProj;

  int _hiddenSize;
  int _intermediateSize;

  MLP();
};

class Attention : public Module {
 public:
  static std::shared_ptr<Attention> create(const Context &ctx, const LlamaConfig &config);

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;
  Tensor forward(StateMap &past, Tensor input) const;

 private:
  std::shared_ptr<Linear> _qkvProj;
  std::shared_ptr<Linear> _outProj;
  Tensor _roPE;

  std::string _namePastK;
  std::string _namePastV;
  std::string _namePastLen;

  int _hiddenSize;
  int _numHead;
  int _numKeyValueHead;
  int _headDim;
  int _maxCtxLen;
  bool _hasProjBias;

  Attention();

  // get past context length.
  int getCtxLength(const StateMap &past) const;
  Tensor applyRoPE(Tensor x, Tensor roPE) const;
  Tensor rotateHalf(Tensor x) const;
};

class DecodeLayer : public Module {
 public:
  static std::shared_ptr<DecodeLayer> create(const Context &ctx, const LlamaConfig &config);

  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;
  Tensor forward(StateMap &past, Tensor input) const;

 private:
  std::shared_ptr<RMSNorm> _inputNorm;
  std::shared_ptr<RMSNorm> _postAttnNorm;
  std::shared_ptr<Attention> _attn;
  std::shared_ptr<MLP> _mlp;

  DecodeLayer() = default;
};

class LlamaModel : public Module {
 public:
  static constexpr char RoPECtxKey[] = "rope_name";

  static std::shared_ptr<LlamaModel> create(const Context &ctx, LlamaConfig config);
  void initParameters(const StateMap &stateDict) override;
  void initParameters(lut::Random *generator, DType weightType) override;

  Tensor forward(StateMap &past, Tensor input) const;
  Tensor forwardLmHead(Tensor hidden) const;
  int getOutputDim() const;

 private:
  LlamaConfig _config;
  std::shared_ptr<Embedding> _embedding;
  std::shared_ptr<RMSNorm> _norm;
  std::vector<std::shared_ptr<DecodeLayer>> _layers;
  std::shared_ptr<Linear> _outProj;

  LlamaModel() = default;
};

class LlamaModelForGeneration : public ModelForGeneration {
 public:
  static std::shared_ptr<LlamaModelForGeneration> fromPackage(
      const Context &ctx,
      lut::ZipFile *package);

  Tensor prefill(StateMap &past, const Prompt &prompt) const override;
  Tensor decode(StateMap &past, LongType inputToken) const override;

  bool isStopToken(int tokenId) const override;
  const char *getName() const override;
  Device getDevice() const override;
  int getOutputDim() const override;
  Prompt buildPrompt(lut::Span<const Message> history) const override;

 protected:
  std::shared_ptr<LlamaModel> _model;
  std::string _modelName;
  int _eotId;

  LlamaModelForGeneration();
  Tensor buildInput(const Prompt &prompt) const;
};

}  // namespace llama
}  // namespace libllm