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
#include <string>
#include <unordered_map>

#include "libllm/model_for_generation.h"
#include "libllm/prompt.h"
#include "lutil/random.h"

namespace libllm {

struct GenerationConfig {
  int topK;
  float topP;
  float temperature;

  GenerationConfig();
};

/// @brief Given model and the generation config, generate tokens.
class Generator {
 public:
  enum { Sampling, Whisper };

  virtual ~Generator() = default;

  /// @brief set the prompt to prefill.
  /// @param prompt the prompt;
  virtual void setPrompt(const Prompt &prompt) = 0;

  /// @brief generate next token. Return false if generation is finished.
  /// @return if generation is finished.
  virtual bool generate() = 0;

  /// @brief get the piece of current token.
  /// @return piece of current token.
  virtual std::string getToken() = 0;

  /// @brief get the display name of current token.
  /// @return name of current token.
  virtual std::string getTokenName() = 0;
};

class BaseGenerator : public Generator {
 public:
  BaseGenerator(std::shared_ptr<ModelForGeneration> model);
  ~BaseGenerator() = default;

  bool generate() override;
  std::string getToken() override;
  std::string getTokenName() override;
  void setPrompt(const Prompt &prompt) override;

 protected:
  Prompt _prompt;
  StateMap _past;
  std::shared_ptr<ModelForGeneration> _model;
  int _currentToken;

  virtual int searchToken(const Tensor &logits) = 0;
};

class Sampler {
 public:
  Sampler(int topK, float topP);

  int sample(const Tensor &distribution);

 private:
  lut::Random _random;
  int _topK;
  float _topP;
  std::vector<std::pair<int, float>> _topBuffer;

  std::vector<int> getTopK(const Tensor &distribution);
  std::vector<int> getTopP(const Tensor &distribution, lut::Span<const int> topK);
  int sampleTopP(const Tensor &distribution, lut::Span<const int> topP);
};

// generator by sampling.
class SamplingGenerator : public BaseGenerator {
 public:
  static std::shared_ptr<SamplingGenerator> newGenerator(
      const GenerationConfig &config,
      std::shared_ptr<ModelForGeneration> model);
  ~SamplingGenerator() = default;

 protected:
  int searchToken(const Tensor &logits) override;

 private:
  Sampler _sampler;
  float _temperature;

  SamplingGenerator(const GenerationConfig &config, std::shared_ptr<ModelForGeneration> model);
};

}  // namespace libllm
