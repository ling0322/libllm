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

// request.h needs GenerationConfig from this header, so it stays a forward declaration here.
class Request;

struct GenerationConfig {
  int topK;
  float topP;
  float temperature;

  GenerationConfig();
};

class Sampler {
 public:
  Sampler(int topK, float topP);

  int sample(const fl::Tensor &distribution);

 private:
  lut::Random _random;
  int _topK;
  float _topP;
  std::vector<std::pair<int, float>> _topBuffer;

  std::vector<int> getTopK(const fl::Tensor &distribution);
  std::vector<int> getTopP(const fl::Tensor &distribution, lut::Span<const int> topK);
  int sampleTopP(const fl::Tensor &distribution, lut::Span<const int> topP);
};

/// @brief Drives a model through one prompt, handing out one sampled token at a time.
class Scheduler {
 public:
  static std::shared_ptr<Scheduler> create(
      const GenerationConfig &config,
      std::shared_ptr<ModelForGeneration> model);

  ~Scheduler() = default;

  /// @brief set the request to complete.
  /// @param request the request;
  void setRequest(std::shared_ptr<Request> request);

  /// @brief generate next token. Return false if generation is finished.
  /// @return if generation is finished.
  bool generate();

  /// @brief get the piece of current token.
  /// @return piece of current token.
  std::string getToken();

  /// @brief get the display name of current token.
  /// @return name of current token.
  std::string getTokenName();

 private:
  std::shared_ptr<Request> _request;
  KVCache _past;
  std::shared_ptr<ModelForGeneration> _model;
  int _currentToken;

  Sampler _sampler;
  float _temperature;

  Scheduler(const GenerationConfig &config, std::shared_ptr<ModelForGeneration> model);

  int searchToken(const fl::Tensor &logits);
};

}  // namespace libllm
