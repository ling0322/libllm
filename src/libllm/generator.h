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

#include "libllm/model_for_generation.h"
#include "libllm/prompt.h"
#include "libllm/sampler.h"

namespace libllm {

struct GenerationConfig {
  int topK;
  float topP;
  float temperature;

  GenerationConfig();
};

// LLM text generator
class Generator {
 public:
  Generator(GenerationConfig config, std::shared_ptr<ModelForGeneration> model);

  void forwardPrompt(const Prompt &prompt);

  // generate the next word (token). Returns nullptr if the generation is finished.
  const char *nextToken();

  bool stopped() const;

 private:
  GenerationConfig _config;
  Sampler _sampler;
  StateMap _past;
  std::shared_ptr<ModelForGeneration> _model;
  int _currentToken;

  int sampleToken(const Tensor &logits);
};

}  // namespace libllm
