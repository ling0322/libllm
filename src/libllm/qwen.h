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
#include "libllm/lut/ini_config.h"
#include "libllm/llama.h"
#include "libllm/model_for_generation.h"

namespace libllm {
namespace qwen {

/// @brief The Qwen model. Model structure of qwen is similiar to llama, the only difference is
/// stopping criteria. So, we re-use the llama model and add specific logic for the stop tokens
/// here.
class QwenModelForGeneration : public llama::LlamaModelForGeneration {
 public:
  static std::shared_ptr<QwenModelForGeneration> fromConfig(
      const Context &ctx,
      const lut::IniConfig &config);

  // noncopyable
  QwenModelForGeneration(QwenModelForGeneration &) = delete;
  QwenModelForGeneration &operator=(QwenModelForGeneration &) = delete;

  // override LlamaModelForGeneration
  bool isStopToken(int tokenId) const override;

 private:
  int _imStartId;
  int _imEndId;

  QwenModelForGeneration();
};

}  // namespace qwen
}  // namespace libllm
