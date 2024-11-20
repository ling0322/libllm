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

#include "libllm/llama.h"
#include "libllm/model_for_generation.h"
#include "lutil/ini_config.h"

namespace libllm {
namespace index {

/// @brief The Bilibili index model. Model structure of index is the same as llama, the only
/// difference is the prompt.
class IndexModelForGeneration : public llama::LlamaModelForGeneration {
 public:
  static std::shared_ptr<IndexModelForGeneration> fromPackage(
      const Context &ctx,
      lut::ZipFile *package);

  // noncopyable
  IndexModelForGeneration(IndexModelForGeneration &) = delete;
  IndexModelForGeneration &operator=(IndexModelForGeneration &) = delete;

  // override LlamaModelForGeneration
  Prompt buildPrompt(lut::Span<const Message> history) const override;

 private:
  IndexModelForGeneration() = default;
};

}  // namespace index
}  // namespace libllm
