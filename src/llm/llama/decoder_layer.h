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
#include "ly/context.h"
#include "ly/nn/rms_norm.h"
#include "llm/llama/attention.h"
#include "llm/llama/llama_config.h"
#include "llm/llama/mlp.h"

namespace libllm {
namespace llama {

class DecodeLayer : public ly::nn::Module {
 public:
  static std::shared_ptr<DecodeLayer> create(const ly::Context &ctx, const LlamaConfig &config);

  void initParameters(const ly::StateMap &stateDict) override;
  ly::Tensor forward(ly::StateMap &past, ly::Tensor input) const;

 private:
  std::shared_ptr<ly::nn::RMSNorm> _inputNorm;
  std::shared_ptr<ly::nn::RMSNorm> _postAttnNorm;
  std::shared_ptr<Attention> _attn;
  std::shared_ptr<MLP> _mlp;

  DecodeLayer() = default;
};

}  // namespace llama
}  // namespace libllm
