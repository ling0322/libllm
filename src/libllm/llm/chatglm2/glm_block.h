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

#include "llyn/llyn.h"
#include "lyutil/ini_config.h"
#include "llm/chatglm2/chatglm2_config.h"
#include "llm/chatglm2/mlp.h"
#include "llm/chatglm2/self_attention.h"

namespace libllm {
namespace chatglm2 {
    
class GLMBlock : public llyn::nn::Module {
 public:
  static std::unique_ptr<GLMBlock> create(const llyn::Context &ctx, ChatGLM2Config config);

  void initParameters(const llyn::StateMap &stateDict) override;
  llyn::Tensor forward(llyn::StateMap &past, llyn::Tensor input, llyn::Tensor roPE) const;

 private:
  std::unique_ptr<llyn::nn::RMSNorm> _inputNorm;
  std::unique_ptr<llyn::nn::RMSNorm> _attnNorm;
  std::unique_ptr<SelfAttention> _attn;
  std::unique_ptr<MLP> _mlp;
  llyn::Context _ctx;

  GLMBlock() = default;
};

}  // namespace chatglm2
}  // namespace libllm
