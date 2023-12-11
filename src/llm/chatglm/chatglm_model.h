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

#include "ly/ly.h"
#include "lyutil/ini_config.h"
#include "llm/common/model_for_generation.h"
#include "llm/chatglm/chatglm_config.h"
#include "llm/chatglm/glm_block.h"

namespace libllm {
namespace chatglm {

// The ChatGLM2 model.
class ChatGlmModel : public ly::nn::Module {
 public:
  // create ChatGLM2 Model.
  static std::unique_ptr<ChatGlmModel> create(const ly::Context &ctx, ChatGlmConfig config);

  // implement interface nn::Module
  void initParameters(const ly::StateMap &state_dict) override;

  ly::Tensor forward(ly::StateMap &past, ly::Tensor input) const;
  ly::Tensor forwardHidden(ly::Tensor hiddenState) const;

 private:
  ChatGlmConfig _config;

  static constexpr char ChatGlm2[] = "chatglm";
  static constexpr char Embd[] = "embd";
  static constexpr char RoPE[] = "rope";
  static constexpr char Block[] = "block";
  static constexpr char FinalNorm[] = "final_norm";
  static constexpr char OutputWeight[] = "output_weight";

  std::unique_ptr<ly::nn::Embedding> _embedding;
  std::vector<std::unique_ptr<GLMBlock>> _blocks;
  std::unique_ptr<ly::nn::RMSNorm> _finalNorm;
  ly::Tensor _rope;
  ly::Tensor _output;

  ChatGlmModel();
};

}  // namespace chatglm
}  // namespace libllm
