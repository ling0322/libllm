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
#include "llm/chatglm/chatglm_config.h"

namespace libllm {
namespace chatglm {

class SelfAttention : public ly::nn::Module {
 public:
  static std::unique_ptr<SelfAttention> create(const ly::Context &ctx, ChatGlmConfig config);

  // implement interface nn::Module
  void initParameters(const ly::StateMap &state_dict) override;

  ly::Tensor forward(ly::StateMap &past, ly::Tensor input, ly::Tensor roPE) const;

 private:
  std::unique_ptr<ly::nn::Linear> _qkvProj;
  ly::Tensor _denseWeight;

  int _kvProjDim;
  int _qProjDim;
  int _hiddenSizePerHead;

  std::string _namePastK;
  std::string _namePastV;
  std::string _namePastLength;

  SelfAttention() = default;

  int getCtxLength(ly::StateMap *past) const;
};

}  // namespace chatglm
}  // namespace libllm
