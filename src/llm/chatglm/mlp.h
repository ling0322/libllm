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
#include "llm/chatglm/chatglm_config.h"
#include "llm/chatglm/self_attention.h"

namespace libllm {
namespace chatglm {
    
class MLP : public ly::nn::Module {
 public:
  static std::unique_ptr<MLP> create(const ly::Context &ctx, ChatGlmConfig config);

  // implement interface nn::Module
  void initParameters(const ly::StateMap &state_dict) override;
  void initParameters(lut::Random *generator, ly::DType weightType) override;

  ly::Tensor forward(const ly::Tensor &input) const;

 private:
  ly::Tensor _dense1Weight;
  ly::Tensor _dense2Weight;

  int _hiddenSize;
  int _ffnHiddenSize;

  MLP() = default;
};

}  // namespace chatglm
}  // namespace libllm
