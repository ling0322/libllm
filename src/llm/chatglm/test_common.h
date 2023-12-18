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
#include "ly/nn/test_helper.h"

namespace libllm {
namespace chatglm {

class TestCommon {
 public:
  static ChatGlmConfig getConfig() {
    ChatGlmConfig config;
    config.ffnHiddenSize = 512;
    config.hiddenSize = 256;
    config.hiddenSizePerAttentionHead = 64;
    config.kvChannels = 64;
    config.multiQueryGroupNum = 2;
    config.normEps = 1e-5;
    config.numLayers = 2;
    config.seqLength = 8192;
    config.symbolEOS = 2;
    config.symbolGMask = 98;
    config.symbolSOP = 99;
    config.vocabSize = 100;

    return config;
  }
};
}  // namespace chatglm
}  // namespace libllm
