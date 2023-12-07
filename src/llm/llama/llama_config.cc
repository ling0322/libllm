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

#include "llm/llama/llama_config.h"

#include "llm/common/constants.h"

namespace libllm {
namespace llama {

LlamaConfig::LlamaConfig() :
    hiddenSize(0),
    numHeads(0),
    intermediateSize(0),
    normEps(0.0f),
    numLayers(0),
    vocabSize(0),
    maxContextLength(0) {}

LlamaConfig LlamaConfig::loadConfig(const lut::IniConfig &iniConfig) {
  const lut::IniSection &section = iniConfig.getSection(Llama2Section);
  LlamaConfig config;

  config.hiddenSize = section.getInt("hidden_size");
  config.numHeads = section.getInt("num_heads");
  config.intermediateSize = section.getInt("intermediate_size");
  config.normEps = section.getFloat("norm_eps");
  config.numLayers = section.getInt("num_layers");
  config.vocabSize = section.getInt("vocab_size");
  config.maxContextLength = section.getInt("max_ctx_length");

  return config;
}

}  // namespace llama
}  // namespace libllm
