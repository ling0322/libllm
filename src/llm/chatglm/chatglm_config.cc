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

#include "llm/chatglm/chatglm_config.h"


namespace libllm {
namespace chatglm {

constexpr char ChatGlmConfig::kSection[];

ChatGlmConfig::ChatGlmConfig()
    : hiddenSize(0),
      vocabSize(0),
      kvChannels(0),
      seqLength(0),
      hiddenSizePerAttentionHead(0),
      multiQueryGroupNum(0),
      normEps(0.0f),
      numLayers(0),
      symbolGMask(0),
      symbolSOP(0),
      symbolEOS(0) {}

ChatGlmConfig ChatGlmConfig::loadConfig(const lut::IniConfig &ini) {
  const lut::IniSection &section = ini.getSection(kSection);

  ChatGlmConfig config;
  config.hiddenSize = section.getInt("hidden_size");
  config.vocabSize = section.getInt("vocab_size");
  config.kvChannels = section.getInt("kv_channels");
  config.seqLength = section.getInt("seq_length");
  config.hiddenSizePerAttentionHead = section.getInt("hidden_size_per_attention_head");
  config.multiQueryGroupNum = section.getInt("multi_query_group_num");
  config.normEps = section.getFloat("norm_eps");
  config.ffnHiddenSize = section.getInt("ffn_hidden_size");
  config.numLayers = section.getInt("num_layers");
  config.symbolGMask = section.getInt("symbol_gmask");
  config.symbolSOP = section.getInt("symbol_sop");
  config.symbolEOS = section.getInt("symbol_eos");

  return config;
}

}  // namespace chatglm
}  // namespace libllm
