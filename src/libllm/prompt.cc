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

#include "libllm/prompt.h"

namespace libllm {

PromptBlock::PromptBlock()
    : waveFormat(WaveFormat::Unknown),
      blockType(Type::Unknown) {
}

std::string PromptBlock::typeToString(Type blockType) {
  switch (blockType) {
    case Type::ControlToken:
      return "controlToken";
    case Type::Text:
      return "text";
    case Type::Wave:
      return "wave";
    case Type::Unknown:
      return "unknown";
    default:
      NOT_IMPL();
  }
}

void Prompt::appendText(const std::string &text) {
  PromptBlock block;
  block.text = text;
  block.blockType = PromptBlock::Text;

  _blocks.emplace_back(std::move(block));
}

void Prompt::appendControlToken(const std::string &controlToken) {
  PromptBlock block;
  block.text = controlToken;
  block.blockType = PromptBlock::ControlToken;

  _blocks.emplace_back(std::move(block));
}

void Prompt::appendWave(lut::Span<const Byte> payload, WaveFormat format) {
  PromptBlock block;
  block.data = std::vector<Byte>(payload.begin(), payload.end());
  block.waveFormat = format;
  block.blockType = PromptBlock::Wave;

  _blocks.emplace_back(std::move(block));
}

bool Prompt::empty() const {
  return _blocks.empty();
}

lut::Span<const PromptBlock> Prompt::getBlocks() const {
  return _blocks;
}

}  // namespace libllm