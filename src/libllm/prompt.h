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

#include <stdint.h>

#include <string>
#include <vector>

#include "libllm/dtype.h"
#include "libllm/wave.h"

namespace libllm {

struct PromptBlock {
  enum Type {
    Text,
    ControlToken,
    Wave,
    Unknown,
  };

  std::string text;
  std::vector<Byte> data;
  WaveFormat waveFormat;
  Type blockType;

  PromptBlock();
  static std::string typeToString(Type blockType);
};

class Prompt {
 public:
  void appendText(const std::string &text);
  void appendControlToken(const std::string &controlToken);
  void appendWave(lut::Span<const Byte> payload, WaveFormat format);

  bool empty() const;

  lut::Span<const PromptBlock> getBlocks() const;

 private:
  std::vector<PromptBlock> _blocks;
};

}  // namespace libllm
