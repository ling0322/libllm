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

#include "libllm/state_map.h"
#include "libllm/tensor.h"
#include "libllm/tokenizer.h"

namespace libllm {

// base class for language model.
class ModelForGeneration {
 public:
  virtual ~ModelForGeneration() = default;

  // Forward input token ids through this language model, update the `past` state and return the
  // hidden state of last layer.
  // Args:
  //   past (StateMap): key-value cache.
  //   inputs <long>(N, L): prompt token ids.
  // Returns:
  //   <float>(N, L, D): hidden state from last layer.
  virtual Tensor forward(StateMap &past, Tensor input) const = 0;

  // Forward the hidden state from last layer and get the logits. hiddenState is usually the
  // return value of forward().
  // Args:
  //   hidden_state <float>(N, L, D): hidden state from last layer.
  // Returns:
  //   <float>(N, L, V): logits. V is vocabulary size.
  virtual Tensor forwardHidden(Tensor hiddenState) const = 0;

  // build model input from the prompt token-ids.
  virtual Tensor buildInput(const std::vector<LongType> &prompt) const = 0;

  /// @brief Return true if tokenId is a stop token. (stop generating texts)
  /// @param tokenId the token id.
  /// @return if tokenId is a stop token.
  virtual bool isStopToken(int tokenId) const = 0;

  // get model name.
  virtual const char *getName() const = 0;

  /// @brief Get device of the model.
  /// @return the device.
  virtual Device getDevice() const = 0;
};

}  // namespace libllm
