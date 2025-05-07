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

#include "libllm/context.h"
#include "libllm/prompt.h"
#include "libllm/state_map.h"
#include "libllm/tensor.h"
#include "libllm/tokenizer.h"
#include "lutil/zip_file.h"

namespace libllm {

/// @brief logits processor used in the generator.
class LogitsProcessor {
 public:
  virtual ~LogitsProcessor() = default;

  /// @brief tells the logits processor that an token is emitted by input prompt or generator.
  /// @param tokenId the id of token.
  virtual void notifyToken(int tokenId) = 0;

  /// @brief process the logits tensor.
  /// @param logits the logits tensor to process.
  virtual void processLogits(Tensor logits) = 0;
};

// base class for language model.
class ModelForGeneration {
 public:
  static constexpr char ModelConfig[] = "model.ini";

  // Cretae instance of ModelForGeneration from local package file (.llmpkg).
  static std::shared_ptr<ModelForGeneration> fromPackage(const Context &ctx, lut::ZipFile *package);

  virtual ~ModelForGeneration() = default;

  /// @brief Used in the prefill phase. Forward the input prompt through this language model, update
  /// the `past` state and return the logits for the next token.
  /// @param past (StateMap): key-value cache.
  /// @param prompt (Prompt): the input prompt for prefill.
  /// @return  <float>(N, 1, V): hidden state from last layer.
  virtual Tensor prefill(StateMap &past, const Prompt &prompt) const = 0;

  /// @brief Used in the decodeing phase. Forward input token ids through this language model,
  /// update the `past` state and return the logits for the next token.
  /// @param past (StateMap): key-value cache.
  /// @param inputToken (LongType): the input token.
  /// @return  <float>(N, 1, V): hidden state from last layer.
  virtual Tensor decode(StateMap &past, LongType inputToken) const = 0;

  /// @brief Return true if tokenId is a stop token. (stop generating texts)
  /// @param tokenId the token id.
  /// @return if tokenId is a stop token.
  virtual bool isStopToken(int tokenId) const = 0;

  // get model name.
  virtual const char *getName() const = 0;

  /// @brief Get device of the model.
  /// @return the device.
  virtual Device getDevice() const = 0;

  /// @brief get the output dimension of model. This dimention is usually the same as vocabulary
  /// size. But for some specific models, they are different.
  /// @return the output dimension of the model.
  virtual int getOutputDim() const = 0;

  /// @brief build prompt from history messages.
  /// @param history the history.
  /// @return the prompt.
  virtual Prompt buildPrompt(lut::Span<const Message> history) const = 0;

  /// @brief Get the vocabulary (tokenId to token string) of the model.
  /// @return The vocabulary.
  const Vocab *getVocab() const;

 protected:
  std::shared_ptr<Tokenizer> _tokenizer;

  ModelForGeneration() = default;

  /// @brief Initialize the tokenizer.
  /// @param package The model package.
  void initTokenizer(lut::ZipFile *package);

  /// @brief Encode a prompt block and append the tokens into `tokenIds`. It will ONLY process two
  /// types of promptBlock: text and controlToken. Once other type occured, it will fatal directly.
  /// @param block The block to process.
  /// @param tokenIds The vector to append processed tokens.
  void encodePromptBlock(const PromptBlock &block, std::vector<LongType> &tokenIds) const;
};

}  // namespace libllm
