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

#include <memory>

#include "libllm/engine_config.h"
#include "libllm/packed_batch.h"
#include "libllm/prompt.h"
#include "libllm/tokenizer.h"
#include "lutil/zip_file.h"
#include "flint/device.h"
#include "libllm/kv_cache.h"
#include "flint/tensor.h"

namespace libllm {

/// @brief logits processor used in the generator.
class LogitsProcessor {
 public:
  virtual ~LogitsProcessor() = default;

  /// @brief tells the logits processor that an token is emitted by input prompt or generator.
  /// @param tokenId the id of token.
  virtual void notifyToken(int tokenId) = 0;

  /// @brief process the logits tensor.
  /// @param logits the logits fl::Tensor to process.
  virtual void processLogits(fl::Tensor logits) = 0;
};

// base class for language model.
class ModelForGeneration {
 public:
  static constexpr char ModelConfig[] = "model.ini";

  // Cretae instance of ModelForGeneration from local package file (.llmpkg).
  static std::shared_ptr<ModelForGeneration> fromPackage(
      const fl::Device &device,
      lut::ZipFile *package);

  virtual ~ModelForGeneration() = default;

  /// @brief Forward one scheduled batch, update the `past` state and return the logits for the
  /// next token of every sequence in it.
  /// @param past (KVCache): key-value cache.
  /// @param batch the batch a scheduler packed, carrying its tokens.
  /// @return  <float>(numSequences, V): logits of the next token.
  virtual fl::Tensor forward(KVCache &past, const PackedBatch &batch) const = 0;

  /// @brief Used in the prefill phase. Forward the tokens of a request through this language
  /// model, update the `past` state and return the logits for the next token.
  /// @param past (KVCache): key-value cache.
  /// @param tokenIds the tokens to prefill. Must not be empty.
  /// @return  <float>(1, V): logits of the next token.
  virtual fl::Tensor prefill(KVCache &past, lut::Span<const fl::LongType> tokenIds) const = 0;

  /// @brief Encode `prompt` and prefill the tokens it yields.
  /// @param past (KVCache): key-value cache.
  /// @param prompt (Prompt): the input prompt for prefill.
  /// @return  <float>(1, V): logits of the next token.
  fl::Tensor prefill(KVCache &past, const Prompt &prompt) const;

  /// @brief Used in the decodeing phase. Forward input token ids through this language model,
  /// update the `past` state and return the logits for the next token.
  /// @param past (KVCache): key-value cache.
  /// @param inputToken (LongType): the input token.
  /// @return  <float>(1, V): logits of the next token.
  virtual fl::Tensor decode(KVCache &past, fl::LongType inputToken) const = 0;

  /// @brief Return true if tokenId is a stop token. (stop generating texts)
  /// @param tokenId the token id.
  /// @return if tokenId is a stop token.
  virtual bool isStopToken(int tokenId) const = 0;

  // get model name.
  virtual const char *getName() const = 0;

  /// @brief Get fl::Device of the model.
  /// @return the device.
  virtual fl::Device getDevice() const = 0;

  /// @brief get the output dimension of model. This dimention is usually the same as vocabulary
  /// size. But for some specific models, they are different.
  /// @return the output dimension of the model.
  virtual int getOutputDim() const = 0;

  /// @brief Get the model's KV cache layout requirements.
  virtual KVCacheSpec getKVCacheSpec() const = 0;

  /// @brief Forward a dummy batch of `numTokens` tokens, so the device records the peak memory a
  /// full-size forward pass needs. The KV cache is not touched.
  /// @param numTokens Number of tokens in the profiling batch.
  virtual void profileRun(int numTokens) const = 0;

  /// @brief Allocate the paged KV cache storage according to the engine configuration. Requires a
  /// device that reports its memory usage.
  /// @param config The engine configuration.
  void initKVCacheFromConfig(const EngineConfig &config);

  /// @brief Get the KV cache manager owned by this model.
  /// @return A non-owning handle to the KV cache manager. Empty if initKVCacheFromConfig() was not
  /// called.
  std::weak_ptr<KVCacheManager> getKVCacheManager() const;

  /// @brief build prompt from history messages.
  /// @param history the history.
  /// @return the prompt.
  virtual Prompt buildPrompt(lut::Span<const Message> history) const = 0;

  /// @brief Get the vocabulary (tokenId to token string) of the model.
  /// @return The vocabulary.
  const Vocab *getVocab() const;

  /// @brief Encode a prompt into the tokens the model reads.
  /// @param prompt The prompt to encode.
  /// @return The tokens.
  std::vector<fl::LongType> encodePrompt(const Prompt &prompt) const;

  /// @brief Return the number of tokens produced when encoding a prompt.
  int getPromptTokenCount(const Prompt &prompt) const;

 protected:
  std::shared_ptr<Tokenizer> _tokenizer;
  std::shared_ptr<KVCacheManager> _kvCacheManager;

  ModelForGeneration() = default;

  /// @brief Initialize the tokenizer.
  /// @param package The model package.
  void initTokenizer(lut::ZipFile *package);

  /// @brief Encode a prompt block and append the tokens into `tokenIds`. It will ONLY process two
  /// types of promptBlock: text and controlToken. Once other type occured, it will fatal directly.
  /// @param block The block to process.
  /// @param tokenIds The vector to append processed tokens.
  void encodePromptBlock(const PromptBlock &block, std::vector<fl::LongType> &tokenIds) const;
};

}  // namespace libllm
