// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include <string>
#include <vector>

#include "flint/tensor.h"
#include "libllm/scheduler.h"
#include "lutil/span.h"

namespace libllm {

/// @brief One completion the engine is working on. It owns the tokens of the request, the ones of
/// the prompt as well as the ones generated so far, and remembers how many of them the model
/// already forwarded into the KV cache.
class Request {
 public:
  /// @brief Create a request.
  /// @param requestId Identifies the request among the ones the engine holds.
  /// @param promptTokenIds Tokens of the prompt. Must not be empty.
  /// @param config Generation configuration to apply to this request.
  Request(
      std::string requestId,
      std::vector<fl::LongType> promptTokenIds,
      const GenerationConfig &config);

  /// @brief Get the id of this request.
  const std::string &getId() const;

  /// @brief Get the generation configuration of this request.
  const GenerationConfig &getConfig() const;

  /// @brief Get the prompt tokens followed by the tokens generated so far.
  lut::Span<const fl::LongType> getTokenIds() const;

  /// @brief Get the number of tokens the model already forwarded.
  int getNumComputedTokens() const;

  /// @brief Tell that the model forwarded `numTokens` further tokens of this request.
  /// @param numTokens Number of tokens forwarded, never more than the ones still left.
  void advanceComputedTokens(int numTokens);

  /// @brief Get the number of leading tokens of this request whose keys and values the KV cache
  /// already holds. The tokens a forward pass adds on top of it are its query.
  int getContextLength() const;

  /// @brief Set the number of leading tokens whose keys and values the KV cache holds.
  /// @param contextLength The new context length.
  void setContextLength(int contextLength);

  /// @brief Get the KV cache blocks of this request, in the order its tokens fill them.
  lut::Span<const int> getBlockIds() const;

  /// @brief Hand further KV cache blocks to this request. They go after the ones it already has.
  /// @param blockIds Ids of the blocks to add.
  void addBlockIds(lut::Span<const int> blockIds);

  /// @brief Forget the KV cache blocks of this request. The caller returns them to the manager.
  void clearBlockIds();

  /// @brief Append a token the model generated for this request.
  /// @param tokenId The generated token.
  void appendToken(fl::LongType tokenId);

  /// @brief Return true once this request wants no further token.
  bool isFinished() const;

  /// @brief Stop generating for this request.
  void finish();

 private:
  std::string _id;
  GenerationConfig _config;

  // the prompt tokens, followed by the generated ones.
  std::vector<fl::LongType> _tokenIds;
  int _numComputedTokens;

  int _contextLength;
  std::vector<int> _blockIds;

  bool _finished;
};

}  // namespace libllm
