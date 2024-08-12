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

#include <string>

namespace libllm {

// interface for vocabulary class that supports finding token_id by token string or getting
// token_string abd token_piece by id.
class Vocab {
 public:
  // id for invalid token.
  static constexpr int kInvalidToken = -1;

  virtual ~Vocab() = default;

  // find token id by token string or the byte ord. returns unk_id if the token not exist in the
  // vocabulary.
  virtual int findToken(const std::string &piece) const = 0;

  /// @brief find control token id by string. Exception will be thrown of the control token not
  /// exist.
  /// @param name name of the control token.
  /// @return id of the token.
  virtual int findControlToken(const std::string &name) const = 0;

  // get token bytes by token id. The token id should fall within the range of (0, vocab_size).
  // If the id is unknown token, control token or unused token, an empty string will be returned.
  // To get the display form of a token, please use token_string.
  virtual const std::string &getTokenPiece(int token_id) const = 0;

  // get human readable token representation. The token id should fall within the range of
  // (0, vocab_size).
  virtual const std::string &getTokenString(int token_id) const = 0;

  // total number of tokens in the vocabulary.
  virtual int getVocabSize() const = 0;

  // return true if token is a control token.
  virtual bool isControlToken(int tokenId) const = 0;

  // id for unknown token.
  virtual int getUnkId() const = 0;
};

}  // namespace libllm
