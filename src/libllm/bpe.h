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

#include <array>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "libllm/vocab.h"
#include "lutil/ini_config.h"
#include "lutil/noncopyable.h"
#include "lutil/pool.h"
#include "lutil/reader.h"

namespace libllm {

// config for BPE tokenizer.
struct BPEConfig {
  // path of the BPE model.
  std::string modelFile;

  // true if add a space into the begining of text.
  bool addPrefixSpace;

  // true if split by unicode characters before merging. false if split by byte.
  bool splitByUnicode;

  // create the BPE config from ini config.
  static BPEConfig fromIni(const lut::IniSection &config);

  // contructor for the default config.
  BPEConfig();
};

// Store tne data from sentence-piece BPE model.
class BPEModel : public Vocab, private lut::NonCopyable {
 public:
  // token flags.
  static constexpr int kUnknown = 1;
  static constexpr int kControl = 2;
  static constexpr int kByte = 4;
  static constexpr int kUnused = 8;

  // Read the BpeModel from file. It could not read a SPM model directly, instead, we need to
  // convert the SPM model using tokenizer_exporter.py.
  static std::shared_ptr<BPEModel> fromStream(lut::Reader *reader);

  // implement interface Vocab
  int findToken(const std::string &token) const override;
  int findControlToken(const std::string &name) const override;
  const std::string &getTokenPiece(int token_id) const override;
  const std::string &getTokenString(int token_id) const override;
  int getVocabSize() const override;
  bool isControlToken(int tokenId) const override;

  /// @brief Get the id of <unknown> token. If no <unknown> tag exists in the model, return
  /// Vocab::kInvalidToken instead.
  /// @return unknown token id.
  int getUnkId() const override;

  // given the left and right token-ids, find the merged token-id and cost, return as (id, cost).
  // If no merge record found, return (Vocab::kInvalidToken, +inf)
  std::pair<int, float> findMerge(int left, int right) const;

  // token-id for space token.
  int getSpaceId() const;

  // return true if the token is either Unknown, Control, SingleByte or Unsed.
  bool isSpecialToken(int token_id) const;

  // returns flag of a token.
  int32_t getTokenFlag(int token_id) const;

  // get token-id for a single byte.
  int getByteId(int byte) const;
  bool isByteTokenAvailable() const {
    return _isByteTokenAvailable;
  }

 private:
  struct TokenInfo;

  static constexpr int16_t kMagicNumber = 0x55aa;

  std::unordered_map<std::string, const TokenInfo *> _tokenDict;
  std::unordered_map<std::string, const TokenInfo *> _controlTokenDict;
  std::vector<TokenInfo> _tokens;

  // standalone tokens for byte.
  std::array<int, 256> _byteId;
  bool _isByteTokenAvailable;

  int _unkId;
  int _spaceId;

  BPEModel();

  // read model from fp
  void readModel(lut::Reader *fp);
  void readMagicNumber(lut::Reader *fp);
  TokenInfo readRecord(lut::Reader *fp);
  void initModel();
  void checkModel();
};

struct BPEModel::TokenInfo {
  int id;
  float weight;
  std::string tokenPiece;
  std::string tokenString;
  int8_t flag;

  constexpr bool isSpecialToken() const {
    return flag != 0;
  }
};

// String to BPE token-ids encoder.
class BPEEncoder : private lut::NonCopyable {
 public:
  BPEEncoder(const BPEModel *model, const BPEConfig &config);

  // encode string to token ids.
  std::vector<int> encode(const std::string &s);

 private:
  static constexpr int kSymbolPoolBlockSize = 256;

  // symbol linked list
  struct Symbol {
    Symbol *prev;
    Symbol *next;
    int tokenId;

    bool valid() const {
      return tokenId != Vocab::kInvalidToken;
    }
  };

  struct Bigram {
    Symbol *left;
    Symbol *right;
    float cost;
    int mergedTokenId;

    bool operator>(const Bigram &rhs) const {
      return cost > rhs.cost;
    }
  };

  const BPEModel *_model;
  const BPEConfig *_config;
  lut::Pool<Symbol, kSymbolPoolBlockSize> _symbolPool;
  Symbol *_header;
  std::priority_queue<Bigram, std::vector<Bigram>, std::greater<Bigram>> _queue;

  // initialize the symbol linked list from string `s` and store the pointer of header node to
  // `header_`.
  void initSymbolList(const std::string &s);

  // initialize the queue by putting all possible two-bytes bigram to queue
  void initQueue();

  // append a token to the tail of symbol linked-list. Returns the new tail pointer.
  Symbol *appendToken(Symbol *tail, int token_id);

  // split string `s` into a list of single-byte strings.
  std::vector<std::string> splitBytes(const std::string &s);

  // add bigram (left, right) to queue if token left+right exists
  void addBigramIfExist(Symbol *left, Symbol *right);

  // merge bigram (left, right) into one symbol, then clear original left and right symbols and
  // return pointer to the merged one.
  Symbol *mergeBigram(const Bigram &bigram);

  // get the final symbol list from linked list pointered by header_
  std::vector<int> getSymbolList();
};

}  // namespace libllm
