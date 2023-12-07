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

#include <queue>
#include <vector>
#include "lyutil/noncopyable.h"
#include "lyutil/pool.h"
#include "lytok/bpe_config.h"
#include "lytok/bpe_model.h"

namespace lytok {


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

    bool valid() const { return tokenId != Vocab::kInvalidToken; }
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

}  // namespace lytok
