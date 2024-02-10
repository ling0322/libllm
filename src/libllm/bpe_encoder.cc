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

#include "libllm/bpe_encoder.h"

#include "libllm/lut/strings.h"

namespace libllm {

BPEEncoder::BPEEncoder(const BPEModel *model, const BPEConfig &config)
    : _model(model),
      _config(&config),
      _header(nullptr) {}

void BPEEncoder::initQueue() {
  Symbol *p = _header->next,
         *q = p->next;
  while (q) {
    addBigramIfExist(p, q);
    p = q;
    q = q->next;
  }
}

std::vector<int> BPEEncoder::getSymbolList() {
  std::vector<int> tokenIds;
  Symbol *p = _header->next;
  while (p) {
    tokenIds.push_back(p->tokenId);
    p = p->next;
  }

  return tokenIds;
}

std::vector<int> BPEEncoder::encode(const std::string &s) {
  initSymbolList(s);
  initQueue();

  // loop until there is no bigram candidates
  while (!_queue.empty()) {
    Bigram bigram = _queue.top();
    _queue.pop();

    if (bigram.left->valid() && bigram.right->valid()) {
      Symbol *symbol = mergeBigram(bigram);
      addBigramIfExist(symbol->prev, symbol);
      addBigramIfExist(symbol, symbol->next);
    }
  }

  return getSymbolList();
}

void BPEEncoder::addBigramIfExist(Symbol *left, Symbol *right) {
  if (left == _header || right == nullptr ||
      _model->isSpecialToken(right->tokenId) ||
      _model->isSpecialToken(left->tokenId)) {
    return;
  }

  int mergedTokenId;
  float cost;
  std::tie(mergedTokenId, cost) = _model->findMerge(left->tokenId, right->tokenId);
  if (mergedTokenId == Vocab::kInvalidToken) {
    return;
  }

  Bigram bigram;
  bigram.left = left;
  bigram.right = right;
  bigram.cost = cost;
  bigram.mergedTokenId = mergedTokenId;
  _queue.push(bigram);
}

BPEEncoder::Symbol *BPEEncoder::mergeBigram(const Bigram &bigram) {
  Symbol *left = bigram.left;
  Symbol *right = bigram.right;
  Symbol *next = right->next;
  Symbol *prev = left->prev;

  Symbol *merged = _symbolPool.alloc();
  merged->tokenId = bigram.mergedTokenId;
  merged->next = next;
  merged->prev = prev;
  if (next) {
    next->prev = merged;
  }
  // prev do not need to check since there is a header node
  prev->next = merged;

  right->tokenId = Vocab::kInvalidToken;
  right->next = nullptr;
  right->prev = nullptr;

  left->tokenId = Vocab::kInvalidToken;
  left->next = nullptr;
  left->prev = nullptr;

  return merged;
}

std::vector<std::string> BPEEncoder::splitBytes(const std::string &s) {
  std::vector<std::string> l;
  
  char buffer[2] = " ";
  for (char ch : s) {
    buffer[0] = ch;
    l.emplace_back(buffer);
  }

  return l;
}

BPEEncoder::Symbol *BPEEncoder::appendToken(Symbol *tail, int tokenId) {
  Symbol *symbol = _symbolPool.alloc();

  symbol->tokenId = tokenId;
  symbol->prev = tail;
  symbol->next = nullptr;

  tail->next = symbol;
  
  return symbol;
}

void BPEEncoder::initSymbolList(const std::string &s) {
  // preprocess sentence
  Symbol *header = _symbolPool.alloc();
  header->prev = nullptr;
  header->tokenId = Vocab::kInvalidToken;

  // prefix (_)
  Symbol *prefix = nullptr;
  if (_config->addPrefixSpace) {
    prefix = appendToken(header, _model->getSpaceId());
  } else {
    prefix = header;
  }

  Symbol *prev = prefix;
  std::vector<std::string> initialPieces;
  if (_config->splitByUnicode) {
    initialPieces = lut::splitUtf8(s);
  } else {
    initialPieces = splitBytes(s);
  }

  for (const std::string &piece : initialPieces) {
    int tokenId = piece == " " ? _model->getSpaceId() : _model->findToken(piece);
    if (tokenId == _model->getUnkId() && _model->isByteTokenAvailable()) {
      // symbol not found in the vocab, but byte token available.
      // Then, fallback to byte tokens.
      for (char ch : piece) {
        prev = appendToken(prev, _model->getByteId(static_cast<uint8_t>(ch)));
      }
    } else {
      prev = appendToken(prev, tokenId);
    }
  }

  _header = header;
}

}  // namespace libllm
