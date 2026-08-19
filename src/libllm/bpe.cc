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

#include "libllm/bpe.h"

#include <limits>

#include "lutil/error.h"
#include "lutil/strings.h"

namespace libllm {

// -----------------------------------------------------------------------------------------------+
// class BPEConfig                                                                                |
// -----------------------------------------------------------------------------------------------+

BPEConfig::BPEConfig()
    : addPrefixSpace(true),
      splitByUnicode(true) {}

BPEConfig BPEConfig::fromIni(const lut::IniSection &iniSection) {
  BPEConfig config;

  config.addPrefixSpace = iniSection.getBool("add_prefix_space");
  config.splitByUnicode = iniSection.getBool("split_by_unicode");
  config.modelFile = iniSection.getPath("model_file").string();

  return config;
}

// -----------------------------------------------------------------------------------------------+
// class BPEModel                                                                                 |
// -----------------------------------------------------------------------------------------------+

BPEModel::BPEModel()
    : _unkId(kInvalidToken),
      _isByteTokenAvailable(false) {
  std::fill(_byteId.begin(), _byteId.end(), kInvalidToken);
}

std::shared_ptr<BPEModel> BPEModel::fromStream(lut::Reader *reader) {
  std::shared_ptr<BPEModel> model(new BPEModel());

  model->readModel(reader);
  model->checkModel();
  return model;
}

void BPEModel::readModel(lut::Reader *fp) {
  std::string s = fp->readString(4);
  if (s != "LLsp") {
    throw lut::AbortedError("bad format (header)");
  }

  int32_t numTokens = fp->readValue<int32_t>();
  readMagicNumber(fp);

  // read the list of token info
  _tokens.clear();
  for (int tokenId = 0; tokenId < numTokens; ++tokenId) {
    TokenInfo info = readRecord(fp);
    info.id = tokenId;

    _tokens.emplace_back(std::move(info));
  }

  readMagicNumber(fp);
  initModel();
}

void BPEModel::readMagicNumber(lut::Reader *fp) {
  // ensure magic number
  int16_t magic_number = fp->readValue<int16_t>();
  if (magic_number != kMagicNumber) {
    throw lut::AbortedError("bad format (magic number)");
  }
}

void BPEModel::initModel() {
  // build token_dict_ and byte_id_
  _tokenDict.clear();

  for (const TokenInfo &info : _tokens) {
    if (!info.flag) {
      // flag == 0 means it's a normal token (not control token, unknown token
      // or unused token)
      _tokenDict[info.tokenPiece] = &info;
    } else if (info.flag & kByte) {
      // single byte token
      CHECK(info.tokenPiece.size() == 1);
      _isByteTokenAvailable = true;
      _byteId[static_cast<uint8_t>(info.tokenPiece[0])] = info.id;
    } else if (info.flag & kUnknown) {
      if (_unkId != kInvalidToken) {
        throw lut::AbortedError("bad format (too many unknown tokens)");
      }
      _unkId = info.id;

      _controlTokenDict[info.tokenString] = &info;
    } else if (info.flag & kControl) {
      _controlTokenDict[info.tokenString] = &info;
    }
  }

  // find id for space character
  auto itSpace = _tokenDict.find(" ");
  if (itSpace == _tokenDict.end()) {
    throw lut::AbortedError("bad format (no symbol for space)");
  }
  _spaceId = itSpace->second->id;
}

BPEModel::TokenInfo BPEModel::readRecord(lut::Reader *fp) {
  TokenInfo info;
  info.flag = fp->readValue<int8_t>();

  // raw piece.
  int nBytes = fp->readValue<uint8_t>();
  std::string piece;
  if (nBytes) {
    piece = fp->readString(nBytes);
  }
  info.tokenPiece = std::move(piece);
  if ((info.flag & kByte) && info.tokenPiece.size() != 1) {
    throw lut::AbortedError("bad format (byte)");
  }

  // piece display.
  nBytes = fp->readValue<uint8_t>();
  std::string pieceDisplay;
  if (nBytes) {
    pieceDisplay = fp->readString(nBytes);
  }
  info.tokenString = std::move(pieceDisplay);

  // weight.
  info.weight = fp->readValue<float>();

  return info;
}

void BPEModel::checkModel() {
  for (int ch = 0; ch < 256 && _isByteTokenAvailable; ++ch) {
    if (_byteId[ch] == kInvalidToken)
      THROW(Aborted, lut::sprintf("bad format, byte %d not exist in model", ch));
  }
}

const std::string &BPEModel::getTokenString(int tokenId) const {
  CHECK(tokenId >= 0 && tokenId < getVocabSize());

  return _tokens[tokenId].tokenString;
}

const std::string &BPEModel::getTokenPiece(int tokenId) const {
  CHECK(tokenId >= 0 && tokenId < getVocabSize());

  return _tokens[tokenId].tokenPiece;
}

int BPEModel::findToken(const std::string &token) const {
  auto it = _tokenDict.find(token);
  if (it == _tokenDict.end()) {
    return getUnkId();
  }

  return it->second->id;
}

int BPEModel::findControlToken(const std::string &name) const {
  auto it = _controlTokenDict.find(name);
  if (it == _controlTokenDict.end()) {
    throw lut::AbortedError("control token not found: " + name);
  }

  return it->second->id;
}

int BPEModel::getByteId(int ord) const {
  CHECK(ord >= 0 && ord < 256);
  return _byteId[ord];
}

int BPEModel::getVocabSize() const {
  return static_cast<int>(_tokens.size());
}

int BPEModel::getUnkId() const {
  return _unkId;
}

int BPEModel::getSpaceId() const {
  return _spaceId;
}

bool BPEModel::isSpecialToken(int token_id) const {
  return _tokens[token_id].isSpecialToken();
}

bool BPEModel::isControlToken(int tokenId) const {
  return _tokens[tokenId].isSpecialToken();
}

int32_t BPEModel::getTokenFlag(int token_id) const {
  return _tokens[token_id].flag;
}

std::pair<int, float> BPEModel::findMerge(int left, int right) const {
  std::string left_tok = _tokens[left].tokenPiece;
  std::string right_tok = _tokens[right].tokenPiece;
  std::string tok = left_tok + right_tok;

  auto it = _tokenDict.find(tok);
  if (it == _tokenDict.end()) {
    return std::make_pair(Vocab::kInvalidToken, std::numeric_limits<float>::infinity());
  } else {
    return std::make_pair(it->second->id, -it->second->weight);
  }
}

// -----------------------------------------------------------------------------------------------+
// class BPEEncoder                                                                               |
// -----------------------------------------------------------------------------------------------+

BPEEncoder::BPEEncoder(const BPEModel *model, const BPEConfig &config)
    : _model(model),
      _config(&config),
      _header(nullptr) {
}

void BPEEncoder::initQueue() {
  Symbol *p = _header->next, *q = p->next;
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
  if (left == _header || right == nullptr || _model->isSpecialToken(right->tokenId) ||
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
