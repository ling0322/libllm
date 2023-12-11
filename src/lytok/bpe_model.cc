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

#include "lytok/bpe_model.h"

#include <limits>
#include "lyutil/error.h"
#include "lyutil/strings.h"

namespace lytok {

BPEModel::BPEModel() : _unkId(kInvalidToken), _isByteTokenAvailable(false) {
  std::fill(_byteId.begin(), _byteId.end(), kInvalidToken);
}

std::unique_ptr<BPEModel> BPEModel::create(const std::string &filename) {
  std::unique_ptr<BPEModel> model(new BPEModel());
  LOG(INFO) << "read tokenizer from " << filename;

  auto fp = lut::ReadableFile::open(filename);

  model->readModel(fp.get());
  model->checkModel();
  return model;
}

void BPEModel::readModel(lut::ReadableFile *fp) {
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

void BPEModel::readMagicNumber(lut::ReadableFile *fp) {
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
      LOG(INFO) << "normal token found: " <<  info.id << ", " << info.tokenString << ", " << info.tokenString;
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
    } else if (info.flag & kControl) {
      LOG(INFO) << "control token found: " <<  info.tokenPiece << ", " << info.tokenString; 
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

BPEModel::TokenInfo BPEModel::readRecord(lut::ReadableFile *fp) {
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
      throw lut::AbortedError(lut::sprintf("bad format, byte %d not exist in model", ch));
  }

  if (_unkId == kInvalidToken) {
    throw lut::AbortedError("bad model (no unknown token)");
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

}  // namespace lytok
