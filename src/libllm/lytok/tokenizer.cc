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

#include "lytok/tokenizer.h"

#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "lytok/bpe_config.h"
#include "lytok/bpe_encoder.h"
#include "lytok/bpe_model.h"


namespace lytok {

constexpr int Vocab::kInvalidToken;

// BPE tokenizer.
class BPETokenizer : public Tokenizer {
 public:
  static std::unique_ptr<BPETokenizer> create(const BPEConfig &config);

  // implement interface Tokenizer
  std::vector<int> encode(const std::string &s) const override;
  const Vocab *getVocab() const override;

 private:
  std::unique_ptr<BPEModel> _model;
  BPEConfig _config;

  BPETokenizer();
};

BPETokenizer::BPETokenizer() {}

std::unique_ptr<BPETokenizer> BPETokenizer::create(const BPEConfig &config) {
  auto model = BPEModel::create(config.modelFile);

  std::unique_ptr<BPETokenizer> tokenizer{new BPETokenizer()};
  tokenizer->_model = std::move(model);
  tokenizer->_config = config;

  return tokenizer;
}

std::vector<int> BPETokenizer::encode(const std::string &s) const {
  BPEEncoder encoder(_model.get(), _config);
  return encoder.encode(s);
}

const Vocab *BPETokenizer::getVocab() const {
  return _model.get();
}

// -- class Tokenizer ----------

std::unique_ptr<Tokenizer> Tokenizer::create(const ly::IniSection &config) {
  std::string type = config.getString("type");
  if (type == "bpe") {
    auto bpe_config = BPEConfig::fromIni(config);

    return BPETokenizer::create(*bpe_config);
  } else {
    throw ly::AbortedError(ly::sprintf("invalid tokenizer type: %s", type));
  }
}

}  // namespace lytok
