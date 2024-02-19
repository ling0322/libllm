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

#include "libllm/tokenizer.h"

#include "libllm/lut/error.h"
#include "libllm/lut/strings.h"
#include "libllm/bpe_config.h"
#include "libllm/bpe_encoder.h"
#include "libllm/bpe_model.h"


namespace libllm {

constexpr int Vocab::kInvalidToken;
constexpr char Tokenizer::ConfigFile[];

// BPE tokenizer.
class BPETokenizer : public Tokenizer {
 public:
  static std::unique_ptr<BPETokenizer> fromStream(lut::Reader *fp, BPEConfig config);

  // implement interface Tokenizer
  std::vector<int> encode(const std::string &s) const override;
  const Vocab *getVocab() const override;

 private:
  std::shared_ptr<BPEModel> _model;
  BPEConfig _config;

  BPETokenizer();
};

BPETokenizer::BPETokenizer() {}

std::unique_ptr<BPETokenizer> BPETokenizer::fromStream(lut::Reader *reader, BPEConfig config) {
  std::unique_ptr<BPETokenizer> tokenizer{new BPETokenizer()};
  tokenizer->_model = BPEModel::fromStream(reader);
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

std::shared_ptr<Tokenizer> Tokenizer::fromPackage(lut::ZipFile *package) {
  std::shared_ptr<lut::IniConfig> ini = lut::IniConfig::fromStream(
      package->open(ConfigFile).get());

  lut::IniSection tokenizerSection = ini->getSection("tokenizer");
  std::string type = tokenizerSection.getString("type");
  if (type == "bpe") {
    BPEConfig config = BPEConfig::fromIni(tokenizerSection);
    return BPETokenizer::fromStream(package->open(config.modelFile).get(), config);
  } else {
    throw lut::AbortedError(lut::sprintf("invalid tokenizer type: %s", type));
  }
}

}  // namespace libllm
