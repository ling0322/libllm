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

#include "catch2/catch_amalgamated.hpp"
#include "lyutil/ini_config.h"
#include "lyutil/reader.h"
#include "lyutil/strings.h"
#include "lytok/tokenizer.h"

using namespace tokenify;


std::vector<std::string> encodeAsPieces(const Tokenizer *tokenizer, const std::string &s) {
  std::vector<std::string> tokens;
  std::vector<int> tokenIds = tokenizer->encode(s);

  const Vocab *vocab = tokenizer->getVocab();
  for (int tokenId : tokenIds) {
    tokens.emplace_back(vocab->getTokenString(tokenId));
  }
  return tokens;
}

void testTokenizer(const std::string &ini_file, const std::string &test_case) {
  auto config = ly::IniConfig::read(ini_file);
  const ly::IniSection &section = config->getSection("tokenizer");

  auto tokenizer = Tokenizer::create(section);
  auto fp = ly::ReadableFile::open(test_case);
  ly::Scanner scanner(fp.get());
  while (scanner.scan()) {
    std::string line = ly::trimRight(scanner.getText(), "\r\n");
    auto row = ly::split(line, "\t");
    CATCH_REQUIRE(row.size() == 2);

    auto sentence = std::move(row[0]);
    auto ref_pieces = ly::split(ly::trim(row[1]), " ");
    auto pieces = encodeAsPieces(tokenizer.get(), sentence);

    CATCH_REQUIRE(pieces == ref_pieces);
  }
}

CATCH_TEST_CASE"tokenizer works", "[tokenify][tokenizer]") {
  testTokenizer("data/test/gpt2_bpe.tokenizer.ini",
                "data/test/gpt2_bpe.tokenizer.test_cases.txt");

  testTokenizer("data/test/llama_spm.tokenizer.ini",
                "data/test/llama_spm.tokenizer.test_cases.txt");

  testTokenizer("data/test/chatglm2.tokenizer.ini",
                "data/test/chatglm2.tokenizer.test_cases.txt");
}
