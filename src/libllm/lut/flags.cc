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

#include "libllm/lut/flags.h"

#include "libllm/lut/error.h"
#include "libllm/lut/log.h"
#include "libllm/lut/strings.h"

namespace lut {

class StrFlagParser : public Flags::Parser {
 public:
  StrFlagParser(std::string *s, const std::string &description);

  void parse(const std::string &arg) override;
  std::string getUsage() const override;
  std::string getType() const override;

 private:
  std::string *_s;
  std::string _usage;
};

StrFlagParser::StrFlagParser(std::string *s, const std::string &usage) :
    _s(s),
    _usage(usage) {}

void StrFlagParser::parse(const std::string &arg) {
  *_s = arg;
}

std::string StrFlagParser::getUsage() const {
  return _usage;
}

std::string StrFlagParser::getType() const {
  return "string";
}

Flags::Flags(const std::string &usage) : _usage(usage) {}

void Flags::define(const std::string &flag, std::string *s, const std::string &usage) {
  CHECK(_parsers.find(flag) == _parsers.end());
  CHECK(s);
  CHECK((!flag.empty()) && flag[0] == '-');

  _parsers.emplace(flag, std::make_unique<StrFlagParser>(s, usage));
}

void Flags::parse(int argc, char *argv[]) {
  constexpr int FlagParserStateBegin = 0;
  constexpr int FlagParserStateFlag = 1;

  int state = FlagParserStateBegin;
  std::string flag;
  for (int i = 0; i < argc; ++i) {
    std::string arg = lut::trim(argv[i]);
    if (state == FlagParserStateBegin) {
      if (arg.empty()) {
        continue;
      } else if (arg.front() == '-') {
        flag = arg;
        state = FlagParserStateFlag;
      } else {
        _positionalArgs.emplace_back(std::move(arg));
      }
    } else if (state == FlagParserStateFlag) {
      auto parserIt = _parsers.find(flag);
      if (parserIt == _parsers.end())
        throw InvalidArgError(lut::sprintf("unexpected flag: %s", flag));
      parserIt->second->parse(arg);
      state = FlagParserStateBegin;
    } else {
      NOT_IMPL();
    }
  }

  if (state == FlagParserStateFlag)
    throw InvalidArgError(lut::sprintf("no value for flag: %s", flag));
}

void Flags::printUsage() const {
  puts(_usage.c_str());
  for (const auto &it : _parsers) {
    printf("%s %s\n", it.first.c_str(), it.second->getType().c_str());
    std::string usage = it.second->getUsage();
    if (!usage.empty())
      printf("    %s\n", usage.c_str());
  }
}

} // namespace lut
