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

#include "lyutil/flags.h"

#include "lyutil/error.h"
#include "lyutil/log.h"
#include "lyutil/strings.h"

namespace ly {

class StrFlagParser : public Flags::Parser {
 public:
  StrFlagParser(std::string *s, const std::string &description);

  void parse(const std::string &arg) override;
  std::string getDescription() const override;

 private:
  std::string *_s;
  std::string _description;
};

StrFlagParser::StrFlagParser(std::string *s, const std::string &description) :
    _s(s),
    _description(description) {}

void StrFlagParser::parse(const std::string &arg) {
  *_s = arg;
}

std::string StrFlagParser::getDescription() const {
  return _description;
}

void Flags::define(const std::string &flag, std::string *s, const std::string &description) {
  CHECK(_parsers.find(flag) == _parsers.end());
  CHECK(s);
  CHECK((!flag.empty()) && flag[0] == '-');

  _parsers.emplace(flag, std::make_unique<StrFlagParser>(s, description));
}

void Flags::parse(int argc, char *argv[]) {
  constexpr int FlagParserStateBegin = 0;
  constexpr int FlagParserStateFlag = 1;

  int state = FlagParserStateBegin;
  std::string flag;
  for (int i = 0; i < argc; ++i) {
    std::string arg = ly::trim(argv[argc]);
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
        throw InvalidArgError(ly::sprintf("unexpected flag: %s", flag));
      parserIt->second->parse(arg);
      state = FlagParserStateBegin;
    } else {
      NOT_IMPL();
    }
  }

  if (state == FlagParserStateFlag)
    throw InvalidArgError(ly::sprintf("no value for flag: %s", flag));
}

} // namespace ly
