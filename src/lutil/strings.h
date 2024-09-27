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

#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "lutil/internal/sprintf.h"

namespace lut {

std::string toUtf8(const std::u16string &u16s);
std::string toUtf8(const std::wstring &ws);
std::string toUtf8(const std::u32string &u32s);
std::u16string toUtf16(const std::string &s);
std::u32string toUtf32(const std::string &s);
std::wstring toWide(const std::string &s);

std::string trimLeft(const std::string &s, const char *chars = " \t\r\n");
std::string trimRight(const std::string &s, const char *chars = " \t\r\n");
std::string trim(const std::string &s, const char *chars = " \t\r\n");
std::vector<std::string> split(const std::string &str, const std::string &delim);

std::string replace(const std::string &s, const std::string &old, const std::string &repl);
std::string toLower(const std::string &s);

// string to int, float... throw AbortedException if parsing failed.
int parseInt(const std::string &s);
float parseFloat(const std::string &s);
bool parseBool(const std::string &s);

// split s string into utf-8 characters (string),
std::vector<std::string> splitUtf8(const std::string &s);

// String formatting, for example:
//   lut::sprintf("%s %d", "foo", 233);
template<typename... Args>
inline std::string sprintf(const std::string &fmt, Args &&...args) {
  return internal::sprintf0(std::stringstream(), fmt.c_str(), std::forward<Args>(args)...);
}

}  // namespace lut
