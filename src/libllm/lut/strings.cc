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

#include "libllm/lut/strings.h"

#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <vector>
#include "../../../third_party/utfcpp/utfcpp.h"
#include "libllm/lut/log.h"

namespace lut {

namespace {

template<typename I>
I findFirstNotMatch(I begin, I end, const char *chars) {
  I it = begin;
  for (; it < end; ++it) {
    const char *pch = chars;
    for (; *pch; ++pch) {
      if (*it == *pch) {
        break;
      }
    }

    if (*pch == '\0') {
      break;
    }
  }

  return it;
}

}  // namespace



#if !defined(WCHAR_MAX)
#error WCHAR_MAX not defined!
#endif
#if defined(_MSC_VER) && _MSC_VER <= 1310
#define AL_INTERNAL_WCHAR_IS_UTF16
#elif WCHAR_MAX > 0x10000
#define AL_INTERNAL_WCHAR_IS_UTF32
#else
#define AL_INTERNAL_WCHAR_IS_UTF16
#endif

std::string toUtf8(const std::u16string &u16s) {
  std::string s;
  utf8::utf16to8(u16s.begin(), u16s.end(), std::back_inserter(s));
  return s;
}

std::u16string toUtf16(const std::string &s) {
  std::u16string u16s;
  utf8::utf8to16(s.begin(), s.end(), std::back_inserter(u16s));
  return u16s;
}

std::string toUtf8(const std::u32string &u32s) {
  std::string s;
  utf8::utf32to8(u32s.begin(), u32s.end(), std::back_inserter(s));
  return s;
}

std::u32string toUtf32(const std::string &s) {
  std::u32string u32s;
  utf8::utf8to32(s.begin(), s.end(), std::back_inserter(u32s));
  return u32s;
}

std::wstring toWide(const std::string &s) {
  std::wstring ws;

#if defined(AL_INTERNAL_WCHAR_IS_UTF32)
  utf8::utf8to32(s.begin(), s.end(), std::back_inserter(ws));
#elif defined(AL_INTERNAL_WCHAR_IS_UTF16)
  utf8::utf8to16(s.begin(), s.end(), std::back_inserter(ws));
#else
#error macro AL_INTERNAL_WCHAR_IS_ not defined
#endif

  return ws;
}

std::string toUtf8(const std::wstring &ws) {
  std::string s;

#if defined(AL_INTERNAL_WCHAR_IS_UTF32)
  utf8::utf32to8(ws.begin(), ws.end(), std::back_inserter(s));
#elif defined(AL_INTERNAL_WCHAR_IS_UTF16)
  utf8::utf16to8(ws.begin(), ws.end(), std::back_inserter(s));
#else
#error macro AL_INTERNAL_WCHAR_IS_ not defined
#endif

  return s;
}

std::string trimLeft(const std::string &s, const char *chars) {
  auto it = findFirstNotMatch(s.begin(), s.end(), chars);
  return std::string(it, s.end());
}

std::string trimRight(const std::string &s, const char *chars) {
  auto it = findFirstNotMatch(s.rbegin(), s.rend(), chars);
  auto n_deleted = it - s.rbegin();
  return std::string(s.begin(), s.end() - n_deleted);
}

std::string trim(const std::string &s, const char *chars) {
  auto it_begin = findFirstNotMatch(s.begin(), s.end(), chars);
  if (it_begin == s.end()) {
    return "";
  }

  auto it_r = findFirstNotMatch(s.rbegin(), s.rend(), chars);
  auto n_deleted = it_r - s.rbegin();
  auto it_end = s.end() - n_deleted;

  CHECK(it_end > it_begin);
  return std::string(it_begin, it_end);
}

std::vector<std::string> split(const std::string &str, const std::string &delim) {
  std::vector<std::string> fields;
  size_t start = 0;
  size_t pos = 0;
  while ((pos = str.find(delim, start)) != std::string::npos) {
    fields.emplace_back(str.cbegin() + start, str.cbegin() + pos);
    start = pos + delim.size();
  }
  
  fields.emplace_back(str.cbegin() + start, str.cend());
  return fields;
}

std::string toLower(const std::string &s) {
  std::string lower(s.begin(), s.end());
  std::transform(lower.begin(), lower.end(), lower.begin(), tolower);
  return lower;
}

int atoi(const std::string &s) {
  char *p = nullptr;
  long v = strtol(s.c_str(), &p, 0);
  if (*p == '\0') {
    return static_cast<int>(v);
  } else {
    throw AbortedError(sprintf("invalid integer string: %s", s));
  }
}

float atof(const std::string &s) {
  char *p = nullptr;
  double v = strtof(s.c_str(), &p);
  if (*p == '\0') {
    return static_cast<float>(v);
  } else {
    throw AbortedError(sprintf("invalid float string: %s", s));
  }
}

std::string replace(const std::string &from, const std::string &old, const std::string &repl) {
  size_t pos = 0;
  std::string s = from;
  while((pos = s.find(old, pos)) != std::string::npos) {
    s.replace(pos, old.length(), repl);
    pos += repl.length();
  }

  return s;
}

std::vector<std::string> splitUtf8(const std::string &s) {
  std::vector<std::string> utf8Chars;

  std::string::const_iterator begin = s.begin();
  uint32_t cp = 0;
  char singleChar[] = " ";
  while (begin < s.end()) {
    std::string::const_iterator next = begin;
    try {
      uint32_t cp = utf8::next(next, s.end());
      utf8Chars.emplace_back(begin, next);
      begin = next;
    } catch (const AbortedError &) {
      singleChar[0] = *begin;
      utf8Chars.emplace_back(singleChar);
      ++begin;
    }
  }

  return utf8Chars;
}

template<>
std::string stox<std::string>(const std::string &s) {
  return s;
}

template<>
float stox<float>(const std::string &s) {
  return atof(s);
}

template<>
int stox<int>(const std::string &s) {
  return atoi(s);
}

template<>
bool stox<bool>(const std::string &s) {
  std::string sl = toLower(s);
  if (sl == "true" || sl == "1") {
    return true;
  } else if (sl == "false" || sl == "0") {
    return false;
  } else {
    throw AbortedError(lut::sprintf("invalid bool value: %s", s));
  }

  // never reach here.
  NOT_IMPL();
  return false;
}

template std::string stox<std::string>(const std::string &s);
template float stox<float>(const std::string &s);
template int stox<int>(const std::string &s);
template bool stox<bool>(const std::string &s);

} // namespace lut
