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

namespace ly {
namespace internal {

// internal functions for Sprintf()
constexpr int kSprintfMaxWeight = 200;
char sprintfParseFormat(const char **pp_string, std::stringstream &ss);

template<typename T>
bool sprintfCheckType(char type_specifier) {
  switch (type_specifier) {
    case 'd':
    case 'i':
    case 'u':
    case 'x':
    case 'X':
      return std::is_integral<typename std::decay<T>::type>::value &&
             !std::is_same<typename std::decay<T>::type, char>::value;
    case 'p':
      return std::is_pointer<typename std::decay<T>::type>::value;
    case 'e':
    case 'E':
    case 'g':
    case 'G':
    case 'a':
    case 'A':
    case 'f':
      return std::is_floating_point<typename std::decay<T>::type>::value;
    case 's':
      return std::is_same<typename std::decay<T>::type, std::string>::value ||
             std::is_same<typename std::decay<T>::type, char *>::value ||
             std::is_same<typename std::decay<T>::type, const char *>::value;
    case 'c':
      return std::is_same<typename std::decay<T>::type, char>::value;
    case '#':
      return false;
  }

  return true;
}

inline std::string sprintf0(std::stringstream &&ss, const char *pch) {
  while (*pch) {
    if (*pch == '%') {
      char type_specifier = sprintfParseFormat(&pch, ss);
      if (type_specifier != '%') {
        ss << "%!" << type_specifier << "(<null>)";
      } else {
        ss << '%';
      }
    } else {
      ss << *pch++;
    }
  }
  return ss.str();
}
template<typename T, typename... Args>
inline std::string sprintf0(
      std::stringstream &&ss, const char *pch, const T &value, Args &&...args) {
  const auto default_precision = ss.precision();
  const auto default_width = ss.width();
  const auto default_flags = ss.flags();
  const auto default_fill = ss.fill();

  while (*pch != '%' && *pch != '\0') {
    ss << *pch++;
  }

  bool type_correct;
  char type_specifier;
  const char *pch_fmtb = pch;
  if (*pch) {
    type_specifier = sprintfParseFormat(&pch, ss);
    if (type_specifier == '%') {
      ss << '%';
      return sprintf0(std::move(ss), pch, value, std::forward<Args>(args)...);
    }
    type_correct = sprintfCheckType<T>(type_specifier);
    if (type_correct) {
      ss << value;
    }
  } else {
    type_specifier = '_';
    type_correct = false;
  }

  ss.setf(default_flags);
  ss.precision(default_precision);
  ss.width(default_width);
  ss.fill(default_fill);

  if (type_specifier == '#') {
    pch_fmtb++;
    ss << "%!" << std::string(pch_fmtb, pch) <<  "(" << value << ")";
  } else if (!type_correct) {
    ss << "%!" << type_specifier << "(" << value << ")";
  }

  return sprintf0(std::move(ss), pch, std::forward<Args>(args)...);
}

}  // namespace internal
} // namespace ly

