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

#include "lutil/internal/sprintf.h"

#include <string.h>

#include <iomanip>

#include "lutil/log.h"

namespace lut {
namespace internal {

int readDigit(const char **ppch, char *buf, int buf_size) {
  const char *pch = *ppch;
  char *pbuf = buf;

  CHECK(isdigit(*pch));
  *pbuf = *pch;
  ++pbuf;
  ++pch;

  while (isdigit(*pch)) {
    int digit_len = static_cast<int>(pbuf - buf);
    if (digit_len >= buf_size - 1) {
      return kSprintfMaxWeight;
    }

    *pbuf = *pch;
    ++pbuf;
    ++pch;
  }

  *pbuf = '\0';
  *ppch = pch;

  if (strlen(buf) > 3) return kSprintfMaxWeight;
  int n = atoi(buf);

  return n < kSprintfMaxWeight ? n : kSprintfMaxWeight;
}

// parse format string and apply to ss
char sprintfParseFormat(const char **pp_string, std::stringstream &ss) {
  char digit_buffer[32];
  std::string format_string;
  const char *pch = *pp_string;

  CHECK(*pch == '%');
  format_string.push_back(*pch);
  ++pch;

  // %
  if (*pch == '%') {
    *pp_string = ++pch;
    return '%';
  }

  // flags
  switch (*pch) {
    case '-':
      ss << std::left;
      ++pch;
      break;
    case '+':
      ss << std::showpos;
      ++pch;
      break;
    case ' ':
      ++pch;
      break;
    case '0':
      ss << std::setfill('0');
      ++pch;
      break;
  }

  // width
  if (isdigit(*pch)) {
    int n = readDigit(&pch, digit_buffer, sizeof(digit_buffer));
    ss << std::setw(n);
  }

  // precision
  if (*pch == '.') {
    ++pch;

    if (isdigit(*pch)) {
      int n = readDigit(&pch, digit_buffer, sizeof(digit_buffer));
      ss << std::setprecision(n);
    } else {
      *pp_string = ++pch;
      return '#';
    }
  }

  // specifier
  char type_specifier = *pch;
  switch (*pch) {
    case 'd':
    case 'i':
    case 'u':
      ss << std::dec;
      break;
    case 'x':
    case 'p':
      ss << std::hex;
      break;
    case 'X':
      ss << std::hex << std::uppercase;
      break;
    case 'e':
      ss << std::scientific;
      break;
    case 'E':
      ss << std::scientific << std::uppercase;
      break;
    case 'g':
      ss << std::defaultfloat;
      break;
    case 'G':
      ss << std::defaultfloat << std::uppercase;
      break;
    case 'a':
      ss << std::hexfloat;
      break;
    case 'A':
      ss << std::hexfloat << std::uppercase;
      break;
    case 'f':
      ss << std::fixed;
      break;
    case 's':
    case 'c':
      break;
    default:
      *pp_string = *pch ? pch + 1 : pch;
      return '#';
  }
  ++pch;

  *pp_string = pch;
  return type_specifier;
}

}  // namespace internal
}  // namespace lut
