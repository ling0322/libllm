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

#include <stdint.h>

#include <string>

namespace lut {

double now();

class Duration {
 public:
  constexpr Duration()
      : _nanoseconds(0) {
  }

  static constexpr int64_t Millisecond = 1000000;
  static constexpr int64_t Second = 1000 * Millisecond;
  static constexpr int64_t Minute = 60 * Second;
  static constexpr int64_t Hour = 60 * Minute;

  static Duration nanoseconds(int64_t ns) {
    return Duration(ns);
  }

  static Duration milliseconds(int64_t ms) {
    return Duration(ms * 1000000);
  }

  static Duration seconds(int64_t s) {
    return Duration(s * 1000000000);
  }

  constexpr int64_t totalNanoseconds() const {
    return _nanoseconds;
  }

  constexpr Duration operator*(Duration rhs) const {
    return Duration(_nanoseconds * rhs._nanoseconds);
  }

  constexpr Duration operator+(Duration rhs) const {
    return Duration(_nanoseconds + rhs._nanoseconds);
  }

  constexpr Duration &operator+=(Duration rhs) {
    _nanoseconds = rhs._nanoseconds;
    return *this;
  }

  constexpr Duration operator-(Duration rhs) const {
    return Duration(_nanoseconds - rhs._nanoseconds);
  }

  constexpr bool operator>(Duration rhs) const {
    return _nanoseconds > rhs._nanoseconds;
  }

  constexpr bool operator>=(Duration rhs) const {
    return _nanoseconds >= rhs._nanoseconds;
  }

  constexpr bool operator<(Duration rhs) const {
    return _nanoseconds < rhs._nanoseconds;
  }

  constexpr bool operator<=(Duration rhs) const {
    return _nanoseconds <= rhs._nanoseconds;
  }

  constexpr bool operator==(Duration rhs) const {
    return _nanoseconds == rhs._nanoseconds;
  }

  std::string toString() const;

 private:
  int64_t _nanoseconds;

  constexpr Duration(int64_t nanoseconds)
      : _nanoseconds(nanoseconds) {
  }
};

}  // namespace lut
