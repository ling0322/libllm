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

#include "lutil/time.h"

#include <stdint.h>

#include <chrono>

#include "lutil/strings.h"

namespace lut {

double now() {
  auto t = std::chrono::high_resolution_clock::now().time_since_epoch();
  int64_t ns = t / std::chrono::nanoseconds(1);
  return ns / 1000000000.0;
}

std::string Duration::toString() const {
  int64_t ns = _nanoseconds;
  int64_t hours = ns / Hour;
  ns %= Hour;

  int64_t minutes = ns / Minute;
  ns %= Minute;

  int64_t seconds = ns / Second;
  ns %= Second;

  int64_t ms = ns / Millisecond;
  return lut::sprintf("%02d:%02d:%02d.%03d", hours, minutes, seconds, ms);
}

}  // namespace lut
