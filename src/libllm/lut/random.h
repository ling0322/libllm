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
#include "libllm/lut/span.h"

namespace lut {

// random number generator.
class Random {
 public:
  static constexpr int32_t RandMax = 2147483647;  // 2^31-1
  static constexpr float PI = 3.1415926f;

  // initialize the random number generator by current time.
  Random();
  Random(uint64_t seed);

  // fill `l` with a list of float numbers in range [0, 1) or [min, max).
  void fill(Span<float> l);
  void fill(Span<float> l, float min, float max);

  // fill `l` with floats sampled from Gaussian distribution. l.size() % 2 MUST equal to 0.
  void fillGaussian(Span<float> l, float mean = 0.0f, float sigma = 1.0f);

  // fill `l` with a list of uint8_t numbers in range [0, 255].
  void fillUInt8(Span<uint8_t> l);

  // fill `l` with a list of int8_t numbers in range [-128, 127].
  void fillInt8(Span<int8_t> l);
  void fillInt8(Span<int8_t> l, int8_t min, int8_t max);

  // return next random int number in range [0, RandMax).
  int32_t nextInt();

  // return next random float number in range [0, 1).
  float nextFloat();

  // reset the random number generator.
  void reset(uint64_t seed);

 private:
  uint64_t _x;
};

} // namespace lut
