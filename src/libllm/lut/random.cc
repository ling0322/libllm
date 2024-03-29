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

#include "libllm/lut/random.h"

#include <math.h>

namespace lut {

Random::Random() {
  uint64_t seed = static_cast<uint64_t>(time(nullptr));
  _x = seed % RandMax;
}

Random::Random(uint64_t seed) : _x(seed % RandMax) {}

void Random::fill(Span<float> l, float min, float max) {
  for (float &v : l) {
    v = nextFloat();
    v = min + (max - min) * v;
  }
}

void Random::fill(Span<float> l) {
  return fill(l, 0.0f, 1.0f);
}

void Random::fillUInt8(Span<uint8_t> l) {
  for (uint8_t &v : l) {
    v = nextInt() % 256;
  }
}

void Random::fillInt8(Span<int8_t> l, int8_t min, int8_t max) {
  for (int8_t &v : l) {
    v = nextInt() % (max - min + 1) + min;
  }
}

void Random::fillInt8(Span<int8_t> l) {
  for (int8_t &v : l) {
    v = nextInt() % 256 - 128;
  }
}


int32_t Random::nextInt() {
  _x = (48271 * _x) % RandMax;
  return static_cast<int32_t>(_x);
}

float Random::nextFloat() {
  return static_cast<float>(static_cast<double>(nextInt()) / RandMax);
}

void Random::reset(uint64_t seed) {
  _x = seed % RandMax;
}

void Random::fillGaussian(Span<float> l, float mean, float sigma) {
  std::vector<float> U(l.size() * 2);
  fill(lut::makeSpan(U), 0.0f, 1.0f);

  for (int64_t i = 0; i < l.size(); ++i) {
    l[i] = cosf(2 * PI * U[i]) * sqrtf(-2 * logf(U[l.size() + i]));
    l[i] = mean + l[i] * sigma;
  }
}

} // namespace lut
