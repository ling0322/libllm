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

#include "lut/half.h"

namespace lut {

// these float and half converion code are from https://gist.github.com/rygorous/2156668
// License: https://creativecommons.org/publicdomain/zero/1.0/

typedef unsigned int uint;

union FP32 {
  uint u;
  float f;
  struct {
    uint Mantissa : 23;
    uint Exponent : 8;
    uint Sign : 1;
  };
};

union FP16 {
  unsigned short u;
  struct {
    uint Mantissa : 10;
    uint Exponent : 5;
    uint Sign : 1;
  };
};

FP16 float_to_half_fast3(FP32 f) {
  constexpr FP32 f32infty = {255 << 23};
  constexpr FP32 f16infty = {31 << 23};
  constexpr FP32 magic = {15 << 23};
  constexpr uint sign_mask = 0x80000000u;
  constexpr uint round_mask = ~0xfffu;
  FP16 o = {0};

  uint sign = f.u & sign_mask;
  f.u ^= sign;

  // NOTE all the integer compares in this function can be safely
  // compiled into signed compares since all operands are below
  // 0x80000000. Important if you want fast straight SSE2 code
  // (since there's no unsigned PCMPGTD).

  if (f.u >= f32infty.u) {                       // Inf or NaN (all exponent bits set)
    o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;  // NaN->qNaN and Inf->Inf
  } else {                                       // (De)normalized number or zero
    f.u &= round_mask;
    f.f *= magic.f;
    f.u -= round_mask;
    if (f.u > f16infty.u) f.u = f16infty.u;  // Clamp to signed infinity if overflowed

    o.u = f.u >> 13;  // Take the bits!
  }

  o.u |= sign >> 16;
  return o;
}

// from half->float code - just for verification.
FP32 half_to_float(FP16 h) {
  constexpr FP32 magic = {113 << 23};
  constexpr uint shifted_exp = 0x7c00 << 13;  // exponent mask after shift
  FP32 o;

  o.u = (h.u & 0x7fff) << 13;    // exponent/mantissa bits
  uint exp = shifted_exp & o.u;  // just the exponent
  o.u += (127 - 15) << 23;       // exponent adjust

  // handle exponent special cases
  if (exp == shifted_exp) {   // Inf/NaN?
    o.u += (128 - 16) << 23;  // extra exp adjust
  } else if (exp == 0) {      // Zero/Denormal?
    o.u += 1 << 23;           // extra exp adjust
    o.f -= magic.f;           // renormalize
  }

  o.u |= (h.u & 0x8000) << 16;  // sign bit
  return o;
}

uint16_t cvtss_sh(float v) {
  FP32 fv;
  fv.f = v;
  return float_to_half_fast3(fv).u;
}

float cvtsh_ss(uint16_t v) {
  return half_to_float({v}).f;
}

}  // namespace lut