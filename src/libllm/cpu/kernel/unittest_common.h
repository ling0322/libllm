// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "catch2/catch_amalgamated.hpp"

#include <omp.h>
#include <math.h>
#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/cpu/kernel/util.h"
#include "libllm/lut/half.h"
#include "libllm/lut/random.h"
#include "libllm/lut/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

constexpr uint32_t MagicNumber = 0x55aa;

inline bool isClose(float l, float r, float atol = 1e-5, float rtol = 1e-5) {
  bool close = fabs(l - r) <= atol + rtol * fabs(r);
  if (!close) {
    printf("%f vs %f\n", l, r);
  }
  return close;
}

inline bool isClose(Float16 l, Float16 r, float atol = 1e-5, float rtol = 1e-5) {
  float lf = cvt_h2s(l);
  float rf = cvt_h2s(r);
  bool close = fabs(lf - rf) <= atol + rtol * std::max(fabs(rf), fabs(lf));
  if (!close) {
    printf("%f vs %f\n", lf, rf);
  }
  return close;
}

template<typename T>
bool isClose(lut::Span<const T> A,
             lut::Span<const T> B,
             float atol = 1e-5,
             float rtol = 1e-5) {
  if (A.size() != B.size()) 
    return false;

  for (int i = 0; i < A.size(); ++i) {
    if (!isClose(A[i], B[i], atol, rtol)) {
      return false;
    }
  }

  return true;
}

inline float toFloat(float x) {
  return x;
}
inline float toFloat(Float16 x) {
  return cvt_h2s(x);
}

template<typename T>
float getMSE(lut::Span<const T> A, lut::Span<const T> B) {
  CHECK(A.size() == B.size());

  double sum = 0;
  for (int i = 0; i < A.size(); ++i) {
    double d = toFloat(A[i]) - toFloat(B[i]);
    sum += d * d;
  }
  double mse = sum / A.size();
  return static_cast<float>(mse);
}

template<typename T>
float getVar(lut::Span<const T> A) {
  double sum = 0;
  for (int i = 0; i < A.size(); ++i) {
    sum += toFloat(A[i]);
  }

  double mean = sum / A.size();
  sum = 0;
  for (int i = 0; i < A.size(); ++i) {
    double d = toFloat(A[i]) - mean;
    sum += d * d;
  }
  double var = sum / A.size();
  return static_cast<float>(var);
}

template<typename T>
float getMeanAbs(lut::Span<const T> A) {
  double sum = 0;
  for (int i = 0; i < A.size(); ++i) {
    sum += fabs(cvtf<float>(A[i]));
  }

  return static_cast<float>(sum / A.size());
}

template<typename T>
float getMaxDiff(lut::Span<const T> A, lut::Span<const T> B) {
  CHECK(A.size() == B.size());

  float maxDiff = 0;
  for (int i = 0; i < A.size(); ++i) {
    float diff = fabs(cvtf<float>(A[i] - B[i]));
    if (diff > maxDiff) {
      maxDiff = diff;
    }
  }

  return maxDiff;
}

template<typename T>
float getRSquared(lut::Span<const T> predict, lut::Span<const T> correct) {
  return 1.0f - getMSE<T>(predict, correct) / getVar<T>(predict);
}


inline void fillRandom(
    lut::Random *r,
    lut::Span<Float16> v) {
  std::vector<float> vf(v.size());
  r->fill(lut::makeSpan(vf), -1, 1);
  std::transform(vf.begin(), vf.end(), v.begin(), [](float v){
    return cvt_s2h(v);
  });
}

inline void fillRandom(
    lut::Random *r,
    lut::Span<float> v) {
  r->fill(v, -1, 1);
}

inline void fillRandomQInt4(
    lut::Random *r,
    lut::Span<UInt4x2> qdata,
    lut::Span<Float16> qscale,
    lut::Span<UInt4x2> qzero) {
  std::vector<float> vf(qdata.size() * 2);
  r->fill(lut::makeSpan(vf), -1.0f, 1.0f);

  quantFloatToQInt4(vf, qdata, qscale, qzero);
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
