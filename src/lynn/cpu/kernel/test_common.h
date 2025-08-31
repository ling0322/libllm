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

#include <math.h>

#include "catch2/catch_amalgamated.hpp"
#include "lutil/half.h"
#include "lutil/log.h"
#include "lutil/random.h"
#include "lynn/cpu/kernel/abstract.h"
#include "lynn/cpu/kernel/asimdhp.h"
#include "lynn/cpu/kernel/avx2.h"
#include "lynn/cpu/kernel/avx512.h"
#include "lynn/cpu/kernel/fallback.h"
#include "lynn/cpu/kernel/util.h"

namespace ly {
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
bool isClose(lut::Span<const T> A, lut::Span<const T> B, float atol = 1e-5, float rtol = 1e-5) {
  if (A.size() != B.size()) return false;

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

inline void fillRandom(lut::Random *r, lut::Span<Float16> v) {
  std::vector<float> vf(v.size());
  r->fill(lut::makeSpan(vf), -1, 1);
  std::transform(vf.begin(), vf.end(), v.begin(), [](float v) { return cvt_s2h(v); });
}

inline void fillRandom(lut::Random *r, lut::Span<float> v) {
  r->fill(v, -1, 1);
}

template<typename T>
inline void fillZero(lut::Span<T> v) {
  memset(v.data(), 0, sizeof(T) * v.size());
}

template<
    typename ElementA,
    typename ElementB,
    typename ElementC,
    int MR,
    int NR,
    CpuMathBackend TYPE>
struct GemmMicroKernelTester {
  void test(int k, float rtol = 0.01) {
    std::vector<ElementA> A(MR * k);
    std::vector<ElementB> B(NR * k);
    std::vector<ElementC> C(MR * NR);
    std::vector<ElementC> Cr(MR * NR);

    lut::Random random(MagicNumber);
    fillRandom(&random, lut::makeSpan<ElementA>(A));
    fillRandom(&random, lut::makeSpan<ElementB>(B));
    fillRandom(&random, lut::makeSpan<ElementC>(C));
    std::copy(C.begin(), C.end(), Cr.begin());

    gemmKernel<ElementA, ElementB, ElementC, MR, NR, TYPE>(k, A.data(), B.data(), C.data(), NR);
    gemmKernel<ElementA, ElementB, ElementC, MR, NR, CpuMathBackend::FALLBACK>(
        k,
        A.data(),
        B.data(),
        Cr.data(),
        NR);

    CATCH_REQUIRE(getMaxDiff<ElementC>(C, Cr) / getMeanAbs<ElementC>(Cr) < 0.05);
  }
};

template<typename ElementA, typename ElementX, typename ElementY, CpuMathBackend TYPE>
struct AxpyKernelTester {
  void test(int n) {
    std::vector<ElementX> x(n);
    std::vector<ElementY> y(n);
    std::vector<ElementY> yr(n);

    lut::Random random(MagicNumber);
    fillRandom(&random, lut::makeSpan<ElementX>(x));
    ElementA a = x[0];

    axpyKernel<ElementA, ElementX, ElementY, TYPE>(n, a, x.data(), 0, y.data());
    axpyKernel<ElementA, ElementX, ElementY, CpuMathBackend::FALLBACK>(
        n,
        a,
        x.data(),
        0,
        yr.data());

    CATCH_REQUIRE(isClose<ElementY>(yr, y));
  }
};

template<typename ElementA, typename ElementC, CpuMathBackend TYPE>
struct CvtKernelTester {
  void test(int n, int offsetX = 0) {
    std::vector<ElementA> x(n);
    std::vector<ElementC> y(n);
    std::vector<ElementC> yr(n);

    lut::Random random(MagicNumber);
    fillRandom(&random, lut::makeSpan<ElementA>(x));
    fillZero(lut::makeSpan<ElementC>(y));
    fillZero(lut::makeSpan<ElementC>(yr));

    n -= offsetX;
    CHECK(n > 0);

    cvtKernel<ElementA, ElementC, TYPE>(n, x.data(), offsetX, y.data(), 0);
    cvtKernel<ElementA, ElementC, CpuMathBackend::FALLBACK>(n, x.data(), offsetX, yr.data(), 0);

    CATCH_REQUIRE(isClose<ElementC>(y, yr));
  }
};

template<typename ElementA, typename ElementX, typename ElementY, CpuMathBackend TYPE>
struct DotKernelTester {
  float _rtol;

  DotKernelTester(float rtol = 5e-2)
      : _rtol(rtol) {
  }

  void test(int n, int offsetY = 0) {
    std::vector<ElementX> x(n);
    std::vector<ElementY> y(n);

    lut::Random random(MagicNumber);
    fillRandom(&random, lut::makeSpan<ElementX>(x));
    fillRandom(&random, lut::makeSpan<ElementY>(y));

    n -= offsetY;
    CHECK(n > 0);

    ElementA z = dotKernel<ElementA, ElementX, ElementY, TYPE>(n, x.data(), y.data(), offsetY);
    ElementA zr = dotKernel<ElementA, ElementX, ElementY, CpuMathBackend::FALLBACK>(
        n,
        x.data(),
        y.data(),
        offsetY);

    CATCH_REQUIRE(isClose(z, zr, 0, _rtol));
  }
};

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace ly
