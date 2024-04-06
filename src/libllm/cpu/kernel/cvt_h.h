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

#include <omp.h>
#include "libllm/cpu/kernel/interfaces.h"
#include "libllm/cpu/kernel/kernel_h.h"
#include "libllm/lut/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

template<class TKernel, typename TX, typename TY, Mode MODE>
class CvtCommon {
 public:
  static void apply(int64_t n, const TX *x, TY *y) {
    int nb = (n + CvtMinElemPerThread - 1) / CvtMinElemPerThread;

    if (MODE == Mode::OMP && nb > 1) {
      int nr = (n - 1) % CvtMinElemPerThread + 1;
      int numThreads = std::min(nb, omp_get_max_threads());

      #pragma omp parallel for num_threads(numThreads)
      for (int i = 0; i < nb; ++i) {
        int ne = (i == nb - 1) ? nr : CvtMinElemPerThread;
        TKernel::apply(
            ne,
            x + i * CvtMinElemPerThread,
            y + i * CvtMinElemPerThread);
      }
    } else {
      TKernel::apply(n, x, y);
    }
  }
};

class CvtHalf {
 public:
  virtual ~CvtHalf() = default;
  virtual void cvtHalfToFloat(int64_t n, const Float16 *x, float *y) const = 0;
  virtual void cvtFloatToHalf(int64_t n, const float *x, Float16 *y) const = 0;
};

template<class THalfToFloatKernel, class TFloatToHalfKernel, Mode MODE>
class CvtHalfImpl : public CvtHalf {
 public:
  void cvtHalfToFloat(int64_t n, const Float16 *x, float *y) const override {
    CvtCommon<THalfToFloatKernel, Float16, float, MODE>::apply(n, x, y);
  }

  void cvtFloatToHalf(int64_t n, const float *x, Float16 *y) const override {
    CvtCommon<TFloatToHalfKernel, float, Float16, MODE>::apply(n, x, y);
  }
};

typedef CvtHalfImpl<CvtHalfToFloatAvx2Kernel, CvtFloatToHalfFallbackKernel, Mode::OMP>
    CvtHalfAvx2OMP;
typedef CvtHalfImpl<CvtHalfToFloatAsimdhpKernel, CvtFloatToHalfAsimdhpKernel, Mode::OMP>
    CvtHalfAsimdhpOMP;
typedef CvtHalfImpl<CvtHalfToFloatFallbackKernel, CvtFloatToHalfFallbackKernel, Mode::OMP>
    CvtHalfFallbackOMP;


}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
