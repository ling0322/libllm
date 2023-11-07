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
#include "lymath/common.h"
#include "lymath/q4kernel.h"

namespace lymath {

class DequantQ4 {
 public:
  virtual ~DequantQ4() = default;
  virtual void apply(int n, PCQ4x2 src, PCFp16 scale, PCInt8 zeroPoint, PFp32 tgt) const = 0;
};

template<class TKernel, Mode MODE>
class DequantQ4Impl : public DequantQ4 {
 public:
  void apply(int n, PCQ4x2 src, PCFp16 scale, PCInt8 zeroPoint, PFp32 tgt) const override {
    CHECK(n % Q4GroupSize == 0);
    int nb = (n + DequantMinElemPerThread - 1) / DequantMinElemPerThread;

    if (MODE == Mode::OMP && nb > 1) {
      int nr = (n - 1) % DequantMinElemPerThread + 1;
      int numThreads = std::min(nb, omp_get_max_threads());

      #pragma omp parallel for num_threads(numThreads)
      for (int i = 0; i < nb; ++i) {
        int ne = (i == nb - 1) ? nr : DequantMinElemPerThread;
        TKernel::apply(
            ne,
            src + i * DequantMinElemPerThread / 2,
            scale + i * DequantMinElemPerThread / Q4GroupSize,
            zeroPoint + i * DequantMinElemPerThread / Q4GroupSize,
            tgt + i * DequantMinElemPerThread);
      }
    } else {
      TKernel::apply(n, src, scale, zeroPoint, tgt);
    }
  }
};

typedef DequantQ4Impl<DequantQ4Avx2Kernel, Mode::SingleThread> DequantQ4Avx2;
typedef DequantQ4Impl<DequantQ4FallbackKernel, Mode::SingleThread> DequantQ4Fallback;
typedef DequantQ4Impl<DequantQ4Avx2Kernel, Mode::OMP> DequantQ4Avx2OMP;
typedef DequantQ4Impl<DequantQ4FallbackKernel, Mode::OMP> DequantQ4FallbackOMP;

}  // namespace lymath
