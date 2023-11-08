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
#include <stdint.h>
#include <memory>
#include "lymath/common.h"
#include "lymath/q4sym_kernel.h"
#include "lyutil/log.h"

namespace lymath {

class DequantQ4Sym {
 public:
  virtual ~DequantQ4Sym() = default;
  virtual void apply(int n, PCQ4x2 src, PCFp16 scale, PFp32 tgt) const = 0;
};

template<class Kernel, Mode MODE>
class DequantQ4SymImpl : public DequantQ4Sym {
 public:
  void apply(int n, PCQ4x2 src, PCFp16 scale, PFp32 tgt) const override {
    CHECK(n % Q4GroupSize == 0);
    int nb = (n + DequantMinElemPerThread - 1) / DequantMinElemPerThread;

    if (MODE == Mode::OMP && nb > 1) {
      int nr = (n - 1) % DequantMinElemPerThread + 1;
      int numThreads = std::min(nb, omp_get_max_threads());

      #pragma omp parallel for num_threads(numThreads)
      for (int i = 0; i < nb; ++i) {
        int ne = (i == nb - 1) ? nr : DequantMinElemPerThread;
        Kernel::apply(
            ne,
            src + i * DequantMinElemPerThread / 2,
            scale + i * DequantMinElemPerThread / Q4GroupSize,
            tgt + i * DequantMinElemPerThread);
      }
    } else {
      Kernel::apply(n, src, scale, tgt);
    }
  }
};

typedef DequantQ4SymImpl<DequantQ4SymAvx2Knl, Mode::SingleThread> DequantQ4SymAvx2;
typedef DequantQ4SymImpl<DequantQ4SymFallbackKnl, Mode::SingleThread> DequantQ4SymFallback;
typedef DequantQ4SymImpl<DequantQ4SymAvx2Knl, Mode::OMP> DequantQ4SymAvx2OMP;
typedef DequantQ4SymImpl<DequantQ4SymFallbackKnl, Mode::OMP> DequantQ4SymFallbackOMP;

}  // namespace lymath
