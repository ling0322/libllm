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

#include "libllm/cpu/kernel/abstract.h"
#include "libllm/mp.h"
#include "lutil/log.h"

namespace libllm {
namespace op {
namespace cpu {
namespace kernel {

template<typename ElementA, typename ElementC, CpuMathBackend TYPE, Mode MODE>
void cvt(int64_t n, const ElementA *x, int64_t offsetX, ElementC *y, int64_t offsetY) {
  int nb = (n + CvtMinElemPerThread - 1) / CvtMinElemPerThread;

  if (MODE == Mode::OMP && nb > 1) {
    int nr = (n - 1) % CvtMinElemPerThread + 1;
    int numThreads = std::min(nb, MP::getMaxThreads());

    MP::parallelFor({nb}, numThreads, [nb, nr, x, offsetX, y, offsetY](MP::Partition partition) {
      for (int i : partition.getRange()) {
        int ne = (i == nb - 1) ? nr : CvtMinElemPerThread;
        cvtKernel<ElementA, ElementC, TYPE>(
            ne,
            x,
            offsetX + i * CvtMinElemPerThread,
            y,
            offsetY + i * CvtMinElemPerThread);
      }
    });
  } else {
    cvtKernel<ElementA, ElementC, TYPE>(n, x, offsetX, y, offsetY);
  }
}

}  // namespace kernel
}  // namespace cpu
}  // namespace op
}  // namespace libllm
