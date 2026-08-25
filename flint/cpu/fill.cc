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

#include "flint/cpu/fill.h"

#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/tensor.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

template<typename T>
void fillKernel(Tensor A, float value) {
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(A);
  int numRows = vC.getLength();
#pragma omp parallel for schedule(dynamic, 1)
  for (int j = 0; j < numRows; ++j) {
    TensorAccessor<T, 1> c = vC.getTensor(j);

    for (int i = 0; i < c.getShape(0); ++i) {
      c[i] = value;
    }
  }
}

void fill(Tensor src, float value) {
  if (src.getDType() == DType::kFloat) {
    if (src.getNumEl() == 1) {
      *getDataPtrCpu<float>(src) = value;
    } else {
      fillKernel<float>(src, value);
    }
    return;
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  if (src.getDType() == DType::kFloat16) {
    if (src.getNumEl() == 1) {
      *getDataPtrCpu<Float16>(src) = value;
    } else {
      fillKernel<Float16>(src, value);
    }
    return;
  }
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
