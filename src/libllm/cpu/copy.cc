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

#include "libllm/cpu/copy.h"

#include "libllm/cpu/accessor.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T>
void copyKernel(const Tensor &src, Tensor &dest) {
  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(src);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(dest);
  CHECK(vA.getLength() == vC.getLength());

  #pragma omp parallel for
  for (int j = 0; j < vA.getLength(); ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    copyVector(c, a);
  }
}

void copy(const Tensor &src, Tensor &dest) {
  if (src.getDType() == DType::kFloat) {
    copyKernel<float>(src, dest);
  } else {
    NOT_IMPL();
  }
}

}  // cpu
}  // op
}  // libllm
