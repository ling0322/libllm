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

#include "llyn/cpu/mul.h"

#include "llyn/cpu/subtensor.h"
#include "llyn/cpu/subtensor_list.h"
#include "llyn/cpu/tensor.h"

namespace llyn {
namespace cpu {

Tensor mul(const Tensor &A, const float k) {
  if (A.getDType() == DType::kFloat) return mulFp32(Subtensor<const float>::fromTensor(A), k);

  NOT_IMPL();
  return Tensor();
}

Tensor mulFp32(Subtensor<const float> A, float k) {
  Tensor C = tensor(A.getShape(), DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);

  SubtensorList<const float> vAs = getVectorList(A);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());

  #pragma omp parallel for
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    for (int i = 0; i < vA.dimension(0); ++i) {
      vC.elem(i) = k * vA.elem(i);
    }
  }
  return C;
}

}  // cpu
}  // flint
