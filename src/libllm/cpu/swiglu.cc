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

#include "libllm/cpu/swiglu.h"

#include <math.h>
#include "libllm/cpu/subtensor.h"
#include "libllm/cpu/subtensor_list.h"
#include "libllm/cpu/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

Tensor swiglu(const Tensor &A) {
  CHECK(A.getShape(-1) % 2 == 0);

  if (A.getDType() == DType::kFloat) return swigluFp32(A);

  NOT_IMPL();
}

Tensor swigluFp32(const Tensor &A) {
  CHECK(A.getDType() == DType::kFloat);
  std::vector<int> shapeA = A.getShape();
  shapeA.back() /= 2;
  Tensor C = tensor(shapeA, DType::kFloat);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  Subtensor<const float> As = Subtensor<const float>::fromTensor(A);

  SubtensorList<const float> vAs = getVectorList(As);
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());

  #pragma omp parallel for
  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<float> vC = vCs.getSubtensor(j);

    int n = vC.dimension(0);
    for (int i = 0; i < n; ++i) {
      float x = vA.data[i];
      x *= 1.0f / (1 + expf(-x));
      x *= vA.data[i + n];
      vC.data[i] = x;
    }
  }
  return C;
}

}  // cpu
}  // op
}  // ly

