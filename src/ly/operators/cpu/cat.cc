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

#include "ly/operators/cpu/cat.h"

#include "ly/operators/cpu/copy.h"
#include "ly/operators/cpu/tensor.h"
#include "ly/operators/cpu/subtensor.h"
#include "ly/operators/cpu/subtensor_list.h"

namespace ly {
namespace op {
namespace cpu {

Tensor cat(Tensor A, Tensor B, int dim) {
  CHECK(A.getDim() == B.getDim());

  std::vector<int> shape;
  for (int d = 0; d < A.getDim(); ++d) {
    if (d == dim) {
      shape.push_back(A.getShape(d) + B.getShape(d));
    } else {
      CHECK(A.getShape(d) == B.getShape(d));
      shape.push_back(A.getShape(d));
    }
  }

  Tensor C = cpu::tensor(lut::makeConstSpan(shape), A.getDType());
  switch (A.getDType()) {
    case DType::kFloat:
      catFp32(
          Subtensor<const float>::fromTensor(A),
          Subtensor<const float>::fromTensor(B),
          dim,
          Subtensor<float>::fromTensor(C));
      break;
    default:
      NOT_IMPL();
  }

  return C;
}

void catFp32(Subtensor<const float> A, Subtensor<const float> B, int dim, Subtensor<float> C) {
  CHECK(A.rank() == B.rank() && A.rank() == C.rank());

  int tensorDim = A.rank() - dim;

  SubtensorList<const float> tAs = getTensorList(A, tensorDim);
  SubtensorList<const float> tBs = getTensorList(B, tensorDim);
  SubtensorList<float> tCs = getTensorList(C, tensorDim);
  CHECK(tAs.getSize() == tBs.getSize() && tAs.getSize() == tCs.getSize());

  #pragma omp parallel for
  for (int i = 0; i < tAs.getSize(); ++i) {
    Subtensor<const float> tA = tAs.getSubtensor(i);
    Subtensor<const float> tB = tBs.getSubtensor(i);
    Subtensor<float> tC = tCs.getSubtensor(i);
    CHECK(tC.dimension(0) == tA.dimension(0) + tB.dimension(0));

    for (int j = 0; j < tA.dimension(0); ++j)
      copyFp32(tA.subtensor(j), tC.subtensor(j));
    for (int j = 0; j < tB.dimension(0); ++j)
      copyFp32(tB.subtensor(j), tC.subtensor(j + tA.dimension(0)));
  }
}

}  // cpu
}  // op
}  // ly
