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

#include "libllm/tensor.h"

#include <cmath>
#include "libllm/cpu/accessor.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/tensor.h"
#include "libllm/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T>
float maxDiffKernel(Tensor A, Tensor B) {
  A.throwIfInvalidShape(B.getShape());

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<const T, 1> vB = TensorList<const T, 1>::fromTensor(B);
  CHECK(vA.getLength() == vB.getLength());

  float maxDiff = 0.0f;
  for (int j = 0; j < vA.getLength(); ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<const T, 1> b = vB.getTensor(j);

    for (int i = 0; i < a.getShape(0); ++i) {
      float diff = fabs(static_cast<float>(a[i]) - static_cast<float>(b[i]));
      if (diff > maxDiff) maxDiff = diff;
    }
  }
  
  return maxDiff;
}

template<typename T>
float meanAbsKernel(Tensor A) {
  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);

  bool ok = true;
  double sum = 0.0;
  for (int j = 0; j < vA.getLength(); ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);

    for (int i = 0; i < a.getShape(0); ++i) {
      sum += fabs(a[i]);
    }
  }
  
  return static_cast<float>(sum / A.getNumEl());
}

template<typename T>
bool allCloseKernel(Tensor A, Tensor B, float rtol, float atol) {
  return maxDiffKernel<T>(A, B) / meanAbsKernel<T>(B) < rtol || maxDiffKernel<T>(A, B) < atol;
}

bool allClose(Tensor A, Tensor B, float rtol, float atol) {
  if (A.getDType() == DType::kFloat) return allCloseKernel<float>(A, B, rtol, atol);

  NOT_IMPL();
}

}  // cpu
}  // op
}  // libllm
