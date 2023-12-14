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

#include <algorithm>
#include "ly/operators/cpu/fingerprint.h"
#include "ly/operators/cpu/subtensor.h"
#include "ly/operators/cpu/tensor.h"

namespace ly {
namespace op {
namespace cpu {

/// Extract fingerprint vector from a tensor. This method is used in unittest to quick compate two
/// tensors.
Tensor fingerprint(Tensor A) {
  switch (A.getDim()) {
    case 1:
      return fingerprint1D(A);
    case 2:
      return fingerprint2D(A);
    case 3:
      return fingerprint3D(A);
    case 4:
      return fingerprint4D(A);
    default:
      NOT_IMPL();
  }
}

float getFingerprintElem(Tensor A, int d0) {
  CHECK(A.getDim() == 1);
  Subtensor<float> sA = Subtensor<float>::fromTensor(A);
  if (d0 < 0) {
    d0 = sA.dimension(0) + d0;
  }
  CHECK(d0 < sA.dimension(0) && d0 >= 0);
  return sA.elem(d0);
}

float getFingerprintElem(Tensor A, int d0, int d1) {
  CHECK(A.getDim() == 2);
  if (d0 > 0) {
    d0 = std::min(d0, A.getShape(0) - 1);
  } else {
    d0 = std::max(d0, -A.getShape(0));
  }
  return getFingerprintElem(A.subtensor(d0), d1);
}

float getFingerprintElem(Tensor A, int d0, int d1, int d2) {
  CHECK(A.getDim() == 3);
  if (d0 > 0) {
    d0 = std::min(d0, A.getShape(0) - 1);
  } else {
    d0 = std::max(d0, -A.getShape(0));
  }
  return getFingerprintElem(A.subtensor(d0), d1, d2);
}

float getFingerprintElem(Tensor A, int d0, int d1, int d2, int d3) {
  CHECK(A.getDim() == 4);
  if (d0 > 0) {
    d0 = std::min(d0, A.getShape(0) - 1);
  } else {
    d0 = std::max(d0, -A.getShape(0));
  }
  return getFingerprintElem(A.subtensor(d0), d1, d2, d3);
}

Tensor fingerprint1D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::kFloat);
  Subtensor<float> sC = Subtensor<float>::fromTensor(C);

  sC.elem(0) = getFingerprintElem(A, 0);
  sC.elem(1) = getFingerprintElem(A, 1);
  sC.elem(2) = getFingerprintElem(A, 2);
  sC.elem(3) = getFingerprintElem(A, 3);
  sC.elem(4) = getFingerprintElem(A, -4);
  sC.elem(5) = getFingerprintElem(A, -3);
  sC.elem(6) = getFingerprintElem(A, -2);
  sC.elem(7) = getFingerprintElem(A, -1);


  return C;
}

Tensor fingerprint2D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::kFloat);
  Subtensor<float> sC = Subtensor<float>::fromTensor(C);

  sC.elem(0) = getFingerprintElem(A, 0, 0);
  sC.elem(1) = getFingerprintElem(A, 1, 1);
  sC.elem(2) = getFingerprintElem(A, 2, 2);
  sC.elem(3) = getFingerprintElem(A, 3, 3);
  sC.elem(4) = getFingerprintElem(A, -4, -4);
  sC.elem(5) = getFingerprintElem(A, -3, -3);
  sC.elem(6) = getFingerprintElem(A, -2, -2);
  sC.elem(7) = getFingerprintElem(A, -1, -1);

  return C;
}

Tensor fingerprint3D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::kFloat);
  Subtensor<float> sC = Subtensor<float>::fromTensor(C);

  sC.elem(0) = getFingerprintElem(A, 0, 0, 0);
  sC.elem(1) = getFingerprintElem(A, 1, 1, 1);
  sC.elem(2) = getFingerprintElem(A, 2, 2, 2);
  sC.elem(3) = getFingerprintElem(A, 3, 3, 3);
  sC.elem(4) = getFingerprintElem(A, -4, -4, -4);
  sC.elem(5) = getFingerprintElem(A, -3, -3, -3);
  sC.elem(6) = getFingerprintElem(A, -2, -2, -2);
  sC.elem(7) = getFingerprintElem(A, -1, -1, -1);

  return C;
}

Tensor fingerprint4D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::kFloat);
  Subtensor<float> sC = Subtensor<float>::fromTensor(C);

  sC.elem(0) = getFingerprintElem(A, 0, 0, 0, 0);
  sC.elem(1) = getFingerprintElem(A, 1, 1, 1, 1);
  sC.elem(2) = getFingerprintElem(A, 2, 2, 2, 2);
  sC.elem(3) = getFingerprintElem(A, 3, 3, 3, 3);
  sC.elem(4) = getFingerprintElem(A, -4, -4, -4, -4);
  sC.elem(5) = getFingerprintElem(A, -3, -3, -3, -3);
  sC.elem(6) = getFingerprintElem(A, -2, -2, -2, -2);
  sC.elem(7) = getFingerprintElem(A, -1, -1, -1, -1);

  return C;
}

}  // cpu
}  // op
}  // ly
