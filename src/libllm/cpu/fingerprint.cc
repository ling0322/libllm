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


#include "libllm/cpu/fingerprint.h"

#include <algorithm>
#include "libllm/cpu/accessor.h"
#include "libllm/cpu/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T>
T getFingerprintElem(Tensor A, int d0) {
  d0 = d0 > 0 ? std::min(d0, A.getShape(0) - 1) : std::max(d0, -A.getShape(0));
  if (d0 < 0) {
    d0 = A.getShape(0) + d0;
  }

  TensorAccessor<T, 1> a = A;
  CHECK(d0 < a.getShape(0) && d0 >= 0);
  return a[d0];
}

template<typename T>
T getFingerprintElem(Tensor A, int d0, int d1) {
  d0 = d0 > 0 ? std::min(d0, A.getShape(0) - 1) : std::max(d0, -A.getShape(0));
  return getFingerprintElem<T>(A.subtensor(d0), d1);
}

template<typename T>
T getFingerprintElem(Tensor A, int d0, int d1, int d2) {
  d0 = d0 > 0 ? std::min(d0, A.getShape(0) - 1) : std::max(d0, -A.getShape(0));
  return getFingerprintElem<T>(A.subtensor(d0), d1, d2);
}

template<typename T>
T getFingerprintElem(Tensor A, int d0, int d1, int d2, int d3) {
  d0 = d0 > 0 ? std::min(d0, A.getShape(0) - 1) : std::max(d0, -A.getShape(0));
  return getFingerprintElem<T>(A.subtensor(d0), d1, d2, d3);
}

template<typename T>
Tensor fingerprint1D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::getType<T>());
  TensorAccessor<T, 1> c = C;

  c[0] = getFingerprintElem<T>(A, 0);
  c[1] = getFingerprintElem<T>(A, 1);
  c[2] = getFingerprintElem<T>(A, 2);
  c[3] = getFingerprintElem<T>(A, 3);
  c[4] = getFingerprintElem<T>(A, -4);
  c[5] = getFingerprintElem<T>(A, -3);
  c[6] = getFingerprintElem<T>(A, -2);
  c[7] = getFingerprintElem<T>(A, -1);

  return C;
}

template<typename T>
Tensor fingerprint2D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::getType<T>());
  TensorAccessor<T, 1> c = C;

  c[0] = getFingerprintElem<T>(A, 0, 0);
  c[1] = getFingerprintElem<T>(A, 1, 1);
  c[2] = getFingerprintElem<T>(A, 2, 2);
  c[3] = getFingerprintElem<T>(A, 3, 3);
  c[4] = getFingerprintElem<T>(A, -4, -4);
  c[5] = getFingerprintElem<T>(A, -3, -3);
  c[6] = getFingerprintElem<T>(A, -2, -2);
  c[7] = getFingerprintElem<T>(A, -1, -1);

  return C;
}

template<typename T>
Tensor fingerprint3D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::getType<T>());
  TensorAccessor<T, 1> c = C;

  c[0] = getFingerprintElem<T>(A, 0, 0, 0);
  c[1] = getFingerprintElem<T>(A, 1, 1, 1);
  c[2] = getFingerprintElem<T>(A, 2, 2, 2);
  c[3] = getFingerprintElem<T>(A, 3, 3, 3);
  c[4] = getFingerprintElem<T>(A, -4, -4, -4);
  c[5] = getFingerprintElem<T>(A, -3, -3, -3);
  c[6] = getFingerprintElem<T>(A, -2, -2, -2);
  c[7] = getFingerprintElem<T>(A, -1, -1, -1);

  return C;
}

template<typename T>
Tensor fingerprint4D(Tensor A) {
  Tensor C = op::cpu::tensor({8}, DType::getType<T>());
  TensorAccessor<T, 1> c = C;

  c[0] = getFingerprintElem<T>(A, 0, 0, 0, 0);
  c[1] = getFingerprintElem<T>(A, 1, 1, 1, 1);
  c[2] = getFingerprintElem<T>(A, 2, 2, 2, 2);
  c[3] = getFingerprintElem<T>(A, 3, 3, 3, 3);
  c[4] = getFingerprintElem<T>(A, -4, -4, -4, -4);
  c[5] = getFingerprintElem<T>(A, -3, -3, -3, -3);
  c[6] = getFingerprintElem<T>(A, -2, -2, -2, -2);
  c[7] = getFingerprintElem<T>(A, -1, -1, -1, -1);

  return C;
}

template<typename T>
Tensor fingerprintKernel(Tensor A) {
  switch (A.getDim()) {
    case 1:
      return fingerprint1D<T>(A);
    case 2:
      return fingerprint2D<T>(A);
    case 3:
      return fingerprint3D<T>(A);
    case 4:
      return fingerprint4D<T>(A);
    default:
      NOT_IMPL();
  }
}

Tensor fingerprint(Tensor A) {
  if (A.getDType() == DType::kFloat) return fingerprintKernel<float>(A);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16) return fingerprintKernel<Float16>(A);
#endif

  NOT_IMPL();
}

}  // cpu
}  // op
}  // libllm
