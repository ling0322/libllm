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

#include "lynn/cpu/swiglu.h"

#include <cmath>

#include "lutil/thread_pool.h"
#include "lynn/cpu/accessor.h"
#include "lynn/cpu/tensor.h"
#include "lynn/mp.h"

namespace ly {
namespace op {
namespace cpu {

template<typename T>
Tensor swigluKernel(const Tensor &A) {
  std::vector<int> shapeC = A.getShape();
  shapeC.back() /= 2;
  Tensor C = tensor(shapeC, DType::getType<T>());

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  MP::parallelFor(vA.getLength(), [&vA, &vC](MP::Context ctx) {
    int j = ctx.getBlockIdx();
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    int n = c.getShape(0);
    for (int i = 0; i < n; ++i) {
      T x = a[i];
      x *= 1.0f / (1 + expf(-x));
      x *= a[i + n];
      c[i] = x;
    }
  });

  return C;
}

Tensor swiglu(const Tensor &A) {
  CHECK(A.getShape(-1) % 2 == 0);

  if (A.getDType() == DType::kFloat) return swigluKernel<float>(A);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16) return swigluKernel<Float16>(A);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
