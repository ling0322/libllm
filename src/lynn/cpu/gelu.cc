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

#include "lynn/cpu/gelu.h"

#include <math.h>

#include "lutil/thread_pool.h"
#include "lynn/cpu/accessor.h"
#include "lynn/cpu/tensor.h"
#include "lynn/mp.h"

namespace libllm {
namespace op {
namespace cpu {

constexpr float Sqrt2 = 1.4142136f;

template<typename T>
Tensor geluKernel(const Tensor &A) {
  Tensor C = tensor(A.getShape(), DType::getType<T>());

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  MP::parallelFor(vA.getLength(), [&vA, &vC](MP::Context ctx) {
    TensorAccessor<const T, 1> a = vA.getTensor(ctx.getBlockIdx());
    TensorAccessor<T, 1> c = vC.getTensor(ctx.getBlockIdx());

    int n = c.getShape(0);
    for (int i = 0; i < n; ++i) {
      float x = a[i];
      x = x * 0.5f * (1.0f + erf(x / Sqrt2));
      c[i] = T(x);
    }
  });

  return C;
}

Tensor gelu(const Tensor &A) {
  CHECK(A.getShape(-1) % 2 == 0);

  if (A.getDType() == DType::kFloat) return geluKernel<float>(A);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16) return geluKernel<Float16>(A);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
