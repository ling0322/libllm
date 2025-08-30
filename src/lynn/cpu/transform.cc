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

#include "lynn/cpu/transform.h"

#include "lynn/cpu/accessor.h"
#include "lynn/cpu/common.h"
#include "lynn/cpu/tensor.h"
#include "lynn/mp.h"
#include "lynn/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T>
Tensor transformKernel(const Tensor &A, float alpha, float beta) {
  Tensor C = tensorLike(A);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  MP::parallelFor(vA.getLength(), [&vA, &vC, alpha, beta](MP::Context ctx) {
    TensorAccessor<const T, 1> a = vA.getTensor(ctx.getBlockIdx());
    TensorAccessor<T, 1> c = vC.getTensor(ctx.getBlockIdx());

    for (int i = 0; i < a.getShape(0); ++i) {
      c[i] = a[i] * static_cast<T>(alpha) + static_cast<T>(beta);
    }
  });

  return C;
}

Tensor transform(const Tensor &src, float alpha, float beta) {
  if (src.getDType() == DType::kFloat) return transformKernel<float>(src, alpha, beta);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (src.getDType() == DType::kFloat16) return transformKernel<Float16>(src, alpha, beta);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
