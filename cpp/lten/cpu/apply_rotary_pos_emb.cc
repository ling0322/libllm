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

#include "lten/cpu/apply_rotary_pos_emb.h"

#include "lten/cpu/accessor.h"
#include "lten/cpu/tensor.h"

namespace lten {
namespace op {
namespace cpu {

template<typename T>
Tensor applyRotaryPosEmbKernel(const Tensor &input, const Tensor &roPE) {
  CHECK(roPE.getDType() == DType::kFloat || roPE.getDType() == DType::kFloat16);

  Tensor C = zerosLike(input);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(input);
  TensorList<const T, 1> vR = TensorList<const T, 1>::fromTensor(roPE);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());
  CHECK(vA.getLength() == vR.getLength());

  for (int j = 0; j < vA.getLength(); ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<const T, 1> r = vR.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    for (int i = 0; i < a.getShape(0); i += 2) {
      c[i + 0] = a[i + 0] * r[i + 0] - a[i + 1] * r[i + 1];
      c[i + 1] = a[i + 1] * r[i + 0] + a[i + 0] * r[i + 1];
    }
  }

  return C;
}

Tensor applyRotaryPosEmb(const Tensor &input, Tensor roPE) {
  CHECK(input.getDim() == 4 && roPE.getDim() == 3 && roPE.isContiguous());
  CHECK(input.getShape(1) == roPE.getShape(0) && input.getShape(3) == roPE.getShape(2));

  roPE = roPE.unsqueeze(0);
  roPE = roPE.expand({input.getShape(0), roPE.getShape(1), input.getShape(2), roPE.getShape(3)});

  if (input.getDType() == DType::kFloat) return applyRotaryPosEmbKernel<float>(input, roPE);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (input.getDType() == DType::kFloat16) return applyRotaryPosEmbKernel<Float16>(input, roPE);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace lten
