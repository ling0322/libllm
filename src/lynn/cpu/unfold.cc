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

#include "lynn/cpu/unfold.h"

#include "lynn/cpu/accessor.h"
#include "lynn/cpu/common.h"
#include "lynn/cpu/copy.h"
#include "lynn/cpu/tensor.h"
#include "lynn/mp.h"

namespace ly {
namespace op {
namespace cpu {

template<typename T>
void unfold1DKernel(const Tensor &src, Tensor &dest, int kernelSize, int stride) {
  TensorList<const T, 2> mA = TensorList<const T, 2>::fromTensor(src);
  TensorList<T, 2> mC = TensorList<T, 2>::fromTensor(dest);
  CHECK(mA.getLength() == mC.getLength());

  for (int i = 0; i < mA.getLength(); ++i) {
    TensorAccessor<const T, 2> vA = mA.getTensor(i);
    TensorAccessor<T, 2> vC = mC.getTensor(i);
    CHECK(vA.getShape(0) / stride == vC.getShape(0));

    MP::parallelFor(vC.getShape(0), [&vA, &vC, kernelSize, stride](MP::Context ctx) {
      int j = ctx.getBlockIdx();
      int kernekIdxBegin = -(kernelSize / 2);
      int kernekIdxEnd = (kernelSize - 1) / 2;
      int numChannels = vA.getShape(1);
      int numInFrames = vA.getShape(0);

      for (int d = 0; d < numChannels; ++d) {
        for (int k = kernekIdxBegin; k <= kernekIdxEnd; ++k) {
          int srcIdx = j * stride + k;
          int offset = k - kernekIdxBegin;
          if (srcIdx < 0 || srcIdx >= numInFrames) {
            // padding.
            vC[j][d * kernelSize + offset] = 0.0f;
          } else {
            vC[j][d * kernelSize + offset] = vA[srcIdx][d];
          }
        }
      }
    });
  }
}

Tensor unfold(const Tensor &src, int kernelSize, int stride) {
  CHECK(src.getDim() >= 2);
  CHECK(src.getShape(-1) >= kernelSize);

  std::vector<Tensor::ShapeType> shape = src.getShape();
  shape.back() *= kernelSize;
  shape[shape.size() - 2] /= stride;

  Tensor dest = op::cpu::tensor(shape, src.getDType());

  if (src.getDType() == DType::kFloat) {
    unfold1DKernel<float>(src, dest, kernelSize, stride);
  } else if (src.getDType() == DType::kFloat16) {
#if LUT_CPU_ARCH == LUT_AARCH64
    unfold1DKernel<Float16>(src, dest, kernelSize, stride);
#else
    NOT_IMPL();
#endif
  } else {
    NOT_IMPL();
  }

  return dest;
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
