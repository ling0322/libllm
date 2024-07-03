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

#include "libllm/cpu/unfold.h"

#include "libllm/cpu/accessor.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/copy.h"
#include "libllm/cpu/tensor.h"
#include "libllm/mp.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T>
void unfold1DKernel(const Tensor &src, Tensor &dest, int kernelSize) {
  TensorList<const T, 2> mA = TensorList<const T, 2>::fromTensor(src);
  TensorList<T, 2> mC = TensorList<T, 2>::fromTensor(dest);
  CHECK(mA.getLength() == mC.getLength());

  for (int i = 0; i < mA.getLength(); ++i) {
    TensorAccessor<const T, 2> vA = mA.getTensor(i);
    TensorAccessor<T, 2> vC = mC.getTensor(i);
    CHECK(vA.getShape(0) == vC.getShape(0));

    MP::parallelFor({vC.getShape(0)}, [&vA, &vC, kernelSize](MP::Partition partition) {
      int kernekIdxBegin = -(kernelSize / 2);
      int kernekIdxEnd = (kernelSize - 1) / 2;

      for (int j : partition.getRange()) {
        int numChennels = vA.getShape(1);
        int numInFrames = vA.getShape(0);

        for (int k = kernekIdxBegin; k <= kernekIdxEnd; ++k) {
          int srcIdx = j + k;
          int destDimOffset = (k - kernekIdxBegin) * numChennels;
          if (srcIdx < 0 || srcIdx >= numInFrames) {
            // padding.
            for (int d = 0; d < numChennels; ++d) {
              vC[j][destDimOffset + d] = 0.0f;
            }
          } else {
            for (int d = 0; d < numChennels; ++d) {
              vC[j][destDimOffset + d] = vA[srcIdx][d];
            }
          }
        }
      }
    });
  }
}

Tensor unfold(const Tensor &src, lut::Span<const int> kernelSize) {
  CHECK(kernelSize.size() == 1) << "only unfold1D is supported.";
  CHECK(src.getDim() >= 2);
  CHECK(src.getShape(-1) >= kernelSize[0]);

  std::vector<Tensor::ShapeType> shape = src.getShape();
  shape.back() *= kernelSize[0];

  Tensor dest = op::cpu::tensor(shape, src.getDType());

  if (src.getDType() == DType::kFloat) {
    unfold1DKernel<float>(src, dest, kernelSize[0]);
  } else if (src.getDType() == DType::kFloat16) {
#if LUT_CPU_ARCH == LUT_AARCH64
    unfold1DKernel<Float16>(src, dest, kernelSize[0]);
#else
    NOT_IMPL();
#endif
  } else {
    NOT_IMPL();
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
