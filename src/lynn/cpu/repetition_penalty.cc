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

#include "lynn/cpu/repetition_penalty.h"

#include "lynn/cpu/accessor.h"
#include "lynn/cpu/tensor.h"
#include "lynn/mp.h"
#include "lynn/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T>
void repetitionPenalty2DKernel(Tensor logits, Tensor history, float weight) {
  CHECK(logits.getDim() == 2 && history.getDim() == 2);
  CHECK(logits.getShape(0) == history.getShape(0));

  TensorList<T, 1> vA = TensorList<T, 1>::fromTensor(logits);
  TensorList<const LongType, 1> vH = TensorList<const LongType, 1>::fromTensor(history);
  CHECK(vA.getLength() == vH.getLength());

  MP::parallelFor(vA.getLength(), [&vA, &vH, weight](MP::Context ctx) {
    TensorAccessor<T, 1> a = vA.getTensor(ctx.getBlockIdx());
    TensorAccessor<const LongType, 1> h = vH.getTensor(ctx.getBlockIdx());

    // gather. Avoid the same logit penalizing twice.
    std::vector<T> scores(h.getShape(0));
    for (int i = 0; i < h.getShape(0); ++i) {
      LongType logitsIdx = h[i];
      CHECK(logitsIdx < a.getShape(0));

      T v = a[logitsIdx];
      if (v > 0) {
        v /= weight;
      } else if (v < 0) {
        v *= weight;
      }

      scores[i] = v;
    };

    // scatter
    for (int i = 0; i < h.getShape(0); ++i) {
      LongType logitsIdx = h[i];
      a[logitsIdx] = scores[i];
    };
  });
}

void repetitionPenalty(Tensor logits, Tensor history, float weight) {
  if (logits.getDType() == DType::kFloat && logits.getDim() == 2)
    repetitionPenalty2DKernel<float>(logits, history, weight);
  else if (logits.getDType() == DType::kFloat && logits.getDim() == 1)
    repetitionPenalty2DKernel<float>(logits.unsqueeze(0), history.unsqueeze(0), weight);
#if LUT_CPU_ARCH == LUT_AARCH64
  else if (logits.getDType() == DType::kFloat16 && logits.getDim() == 2)
    repetitionPenalty2DKernel<Float16>(logits, history, weight);
  else if (logits.getDType() == DType::kFloat16 && logits.getDim() == 1)
    repetitionPenalty2DKernel<Float16>(logits.unsqueeze(0), history.unsqueeze(0), weight);
#endif
  else
    NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
