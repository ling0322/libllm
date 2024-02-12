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

#include "libllm/cpu/apply_rotary_pos_emb.h"

#include "libllm/cpu/subtensor.h"
#include "libllm/cpu/subtensor_list.h"
#include "libllm/cpu/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

Tensor applyRotaryPosEmbFp32(const Tensor &input, const Tensor &roPE) {
  CHECK(roPE.getDType() == DType::kFloat);

  Tensor C = zerosLike(input);
  Subtensor<float> Cs = Subtensor<float>::fromTensor(C);
  
  SubtensorList<const float> vAs = getVectorList(Subtensor<const float>::fromTensor(input));
  SubtensorList<const float> vRs = getVectorList(Subtensor<const float>::fromTensor(roPE));
  SubtensorList<float> vCs = getVectorList(Cs);
  CHECK(vAs.getSize() == vCs.getSize());
  CHECK(vAs.getSize() == vRs.getSize());

  for (int j = 0; j < vAs.getSize(); ++j) {
    Subtensor<const float> vA = vAs.getSubtensor(j);
    Subtensor<const float> vR = vRs.getSubtensor(j % vRs.getSize());  // (D/2, 2) matrix
    Subtensor<float> vC = vCs.getSubtensor(j);

    int n = vA.dimension(0) / 2;
    const float *pA = vA.data;
    const float *pR = vR.data;
    float *pC = vC.data;
    for (int i = 0; i < n; ++i) {
      pC[0] = pA[0] * pR[0] - pA[1] * pR[1];
      pC[1] = pA[1] * pR[0] + pA[0] * pR[1];
      pA += 2;
      pR += 2;
      pC += 2;
    }
  }

  return C;
}

Tensor applyRotaryPosEmb(const Tensor &input, Tensor roPE) {
  CHECK(input.getDim() == 4 && roPE.getDim() == 3 && roPE.isContiguous());
  CHECK(input.getShape(1) == roPE.getShape(0) && input.getShape(3) == roPE.getShape(2));

  roPE = roPE.unsqueeze(0);
  roPE = roPE.expand({input.getShape(0), roPE.getShape(1), input.getShape(2), roPE.getShape(3)});

  if (input.getDType() == DType::kFloat) return applyRotaryPosEmbFp32(input, roPE);

  NOT_IMPL();
}

}  // cpu
}  // op
}  // ly
