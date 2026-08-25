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

#include "flint/cpu/causal_conv1d.h"

#include <omp.h>

#include "flint/cpu/common.h"
#include "flint/cpu/tensor.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

Tensor causalConv1d(const Tensor &input, const Tensor &weight, const Tensor &cuSeqlens) {
  CHECK(input.getDim() == 2 && weight.getDim() == 2 && cuSeqlens.getDim() == 1);
  CHECK(input.getDType() == DType::kFloat && weight.getDType() == DType::kFloat);
  CHECK(cuSeqlens.getDType() == DType::kInt32);
  CHECK(input.isContiguous() && weight.isContiguous() && cuSeqlens.isContiguous());

  int numTokens = input.getShape(0);
  int numChannels = input.getShape(1);
  int kernelSize = weight.getShape(1);
  CHECK(weight.getShape(0) == numChannels && kernelSize > 0);

  int numSequences = cuSeqlens.getShape(0) - 1;
  CHECK(numSequences >= 0);

  Tensor output = op::cpu::tensor({numTokens, numChannels}, DType::kFloat);
  const float *in = getDataPtrCpu<float>(input);
  const float *w = getDataPtrCpu<float>(weight);
  const IntType *offsets = getDataPtrCpu<IntType>(cuSeqlens);
  float *out = getDataPtrCpu<float>(output);

#pragma omp parallel for num_threads(omp_get_max_threads())
  for (int sequence = 0; sequence < numSequences; ++sequence) {
    int begin = offsets[sequence];
    int end = offsets[sequence + 1];
    CHECK(begin >= 0 && begin <= end && end <= numTokens);

    for (int token = begin; token < end; ++token) {
      for (int channel = 0; channel < numChannels; ++channel) {
        float sum = 0.0f;
        for (int tap = 0; tap < kernelSize; ++tap) {
          // The window ends at `token`, so tap kernelSize-1 is the current position. Anything
          // reaching back past the start of this sequence reads as zero rather than borrowing
          // the tail of the sequence packed before it.
          int source = token - (kernelSize - 1) + tap;
          if (source < begin) continue;
          sum += w[channel * kernelSize + tap] * in[source * numChannels + channel];
        }
        out[token * numChannels + channel] = sum;
      }
    }
  }

  return output;
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
