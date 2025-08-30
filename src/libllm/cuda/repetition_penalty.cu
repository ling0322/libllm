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

#include <cuda_fp16.h>

#include "libllm/cuda/common.h"
#include "libllm/cuda/repetition_penalty.h"

namespace libllm {
namespace op {
namespace cuda {

__device__ constexpr int MaxHistory = 64;

__global__ void repetitionPenalty2DKernel(
    PackedTensorAccessor<half, 2> logits,
    PackedTensorAccessor<const LongType, 2> history,
    float weight) {
  assert(logits.getShape(0) == history.getShape(0));

  int batchIdx = blockIdx.x;
  int historyIdx = threadIdx.x;

  if (historyIdx >= history.getShape(1)) {
    return;
  }

  // gather
  LongType logitsIdx = history[batchIdx][historyIdx];
  assert(logitsIdx < logits.getShape(1));

  float score = logits[batchIdx][logitsIdx];
  if (score > 0) {
    score /= weight;
  } else if (score < 0) {
    score *= weight;
  }

  logits[batchIdx][logitsIdx] = score;
}

void repetitionPenalty2D(Tensor logits, Tensor history, float weight) {
  CHECK(history.getShape(1) < MaxHistory);

  repetitionPenalty2DKernel<<<logits.getShape(0), MaxHistory>>>(logits, history, weight);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

void repetitionPenalty1D(const Tensor &logits, const Tensor &history, float weight) {
  repetitionPenalty2D(logits.unsqueeze(0), history.unsqueeze(0), weight);
}

void repetitionPenalty(Tensor logits, Tensor history, float weight) {
  if (logits.getDim() == 2)
    repetitionPenalty2D(logits, history, weight);
  else if (logits.getDim() == 1)
    repetitionPenalty1D(logits, history, weight);
  else
    NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
