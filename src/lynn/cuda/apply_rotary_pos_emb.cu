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

#include <cuda_fp16.h>

#include "lynn/cpu/view.h"
#include "lynn/cuda/accessor.h"
#include "lynn/cuda/apply_rotary_pos_emb.h"
#include "lynn/cuda/common.h"

namespace ly {
namespace op {
namespace cuda {

__global__ void applyRotaryPosEmbKernel4D(
    PackedTensorAccessor<const half, 4> A,
    PackedTensorAccessor<const half, 2> roPE,
    PackedTensorAccessor<half, 4> C) {
  // dimensions for A and C are (N, L, H, D)
  int dD = blockIdx.x * blockDim.x + threadIdx.x;

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  dim3 dy = splitIndexToDim3(y, A.getSize());
  int dN = dy.z;
  int dL = dy.y;
  int dH = dy.x;

  if (dN >= A.getShape(0) || dL >= A.getShape(1) || dH >= A.getShape(2) ||
      dD >= A.getShape(3) / 2) {
    return;
  }

  TensorAccessor<const half, 1> vA = A[dN][dL][dH];
  TensorAccessor<half, 1> vC = C[dN][dL][dH];
  half cosTheta = roPE[dL][dD * 2];
  half sinTheta = roPE[dL][dD * 2 + 1];
  half a0 = vA[dD * 2];
  half a1 = vA[dD * 2 + 1];
  vC[dD * 2] = a0 * cosTheta - a1 * sinTheta;
  vC[dD * 2 + 1] = a1 * cosTheta + a0 * sinTheta;
}

Tensor applyRotaryPosEmb4D(const Tensor &tensor, const Tensor &rope) {
  CHECK(tensor.getDim() == 4);
  CHECK(rope.getShape(0) == tensor.getShape(1) && rope.getShape(1) == tensor.getShape(3));

  Tensor C = createCudaTensorHalf(tensor.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.y = tensor.getShape(0) * tensor.getShape(1) * tensor.getShape(2);
  d.x = (tensor.getShape(3) / 2 + blockSize - 1) / blockSize;

  applyRotaryPosEmbKernel4D<<<d, blockSize>>>(tensor, rope, C);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor applyRotaryPosEmb(const Tensor &tensor, Tensor roPE) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  CHECK(roPE.getDevice().getType() == Device::kCuda);
  CHECK(tensor.getShape(-1) % 2 == 0);
  CHECK(roPE.getDim() == 3 && roPE.getShape(1) == 1);

  roPE = op::cpu::view(roPE, {roPE.getShape(0), roPE.getShape(2)});
  if (tensor.getDim() == 4) return applyRotaryPosEmb4D(tensor, roPE);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
