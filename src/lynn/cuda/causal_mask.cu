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
#include <math.h>

#include "lynn/cuda/accessor.h"
#include "lynn/cuda/causal_mask.h"
#include "lynn/cuda/common.h"

namespace ly {
namespace op {
namespace cuda {

__global__ void causalMaskKernel(PackedTensorAccessor<half, 2> mask) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (y < mask.getShape(0) && x < mask.getShape(1)) {
    mask[y][x] = x > y ? -INFINITY : 0;
  }
}

Tensor causalMask(int size) {
  Tensor C = createCudaTensorHalf({size, size});

  constexpr int blockSize = 256;
  dim3 d;
  d.y = C.getShape(0);
  d.x = (C.getShape(1) + blockSize - 1) / blockSize;

  causalMaskKernel<<<d, blockSize>>>(C);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
