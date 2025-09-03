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

#include "lynn/cuda/accessor.h"
#include "lynn/cuda/common.h"
#include "lynn/cuda/unfold.h"

namespace ly {
namespace op {
namespace cuda {

__global__ void unfold1DKernel3D(
    PackedTensorAccessor<const half, 3> A,
    PackedTensorAccessor<half, 3> C,
    int kernelSize,
    int stride) {
  // x, y, z is the dimensions of C.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= C.getShape(2) || y >= C.getShape(1) || z >= C.getShape(0)) {
    return;
  }

  assert(A.getShape(2) * kernelSize == C.getShape(2));
  assert(A.getShape(1) / stride == C.getShape(1));

  int kernelIdxBegin = -(kernelSize / 2);

  int yA = y * stride + kernelIdxBegin + x % kernelSize;
  int xA = x / kernelSize;

  if (yA >= 0 && yA < A.getShape(1)) {
    C[z][y][x] = A[z][yA][xA];
  } else {
    C[z][y][x] = 0.0;
  }
}

Tensor unfold(const Tensor &src, int kernelSize, int stride) {
  CHECK(src.getDim() == 3);
  CHECK(src.getShape(-1) >= kernelSize);

  std::vector<Tensor::ShapeType> shape = src.getShape();
  shape.back() *= kernelSize;
  shape[shape.size() - 2] /= stride;

  Tensor C = createCudaTensorHalf(shape);

  constexpr int blockSize = 512;
  dim3 d;
  d.z = C.getShape(0);
  d.y = C.getShape(1);
  d.x = (C.getShape(2) + blockSize - 1) / blockSize;

  unfold1DKernel3D<<<d, blockSize>>>(src, C, kernelSize, stride);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
