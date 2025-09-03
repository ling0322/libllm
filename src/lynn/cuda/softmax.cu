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
#include "lynn/cuda/binary.h"
#include "lynn/cuda/common.h"
#include "lynn/cuda/reduce.h"
#include "lynn/functional.h"

namespace ly {
namespace op {
namespace cuda {

template<typename T>
__global__ void softmaxKernel3D(
    PackedTensorAccessor<const T, 3> input,
    PackedTensorAccessor<const float, 2> sumExp,
    PackedTensorAccessor<T, 3> output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < input.getShape(2) && y < input.getShape(1) && z < input.getShape(0)) {
    float el = static_cast<float>(input[z][y][x]);
    output[z][y][x] = static_cast<T>(expf(el - logf(sumExp[z][y])));
  }
}

Tensor softmaxHalf3D(Tensor A) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 3);

  Tensor max = reduceLastDim(A, DType::kFloat16, MapReduceType::MAX);
  max = max.view({max.getShape(0), max.getShape(1), 1});

  A = cuda::applyBinaryOp(BinaryOp::SUB, A, max);

  Tensor sumExp = reduceLastDim(A, DType::kFloat, MapReduceType::SUM_EXP);
  Tensor C = createCudaTensorHalf(A.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.z = A.getShape(0);
  d.y = A.getShape(1);
  d.x = (A.getShape(2) + blockSize - 1) / blockSize;

  softmaxKernel3D<half><<<d, blockSize>>>(A, sumExp, C);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor softmaxHalf1D(Tensor A) {
  Tensor xA = A.view({1, 1, A.getShape(0)});
  Tensor C = softmaxHalf3D(xA);

  return C.view({C.getShape(2)});
}

Tensor softmaxHalf2D(Tensor A) {
  Tensor xA = A.view({1, A.getShape(0), A.getShape(1)});
  Tensor C = softmaxHalf3D(xA);

  return C.view({C.getShape(1), C.getShape(2)});
}

Tensor softmaxHalf4D(Tensor A) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(2), A.getShape(3)});
  Tensor C = softmaxHalf3D(xA);

  return C.view(shape);
}

Tensor softmaxHalf(Tensor A) {
  if (A.getDim() == 1) return softmaxHalf1D(A);
  if (A.getDim() == 2) return softmaxHalf2D(A);
  if (A.getDim() == 3) return softmaxHalf3D(A);
  if (A.getDim() == 4) return softmaxHalf4D(A);

  NOT_IMPL();
}

Tensor softmax(Tensor A) {
  if (A.getDType() == DType::kFloat16) return softmaxHalf(A);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
