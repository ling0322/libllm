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

#include "lten/cuda/common.h"
#include "lten/cuda/reduce.h"
#include "lten/cuda/rms_norm.h"

namespace lten {
namespace op {
namespace cuda {

__global__ void rmsNormKernel3D(
    PackedSubtensor<const half, 3> inputTensor,
    PackedSubtensor<const float, 2> sumSquare,
    PackedSubtensor<const half, 1> weight,
    PackedSubtensor<half, 3> outputTensor,
    float eps) {
  assert(inputTensor.getShape(0) == outputTensor.getShape(0));
  assert(inputTensor.getShape(1) == outputTensor.getShape(1));
  assert(inputTensor.getShape(2) == outputTensor.getShape(2));
  assert(inputTensor.getShape(0) == sumSquare.getShape(0));
  assert(inputTensor.getShape(1) == sumSquare.getShape(1));
  assert(inputTensor.getShape(2) == weight.getShape(0));

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (z < inputTensor.getShape(0) && y < inputTensor.getShape(1) && x < inputTensor.getShape(2)) {
    half rms = __float2half(sqrtf(sumSquare[z][y] / __int2float_rd(weight.getShape(0)) + eps));
    outputTensor[z][y][x] = inputTensor[z][y][x] * weight[x] / rms;
  }
}

Tensor rmsNorm3D(const Tensor &tensor, const Tensor &weight, float eps) {
  Tensor reduceNorm2 = op::cuda::reduceHalfToSingle3D(tensor, MapReduceType::SUM_SQUARE_FP16_FP32);
  CHECK(reduceNorm2.getDim() == 2);

  Tensor C = createCudaTensorHalf(tensor.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.z = C.getShape(0);
  d.y = C.getShape(1);
  d.x = (C.getShape(2) + blockSize - 1) / blockSize;

  rmsNormKernel3D<<<d, blockSize>>>(tensor, reduceNorm2, weight, C, eps);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor rmsNorm(const Tensor &tensor, const Tensor &weight, float eps) {
  CHECK(weight.getDim() == 1);
  CHECK(tensor.getShape(-1) == weight.getShape(0));
  CHECK(weight.getDevice().getType() == Device::kCuda);

  if (tensor.getDim() == 3) return rmsNorm3D(tensor, weight, eps);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace lten
