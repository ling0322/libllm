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

#include "lynn/cuda/binary.h"
#include "lynn/cuda/binary_scalar.h"
#include "lynn/cuda/cast.h"
#include "lynn/cuda/common.h"
#include "lynn/cuda/layer_norm.h"
#include "lynn/cuda/reduce.h"
#include "lynn/functional.h"

namespace libllm {
namespace op {
namespace cuda {

__global__ void layerNormKernel3D(
    PackedTensorAccessor<const half, 3> inputTensor,
    PackedTensorAccessor<const half, 2> mean,
    PackedTensorAccessor<const float, 2> sumDiffSquare,
    PackedTensorAccessor<const half, 1> weight,
    PackedTensorAccessor<const half, 1> bias,
    PackedTensorAccessor<half, 3> outputTensor,
    float eps) {
  assert(inputTensor.getShape(0) == outputTensor.getShape(0));
  assert(inputTensor.getShape(1) == outputTensor.getShape(1));
  assert(inputTensor.getShape(2) == outputTensor.getShape(2));
  assert(inputTensor.getShape(0) == mean.getShape(0));
  assert(inputTensor.getShape(1) == mean.getShape(1));
  assert(inputTensor.getShape(0) == sumDiffSquare.getShape(0));
  assert(inputTensor.getShape(1) == sumDiffSquare.getShape(1));
  assert(inputTensor.getShape(2) == weight.getShape(0));
  assert(inputTensor.getShape(2) == bias.getShape(0));

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (z < inputTensor.getShape(0) && y < inputTensor.getShape(1) && x < inputTensor.getShape(2)) {
    float meanVal = mean[z][y];
    float varVal = sumDiffSquare[z][y] / inputTensor.getShape(2);
    float sd = sqrt(varVal + eps);
    float elem = inputTensor[z][y][x];
    elem = (elem - meanVal) / sd;
    elem = elem * __half2float(weight[x]) + __half2float(bias[x]);
    outputTensor[z][y][x] = elem;
  }
}

Tensor layerNorm3D(Tensor tensor, Tensor weight, Tensor bias, float eps) {
  CHECK(tensor.getShape(-1) == weight.getShape(0) && bias.getShape(0) == weight.getShape(0));

  Tensor reduceSum = op::cuda::reduceLastDim(tensor, DType::kFloat, MapReduceType::SUM);
  CHECK(reduceSum.getDim() == 2);
  reduceSum = op::cuda::castFloatToHalf(reduceSum);

  Tensor mean = applyBinaryScalarOp(BinaryScalarOp::DIV, reduceSum, tensor.getShape(2));
  Tensor diff = op::cuda::applyBinaryOp(BinaryOp::SUB, tensor, mean.unsqueeze(2));

  Tensor sumDiffSquare = op::cuda::reduceLastDim(diff, DType::kFloat, MapReduceType::SUM_SQUARE);
  Tensor C = createCudaTensorHalf(tensor.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.z = C.getShape(0);
  d.y = C.getShape(1);
  d.x = (C.getShape(2) + blockSize - 1) / blockSize;

  layerNormKernel3D<<<d, blockSize>>>(tensor, mean, sumDiffSquare, weight, bias, C, eps);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor layerNorm(Tensor tensor, Tensor weight, Tensor bias, float eps) {
  CHECK(weight.getDim() == 1 && bias.getDim() == 1);
  CHECK(tensor.getShape(-1) == weight.getShape(0));
  CHECK(tensor.getShape(-1) == bias.getShape(0));
  CHECK(weight.getDevice().getType() == Device::kCuda);

  if (tensor.getDim() == 3) return layerNorm3D(tensor, weight, bias, eps);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
