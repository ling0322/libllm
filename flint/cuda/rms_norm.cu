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
#include <cub/block/block_reduce.cuh>

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/rms_norm.h"

namespace fl {
namespace op {
namespace cuda {

template<int BLOCK_SIZE, bool VECTORIZED>
__global__ void rmsNormFusedKernel(
    const half *__restrict__ input,
    const half *__restrict__ weight,
    half *__restrict__ output,
    int hiddenSize,
    float eps) {
  int row = blockIdx.x;
  int rowOffset = row * hiddenSize;
  float sumSquare = 0.0f;

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    int width = hiddenSize / 2;
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      sumSquare += value.x * value.x + value.y * value.y;
    }
  } else {
    for (int i = threadIdx.x; i < hiddenSize; i += BLOCK_SIZE) {
      float value = __half2float(input[rowOffset + i]);
      sumSquare += value * value;
    }
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float invRms;
  sumSquare = BlockReduce(tempStorage).Sum(sumSquare);
  if (threadIdx.x == 0) {
    invRms = rsqrtf(sumSquare / hiddenSize + eps);
  }
  __syncthreads();

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    const half2 *weight2 = reinterpret_cast<const half2 *>(weight);
    half2 *output2 = reinterpret_cast<half2 *>(output + rowOffset);
    int width = hiddenSize / 2;
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      float2 scale = __half22float2(weight2[i]);
      output2[i] = __floats2half2_rn(
          value.x * scale.x * invRms,
          value.y * scale.y * invRms);
    }
  } else {
    for (int i = threadIdx.x; i < hiddenSize; i += BLOCK_SIZE) {
      float value = __half2float(input[rowOffset + i]);
      float scale = __half2float(weight[i]);
      output[rowOffset + i] = __float2half(value * scale * invRms);
    }
  }
}

template<int BLOCK_SIZE>
__global__ void rmsNormStridedKernel(
    PackedTensorAccessor<const half, 3> inputTensor,
    PackedTensorAccessor<const half, 1> weight,
    PackedTensorAccessor<half, 3> outputTensor,
    float eps) {
  int hiddenSize = inputTensor.getShape(2);
  int y = blockIdx.x % inputTensor.getShape(1);
  int z = blockIdx.x / inputTensor.getShape(1);

  float sumSquare = 0.0f;
  for (int x = threadIdx.x; x < hiddenSize; x += BLOCK_SIZE) {
    float value = __half2float(inputTensor[z][y][x]);
    sumSquare += value * value;
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float invRms;
  sumSquare = BlockReduce(tempStorage).Sum(sumSquare);
  if (threadIdx.x == 0) {
    invRms = rsqrtf(sumSquare / hiddenSize + eps);
  }
  __syncthreads();

  for (int x = threadIdx.x; x < hiddenSize; x += BLOCK_SIZE) {
    float value = __half2float(inputTensor[z][y][x]);
    float scale = __half2float(weight[x]);
    outputTensor[z][y][x] = __float2half(value * scale * invRms);
  }
}

Tensor rmsNormStrided3D(const Tensor &tensor, const Tensor &weight, float eps) {
  Tensor C = createCudaTensorHalf(tensor.getShape());

  constexpr int blockSize = 256;
  int rows = tensor.getShape(0) * tensor.getShape(1);
  rmsNormStridedKernel<blockSize><<<rows, blockSize>>>(tensor, weight, C, eps);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor rmsNormContiguous3D(const Tensor &tensor, const Tensor &weight, float eps) {
  int hiddenSize = tensor.getShape(2);
  int64_t numel = tensor.getNumEl();
  CHECK(numel < std::numeric_limits<int>::max());
  int64_t rows64 = numel / hiddenSize;
  CHECK(rows64 < std::numeric_limits<int>::max());
  int rows = static_cast<int>(rows64);

  Tensor C = createCudaTensorHalf(tensor.getShape());
  const half *input = getDataPtrCuda<half>(tensor);
  const half *weightData = getDataPtrCuda<half>(weight);
  half *output = getDataPtrCuda<half>(C);

  constexpr int blockSize = 256;
  bool useHalf2 = hiddenSize % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(input) % alignof(half2) == 0 &&
                  reinterpret_cast<uintptr_t>(weightData) % alignof(half2) == 0 &&
                  reinterpret_cast<uintptr_t>(output) % alignof(half2) == 0;
  if (useHalf2) {
    rmsNormFusedKernel<blockSize, true>
        <<<rows, blockSize>>>(input, weightData, output, hiddenSize, eps);
  } else {
    rmsNormFusedKernel<blockSize, false>
        <<<rows, blockSize>>>(input, weightData, output, hiddenSize, eps);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor rmsNorm3D(const Tensor &tensor, const Tensor &weight, float eps) {
  if (tensor.isContiguous() && weight.isContiguous()) {
    return rmsNormContiguous3D(tensor, weight, eps);
  }
  return rmsNormStrided3D(tensor, weight, eps);
}

Tensor rmsNorm(const Tensor &tensor, const Tensor &weight, float eps) {
  CHECK(weight.getDim() == 1);
  CHECK(tensor.getShape(-1) == weight.getShape(0));
  CHECK(weight.getDevice().getType() == Device::kCuda);

  if (tensor.getDim() == 3) return rmsNorm3D(tensor, weight, eps);

  // a packed batch is 2D, the kernels index it as a 3D tensor with one leading row.
  if (tensor.getDim() == 2) return rmsNorm3D(tensor.unsqueeze(0), weight, eps).subtensor(0);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
