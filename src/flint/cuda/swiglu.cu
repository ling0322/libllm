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

#include "flint/cuda/common.h"
#include "flint/cuda/swiglu.h"

namespace fl {
namespace op {
namespace cuda {

template<bool VECTORIZED>
__global__ void swigluContiguousKernel(
    const half *__restrict__ input,
    half *__restrict__ output,
    int inputWidth,
    int outputWidth) {
  int row = blockIdx.z * gridDim.y + blockIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if constexpr (VECTORIZED) {
    int x2 = x * 2;
    if (x2 >= outputWidth) return;

    int inputOffset = row * inputWidth + x2;
    int outputOffset = row * outputWidth + x2;
    float2 gate = __half22float2(*reinterpret_cast<const half2 *>(input + inputOffset));
    float2 value =
        __half22float2(*reinterpret_cast<const half2 *>(input + inputOffset + outputWidth));
    *reinterpret_cast<half2 *>(output + outputOffset) = __floats2half2_rn(
        value.x * gate.x / (1.0f + expf(-gate.x)),
        value.y * gate.y / (1.0f + expf(-gate.y)));
  } else {
    if (x >= outputWidth) return;

    int inputOffset = row * inputWidth + x;
    float gate = __half2float(input[inputOffset]);
    float value = __half2float(input[inputOffset + outputWidth]);
    output[row * outputWidth + x] =
        __float2half(value * gate / (1.0f + expf(-gate)));
  }
}

template<bool VECTORIZED>
__global__ void swigluStridedKernel(
    const half *__restrict__ input,
    half *__restrict__ output,
    int inputStride0,
    int inputStride1,
    int inputStride2,
    int outputWidth) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int inputOffset = z * inputStride0 + y * inputStride1;
  int outputOffset = (z * gridDim.y + y) * outputWidth;

  if constexpr (VECTORIZED) {
    int x2 = x * 2;
    if (x2 >= outputWidth) return;

    float2 gate = __half22float2(
        *reinterpret_cast<const half2 *>(input + inputOffset + x2));
    float2 value = __half22float2(
        *reinterpret_cast<const half2 *>(input + inputOffset + outputWidth + x2));
    *reinterpret_cast<half2 *>(output + outputOffset + x2) = __floats2half2_rn(
        value.x * gate.x / (1.0f + expf(-gate.x)),
        value.y * gate.y / (1.0f + expf(-gate.y)));
  } else {
    if (x >= outputWidth) return;

    int gateOffset = inputOffset + x * inputStride2;
    float gate = __half2float(input[gateOffset]);
    float value = __half2float(input[gateOffset + outputWidth * inputStride2]);
    output[outputOffset + x] =
        __float2half(value * gate / (1.0f + expf(-gate)));
  }
}

Tensor swiglu3D(const Tensor &tensor) {
  std::vector<Tensor::ShapeType> shapeC = tensor.getShape();
  shapeC.back() /= 2;

  Tensor C = createCudaTensorHalf(shapeC);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = C.getShape(0);
  d.y = C.getShape(1);
  if (tensor.isContiguous()) {
    const half *input = getDataPtrCuda<half>(tensor);
    half *output = getDataPtrCuda<half>(C);
    int inputWidth = tensor.getShape(2);
    int outputWidth = C.getShape(2);
    bool useHalf2 = outputWidth % 2 == 0 &&
                    reinterpret_cast<uintptr_t>(input) % alignof(half2) == 0 &&
                    reinterpret_cast<uintptr_t>(output) % alignof(half2) == 0;
    if (useHalf2) {
      d.x = (outputWidth / 2 + blockSize - 1) / blockSize;
      swigluContiguousKernel<true><<<d, blockSize>>>(input, output, inputWidth, outputWidth);
    } else {
      d.x = (outputWidth + blockSize - 1) / blockSize;
      swigluContiguousKernel<false><<<d, blockSize>>>(input, output, inputWidth, outputWidth);
    }
  } else {
    const half *input = getDataPtrCuda<half>(tensor);
    half *output = getDataPtrCuda<half>(C);
    int inputStride0 = tensor.getStride(0);
    int inputStride1 = tensor.getStride(1);
    int inputStride2 = tensor.getStride(2);
    int outputWidth = C.getShape(2);
    bool useHalf2 = inputStride2 == 1 && outputWidth % 2 == 0 && inputStride0 % 2 == 0 &&
                    inputStride1 % 2 == 0 &&
                    reinterpret_cast<uintptr_t>(input) % alignof(half2) == 0 &&
                    reinterpret_cast<uintptr_t>(output) % alignof(half2) == 0;
    if (useHalf2) {
      d.x = (outputWidth / 2 + blockSize - 1) / blockSize;
      swigluStridedKernel<true><<<d, blockSize>>>(
          input, output, inputStride0, inputStride1, inputStride2, outputWidth);
    } else {
      d.x = (outputWidth + blockSize - 1) / blockSize;
      swigluStridedKernel<false><<<d, blockSize>>>(
          input, output, inputStride0, inputStride1, inputStride2, outputWidth);
    }
  }
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor swiglu(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  CHECK(tensor.getShape(-1) % 2 == 0);

  if (tensor.getDim() == 3) return swiglu3D(tensor);

  // a packed batch is 2D, the kernels index it as a 3D tensor with one leading row.
  if (tensor.getDim() == 2) return swiglu3D(tensor.unsqueeze(0)).subtensor(0);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
