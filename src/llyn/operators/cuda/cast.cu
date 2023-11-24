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

#include "llyn/operators/cuda/cast.h"

#include <cuda_fp16.h>
#include "llyn/operators/cuda/common.h"

namespace llyn {
namespace op {
namespace cuda {

__global__ void castFloatToHalfKernel(int n, const float *src, half *dest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  dest[idx] = __float2half(src[idx]);
}

__global__ void castHalfToFloatKernel(int n, const half *src, float *dest) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  dest[idx] = __half2float(src[idx]);
}

Tensor castFloatToHalf(const Tensor &tensor) {
  LL_CHECK_CONTIGUOUS(tensor);

  Tensor tgtTensor = createCudaTensorHalf(tensor.getShape());
  const float *src = tensor.getData<float>();
  half *dest = (half *)tgtTensor.getData<Float16>();

  int64_t numel = tensor.getNumEl();
  constexpr int blockSize = 256;
  int64_t nb = (numel + blockSize - 1) / blockSize;
  castFloatToHalfKernel<<<nb, blockSize>>>(numel, src, dest);
  checkCudaError(cudaGetLastError());

  return tgtTensor;
}

Tensor castHalfToFloat(const Tensor &tensor) {
  LL_CHECK_CONTIGUOUS(tensor);

  Tensor tgtTensor = createCudaTensorFloat(tensor.getShape());
  const half *src = (half *)tensor.getData<Float16>();
  float *dest = tgtTensor.getData<float>();

  int64_t numel = tensor.getNumEl();
  constexpr int blockSize = 256;
  int64_t nb = (numel + blockSize - 1) / blockSize;
  castHalfToFloatKernel<<<nb, blockSize>>>(numel, src, dest);
  checkCudaError(cudaGetLastError());

  return tgtTensor;
}

Tensor cast(const Tensor &tensor, DType dtype) {
  if (tensor.getDType() == dtype) return tensor;
  if (tensor.getDType() == DType::kFloat16 && dtype == DType::kFloat)
    return castHalfToFloat(tensor);
  if (tensor.getDType() == DType::kFloat && dtype == DType::kFloat16)
    return castFloatToHalf(tensor);

  NOT_IMPL();
}

}  // cuda
}  // op
}  // llyn
