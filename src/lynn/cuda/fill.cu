// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include "lynn/cuda/common.h"
#include "lynn/cuda/fill.h"

namespace ly {
namespace op {
namespace cuda {

template<typename scalar_t>
__global__ void fillContigKernel(scalar_t *__restrict__ data, int numel, scalar_t v) {
  int stride = (int64_t)blockDim.x * gridDim.x;
  int idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    data[idx] = v;
  }
}

template<typename scalar_t, int DIM>
__global__ void fillGenericKernel(PackedTensorAccessor<scalar_t, DIM> A, int numel, scalar_t v) {
  int stride = (int64_t)blockDim.x * gridDim.x;
  int idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    A.getElemByIndex(idx) = v;
  }
}

template<typename T>
void fillImpl(Tensor &tensor, T v) {
  int64_t numel64 = tensor.getNumEl();
  CHECK(numel64 < std::numeric_limits<int>::max());
  int numel = static_cast<int>(numel64);

  constexpr int blockSize = 256;
  dim3 grid = getGrid1D(numel, blockSize);
  int d = tensor.getDim();

  if (tensor.isContiguous()) {
    fillContigKernel<T><<<grid, blockSize>>>(tensor.getInternalData()->getData<T>(), numel, v);
  } else {
    if (d == 1)
      fillGenericKernel<T, 1><<<grid, blockSize>>>(tensor, numel, v);
    else if (d == 2)
      fillGenericKernel<T, 2><<<grid, blockSize>>>(tensor, numel, v);
    else if (d == 3)
      fillGenericKernel<T, 3><<<grid, blockSize>>>(tensor, numel, v);
    else if (d == 4)
      fillGenericKernel<T, 4><<<grid, blockSize>>>(tensor, numel, v);
    else
      NOT_IMPL();
  }

  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}
void fill(Tensor A, float value) {
  CHECK(A.getDevice().getType() == Device::kCuda);

  if (A.getDType() == DType::kFloat16)
    fillImpl<half>(A, __float2half(value));
  else
    NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
