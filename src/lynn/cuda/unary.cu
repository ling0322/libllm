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

#include <cuda_fp16.h>

#include "lynn/cuda/common.h"
#include "lynn/cuda/unary.h"

namespace ly {
namespace op {
namespace cuda {

__device__ constexpr float Sqrt2 = 1.4142136f;

template<typename T, UnaryOp OP>
__forceinline__ __device__ T applyUnaryOp(T x) {
  if constexpr (OP == UnaryOp::SQUARE) {
    return x * x;
  } else if constexpr (OP == UnaryOp::GELU && std::is_same_v<T, float>) {
    return x * 0.5f * (1.0f + erf(x / Sqrt2));
  } else if constexpr (OP == UnaryOp::GELU && std::is_same_v<T, half>) {
    return __float2half(float(x) * 0.5f * (1.0f + erf(float(x) / Sqrt2)));
  } else {
    __trap();
  }
}

template<typename scalar_t, UnaryOp OP>
__global__ void unaryContigKernel(
    const scalar_t *__restrict__ in,
    scalar_t *__restrict__ out,
    int numel) {
  int stride = (int64_t)blockDim.x * gridDim.x;
  int idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    out[idx] = applyUnaryOp<scalar_t, OP>(in[idx]);
  }
}

template<typename scalar_t, UnaryOp OP, int DIM>
__global__ void unaryGenericKernel(
    PackedTensorAccessor<const scalar_t, DIM> A,
    scalar_t *__restrict__ C,
    int numel) {
  int stride = (int64_t)blockDim.x * gridDim.x;
  int idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    scalar_t v = A.getElemByIndex(idx);
    C[idx] = applyUnaryOp<scalar_t, OP>(v);
  }
}

template<typename T, UnaryOp OP>
Tensor unaryImpl(const Tensor &tensor) {
  int64_t numel64 = tensor.getNumEl();
  CHECK(numel64 < std::numeric_limits<int>::max());
  int numel = static_cast<int>(numel64);

  int d = tensor.getDim();
  Tensor C = createCudaTensor<T>(tensor.getShape());
  T *dataC = C.getData<T>();

  constexpr int blockSize = 256;
  dim3 grid = getGrid1D(numel, blockSize);

  if (tensor.isContiguous()) {
    unaryContigKernel<T, OP><<<grid, blockSize>>>(tensor.getData<T>(), dataC, numel);
  } else {
    if (d == 1)
      unaryGenericKernel<T, OP, 1><<<grid, blockSize>>>(tensor, dataC, numel);
    else if (d == 2)
      unaryGenericKernel<T, OP, 2><<<grid, blockSize>>>(tensor, dataC, numel);
    else if (d == 3)
      unaryGenericKernel<T, OP, 3><<<grid, blockSize>>>(tensor, dataC, numel);
    else if (d == 4)
      unaryGenericKernel<T, OP, 4><<<grid, blockSize>>>(tensor, dataC, numel);
    else
      NOT_IMPL();
  }

  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor applyUnaryOp(UnaryOp op, const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  DType dtype = tensor.getDType();

  if (op == UnaryOp::GELU && dtype == DType::kFloat16)
    return unaryImpl<half, UnaryOp::GELU>(tensor);
  if (op == UnaryOp::SQUARE && dtype == DType::kFloat16)
    return unaryImpl<half, UnaryOp::SQUARE>(tensor);
  if (op == UnaryOp::GELU && dtype == DType::kFloat) return unaryImpl<float, UnaryOp::GELU>(tensor);
  if (op == UnaryOp::SQUARE && dtype == DType::kFloat)
    return unaryImpl<float, UnaryOp::SQUARE>(tensor);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
