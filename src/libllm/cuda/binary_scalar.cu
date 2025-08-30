// The MIT License (MIT)
//
// Copyright (c) 2024-2025 Xiaoyang Chen
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

#include "libllm/cuda/binary_scalar.h"
#include "libllm/cuda/common.h"

namespace libllm {
namespace op {
namespace cuda {

template<typename T, BinaryScalarOp OP>
__forceinline__ __device__ T applyBinaryScalarOp(T a, T b) {
  if constexpr (OP == BinaryScalarOp::ADD) {
    return a + b;
  } else if constexpr (OP == BinaryScalarOp::SUB) {
    return a - b;
  } else if constexpr (OP == BinaryScalarOp::MUL) {
    return a * b;
  } else if constexpr (OP == BinaryScalarOp::DIV) {
    return a / b;
  } else if constexpr (OP == BinaryScalarOp::MOD) {
    return a % b;
  } else {
    __trap();
  }
}

template<typename T, BinaryScalarOp OP>
__global__ void binaryScalarContigKernel(
    const T *__restrict__ in,
    T rhs,
    T *__restrict__ out,
    int numel) {
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    out[idx] = applyBinaryScalarOp<T, OP>(in[idx], rhs);
  }
}

template<typename T, BinaryScalarOp OP, int DIM>
__global__ void binaryScalarGenericKernel(
    PackedTensorAccessor<const T, DIM> A,
    T rhs,
    T *__restrict__ C,
    int numel) {
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    T v = A.getElemByIndex(idx);
    C[idx] = applyBinaryScalarOp<T, OP>(v, rhs);
  }
}

template<typename T, BinaryScalarOp OP>
Tensor binaryScalarImpl(const Tensor &tensor, T rhs) {
  int64_t numel64 = tensor.getNumEl();
  CHECK(numel64 < std::numeric_limits<int>::max());
  int numel = static_cast<int>(numel64);

  int d = tensor.getDim();
  Tensor C = createCudaTensor<T>(tensor.getShape());
  T *dataC = C.getData<T>();

  constexpr int blockSize = 256;
  dim3 grid = getGrid1D(numel, blockSize);

  if (tensor.isContiguous()) {
    binaryScalarContigKernel<T, OP><<<grid, blockSize>>>(tensor.getData<T>(), rhs, dataC, numel);
  } else {
    if (d == 1)
      binaryScalarGenericKernel<T, OP, 1><<<grid, blockSize>>>(tensor, rhs, dataC, numel);
    else if (d == 2)
      binaryScalarGenericKernel<T, OP, 2><<<grid, blockSize>>>(tensor, rhs, dataC, numel);
    else if (d == 3)
      binaryScalarGenericKernel<T, OP, 3><<<grid, blockSize>>>(tensor, rhs, dataC, numel);
    else if (d == 4)
      binaryScalarGenericKernel<T, OP, 4><<<grid, blockSize>>>(tensor, rhs, dataC, numel);
    else
      NOT_IMPL();
  }

  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor applyBinaryScalarOp(BinaryScalarOp op, const Tensor &tensor, float rhs) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  DType dtype = tensor.getDType();

  if (op == BinaryScalarOp::ADD && dtype == DType::kFloat16)
    return binaryScalarImpl<half, BinaryScalarOp::ADD>(tensor, __float2half(rhs));
  if (op == BinaryScalarOp::SUB && dtype == DType::kFloat16)
    return binaryScalarImpl<half, BinaryScalarOp::SUB>(tensor, __float2half(rhs));
  if (op == BinaryScalarOp::MUL && dtype == DType::kFloat16)
    return binaryScalarImpl<half, BinaryScalarOp::MUL>(tensor, __float2half(rhs));
  if (op == BinaryScalarOp::DIV && dtype == DType::kFloat16)
    return binaryScalarImpl<half, BinaryScalarOp::DIV>(tensor, __float2half(rhs));
  if (op == BinaryScalarOp::ADD && dtype == DType::kFloat)
    return binaryScalarImpl<float, BinaryScalarOp::ADD>(tensor, rhs);
  if (op == BinaryScalarOp::SUB && dtype == DType::kFloat)
    return binaryScalarImpl<float, BinaryScalarOp::SUB>(tensor, rhs);
  if (op == BinaryScalarOp::MUL && dtype == DType::kFloat)
    return binaryScalarImpl<float, BinaryScalarOp::MUL>(tensor, rhs);
  if (op == BinaryScalarOp::DIV && dtype == DType::kFloat)
    return binaryScalarImpl<float, BinaryScalarOp::DIV>(tensor, rhs);

  NOT_IMPL();
}

Tensor applyBinaryScalarOpLong(BinaryScalarOp op, const Tensor &tensor, LongType rhs) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  DType dtype = tensor.getDType();

  if (op == BinaryScalarOp::MOD && dtype == DType::kLong)
    return binaryScalarImpl<LongType, BinaryScalarOp::MOD>(tensor, rhs);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
