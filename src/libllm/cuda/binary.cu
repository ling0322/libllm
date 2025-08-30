// The MIT License (MIT)
//
// Copyright (c) 2023-2025 Xiaoyang Chen
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

#include "libllm/cpu/common.h"
#include "libllm/cuda/binary.h"
#include "libllm/cuda/common.h"
#include "libllm/tensor.h"
#include "lutil/span.h"

namespace libllm {
namespace op {
namespace cuda {

template<typename TIn, typename TOut, BinaryOp OP>
__forceinline__ __device__ TOut applyBinaryOp(TIn a, TIn b) {
  if constexpr (OP == BinaryOp::ADD) {
    return a + b;
  } else if constexpr (OP == BinaryOp::SUB) {
    return a - b;
  } else if constexpr (OP == BinaryOp::MUL) {
    return a * b;
  } else if constexpr (OP == BinaryOp::EQUAL) {
    return a == b;
  } else {
    __trap();
  }
}

template<typename TIn, typename TOut, BinaryOp OP>
__global__ void binaryContigKernel(
    const TIn *__restrict__ A,
    const TIn *__restrict__ B,
    TOut *__restrict__ C,
    int numel) {
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    C[idx] = applyBinaryOp<TIn, TOut, OP>(A[idx], B[idx]);
  }
}

template<typename TIn, typename TOut, BinaryOp OP, int DIM>
__global__ void binaryGenericKernel(
    PackedTensorAccessor<const TIn, DIM> A,
    PackedTensorAccessor<const TIn, DIM> B,
    TOut *__restrict__ C,
    int numel) {
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    int offsetA = 0;
    int offsetB = 0;
    int offsetC = idx;
#pragma unroll
    for (int d = DIM - 1; d >= 0; --d) {
      offsetA += (idx % A.getShape(d)) * A.getStride(d);
      offsetB += (idx % A.getShape(d)) * B.getStride(d);
      idx /= A.getShape(d);
    }
    C[offsetC] = applyBinaryOp<TIn, TOut, OP>(A.getData()[offsetA], B.getData()[offsetB]);
  }
}

Tensor broadcastTensor(const Tensor &a, lut::Span<const Tensor::ShapeType> targetShape) {
  Tensor x = op::cpu::expandBatchDims(a, targetShape);
  return x.expand(targetShape);
}

template<typename TIn, typename TOut, BinaryOp OP>
Tensor binaryImpl(const Tensor &A, const Tensor &B) {
  Tensor xB = broadcastTensor(B, A.getShape());
  CHECK(A.getDType() == DType::getType<TIn>() && xB.getDType() == DType::getType<TIn>());
  xB.throwIfInvalidShape(A.getShape(), "B");

  int64_t numel64 = A.getNumEl();
  CHECK(numel64 < std::numeric_limits<int>::max());
  int numel = static_cast<int>(numel64);

  int d = A.getDim();
  Tensor C = createCudaTensor<TOut>(A.getShape());
  const TIn *pA = A.getData<TIn>();
  const TIn *pB = B.getData<TIn>();
  TOut *pC = C.getData<TOut>();

  constexpr int blockSize = 256;
  dim3 grid = getGrid1D(numel, blockSize);

  if (A.isContiguous() && B.isContiguous()) {
    binaryContigKernel<TIn, TOut, OP><<<grid, blockSize>>>(pA, pB, pC, numel);
  } else {
    if (d == 1)
      binaryGenericKernel<TIn, TOut, OP, 1><<<grid, blockSize>>>(A, xB, pC, numel);
    else if (d == 2)
      binaryGenericKernel<TIn, TOut, OP, 2><<<grid, blockSize>>>(A, xB, pC, numel);
    else if (d == 3)
      binaryGenericKernel<TIn, TOut, OP, 3><<<grid, blockSize>>>(A, xB, pC, numel);
    else if (d == 4)
      binaryGenericKernel<TIn, TOut, OP, 4><<<grid, blockSize>>>(A, xB, pC, numel);
    else
      NOT_IMPL();
  }

  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor applyBinaryOp(BinaryOp op, const Tensor &A, const Tensor &B) {
  CHECK(A.getDevice().getType() == Device::kCuda && B.getDevice().getType() == Device::kCuda);
  DType dtype = A.getDType();

  if (op == BinaryOp::ADD && dtype == DType::kFloat16)
    return binaryImpl<half, half, BinaryOp::ADD>(A, B);
  if (op == BinaryOp::SUB && dtype == DType::kFloat16)
    return binaryImpl<half, half, BinaryOp::SUB>(A, B);
  if (op == BinaryOp::MUL && dtype == DType::kFloat16)
    return binaryImpl<half, half, BinaryOp::MUL>(A, B);
  if (op == BinaryOp::ADD && dtype == DType::kFloat)
    return binaryImpl<float, float, BinaryOp::ADD>(A, B);
  if (op == BinaryOp::SUB && dtype == DType::kFloat)
    return binaryImpl<float, float, BinaryOp::SUB>(A, B);
  if (op == BinaryOp::MUL && dtype == DType::kFloat)
    return binaryImpl<float, float, BinaryOp::MUL>(A, B);
  if (op == BinaryOp::EQUAL && dtype == DType::kUInt8)
    return binaryImpl<UInt8, BoolType, BinaryOp::EQUAL>(A, B);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
