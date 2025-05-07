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

#include "libllm/cpu/common.h"
#include "libllm/cuda/binary_op.h"
#include "libllm/cuda/common.h"
#include "libllm/tensor.h"
#include "lutil/span.h"

namespace libllm {
namespace op {
namespace cuda {

template<typename T, BinaryOp OP>
__forceinline__ __device__ T applyOp(T a, T b);

template<>
__forceinline__ __device__ half applyOp<half, BinaryOp::ADD>(half a, half b) {
  return a + b;
}
template<>
__forceinline__ __device__ half applyOp<half, BinaryOp::SUB>(half a, half b) {
  return a - b;
}
template<>
__forceinline__ __device__ half applyOp<half, BinaryOp::MUL>(half a, half b) {
  return a * b;
}

template<typename T, BinaryOp OP>
__global__ void binaryOpKernel2D(
    PackedSubtensor<const half, 2> A,
    PackedSubtensor<const half, 2> B,
    PackedSubtensor<half, 2> C) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < A.getShape(1) && y < A.getShape(0)) {
    C[y][x] = applyOp<T, OP>(A[y][x], B[y][x]);
  }
}

template<typename T, BinaryOp OP>
__global__ void binaryOpKernel3D(
    PackedSubtensor<const half, 3> A,
    PackedSubtensor<const half, 3> B,
    PackedSubtensor<half, 3> C) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < A.getShape(2) && y < A.getShape(1) && z < A.getShape(0)) {
    C[z][y][x] = applyOp<T, OP>(A[z][y][x], B[z][y][x]);
  }
}

template<typename T, BinaryOp OP>
__global__ void binaryOpKernel4D(
    PackedSubtensor<const half, 4> A,
    PackedSubtensor<const half, 4> B,
    PackedSubtensor<half, 4> C) {
  int d3 = blockIdx.x * blockDim.x + threadIdx.x;

  dim3 dz = splitIndexToDim3(blockIdx.y * blockDim.y + threadIdx.y, A.getSize());
  int d2 = dz.x;
  int d1 = dz.y;
  int d0 = dz.z;

  if (d3 < A.getShape(3) && d2 < A.getShape(2) && d1 < A.getShape(1) && d0 < A.getShape(0)) {
    C[d0][d1][d2][d3] = applyOp<T, OP>(A[d0][d1][d2][d3], B[d0][d1][d2][d3]);
  }
}

Tensor broadcastTensor(const Tensor &a, lut::Span<const Tensor::ShapeType> targetShape) {
  Tensor x = op::cpu::expandBatchDims(a, targetShape);
  return x.expand(targetShape);
}

Tensor binaryOpHalf4D(const Tensor &A, const Tensor &B, BinaryOp op) {
  Tensor xB = broadcastTensor(B, A.getShape());
  Tensor C = createCudaTensorHalf(A.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.y = A.getShape(0) * A.getShape(1) * A.getShape(2);
  d.x = (A.getShape(3) + blockSize - 1) / blockSize;

  if (op == BinaryOp::ADD) binaryOpKernel4D<half, BinaryOp::ADD><<<d, blockSize>>>(A, xB, C);
  if (op == BinaryOp::SUB) binaryOpKernel4D<half, BinaryOp::SUB><<<d, blockSize>>>(A, xB, C);
  if (op == BinaryOp::MUL) binaryOpKernel4D<half, BinaryOp::MUL><<<d, blockSize>>>(A, xB, C);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor binaryOpHalf3D(const Tensor &A, const Tensor &B, BinaryOp op) {
  Tensor xB = broadcastTensor(B, A.getShape());
  Tensor C = createCudaTensorHalf(A.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.z = A.getShape(0);
  d.y = A.getShape(1);
  d.x = (A.getShape(2) + blockSize - 1) / blockSize;

  if (op == BinaryOp::ADD) binaryOpKernel3D<half, BinaryOp::ADD><<<d, blockSize>>>(A, xB, C);
  if (op == BinaryOp::SUB) binaryOpKernel3D<half, BinaryOp::SUB><<<d, blockSize>>>(A, xB, C);
  if (op == BinaryOp::MUL) binaryOpKernel3D<half, BinaryOp::MUL><<<d, blockSize>>>(A, xB, C);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor binaryOpHalf2D(const Tensor &A, const Tensor &B, BinaryOp op) {
  Tensor xB = broadcastTensor(B, A.getShape());
  Tensor C = createCudaTensorHalf(A.getShape());

  constexpr int blockSize = 256;
  dim3 d;
  d.y = A.getShape(0);
  d.x = (A.getShape(1) + blockSize - 1) / blockSize;

  if (op == BinaryOp::ADD) binaryOpKernel2D<half, BinaryOp::ADD><<<d, blockSize>>>(A, xB, C);
  if (op == BinaryOp::SUB) binaryOpKernel2D<half, BinaryOp::SUB><<<d, blockSize>>>(A, xB, C);
  if (op == BinaryOp::MUL) binaryOpKernel2D<half, BinaryOp::MUL><<<d, blockSize>>>(A, xB, C);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor binaryOpHalf(const Tensor &A, const Tensor &B, BinaryOp op) {
  CHECK(A.getDType() == DType::kFloat16 && B.getDType() == DType::kFloat16);

  if (A.getDim() == 2) return binaryOpHalf2D(A, B, op);
  if (A.getDim() == 3) return binaryOpHalf3D(A, B, op);
  if (A.getDim() == 4) return binaryOpHalf4D(A, B, op);

  NOT_IMPL();
}

// apply C <- BinaryOp(A, B)
Tensor binaryOp(const Tensor &A, const Tensor &B, BinaryOp op) {
  if (A.getDType() == DType::kFloat16) return binaryOpHalf(A, B, op);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
