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

// Inspired by https://github.com/ankan-ban/llama_cu_awq and https://github.com/mit-han-lab/llm-awq

#include <cuda_fp16.h>

#include "libllm/cuda/common.h"
#include "libllm/tensor.h"

namespace libllm {
namespace op {
namespace cuda {

// 8 * word
union OWORD {
  uint4 vec;
  uint32_t u[4];
  half h[8];
};

int divUp(int a, int b) {
  return (a - 1) / b + 1;
}

// load with cache streaming.
__forceinline__ __device__ uint4 ldcsv4u32(const void *src) {
  return __ldcs((const uint4 *)src);
}

__forceinline__ __device__ uint4 ldgv4u32(const void *src) {
  return *((const uint4 *)src);
}

__device__ __forceinline__ float wrapReduceSum(float sum) {
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

__global__ void mat_vec_kernel(
    half *y,
    const half *__restrict__ x,
    const half *__restrict__ A,
    int n,
    int d,
    int lda) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= d) return;

  constexpr int Vec = 8;
  constexpr int WrapSize = 32;
  int startIdx = threadIdx.x * Vec;

  float sum = 0;
  for (int i = startIdx; i < n; i += Vec * WrapSize) {
    OWORD packA;
    OWORD packX;
    packA.vec = ldcsv4u32(&A[row * lda + i]);
    packX.vec = ldgv4u32(&x[i]);
#pragma unroll
    for (int j = 0; j < Vec; j++) sum += float(packA.h[j]) * float(packX.h[j]);
  }

  sum = wrapReduceSum(sum);
  if (threadIdx.x == 0) y[row] = (half)sum;
}

__global__ void
mat_vec_kernel_q4g32(half *y, const half *__restrict__ x, PackedSubtensor2DQInt4x32 A) {
  int numCol = A.getNumCol();
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= A.getNumRow()) return;

  constexpr int VecX = 8;
  constexpr int VecA = 4;
  constexpr int WrapSize = 32;

  int groupPerRow = numCol / QInt4x32::GroupSize;
  int rowGroupIdx = row * groupPerRow;

  float sum = 0;
  int groupIdx = threadIdx.x;
  OWORD packX;  // VecX * half
  for (int i = groupIdx; i < groupPerRow; i += WrapSize) {
    float scale = float(A.getScaleValue(rowGroupIdx + i));
    float zero = float(A.getZeroValue(rowGroupIdx + i));
    const uint32_t *data = reinterpret_cast<const uint32_t *>(A.getData(rowGroupIdx + i));

    // 32 elements: QInt4x32::GroupSize / (VecA * VecX) == 1
#pragma unroll
    for (int k = 0; k < QInt4x32::GroupSize / (VecA * VecX); ++k) {
      // 32 elements
#pragma unroll
      for (int j = 0; j < VecA; ++j) {
        // TODO: Memory Coalescing
        uint32_t packAv8 = data[j];
        packX.vec = ldgv4u32(&x[i * QInt4x32::GroupSize + k * (VecA * VecX) + j * VecX]);

        // 8 elements
#pragma unroll
        for (int el = 0; el < VecX; ++el) {
          sum += (scale * float(packAv8 & 0xf) - zero) * float(packX.h[el]);
          packAv8 = packAv8 >> 4;
        }
      }
    }
  }

  sum = wrapReduceSum(sum);
  if (threadIdx.x == 0) y[row] = (half)sum;
}

Tensor gemvHalf(const Tensor &A, const Tensor &B) {
  int n = A.getShape(1);
  int d = A.getShape(0);

  Tensor C = createCudaTensorHalf({d, 1});

  dim3 block_dim(32, 4);
  dim3 grid_dim(divUp(d, 4), 1);

  mat_vec_kernel<<<grid_dim, block_dim, 0>>>(
      C.getData<half>(),
      B.getData<half>(),
      A.getData<half>(),
      n,
      d,
      n);
  cudaDeviceSynchronize();
  return C;
}

Tensor gemvQ4(const Tensor &A, const Tensor &x) {
  CHECK(A.getShape(1) == x.getShape(0) && x.getShape(1) == 1);
  CHECK(x.getShape(0) % QInt4x32::GroupSize == 0);
  int n = A.getShape(1);
  int d = A.getShape(0);

  Tensor C = createCudaTensorHalf({d, 1});

  dim3 block_dim(32, 4);
  dim3 grid_dim(divUp(d, 4), 1);

  mat_vec_kernel_q4g32<<<grid_dim, block_dim, 0>>>(C.getData<half>(), x.getData<half>(), A);
  cudaDeviceSynchronize();
  return C;
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
