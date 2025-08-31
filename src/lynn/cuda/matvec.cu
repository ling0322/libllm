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

#include "lynn/cuda/common.h"
#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cuda {

// 8 * word
union OWORD {
  uint4 vec;
  uint32_t u[4];
  half h[8];
};
static_assert(sizeof(OWORD) == 16, "invalid size of OWORD");

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

}  // namespace cuda
}  // namespace op
}  // namespace ly
