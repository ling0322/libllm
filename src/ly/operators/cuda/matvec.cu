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
#include "ly/tensor.h"
#include "ly/operators/cuda/common.h"

namespace ly {
namespace op {
namespace cuda {

int divUp(int a, int b) {
    return (a - 1) / b + 1;
} 

// load with cache streaming.
__forceinline__ __device__ void load16byteCS(const void *src, void *dest) {
  uint4 d = __ldcs((const uint4 *)src);
  *((uint4 *)dest) = d;
}

template<typename T>
__forceinline__ __device__ void load16byte(const void *src, T *dest) {
  static_assert(sizeof(T) == 16, "T is not a 16 byte struct");
  uint4 d = *((const uint4 *)src);
  *((uint4 *)dest) = d;
}

__device__ __forceinline__ float wrapReduceSum(float sum) {
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

__global__ void mat_vec_kernel(half* y, const half *__restrict__ x, const half *__restrict__ A, int n, int d, int lda) {
  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= d)
    return;

  constexpr int Vec = 8;
  constexpr int WrapSize = 32;
  int startIdx = threadIdx.x * Vec;

  float sum = 0;
  for (int i = startIdx; i < n; i += Vec * WrapSize) {
    half packA[Vec];
    half packX[Vec];
    load16byteCS(&A[row * lda + i], &packA);
    load16byte<half[Vec]>(&x[i], &packX);

    #pragma unroll
    for (int j = 0; j < Vec; j++)
      sum += float(packA[j]) * float(packX[j]);
  }

  sum = wrapReduceSum(sum);
  if (threadIdx.x == 0)
    y[row] = (half)sum;
}

__global__ void mat_vec_kernel_q4g32(half* y, const half *__restrict__ x, PackedSubtensor2DQ4 A) {
  int numCol = A.getNumCol();
  int lda = A.getNumCol();

  int row = blockIdx.x * blockDim.y + threadIdx.y;
  if (row >= A.getNumRow()) return;

  constexpr int VecX = 8;
  constexpr int VecA = 16;
  constexpr int WrapSize = 32;

  int groupPerRow = numCol / Q4::GroupSize;
  constexpr int bytesPerGroup = Q4::GroupSize / 2;
  int rowGroupIdx = row * groupPerRow;
  const uint8_t *__restrict__ pdata = A.getData(row * groupPerRow);

  float sum = 0;
  int groupIdx = threadIdx.x;
  for (int i = groupIdx; i < groupPerRow; i += WrapSize) {
    float scale = float(A.getScaleValue(rowGroupIdx + i));
    float qzero = float(A.getZeroValue(rowGroupIdx + i));

    // 128 elements
    #pragma unroll
    for (int k = 0; k < 1; ++k) {
      uint32_t packA[VecA];
      load16byteCS(&pdata[i * bytesPerGroup], &packA[0]);
      load16byteCS(&pdata[i * bytesPerGroup + 16], &packA[4]);
      load16byteCS(&pdata[i * bytesPerGroup + 32], &packA[8]);
      load16byteCS(&pdata[i * bytesPerGroup + 48], &packA[12]);

      // 32 elements
      #pragma unroll
      for (int j = 0; j < VecA; ++j) {
        uint32_t packAv8 = packA[j];
        half packX[VecX];
        load16byte<half[VecX]>(&x[i * Q4::GroupSize + k * (VecA * VecX) + j * VecX], &packX);

        // 8 elements
        #pragma unroll
        for (int el = 0; el < VecX; ++el) {
          sum += scale * (float(packAv8 & 0xf) - qzero) * float(packX[el]);
          packAv8 = packAv8 >> 4;
        }
      }
    }
  }

  sum = wrapReduceSum(sum);
  if (threadIdx.x == 0)
    y[row] = (half)sum;
}

Tensor gemvHalf(const Tensor &A, const Tensor &B) {
  int n = A.getShape(1);
  int d = A.getShape(0);

  Tensor C = createCudaTensorHalf({d, 1});

  dim3 block_dim(32, 4);
  dim3 grid_dim(divUp(d, 4), 1);

  mat_vec_kernel <<<grid_dim, block_dim, 0 >>> (C.getData<half>(), B.getData<half>(), A.getData<half>(), n, d, n);
  cudaDeviceSynchronize();
  return C;
}

Tensor gemvQ4(const Tensor &A, const Tensor &x) {
  CHECK(A.getShape(1) == x.getShape(0) && x.getShape(1) == 1);
  CHECK(x.getShape(0) % Q4::GroupSize == 0);
  int n = A.getShape(1);
  int d = A.getShape(0);

  Tensor C = createCudaTensorHalf({d, 1});

  dim3 block_dim(32, 4);
  dim3 grid_dim(divUp(d, 4), 1);

  mat_vec_kernel_q4g32 <<<grid_dim, block_dim, 0 >>> (C.getData<half>(), x.getData<half>(), A);
  cudaDeviceSynchronize();
  return C;
}

}  // cuda
}  // op
}  // ly
