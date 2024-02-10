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

#include "libllm/cuda/reduce.h"

#include <cuda_fp16.h>
#include <math.h>
#include "libllm/cuda/common.h"

namespace libllm {
namespace op {
namespace cuda {

template<typename TIn, typename TOut, ReduceType REDUCE_TYPE>
__device__ __forceinline__ TOut mapper(TIn x);

template<>
__device__ __forceinline__ float mapper<half, float, ReduceType::SUM_EXP_FP16_FP32>(half x) {
  return expf(static_cast<float>(x));
}

template<>
__device__ __forceinline__ float mapper<half, float, ReduceType::SUM_SQUARE_FP16_FP32>(half x) {
  float xs = static_cast<float>(x);
  return xs * xs;
}

template<typename T>
__device__ __forceinline__ T wrapReduceSum(T sum) {
  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);
  return sum;
}

template<typename T>
__device__ __forceinline__ float blockReduceSum256(volatile T *sdata) {
  int tid = threadIdx.x;
  
  if (tid < 128) sdata[tid] += sdata[tid + 128];
  if (tid < 64) sdata[tid] += sdata[tid + 64];
  if (tid < 32) sdata[tid] += sdata[tid + 32];

  return wrapReduceSum<T>(sdata[tid]);
}

// Wrap-level reduce
template<typename TIn, typename TOut, ReduceType REDUCE_TYPE>
__global__ void reduce0Kernel3D(PackedSubtensor<const TIn, 3> input,
                                PackedSubtensor<TOut, 2> output) {
  assert(blockDim.x == 32);
  assert(input.getShape(0) == output.getShape(0));

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  TOut sum = 0;
  for (int offset = 0; offset < input.getShape(2); offset += 32) {
    int idx = offset + threadIdx.x;
    if (idx < input.getShape(2)) {
      sum += mapper<TIn, TOut, REDUCE_TYPE>(input[z][y][idx]);
    }
  }

  sum = wrapReduceSum(sum);
  if (threadIdx.x == 0) output[z][y] = sum;
}

Tensor reduceHalfToSingle3D(Tensor A, ReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 3);

  std::vector<int> shape = A.getShape();
  shape.pop_back();

  Tensor C = createCudaTensorFloat(shape);

  constexpr int blockSize = 32;
  dim3 d;
  d.z = A.getShape(0);
  d.y = A.getShape(1);
  d.x = (A.getShape(2) + blockSize - 1) / blockSize;

  switch (reduceType) {
    case ReduceType::SUM_EXP_FP16_FP32:
      reduce0Kernel3D<half, float, ReduceType::SUM_EXP_FP16_FP32><<<d, blockSize>>>(A, C);
      break;
    case ReduceType::SUM_SQUARE_FP16_FP32:
      reduce0Kernel3D<half, float, ReduceType::SUM_SQUARE_FP16_FP32><<<d, blockSize>>>(A, C);
      break;
    default:
      NOT_IMPL();
  }
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor reduceHalfToSingle(Tensor A, ReduceType reduceType) {
  if (A.getDim() == 3) return reduceHalfToSingle3D(A, reduceType);
  
  NOT_IMPL();
}

Tensor reduce(Tensor A, ReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);

  if (reduceType == ReduceType::SUM_EXP_FP16_FP32) return reduceHalfToSingle(A, reduceType);

  NOT_IMPL();
}


}  // cuda
}  // op
}  // ly
