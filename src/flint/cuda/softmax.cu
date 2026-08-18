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
#include <cub/block/block_reduce.cuh>
#include <math.h>

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/functional.h"

namespace fl {
namespace op {
namespace cuda {

struct MaxFloat {
  __device__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

template<int BLOCK_SIZE, bool VECTORIZED>
__global__ void softmaxFusedKernel(
    const half *__restrict__ input,
    half *__restrict__ output,
    int width) {
  int rowOffset = blockIdx.x * width;
  float threadMax = -INFINITY;

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    int width2 = width / 2;
    for (int i = threadIdx.x; i < width2; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      threadMax = fmaxf(threadMax, fmaxf(value.x, value.y));
    }
  } else {
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      threadMax = fmaxf(threadMax, __half2float(input[rowOffset + i]));
    }
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float rowMax;
  __shared__ float invSum;
  threadMax = BlockReduce(tempStorage).Reduce(threadMax, MaxFloat{});
  if (threadIdx.x == 0) rowMax = threadMax;
  __syncthreads();

  float threadSum = 0.0f;
  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    int width2 = width / 2;
    for (int i = threadIdx.x; i < width2; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      threadSum += expf(value.x - rowMax) + expf(value.y - rowMax);
    }
  } else {
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      threadSum += expf(__half2float(input[rowOffset + i]) - rowMax);
    }
  }

  threadSum = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) invSum = 1.0f / threadSum;
  __syncthreads();

  if constexpr (VECTORIZED) {
    const half2 *input2 = reinterpret_cast<const half2 *>(input + rowOffset);
    half2 *output2 = reinterpret_cast<half2 *>(output + rowOffset);
    int width2 = width / 2;
    for (int i = threadIdx.x; i < width2; i += BLOCK_SIZE) {
      float2 value = __half22float2(input2[i]);
      output2[i] = __floats2half2_rn(
          expf(value.x - rowMax) * invSum,
          expf(value.y - rowMax) * invSum);
    }
  } else {
    for (int i = threadIdx.x; i < width; i += BLOCK_SIZE) {
      output[rowOffset + i] =
          __float2half(expf(__half2float(input[rowOffset + i]) - rowMax) * invSum);
    }
  }
}

template<int BLOCK_SIZE>
__global__ void softmaxStridedKernel(
    PackedTensorAccessor<const half, 3> input,
    PackedTensorAccessor<half, 3> output) {
  int width = input.getShape(2);
  int y = blockIdx.x % input.getShape(1);
  int z = blockIdx.x / input.getShape(1);

  float threadMax = -INFINITY;
  for (int x = threadIdx.x; x < width; x += BLOCK_SIZE) {
    threadMax = fmaxf(threadMax, __half2float(input[z][y][x]));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ float rowMax;
  __shared__ float invSum;
  threadMax = BlockReduce(tempStorage).Reduce(threadMax, MaxFloat{});
  if (threadIdx.x == 0) rowMax = threadMax;
  __syncthreads();

  float threadSum = 0.0f;
  for (int x = threadIdx.x; x < width; x += BLOCK_SIZE) {
    threadSum += expf(__half2float(input[z][y][x]) - rowMax);
  }

  threadSum = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) invSum = 1.0f / threadSum;
  __syncthreads();

  for (int x = threadIdx.x; x < width; x += BLOCK_SIZE) {
    output[z][y][x] =
        __float2half(expf(__half2float(input[z][y][x]) - rowMax) * invSum);
  }
}

Tensor softmaxHalfStrided3D(Tensor A) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 3);

  Tensor C = createCudaTensorHalf(A.getShape());

  constexpr int blockSize = 256;
  int rows = A.getShape(0) * A.getShape(1);
  softmaxStridedKernel<blockSize><<<rows, blockSize>>>(A, C);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor softmaxHalf1D(Tensor A) {
  Tensor xA = A.view({1, 1, A.getShape(0)});
  Tensor C = softmaxHalfStrided3D(xA);

  return C.view({C.getShape(2)});
}

Tensor softmaxHalf2D(Tensor A) {
  Tensor xA = A.view({1, A.getShape(0), A.getShape(1)});
  Tensor C = softmaxHalfStrided3D(xA);

  return C.view({C.getShape(1), C.getShape(2)});
}

Tensor softmaxHalf4D(Tensor A) {
  std::vector<int> shape = A.getShape();

  Tensor xA = A.view({-1, A.getShape(2), A.getShape(3)});
  Tensor C = softmaxHalfStrided3D(xA);

  return C.view(shape);
}

Tensor softmaxHalfContiguous(Tensor A) {
  int width = A.getShape(-1);
  int64_t numel = A.getNumEl();
  CHECK(numel < std::numeric_limits<int>::max());
  int rows = static_cast<int>(numel / width);

  Tensor C = createCudaTensorHalf(A.getShape());
  const half *input = getDataPtrCuda<half>(A);
  half *output = getDataPtrCuda<half>(C);

  constexpr int blockSize = 256;
  bool useHalf2 = width % 2 == 0 &&
                  reinterpret_cast<uintptr_t>(input) % alignof(half2) == 0 &&
                  reinterpret_cast<uintptr_t>(output) % alignof(half2) == 0;
  if (useHalf2) {
    softmaxFusedKernel<blockSize, true><<<rows, blockSize>>>(input, output, width);
  } else {
    softmaxFusedKernel<blockSize, false><<<rows, blockSize>>>(input, output, width);
  }

  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return C;
}

Tensor softmaxHalf(Tensor A) {
  if (A.isContiguous()) return softmaxHalfContiguous(A);
  if (A.getDim() == 1) return softmaxHalf1D(A);
  if (A.getDim() == 2) return softmaxHalf2D(A);
  if (A.getDim() == 3) return softmaxHalfStrided3D(A);
  if (A.getDim() == 4) return softmaxHalf4D(A);

  NOT_IMPL();
}

Tensor softmax(Tensor A) {
  if (A.getDType() == DType::kFloat16) return softmaxHalf(A);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
