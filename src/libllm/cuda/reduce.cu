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

#include <assert.h>
#include <cuda_fp16.h>
#include <math.h>

#include <cub/block/block_reduce.cuh>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "libllm/cuda/common.h"
#include "libllm/cuda/reduce.h"

namespace libllm {
namespace op {
namespace cuda {

enum class MapType { EXP_FP16_FP32, SQUARE_FP16_FP32, IDENTITY, UNKNOWN };
enum class ReduceType { SUM, MAX, UNKNOWN };

__device__ __forceinline__ constexpr MapType getMapType(MapReduceType mapReduceType) {
  switch (mapReduceType) {
    case MapReduceType::SUM_EXP_FP16_FP32:
      return MapType::EXP_FP16_FP32;
    case MapReduceType::SUM_SQUARE_FP16_FP32:
      return MapType::SQUARE_FP16_FP32;
    case MapReduceType::MAX:
      return MapType::IDENTITY;
    default:
      return MapType::UNKNOWN;
  }
}

__device__ __forceinline__ constexpr ReduceType getReduceType(MapReduceType mapReduceType) {
  switch (mapReduceType) {
    case MapReduceType::SUM_EXP_FP16_FP32:
    case MapReduceType::SUM_SQUARE_FP16_FP32:
      return ReduceType::SUM;
    case MapReduceType::MAX:
      return ReduceType::MAX;
    default:
      return ReduceType::UNKNOWN;
  }
}

template<typename T, ReduceType REDUCE_TYPE>
__device__ __forceinline__ T getReduceInitial() {
  switch (REDUCE_TYPE) {
    case ReduceType::SUM:
      return T(0);
    case ReduceType::MAX:
      return -::cuda::std::numeric_limits<float>::infinity();
    default:
      __trap();
  }
}

// Wrap-level reduce
template<typename TIn, typename TOut, MapReduceType REDUCE_TYPE, int BLOCK_DIM>
__global__ void reduce0Kernel3D(
    PackedSubtensor<const TIn, 3> input,
    PackedSubtensor<TOut, 2> output) {
  assert(blockDim.x == BLOCK_DIM);
  assert(input.getShape(0) == output.getShape(0));

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  TOut elemReduce = getReduceInitial<TOut, getReduceType(REDUCE_TYPE)>();
  for (int offset = 0; offset < input.getShape(2); offset += BLOCK_DIM) {
    int idx = offset + threadIdx.x;
    if (idx < input.getShape(2)) {
      // element-wise map.
      TIn elemIn = input[z][y][idx];
      TOut elemMap;
      switch (getMapType(REDUCE_TYPE)) {
        case MapType::EXP_FP16_FP32:
          elemMap = expf(static_cast<float>(elemIn));
          break;
        case MapType::SQUARE_FP16_FP32:
          elemMap = static_cast<float>(elemIn) * static_cast<float>(elemIn);
          break;
        case MapType::IDENTITY:
          elemMap = elemIn;
          break;
        default:
          __trap();
      }

      // element-wise reduce.
      switch (getReduceType(REDUCE_TYPE)) {
        case ReduceType::SUM:
          elemReduce += elemMap;
          break;
        case ReduceType::MAX:
          elemReduce = cub::Max()(elemReduce, elemMap);
          break;
        default:
          __trap();
      }
    }
  }

  using BlockReduce = cub::BlockReduce<TOut, BLOCK_DIM>;
  using TempStorage = typename BlockReduce::TempStorage;

  __shared__ TempStorage tempStorage;
  BlockReduce blockReduce{tempStorage};

  // block-level reduce.
  switch (getReduceType(REDUCE_TYPE)) {
    case ReduceType::SUM:
      elemReduce = blockReduce.Sum(elemReduce);
      break;
    case ReduceType::MAX:
      elemReduce = blockReduce.Reduce(elemReduce, cub::Max());
      break;
    default:
      assert(false);
  }

  if (threadIdx.x == 0) output[z][y] = elemReduce;
}

Tensor reduceHalfToSingle3D(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 3);

  std::vector<int> shape = A.getShape();
  shape.pop_back();

  Tensor C = createCudaTensorFloat(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = A.getShape(0);
  d.y = A.getShape(1);
  d.x = 1;

  switch (reduceType) {
    case MapReduceType::SUM_EXP_FP16_FP32:
      reduce0Kernel3D<half, float, MapReduceType::SUM_EXP_FP16_FP32, blockSize>
          <<<d, blockSize>>>(A, C);
      break;
    case MapReduceType::SUM_SQUARE_FP16_FP32:
      reduce0Kernel3D<half, float, MapReduceType::SUM_SQUARE_FP16_FP32, blockSize>
          <<<d, blockSize>>>(A, C);
      break;
    case MapReduceType::MAX:
      reduce0Kernel3D<half, float, MapReduceType::MAX, blockSize><<<d, blockSize>>>(A, C);
      break;
    default:
      NOT_IMPL();
  }
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor reduceHalf3D(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 3);

  std::vector<int> shape = A.getShape();
  shape.pop_back();

  Tensor C = createCudaTensorHalf(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = A.getShape(0);
  d.y = A.getShape(1);
  d.x = 1;

  switch (reduceType) {
    case MapReduceType::MAX:
      reduce0Kernel3D<half, half, MapReduceType::MAX, blockSize><<<d, blockSize>>>(A, C);
      break;
    default:
      NOT_IMPL();
  }
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor reduceHalfToSingle(Tensor A, MapReduceType reduceType) {
  if (A.getDim() == 3) return reduceHalfToSingle3D(A, reduceType);

  NOT_IMPL();
}

Tensor reduce(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);

  if (reduceType == MapReduceType::SUM_EXP_FP16_FP32) return reduceHalfToSingle(A, reduceType);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
