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
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <cub/cub.cuh>
#include <cuda/std/cmath>
#include <cuda/std/limits>

#include "lynn/cuda/common.h"
#include "lynn/cuda/reduce.h"

namespace ly {
namespace op {
namespace cuda {

template<MapReduceType MR_TYPE, typename TIn, typename TOut>
struct MapOp {
  __device__ TOut operator()(const TIn &x) const {
    if constexpr (
        MR_TYPE == MapReduceType::SUM || MR_TYPE == MapReduceType::MAX ||
        MR_TYPE == MapReduceType::ALL) {
      return TOut(x);
    } else if constexpr (MR_TYPE == MapReduceType::SUM_EXP) {
      return expf(TOut(x));
    } else if constexpr (MR_TYPE == MapReduceType::SUM_SQUARE) {
      return TOut(x) * TOut(x);
    } else {
      __trap();
    }
  }
};

template<MapReduceType MR_TYPE, typename T>
struct ReduceOp {
  __device__ T operator()(const T &a, const T &b) const {
    if constexpr (
        MR_TYPE == MapReduceType::SUM || MR_TYPE == MapReduceType::SUM_EXP ||
        MR_TYPE == MapReduceType::SUM_SQUARE) {
      return a + b;
    } else if constexpr (MR_TYPE == MapReduceType::ALL) {
      return a && b;
    } else if constexpr (MR_TYPE == MapReduceType::MAX) {
      return std::max(a, b);
    } else {
      __trap();
    }
  }
};

template<typename T, MapReduceType MR_TYPE>
__device__ __host__ T getReduceInitial() {
  if constexpr (
      MR_TYPE == MapReduceType::SUM || MR_TYPE == MapReduceType::SUM_EXP ||
      MR_TYPE == MapReduceType::SUM_SQUARE) {
    return T(0);
  } else if constexpr (MR_TYPE == MapReduceType::MAX) {
    return -::cuda::std::numeric_limits<float>::infinity();
  } else if constexpr (MR_TYPE == MapReduceType::ALL) {
    return true;
  } else {
#ifdef __CUDA_ARCH__
    __trap();
#else
    NOT_IMPL();
#endif
  }
}

// Wrap-level reduce
template<typename TIn, typename TOut, MapReduceType REDUCE_TYPE, int BLOCK_DIM>
__global__ void reduce0Kernel3D(
    PackedTensorAccessor<const TIn, 3> input,
    PackedTensorAccessor<TOut, 2> output) {
  assert(blockDim.x == BLOCK_DIM);
  assert(input.getShape(0) == output.getShape(0));
  MapOp<REDUCE_TYPE, TIn, TOut> mapOp;
  ReduceOp<REDUCE_TYPE, TOut> reduceOp;

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  TOut elemReduce = getReduceInitial<TOut, REDUCE_TYPE>();
  for (int offset = 0; offset < input.getShape(2); offset += BLOCK_DIM) {
    int idx = offset + threadIdx.x;
    if (idx < input.getShape(2)) {
      // element-wise map.
      TIn elemIn = input[z][y][idx];
      TOut elemMap = mapOp(elemIn);
      elemReduce = reduceOp(elemReduce, elemMap);
    }
  }

  using BlockReduce = cub::BlockReduce<TOut, BLOCK_DIM>;
  using TempStorage = typename BlockReduce::TempStorage;

  __shared__ TempStorage tempStorage;
  BlockReduce blockReduce{tempStorage};
  elemReduce = blockReduce.Reduce(elemReduce, reduceOp);

  if (threadIdx.x == 0) output[z][y] = elemReduce;
}

template<MapReduceType MR_TYPE, typename TIn, typename TOut>
Tensor reduceAllImpl(const Tensor &A) {
  CHECK(A.isContiguous());
  CHECK(A.getDType() == DType::getType<TIn>());

  int64_t numel = A.getNumEl();

  Tensor C = createCudaTensor<TOut>({1});
  TOut *result = C.getData<TOut>();

  thrust::device_ptr<const TIn> pdata = thrust::device_pointer_cast(A.getData<TIn>());
  thrust::device_vector<TIn> data(pdata, pdata + numel);
  auto iter = thrust::make_transform_iterator(data.begin(), MapOp<MR_TYPE, TIn, TOut>{});

  void *tempStorage = nullptr;
  size_t tempStorageBytes = 0;

  // Step 1: Query temp storage size
  cub::DeviceReduce::Reduce(
      tempStorage,
      tempStorageBytes,
      A.getData<TIn>(),
      result,
      numel,
      ReduceOp<MR_TYPE, TOut>{},
      0.0);

  // Allocate temp storage
  cudaMalloc(&tempStorage, tempStorageBytes);

  // Step 2: Run the reduction
  cub::DeviceReduce::Reduce(
      tempStorage,
      tempStorageBytes,
      iter,
      result,
      numel,
      ReduceOp<MR_TYPE, TOut>{},
      getReduceInitial<TOut, MR_TYPE>());

  // Free temp storage
  cudaFree(tempStorage);

  return C;
}

template<MapReduceType MR_TYPE, typename TIn, typename TOut>
Tensor reduceLastDim3DImpl(Tensor A) {
  CHECK(A.getDType() == DType::getType<TIn>());
  int origDim = A.getDim();

  switch (A.getDim()) {
    case 1:
      A = A.view({1, 1, A.getShape(0)});
      break;
    case 2:
      A = A.view({1, A.getShape(0), A.getShape(1)});
      break;
    case 3:
      break;
    default:
      NOT_IMPL();
  }

  std::vector<int> shape = A.getShape();
  shape.pop_back();

  Tensor C = createCudaTensor<TOut>(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = A.getShape(0);
  d.y = A.getShape(1);
  d.x = 1;

  reduce0Kernel3D<TIn, TOut, MR_TYPE, blockSize><<<d, blockSize>>>(A, C);

  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  switch (origDim) {
    case 1:
      C = C.view({C.getShape(2)});
      break;
    case 2:
      C = C.view({C.getShape(1), C.getShape(2)});
      break;
    case 3:
      break;
    default:
      NOT_IMPL();
  }
  return C;
}

Tensor reduceLastDim(Tensor A, DType outType, MapReduceType reduceType) {
  DType inType = A.getDType();

  if (inType == DType::kFloat16 && outType == DType::kFloat && reduceType == MapReduceType::SUM_EXP)
    return reduceLastDim3DImpl<MapReduceType::SUM_EXP, half, float>(A);
  if (inType == DType::kFloat16 && outType == DType::kFloat && reduceType == MapReduceType::SUM)
    return reduceLastDim3DImpl<MapReduceType::SUM, half, float>(A);
  if (inType == DType::kFloat16 && outType == DType::kFloat16 && reduceType == MapReduceType::MAX)
    return reduceLastDim3DImpl<MapReduceType::MAX, half, half>(A);

  NOT_IMPL();
}

Tensor reduceAll(Tensor A, DType outType, MapReduceType reduceType) {
  DType inType = A.getDType();

  if (inType == DType::kFloat16 && outType == DType::kFloat && reduceType == MapReduceType::SUM)
    return reduceAllImpl<MapReduceType::SUM, half, float>(A);
  if (inType == DType::kFloat && outType == DType::kFloat && reduceType == MapReduceType::SUM)
    return reduceAllImpl<MapReduceType::SUM, float, float>(A);
  if (inType == DType::kBool && outType == DType::kBool && reduceType == MapReduceType::ALL)
    return reduceAllImpl<MapReduceType::ALL, BoolType, BoolType>(A);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
