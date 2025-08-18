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

#include "libllm/cuda/common.h"
#include "libllm/cuda/reduce.h"

namespace libllm {
namespace op {
namespace cuda {

template<MapReduceType MR_TYPE, typename TIn, typename TOut>
struct MapOp {
  __device__ TOut operator()(const TIn &x) const {
    if constexpr (
        MR_TYPE == MapReduceType::SUM_FP16_FP32 || MR_TYPE == MapReduceType::MAX ||
        MR_TYPE == MapReduceType::SUM_FP32) {
      return x;
    } else if constexpr (MR_TYPE == MapReduceType::SUM_EXP_FP16_FP32) {
      return expf(static_cast<float>(x));
    } else if constexpr (MR_TYPE == MapReduceType::SUM_SQUARE_FP16_FP32) {
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
        MR_TYPE == MapReduceType::SUM_FP16_FP32 || MR_TYPE == MapReduceType::SUM_FP32 ||
        MR_TYPE == MapReduceType::SUM_EXP_FP16_FP32 ||
        MR_TYPE == MapReduceType::SUM_SQUARE_FP16_FP32) {
      return a + b;
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
      MR_TYPE == MapReduceType::SUM_FP32 || MR_TYPE == MapReduceType::SUM_FP16_FP32 ||
      MR_TYPE == MapReduceType::SUM_EXP_FP16_FP32 ||
      MR_TYPE == MapReduceType::SUM_SQUARE_FP16_FP32) {
    return T(0);
  } else if constexpr (MR_TYPE == MapReduceType::MAX) {
    return -::cuda::std::numeric_limits<float>::infinity();
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
    PackedSubtensor<const TIn, 3> input,
    PackedSubtensor<TOut, 2> output) {
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
    case MapReduceType::SUM_FP16_FP32:
      reduce0Kernel3D<half, float, MapReduceType::SUM_FP16_FP32, blockSize><<<d, blockSize>>>(A, C);
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

Tensor reduceHalfToSingle2D(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 2);

  int d0 = A.getShape(0);
  int d1 = A.getShape(1);

  Tensor C = reduceHalfToSingle3D(A.view({1, d0, d1}), reduceType);
  return C.view({d0});
}

Tensor reduceHalfToSingle1D(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 1);

  int d0 = A.getShape(0);

  Tensor C = reduceHalfToSingle3D(A.view({1, 1, d0}), reduceType);
  return C.view({1});
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

Tensor reduceHalf2D(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 2);

  int d0 = A.getShape(0);
  int d1 = A.getShape(1);

  Tensor C = reduceHalf3D(A.view({1, d0, d1}), reduceType);
  return C.view({d0});
}

Tensor reduceHalf1D(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);
  CHECK(A.getDim() == 1);

  int d0 = A.getShape(0);

  Tensor C = reduceHalf3D(A.view({1, 1, d0}), reduceType);
  return C.view({1});
}

Tensor reduceHalf(Tensor A, MapReduceType reduceType) {
  if (A.getDim() == 3) return reduceHalf3D(A, reduceType);
  if (A.getDim() == 2) return reduceHalf2D(A, reduceType);
  if (A.getDim() == 1) return reduceHalf1D(A, reduceType);

  NOT_IMPL();
}

Tensor reduceHalfToSingle(Tensor A, MapReduceType reduceType) {
  if (A.getDim() == 3) return reduceHalfToSingle3D(A, reduceType);
  if (A.getDim() == 2) return reduceHalfToSingle2D(A, reduceType);
  if (A.getDim() == 1) return reduceHalfToSingle1D(A, reduceType);

  NOT_IMPL();
}

Tensor reduce(Tensor A, MapReduceType reduceType) {
  CHECK(A.getDType() == DType::kFloat16);

  if (reduceType == MapReduceType::SUM_EXP_FP16_FP32) return reduceHalfToSingle(A, reduceType);
  if (reduceType == MapReduceType::SUM_FP16_FP32) return reduceHalfToSingle(A, reduceType);
  if (reduceType == MapReduceType::MAX) return reduceHalf(A, reduceType);

  NOT_IMPL();
}

Tensor reduceAll(Tensor A, MapReduceType reduceType) {
  if (reduceType == MapReduceType::SUM_FP16_FP32)
    return reduceAllImpl<MapReduceType::SUM_FP16_FP32, half, float>(A);
  if (reduceType == MapReduceType::SUM_FP32)
    return reduceAllImpl<MapReduceType::SUM_FP32, float, float>(A);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
