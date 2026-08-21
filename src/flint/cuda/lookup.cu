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

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/lookup.h"

namespace fl {
namespace op {
namespace cuda {

template<typename T>
__global__ void lookupKernel2D(
  PackedTensorAccessor<const T, 2> embd,
    PackedTensorAccessor<const int64_t, 2> inputs,
  PackedTensorAccessor<T, 3> dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < embd.getShape(1) && y < inputs.getShape(1) && z < inputs.getShape(0)) {
    int index = inputs[z][y];
    assert(index < embd.getShape(0));
    dst[z][y][x] = embd[index][x];
  }
}

template<typename T>
Tensor lookup2D(const Tensor &embdTable, const Tensor &input) {
  std::vector<Tensor::ShapeType> shape = input.getShape();
  shape.push_back(embdTable.getShape(1));
  Tensor dst = createCudaTensor<T>(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = dst.getShape(0);
  d.y = dst.getShape(1);
  d.x = (dst.getShape(2) + blockSize - 1) / blockSize;

  PackedTensorAccessor<const T, 2> sA(embdTable);
  PackedTensorAccessor<const int64_t, 2> sB(input);
  PackedTensorAccessor<T, 3> sC(dst);

  lookupKernel2D<<<d, blockSize>>>(sA, sB, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return dst;
}

template<typename T>
__global__ void lookupKernel1D(
  PackedTensorAccessor<const T, 2> embd,
    PackedTensorAccessor<const int64_t, 1> inputs,
  PackedTensorAccessor<T, 2> dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < embd.getShape(1) && y < inputs.getShape(0)) {
    int index = inputs[y];
    assert(index < embd.getShape(0));
    dst[y][x] = embd[index][x];
  }
}

template<typename T>
Tensor lookup1D(const Tensor &embdTable, const Tensor &input) {
  std::vector<Tensor::ShapeType> shape = input.getShape();
  shape.push_back(embdTable.getShape(1));
  Tensor dst = createCudaTensor<T>(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.y = dst.getShape(0);
  d.x = (dst.getShape(1) + blockSize - 1) / blockSize;

  PackedTensorAccessor<const T, 2> sA(embdTable);
  PackedTensorAccessor<const int64_t, 1> sB(input);
  PackedTensorAccessor<T, 2> sC(dst);

  lookupKernel1D<<<d, blockSize>>>(sA, sB, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return dst;
}

Tensor lookup(const Tensor &embdTable, const Tensor &input) {
  CHECK(input.getDType() == DType::kLong);
  CHECK(input.getDevice().getType() == Device::kCuda);
  CHECK(embdTable.getDevice().getType() == Device::kCuda);
  CHECK(embdTable.getDim() == 2);

  if (input.getDim() == 2 && embdTable.getDType() == DType::kFloat16) {
    return lookup2D<half>(embdTable, input);
  }
  if (input.getDim() == 2 && embdTable.getDType() == DType::kFloat) {
    return lookup2D<float>(embdTable, input);
  }

  // a packed batch of ids is 1D, one embedding row comes out per id.
  if (input.getDim() == 1 && embdTable.getDType() == DType::kFloat16) {
    return lookup1D<half>(embdTable, input);
  }
  if (input.getDim() == 1 && embdTable.getDType() == DType::kFloat) {
    return lookup1D<float>(embdTable, input);
  }

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
