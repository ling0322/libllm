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

#include "ly/operators/cuda/lookup.h"

#include <cuda_fp16.h>
#include "ly/operators/cuda/common.h"

namespace ly {
namespace op {
namespace cuda {

__global__
void lookupHalfKernel2D(PackedSubtensor<const half, 2> embd,
                        PackedSubtensor<const int64_t, 2> inputs,
                        PackedSubtensor<half, 3> dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < embd.getShape(1) && y < inputs.getShape(1) && z < inputs.getShape(0)) {
    int index = inputs[z][y];
    assert(index < embd.getShape(0));
    dst[z][y][x] = embd[index][x];
  }
}

__global__
void lookupQ4Kernel2D(PackedSubtensor2DQ4 embd,
                      PackedSubtensor<const int64_t, 2> inputs,
                      PackedSubtensor<half, 3> dst) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x < embd.getNumCol() / 2 && y < inputs.getShape(1) && z < inputs.getShape(0)) {
    int row = inputs[z][y];
    assert(row < embd.getNumRow());

    int rowGroup = row * embd.getNumCol() / 32;
    int elemGroup = rowGroup + x / 16;

    uint8_t q4elem = embd.getData(rowGroup)[x];
    half scale = embd.getScaleValue(elemGroup);
    int8_t zero = embd.getZeroValue(elemGroup);

    dst[z][y][x * 2] = __hmul(scale, __int2half_rd(static_cast<int>(q4elem & 0xf) - zero));
    dst[z][y][x * 2+ 1] = __hmul(scale, __int2half_rd(static_cast<int>(q4elem >> 4) - zero));
  }
}

Tensor lookup2DHalf(const Tensor &embdTable, const Tensor &input) {
  CHECK(embdTable.getDType() == DType::kFloat16);

  std::vector<Tensor::ShapeType> shape = input.getShape();
  shape.push_back(embdTable.getShape(1));
  Tensor dst = createCudaTensorHalf(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = dst.getShape(0);
  d.y = dst.getShape(1);
  d.x = (dst.getShape(2) + blockSize - 1) / blockSize;

  PackedSubtensor<const half, 2> sA(embdTable);
  PackedSubtensor<const int64_t, 2> sB(input);
  PackedSubtensor<half, 3> sC(dst);

  lookupHalfKernel2D<<<d, blockSize>>>(sA, sB, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return dst;
}

Tensor lookup2DQ4(const Tensor &embdTable, const Tensor &input) {
  CHECK(embdTable.getDType() == DType::kQInt4Group32);

  std::vector<Tensor::ShapeType> shape = input.getShape();
  shape.push_back(embdTable.getShape(1));
  Tensor dst = createCudaTensorHalf(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = dst.getShape(0);
  d.y = dst.getShape(1);

  // div dst.getShape(2) by 2 here since in each kernel we process 2 elements.
  d.x = (dst.getShape(2) / 2 + blockSize - 1) / blockSize;

  PackedSubtensor2DQ4 sA(embdTable);
  PackedSubtensor<const int64_t, 2> sB(input);
  PackedSubtensor<half, 3> sC(dst);

  lookupQ4Kernel2D<<<d, blockSize>>>(sA, sB, sC);
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
    return lookup2DHalf(embdTable, input);
  } if (input.getDim() == 2 && embdTable.getDType() == DType::kQInt4Group32) {
    return lookup2DQ4(embdTable, input);
  } else {
    NOT_IMPL();
  }
}

}  // cuda
}  // op
}  // ly
    
