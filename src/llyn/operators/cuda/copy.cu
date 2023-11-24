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

#include "llyn/operators/cuda/copy.h"

#include "llyn/operators/cuda/common.h"

namespace llyn {
namespace op {
namespace cuda {

template<typename T>
__global__ void copy5DKernel(PackedSubtensor<const T, 5> src, PackedSubtensor<T, 5> dest) {
  int d4 = blockIdx.x * blockDim.x + threadIdx.x;
  int d3 = blockIdx.y * blockDim.y + threadIdx.y;

  dim3 dz = splitIndexToDim3(blockIdx.z * blockDim.z + threadIdx.z, src.getSize());
  int d2 = dz.x;
  int d1 = dz.y;
  int d0 = dz.z;

  const Size *s = src.getSize();

  if (d0 < s[0].shape && d1 < s[1].shape && d2 < s[2].shape && d3 < s[3].shape && 
      d4 < s[4].shape) {
    dest[d0][d1][d2][d3][d4] = src[d0][d1][d2][d3][d4];
  }
}

template<typename T>
void copy5D(const Tensor &src, Tensor &dest) {
  src.throwIfInvalidShape(dest.getShape());

  PackedSubtensor<const T, 5> sA(src);
  PackedSubtensor<T, 5> sC(dest);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = src.getShape(0) * src.getShape(1) * src.getShape(2);
  d.y = src.getShape(3);
  d.x = (src.getShape(4) + blockSize - 1) / blockSize;

  copy5DKernel<T><<<d, blockSize>>>(sA, sC);
  checkCudaError(cudaGetLastError());
}

void copy(const Tensor &src, Tensor &dest) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(dest.getDevice().getType() == Device::kCuda);

  if (src.getDType() == DType::kFloat16 && src.getDim() == 5) return copy5D<half>(src, dest);

  NOT_IMPL();
}

void copyContig(const Tensor &src, Tensor &dest) {
  cudaMemcpy(
      dest.getData<void>(),
      src.getData<void>(),
      src.getDType().getTotalSize(src.getNumEl()),
      cudaMemcpyDeviceToDevice); 
}

}  // cuda
}  // op
}  // llyn
