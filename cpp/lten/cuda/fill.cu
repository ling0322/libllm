// The MIT License (MIT)
//
// Copyright (c) 202r Xiaoyang Chen
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

#include "lten/cuda/common.h"
#include "lten/cuda/fill.h"

namespace lten {
namespace op {
namespace cuda {

template<typename T>
__global__ void fill5DKernel(PackedSubtensor<T, 5> dest, float value) {
  int d4 = blockIdx.x * blockDim.x + threadIdx.x;
  int d3 = blockIdx.y * blockDim.y + threadIdx.y;

  dim3 dz = splitIndexToDim3(blockIdx.z * blockDim.z + threadIdx.z, dest.getSize());
  int d2 = dz.x;
  int d1 = dz.y;
  int d0 = dz.z;

  const Size *s = dest.getSize();

  if (d0 < s[0].shape && d1 < s[1].shape && d2 < s[2].shape && d3 < s[3].shape && d4 < s[4].shape) {
    dest[d0][d1][d2][d3][d4] = value;
  }
}

template<typename T>
__global__ void fill4DKernel(PackedSubtensor<T, 4> dest, float value) {
  int d3 = blockIdx.x * blockDim.x + threadIdx.x;

  dim3 dz = splitIndexToDim3(blockIdx.y * blockDim.y + threadIdx.y, dest.getSize());
  int d2 = dz.x;
  int d1 = dz.y;
  int d0 = dz.z;

  const Size *s = dest.getSize();

  if (d0 < s[0].shape && d1 < s[1].shape && d2 < s[2].shape && d3 < s[3].shape) {
    dest[d0][d1][d2][d3] = value;
  }
}

template<typename T>
__global__ void fill3DKernel(PackedSubtensor<T, 3> dest, float value) {
  int d2 = blockIdx.x * blockDim.x + threadIdx.x;
  int d1 = blockIdx.y * blockDim.y + threadIdx.y;
  int d0 = blockIdx.z * blockDim.z + threadIdx.z;

  const Size *s = dest.getSize();

  if (d0 < s[0].shape && d1 < s[1].shape && d2 < s[2].shape) {
    dest[d0][d1][d2] = value;
  }
}

template<typename T>
void fill5D(Tensor tensor, float value) {
  PackedSubtensor<T, 5> sC(tensor);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = tensor.getShape(0) * tensor.getShape(1) * tensor.getShape(2);
  d.y = tensor.getShape(3);
  d.x = (tensor.getShape(4) + blockSize - 1) / blockSize;

  fill5DKernel<T><<<d, blockSize>>>(sC, value);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void fill4D(Tensor tensor, float value) {
  PackedSubtensor<T, 4> sC(tensor);

  constexpr int blockSize = 256;
  dim3 d;
  d.y = tensor.getShape(0) * tensor.getShape(1) * tensor.getShape(2);
  d.x = (tensor.getShape(3) + blockSize - 1) / blockSize;

  fill4DKernel<T><<<d, blockSize>>>(sC, value);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void fill3D(Tensor tensor, float value) {
  PackedSubtensor<T, 3> sC(tensor);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = tensor.getShape(0);
  d.y = tensor.getShape(1);
  d.x = (tensor.getShape(2) + blockSize - 1) / blockSize;

  fill3DKernel<T><<<d, blockSize>>>(sC, value);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void fill2D(Tensor tensor, float value) {
  int d0 = tensor.getShape(0);
  int d1 = tensor.getShape(1);

  fill3D<T>(tensor.view({1, d0, d1}), value);
}

template<typename T>
void fill1D(Tensor tensor, float value) {
  int d0 = tensor.getShape(0);
  fill3D<T>(tensor.view({1, 1, d0}), value);
}

void fill(Tensor A, float value) {
  CHECK(A.getDevice().getType() == Device::kCuda);

  if (A.getDType() == DType::kFloat16 && A.getDim() == 5) return fill5D<half>(A, value);
  if (A.getDType() == DType::kFloat16 && A.getDim() == 4) return fill4D<half>(A, value);
  if (A.getDType() == DType::kFloat16 && A.getDim() == 3) return fill3D<half>(A, value);
  if (A.getDType() == DType::kFloat16 && A.getDim() == 2) return fill2D<half>(A, value);
  if (A.getDType() == DType::kFloat16 && A.getDim() == 1) return fill1D<half>(A, value);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace lten
