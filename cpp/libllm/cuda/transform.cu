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

#include "libllm/cuda/common.h"
#include "libllm/cuda/transform.h"

namespace libllm {
namespace op {
namespace cuda {

template<typename T>
__global__ void transform5DKernel(
    PackedSubtensor<const T, 5> src,
    T alpha,
    T beta,
    PackedSubtensor<T, 5> dest) {
  int d4 = blockIdx.x * blockDim.x + threadIdx.x;
  int d3 = blockIdx.y * blockDim.y + threadIdx.y;

  dim3 dz = splitIndexToDim3(blockIdx.z * blockDim.z + threadIdx.z, src.getSize());
  int d2 = dz.x;
  int d1 = dz.y;
  int d0 = dz.z;

  const Size *s = src.getSize();

  if (d0 < s[0].shape && d1 < s[1].shape && d2 < s[2].shape && d3 < s[3].shape && d4 < s[4].shape) {
    dest[d0][d1][d2][d3][d4] = alpha * src[d0][d1][d2][d3][d4] + beta;
  }
}

template<typename T>
__global__ void transform4DKernel(
    PackedSubtensor<const T, 4> src,
    T alpha,
    T beta,
    PackedSubtensor<T, 4> dest) {
  int d3 = blockIdx.x * blockDim.x + threadIdx.x;

  dim3 dz = splitIndexToDim3(blockIdx.y * blockDim.y + threadIdx.y, src.getSize());
  int d2 = dz.x;
  int d1 = dz.y;
  int d0 = dz.z;

  const Size *s = src.getSize();

  if (d0 < s[0].shape && d1 < s[1].shape && d2 < s[2].shape && d3 < s[3].shape) {
    dest[d0][d1][d2][d3] = alpha * src[d0][d1][d2][d3] + beta;
  }
}

template<typename T>
__global__ void transform3DKernel(
    PackedSubtensor<const T, 3> src,
    T alpha,
    T beta,
    PackedSubtensor<T, 3> dest) {
  int d2 = blockIdx.x * blockDim.x + threadIdx.x;
  int d1 = blockIdx.y * blockDim.y + threadIdx.y;
  int d0 = blockIdx.z * blockDim.z + threadIdx.z;

  const Size *s = src.getSize();

  if (d0 < s[0].shape && d1 < s[1].shape && d2 < s[2].shape) {
    dest[d0][d1][d2] = alpha * src[d0][d1][d2] + beta;
  }
}

template<typename T>
void transform5D(Tensor src, Tensor dest, T alpha, T beta) {
  src.throwIfInvalidShape(dest.getShape(), "transform5D");

  PackedSubtensor<const T, 5> sA(src);
  PackedSubtensor<T, 5> sC(dest);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = src.getShape(0) * src.getShape(1) * src.getShape(2);
  d.y = src.getShape(3);
  d.x = (src.getShape(4) + blockSize - 1) / blockSize;

  transform5DKernel<T><<<d, blockSize>>>(sA, alpha, beta, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void transform4D(Tensor src, Tensor dest, T alpha, T beta) {
  src.throwIfInvalidShape(dest.getShape(), "transform4D");

  PackedSubtensor<const T, 4> sA(src);
  PackedSubtensor<T, 4> sC(dest);

  constexpr int blockSize = 256;
  dim3 d;
  d.y = src.getShape(0) * src.getShape(1) * src.getShape(2);
  d.x = (src.getShape(3) + blockSize - 1) / blockSize;

  transform4DKernel<T><<<d, blockSize>>>(sA, alpha, beta, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void transform3D(Tensor src, Tensor dest, T alpha, T beta) {
  src.throwIfInvalidShape(dest.getShape(), "transform3D");

  PackedSubtensor<const T, 3> sA(src);
  PackedSubtensor<T, 3> sC(dest);

  constexpr int blockSize = 256;
  dim3 d;
  d.z = src.getShape(0);
  d.y = src.getShape(1);
  d.x = (src.getShape(2) + blockSize - 1) / blockSize;

  transform3DKernel<T><<<d, blockSize>>>(sA, alpha, beta, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

template<typename T>
void transform2D(Tensor src, Tensor dest, T alpha, T beta) {
  src.throwIfInvalidShape(dest.getShape(), "transform2D");

  int d0 = src.getShape(0);
  int d1 = src.getShape(1);

  return transform3D(src.view({1, d0, d1}), dest.view({1, d0, d1}), alpha, beta);
}

template<typename T>
void transform1D(Tensor src, Tensor dest, T alpha, T beta) {
  src.throwIfInvalidShape(dest.getShape(), "transform2D");

  int d0 = src.getShape(0);
  return transform3D(src.view({1, 1, d0}), dest.view({1, 1, d0}), alpha, beta);
}

void transformHalf(Tensor src, Tensor dest, half alpha, half beta) {
  CHECK(src.getDType() == DType::kFloat16);
  CHECK(dest.getDType() == DType::kFloat16);

  if (src.getDim() == 5) return transform5D<half>(src, dest, alpha, beta);
  if (src.getDim() == 4) return transform4D<half>(src, dest, alpha, beta);
  if (src.getDim() == 3) return transform3D<half>(src, dest, alpha, beta);
  if (src.getDim() == 2) return transform2D<half>(src, dest, alpha, beta);
  if (src.getDim() == 1) return transform1D<half>(src, dest, alpha, beta);

  NOT_IMPL();
}

Tensor transform(const Tensor &src, float alpha, float beta) {
  CHECK(src.getDevice().getType() == Device::kCuda);

  if (src.getDType() == DType::kFloat16) {
    Tensor dest = createCudaTensorHalf(src.getShape());
    transformHalf(src, dest, half(alpha), half(beta));
    return dest;
  } else {
    NOT_IMPL();
  }
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
