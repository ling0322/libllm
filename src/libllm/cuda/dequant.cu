// The MIT License (MIT)
//
// Copyright (c) 2023 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda_fp16.h>
#include <cuda_fp4.h>

#include "libllm/cuda/common.h"
#include "libllm/cuda/copy.h"
#include "libllm/cuda/dequant.h"

namespace libllm {
namespace op {
namespace cuda {

__global__ void dequantTensor2DQ4(
    PackedSubtensor2DQInt4x32 qtensor,
    PackedSubtensor<half, 2> destTensor) {
  int64_t numQ4x2 = qtensor.getNumCol() * qtensor.getNumRow() / 2;
  half *destData = destTensor.getData();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int elemOffset = idx; elemOffset < numQ4x2; elemOffset += gridDim.x * blockDim.x) {
    int elemGroup = elemOffset / (QInt4x32::GroupSize / 2);
    uint8_t q4elem = qtensor.getData(elemGroup)[elemOffset % (QInt4x32::GroupSize / 2)];
    half scale = qtensor.getScaleValue(elemGroup);
    half zero = qtensor.getZeroValue(elemGroup);

    destData[elemOffset * 2] = __hsub(
        __hmul(scale, __int2half_rd(static_cast<int>(q4elem & 0xf))),
        zero);
    destData[elemOffset * 2 + 1] = __hsub(
        __hmul(scale, __int2half_rd(static_cast<int>(q4elem >> 4))),
        zero);
  }
}

inline __device__ uint8_t fp32_vec_to_e2m1(float v0, float v1) {
  uint8_t b = __nv_cvt_float2_to_fp4x2(make_float2(v0, v1), __NV_E2M1, cudaRoundNearest);
  return b;
}

__global__ void quantizeHalfToMxfp4Kernel(
    const half *__restrict__ x,
    uint8_t *__restrict__ q,
    uint8_t *__restrict__ scales,
    int64_t N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & (32 - 1);
  int group = idx >> 5;
  int group_start = group << 5;

  float xi = (idx < N) ? static_cast<float>(x[idx]) : 0.0f;

  float a = fabsf(xi);
  unsigned mask = 0xFFFFFFFFu;

  a = fmaxf(a, __shfl_down_sync(mask, a, 16));
  a = fmaxf(a, __shfl_down_sync(mask, a, 8));
  a = fmaxf(a, __shfl_down_sync(mask, a, 4));
  a = fmaxf(a, __shfl_down_sync(mask, a, 2));
  a = fmaxf(a, __shfl_down_sync(mask, a, 1));

  float max_abs = __shfl_sync(mask, a, 0);
  float scale = (max_abs / 3.0) + 1e-8f;
  reinterpret_cast<uint32_t &>(scale) = (reinterpret_cast<uint32_t &>(scale)) & 0x7f800000;

  float v0 = xi / scale;
  float v1 = __shfl_xor_sync(mask, v0, 1);
  if (lane % 2 == 0 && idx < N) {
    q[idx / 2] = fp32_vec_to_e2m1(v0, v1);
  }
  if (lane == 0 && group_start < N) {
    scales[group] = static_cast<uint8_t>(reinterpret_cast<uint32_t &>(scale) >> 23);
  }
}

__global__ void dequantizeMxfp4ToHalfKernel(
    const uint8_t *__restrict__ q,
    const uint8_t *__restrict__ scales,
    half *__restrict__ x,
    int64_t numel) {
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  half2 *__restrict__ x2 = reinterpret_cast<half2 *>(x);

  for (; idx < numel; idx += stride) {
    int group = idx / 16;

    uint8_t fp4x2 = q[idx];
    uint8_t scaleE8m0 = scales[group];

    half2 h2 = __nv_cvt_fp4x2_to_halfraw2(fp4x2, __NV_E2M1);
    half scale = __float2half(__bfloat162float(__nv_cvt_e8m0_to_bf16raw(scaleE8m0)));

    h2.x *= scale;
    h2.y *= scale;

    x2[idx] = h2;
  }
}

Tensor dequantQ4ToHalf(const Tensor &qtensor) {
  CHECK(qtensor.getDType() == DType::kQInt4x32);
  Tensor qT = qtensor;
  bool transQ = qtensor.getStride(0) == 1 && qtensor.getStride(1) != 1;
  if (transQ) qT = qT.transpose(0, 1);

  std::vector<Tensor::ShapeType> shape = qT.getShape();
  Tensor dst = createCudaTensorHalf(shape);

  constexpr int blockSize = 256;

  CHECK(qT.getNumEl() < std::numeric_limits<int32_t>::max());
  dim3 grid = getGrid1D(static_cast<int>(qT.getNumEl()), blockSize);

  PackedSubtensor2DQInt4x32 sA(qT);
  PackedSubtensor<half, 2> sC(dst);

  dequantTensor2DQ4<<<grid, blockSize>>>(sA, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  if (transQ) dst = dst.transpose(0, 1);
  return dst;
}

std::pair<Tensor, Tensor> quantHalfToMxfp4(const Tensor &tensor) {
  CHECK(tensor.getDType() == DType::kFloat16);
  CHECK(tensor.isContiguous());
  CHECK(tensor.getShape(-1) % 32 == 0);

  int64_t numel = tensor.getNumEl();

  dim3 block(256);
  dim3 grid((numel + block.x - 1) / block.x);

  std::vector<Tensor::ShapeType> shape = tensor.getShape();
  shape.back() /= 2;
  Tensor data = createCudaTensorFp4x2(shape);

  shape.back() /= 16;
  Tensor scale = createCudaTensorUInt8(shape);

  quantizeHalfToMxfp4Kernel<<<grid, block>>>(
      tensor.getData<half>(),
      reinterpret_cast<uint8_t *>(data.getData<Fp4E2M0x2>()),
      reinterpret_cast<uint8_t *>(scale.getData<UInt8>()),
      numel);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return std::make_pair(data, scale);
}

Tensor dequandMxfp4ToHalf(const Tensor &fp4, const Tensor &scale) {
  CHECK(fp4.isContiguous() && scale.isContiguous());
  CHECK(fp4.getDim() == scale.getDim());
  CHECK(fp4.getShape(-1) % 16 == 0 && fp4.getShape(-1) / 16 == scale.getShape(-1));

  int64_t numel64 = fp4.getNumEl();
  CHECK(numel64 < std::numeric_limits<int32_t>::max());
  int numel = static_cast<int>(numel64);

  int blockSize = 256;
  dim3 grid = getGrid1D(numel, blockSize);

  std::vector<Tensor::ShapeType> shape = fp4.getShape();
  shape.back() *= 2;
  Tensor C = createCudaTensorHalf(shape);

  dequantizeMxfp4ToHalfKernel<<<grid, blockSize>>>(
      reinterpret_cast<const uint8_t *>(fp4.getData<Fp4E2M0x2>()),
      reinterpret_cast<const uint8_t *>(scale.getData<UInt8>()),
      C.getData<half>(),
      numel);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor toSm1xxScaleBlock(const Tensor &scale) {
  CHECK(scale.getDim() == 2);  //  && scale.getDType() == DType::kUInt8);

  int numRow = scale.getShape(0);
  int numCol = scale.getShape(1);

  CHECK(numRow % 128 == 0 && numCol % 4 == 0);
  Tensor scale0 = scale.view({numRow / 128, 128, numCol / 4, 4}).transpose(1, 2);
  Tensor scale1 = tensorLike(scale0);
  copy(scale0, scale1);

  scale1 = scale1.view({-1, 4, 32, 4}).transpose(1, 2);
  Tensor scale2 = tensorLike(scale1);
  copy(scale1, scale2);

  return scale2;
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
