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

#include "lynn/cuda/common.h"
#include "lynn/cuda/copy.h"
#include "lynn/cuda/dequant.h"

namespace libllm {
namespace op {
namespace cuda {

__global__ void dequantTensor2DQ4(
    PackedSubtensor2DQInt4x32 qtensor,
    PackedTensorAccessor<half, 2> destTensor) {
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

__forceinline__ __device__ int getScaleOffset(int row, int col, int numCol) {
  int tileRow = row / 128;
  int tileCol = col / 4;
  int tileNumCol = numCol / 4;
  int tileOffset = (tileCol + tileRow * tileNumCol) * 512;
  int innerRow = row % 128;
  int innerCol = col % 4;
  int swizzedRow = innerRow % 32;
  int swizzedCol = (innerRow / 32) * 4 + innerCol;
  int offset = tileOffset + swizzedRow * 16 + swizzedCol;

  return offset;
}

template<typename T>
__global__ void swizzleScaleKernel(const T *__restrict__ in, T *__restrict__ out, int m, int n) {
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int numel = n * m;

  for (; idx < numel; idx += stride) {
    int row = idx / n;
    int col = idx % n;

    int offset = getScaleOffset(row, col, n);
    if (offset >= numel) {
      __trap();
    }

    out[offset] = in[idx];
  }
}

__forceinline__ __device__ uint32_t cvtFp32x8ToFp4(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x),
        "f"(array[0].y),
        "f"(array[1].x),
        "f"(array[1].y),
        "f"(array[2].x),
        "f"(array[2].y),
        "f"(array[3].x),
        "f"(array[3].y));
  return val;
}

__forceinline__ __device__ float fastRcp(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

template<bool SWIZZLE_SCALE>
__global__ void quantizeHalfToMxfp4Kernel(
    const half *__restrict__ x,
    __nv_fp4x2_storage_t *__restrict__ q,
    __nv_fp8_storage_t *__restrict__ scales,
    int numRow,
    int numCol) {
  for (int rowIdx = blockIdx.x; rowIdx < numRow; rowIdx += gridDim.x) {
    for (int columnIdx = threadIdx.x; columnIdx < numCol / 8; columnIdx += blockDim.x) {
      int offset = rowIdx * (numCol / 8) + columnIdx;
      PackedOWORD<half2> po = reinterpret_cast<const PackedOWORD<half2> *>(x)[offset];

      half2 maxAbs2 = __habs2(po.v[0]);
      maxAbs2 = __hmax2(maxAbs2, __habs2(po.v[1]));
      maxAbs2 = __hmax2(maxAbs2, __habs2(po.v[2]));
      maxAbs2 = __hmax2(maxAbs2, __habs2(po.v[3]));

      unsigned mask = 0xffffffff;
      maxAbs2 = __hmax2(maxAbs2, __shfl_xor_sync(mask, maxAbs2, 1));
      maxAbs2 = __hmax2(maxAbs2, __shfl_xor_sync(mask, maxAbs2, 2));

      float maxAbs = float(__hmax(maxAbs2.x, maxAbs2.y));
      float scale = maxAbs * fastRcp(6.0);
      uint32_t scaleU32 = reinterpret_cast<uint32_t &>(scale) >> 23;
      __nv_fp8_storage_t e8m0Scale = scaleU32 & 0xff;
      reinterpret_cast<uint32_t &>(scale) = scaleU32 << 23;

      int scaleOffset;
      if constexpr (SWIZZLE_SCALE) {
        scaleOffset = getScaleOffset(rowIdx, columnIdx / 4, numCol / 32);
      } else {
        scaleOffset = rowIdx * (numCol / 32) + (columnIdx / 4);
      }
      scales[scaleOffset] = e8m0Scale;

      float rcpScale = fastRcp(scale);
      float2 f2[4];
      f2[0] = __half22float2(po.v[0]);
      f2[1] = __half22float2(po.v[1]);
      f2[2] = __half22float2(po.v[2]);
      f2[3] = __half22float2(po.v[3]);

      f2[0].x *= rcpScale;
      f2[0].y *= rcpScale;
      f2[1].x *= rcpScale;
      f2[1].y *= rcpScale;
      f2[2].x *= rcpScale;
      f2[2].y *= rcpScale;
      f2[3].x *= rcpScale;
      f2[3].y *= rcpScale;

      uint32_t qfp4x8 = cvtFp32x8ToFp4(f2);
      reinterpret_cast<uint32_t *>(q)[offset] = qfp4x8;
    }
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
  PackedTensorAccessor<half, 2> sC(dst);

  dequantTensor2DQ4<<<grid, blockSize>>>(sA, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  if (transQ) dst = dst.transpose(0, 1);
  return dst;
}

std::pair<Tensor, Tensor> quantHalfToMxfp4(const Tensor &tensor, bool scaleLayout) {
  CHECK(tensor.getDType() == DType::kFloat16);
  CHECK(tensor.isContiguous() && tensor.getDim() == 2);
  CHECK(tensor.getShape(0) % 128 == 0 && tensor.getShape(1) % 128 == 0);

  int64_t numel64 = tensor.getNumEl();
  CHECK(numel64 < std::numeric_limits<int32_t>::max());

  int numRow = tensor.getShape(0);
  int numCol = tensor.getShape(1);
  CHECK(numRow % 128 == 0 && numCol % 128 == 0);

  int numSM = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  int numBlock = std::min(tensor.getShape(0), 4 * numSM);
  int blockSize = std::min(tensor.getShape(1) / 8, 256);

  Tensor q = createCudaTensorFp4x2({numRow, numCol / 2});
  Tensor scales = createCudaTensorUInt8({numRow, numCol / 32});
  if (scaleLayout) {
    scales = scales.view({-1, 32, 16});
  }

  if (scaleLayout) {
    quantizeHalfToMxfp4Kernel<true><<<numBlock, blockSize>>>(
        tensor.getData<half>(),
        reinterpret_cast<uint8_t *>(q.getData<Fp4E2M0x2>()),
        reinterpret_cast<uint8_t *>(scales.getData<UInt8>()),
        numRow,
        numCol);
  } else {
    quantizeHalfToMxfp4Kernel<false><<<numBlock, blockSize>>>(
        tensor.getData<half>(),
        reinterpret_cast<uint8_t *>(q.getData<Fp4E2M0x2>()),
        reinterpret_cast<uint8_t *>(scales.getData<UInt8>()),
        numRow,
        numCol);
  }
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return std::make_pair(q, scales);
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

template<typename T>
Tensor toSm1xxScaleBlockImpl(const Tensor &scale) {
  CHECK(scale.getDim() == 2 && scale.getDType() == DType::getType<T>());

  int numRow = scale.getShape(0);
  int numCol = scale.getShape(1);

  CHECK(numRow % 128 == 0 && numCol % 4 == 0);

  int64_t numel64 = scale.getNumEl();
  CHECK(numel64 < std::numeric_limits<int32_t>::max());
  int numel = static_cast<int>(numel64);

  int blockSize = 256;
  dim3 grid = getGrid1D(numel, blockSize);

  Tensor C = createCudaTensor<T>({scale.getShape(0), scale.getShape(1)});
  C = C.view({-1, 32, 16});

  swizzleScaleKernel<T><<<grid, blockSize>>>(
      reinterpret_cast<const T *>(scale.getData<T>()),
      reinterpret_cast<T *>(C.getData<T>()),
      numRow,
      numCol);

  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return C;
}

Tensor toSm1xxScaleBlock(const Tensor &scale) {
  CHECK(scale.getDevice().getType() == Device::kCuda);
  DType dtype = scale.getDType();

  if (dtype == DType::kLong) return toSm1xxScaleBlockImpl<LongType>(scale);
  if (dtype == DType::kUInt8) return toSm1xxScaleBlockImpl<UInt8>(scale);

  NOT_IMPL();
}

}  // namespace cuda
}  // namespace op
}  // namespace libllm
