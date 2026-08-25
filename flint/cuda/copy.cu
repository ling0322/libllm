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

#include <algorithm>

#include "flint/cuda/accessor.h"
#include "flint/cuda/common.h"
#include "flint/cuda/copy.h"

namespace fl {
namespace op {
namespace cuda {

template<typename T, int DIM>
__global__ void copyNDKernel(
    PackedTensorAccessor<const T, DIM> src,
    PackedTensorAccessor<T, DIM> dest,
    int numel) {
  int stride = blockDim.x * gridDim.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (; idx < numel; idx += stride) {
    dest.getElemByIndex(idx) = src.getElemByIndex(idx);
  }
}

template<typename T, int DIM>
void copyND(const Tensor &src, Tensor &dest) {
  src.throwIfInvalidShape(dest.getShape(), "copyND");

  PackedTensorAccessor<const T, DIM> sA(src);
  PackedTensorAccessor<T, DIM> sC(dest);

  constexpr int blockSize = 256;
  int64_t numel64 = src.getNumEl();
  CHECK(numel64 < std::numeric_limits<int>::max());
  int numel = static_cast<int>(numel64);
  dim3 grid = getGrid1D(numel, blockSize);

  copyNDKernel<T, DIM><<<grid, blockSize>>>(sA, sC, numel);
  LL_CUDA_SYNCHRONIZE();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

void copy(const Tensor &src, Tensor &dest) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(dest.getDevice().getType() == Device::kCuda);

  if (src.getDType() == DType::kFloat16 && src.getDim() == 5) return copyND<half, 5>(src, dest);
  if (src.getDType() == DType::kFloat16 && src.getDim() == 4) return copyND<half, 4>(src, dest);
  if (src.getDType() == DType::kFloat16 && src.getDim() == 3) return copyND<half, 3>(src, dest);
  if (src.getDType() == DType::kFloat16 && src.getDim() == 2) return copyND<half, 2>(src, dest);
  if (src.getDType() == DType::kUInt8 && src.getDim() == 5) return copyND<UInt8, 5>(src, dest);
  if (src.getDType() == DType::kUInt8 && src.getDim() == 4) return copyND<UInt8, 4>(src, dest);
  if (src.getDType() == DType::kUInt8 && src.getDim() == 3) return copyND<UInt8, 3>(src, dest);
  if (src.getDType() == DType::kUInt8 && src.getDim() == 2) return copyND<UInt8, 2>(src, dest);
  if (src.getDType() == DType::kLong && src.getDim() == 5) return copyND<LongType, 5>(src, dest);
  if (src.getDType() == DType::kLong && src.getDim() == 4) return copyND<LongType, 4>(src, dest);
  if (src.getDType() == DType::kLong && src.getDim() == 3) return copyND<LongType, 3>(src, dest);
  if (src.getDType() == DType::kLong && src.getDim() == 2) return copyND<LongType, 2>(src, dest);

  NOT_IMPL();
}

void copyContig(const Tensor &src, Tensor &dest) {
  LL_CHECK_CUDA_STATUS(cudaMemcpy(
      getDataPtrCuda<void>(dest),
      getDataPtrCuda<void>(src),
      src.getDType().getTotalSize(src.getNumEl()),
      cudaMemcpyDeviceToDevice));
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
