// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

#include "flint/cuda/common.h"
#include "flint/cuda/store_kv_cache.h"

namespace fl {
namespace op {
namespace cuda {

__global__ void storeKVCacheHalfKernel(
    const half *__restrict__ k,
    const half *__restrict__ v,
    const int32_t *__restrict__ slotMapping,
    half *__restrict__ keyCache,
    half *__restrict__ valueCache,
    int kTokenStride,
    int vTokenStride,
    int numElementsPerToken,
    int keyCacheBlockStride,
    int keyCacheTokenStride,
    int valueCacheBlockStride,
    int valueCacheTokenStride,
    int cacheBlockShift,
    int cacheBlockMask,
    int numCacheSlots) {
  int token = blockIdx.x;
  __shared__ int keyCacheOffset;
  __shared__ int valueCacheOffset;

  if (threadIdx.x == 0) {
    int slot = slotMapping[token];
    assert(slot >= 0 && slot < numCacheSlots);
    int blockId = slot >> cacheBlockShift;
    int offset = slot & cacheBlockMask;
    keyCacheOffset = blockId * keyCacheBlockStride + offset * keyCacheTokenStride;
    valueCacheOffset = blockId * valueCacheBlockStride + offset * valueCacheTokenStride;
  }
  __syncthreads();

  int kOffset = token * kTokenStride;
  int vOffset = token * vTokenStride;
  for (int i = threadIdx.x; i < numElementsPerToken; i += blockDim.x) {
    keyCache[keyCacheOffset + i] = k[kOffset + i];
    valueCache[valueCacheOffset + i] = v[vOffset + i];
  }
}

void storeKVCache(
    const Tensor &k,
    const Tensor &v,
    Tensor &keyCache,
    Tensor &valueCache,
    const Tensor &slotMapping) {
  CHECK(k.getDevice().getType() == Device::kCuda);
  CHECK(v.getDevice().getType() == Device::kCuda);
  CHECK(keyCache.getDevice().getType() == Device::kCuda);
  CHECK(valueCache.getDevice().getType() == Device::kCuda);
  CHECK(slotMapping.getDevice().getType() == Device::kCuda);
  CHECK(k.getDType() == DType::kFloat16 && v.getDType() == DType::kFloat16);
  CHECK(keyCache.getDType() == DType::kFloat16 && valueCache.getDType() == DType::kFloat16);
  CHECK(slotMapping.getDType() == DType::kInt32);
  CHECK(k.getDim() == 3 && v.getDim() == 3);
  CHECK(keyCache.getDim() == 4 && valueCache.getDim() == 4);
  CHECK(slotMapping.getDim() == 1);

  int numTokens = k.getShape(0);
  int numKeyValueHeads = k.getShape(1);
  int headDim = k.getShape(2);
  CHECK(v.getShape(0) == numTokens && v.getShape(1) == numKeyValueHeads);
  CHECK(v.getShape(2) == headDim);
  CHECK(slotMapping.getShape(0) == numTokens);
  CHECK(keyCache.getShape(2) == numKeyValueHeads && keyCache.getShape(3) == headDim);
  CHECK(valueCache.getShape() == keyCache.getShape());
  CHECK(k.getStride(2) == 1 && k.getStride(1) == headDim);
  CHECK(v.getStride(2) == 1 && v.getStride(1) == headDim);
  CHECK(keyCache.getStride(3) == 1 && keyCache.getStride(2) == headDim);
  CHECK(valueCache.getStride(3) == 1 && valueCache.getStride(2) == headDim);
  CHECK(slotMapping.getStride(0) == 1);
  if (numTokens == 0) return;

  constexpr int numThreads = 256;
  int cacheBlockSize = keyCache.getShape(1);
  CHECK((cacheBlockSize & (cacheBlockSize - 1)) == 0) << "cache block size must be a power of two";
  int cacheBlockShift = 0;
  while ((1 << cacheBlockShift) < cacheBlockSize) ++cacheBlockShift;
  storeKVCacheHalfKernel<<<numTokens, numThreads>>>(
      getDataPtrCuda<half>(k),
      getDataPtrCuda<half>(v),
      getDataPtrCuda<int32_t>(slotMapping),
      getDataPtrCuda<half>(keyCache),
      getDataPtrCuda<half>(valueCache),
      k.getStride(0),
      v.getStride(0),
      numKeyValueHeads * headDim,
      keyCache.getStride(0),
      keyCache.getStride(1),
      valueCache.getStride(0),
      valueCache.getStride(1),
      cacheBlockShift,
      cacheBlockSize - 1,
      keyCache.getShape(0) * cacheBlockSize);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
