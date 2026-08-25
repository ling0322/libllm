// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cuda_fp16.h>

#include "flint/cuda/common.h"
#include "flint/cuda/rotary_embedding.h"

namespace fl {
namespace op {
namespace cuda {

__global__ void rotaryEmbeddingHalfKernel(
    const int64_t *__restrict__ positions,
    half *__restrict__ query,
    half *__restrict__ key,
    const half *__restrict__ rotaryCache,
    int queryTokenStride,
    int keyTokenStride,
    int numQueryHeads,
    int numKeyHeads,
    int headDim,
    int halfDimShift,
    int halfDimMask,
    int cacheStride,
    int maxPositions) {
  int token = blockIdx.x;
  int halfDim = headDim / 2;
  int pairsPerToken = (numQueryHeads + numKeyHeads) * halfDim;
  int position = static_cast<int>(positions[token]);
  assert(position >= 0 && position < maxPositions);

  const half *cache = rotaryCache + position * cacheStride;
  const half *cos = cache;
  const half *sin = cache + headDim;

  for (int pair = threadIdx.x; pair < pairsPerToken; pair += blockDim.x) {
    int head = pair >> halfDimShift;
    int dim = pair & halfDimMask;
    half *vector;
    if (head < numQueryHeads) {
      vector = query + token * queryTokenStride + head * headDim;
    } else {
      vector = key + token * keyTokenStride + (head - numQueryHeads) * headDim;
    }

    half first = vector[dim];
    half second = vector[dim + halfDim];
    vector[dim] = __hsub(__hmul(first, cos[dim]), __hmul(second, sin[dim]));
    vector[dim + halfDim] = __hadd(
        __hmul(second, cos[dim + halfDim]),
        __hmul(first, sin[dim + halfDim]));
  }
}

void rotaryEmbedding(
    const Tensor &positions,
    Tensor &query,
    Tensor &key,
  const Tensor &rotaryCache) {
  CHECK(positions.getDevice().getType() == Device::kCuda);
  CHECK(query.getDevice().getType() == Device::kCuda);
  CHECK(key.getDevice().getType() == Device::kCuda);
  CHECK(rotaryCache.getDevice().getType() == Device::kCuda);
  CHECK(positions.getDType() == DType::kLong);
  CHECK(query.getDType() == DType::kFloat16 && key.getDType() == DType::kFloat16);
  CHECK(rotaryCache.getDType() == DType::kFloat16);
  CHECK(positions.getDim() == 1);
  CHECK(query.getDim() == 3 && key.getDim() == 3);
  CHECK(rotaryCache.getDim() == 2);

  int numTokens = positions.getShape(0);
  int headDim = query.getShape(2);
  CHECK(key.getShape(0) == numTokens && query.getShape(0) == numTokens);
  CHECK(key.getShape(2) == headDim && headDim > 0 && headDim % 2 == 0);
  int halfDim = headDim / 2;
  CHECK((halfDim & (halfDim - 1)) == 0) << "half head dimension must be a power of two";
  CHECK(rotaryCache.getShape(1) == 2 * headDim);
  CHECK(positions.getStride(0) == 1);
  CHECK(query.getStride(2) == 1 && query.getStride(1) == headDim);
  CHECK(key.getStride(2) == 1 && key.getStride(1) == headDim);
  CHECK(rotaryCache.getStride(1) == 1);
  if (numTokens == 0) return;

  constexpr int NumThreads = 256;
  int halfDimShift = 0;
  while ((1 << halfDimShift) < halfDim) ++halfDimShift;
  rotaryEmbeddingHalfKernel<<<numTokens, NumThreads>>>(
      getDataPtrCuda<int64_t>(positions),
      getDataPtrCuda<half>(query),
      getDataPtrCuda<half>(key),
      getDataPtrCuda<half>(rotaryCache),
      query.getStride(0),
      key.getStride(0),
      query.getShape(1),
      key.getShape(1),
      headDim,
      halfDimShift,
      halfDim - 1,
      rotaryCache.getStride(0),
      rotaryCache.getShape(0));
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
}

}  // namespace cuda
}  // namespace op
}  // namespace fl
