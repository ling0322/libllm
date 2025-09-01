// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

#include "lynn/cuda/common.h"
#include "lynn/cuda/fill.h"

namespace ly {
namespace op {
namespace cuda {

template<typename T>
__global__ void arangeKernel(T *__restrict__ out, int n, T start, T step) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = idx; i < n; i += stride) {
    out[i] = start + step * (T)i;
  }
}

Tensor arangeLong(LongType begin, LongType end, LongType step) {
  int64_t numel64 = (end - begin) / step;
  CHECK(numel64 < std::numeric_limits<int32_t>::max());
  int numel = static_cast<int>(numel64);

  Tensor tensor = createCudaTensorLong({numel});

  constexpr int blockSize = 256;
  dim3 grid = getGrid1D(numel, blockSize);

  arangeKernel<LongType>
      <<<grid, blockSize>>>(tensor.getInternalData()->getData<LongType>(), numel, begin, step);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  return tensor;
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
