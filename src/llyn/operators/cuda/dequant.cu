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

#include "llyn/operators/cuda/dequant.h"

#include <cuda_fp16.h>
#include "llyn/operators/cuda/common.h"

namespace llyn {
namespace op {
namespace cuda {

__global__
void dequantTensor2DQ4(PackedSubtensor2DQ4 qtensor,
                       PackedSubtensor<half, 2> destTensor) {
  int64_t numQ4x2 = qtensor.getNumCol() * qtensor.getNumRow() / 2;
  half *destData = destTensor.getData();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int elemOffset = idx; elemOffset < numQ4x2; elemOffset += gridDim.x * blockDim.x) {
    int elemGroup = elemOffset / 16;
    uint8_t q4elem = qtensor.getData()[elemOffset];
    half scale = qtensor.getScale()[elemGroup];
    int8_t bias = qtensor.getBias()[elemGroup];

    destData[elemOffset * 2] = __hmul(scale, __int2half_rd(static_cast<int>(q4elem >> 4) - bias));
    destData[elemOffset * 2 + 1] = __hmul(
        scale, __int2half_rd(static_cast<int>(q4elem & 0xf) - bias));
  }
}

Tensor dequantQ4ToHalf(const Tensor &qtensor) {
  CHECK(qtensor.getDType() == DType::kQInt4Group32);
  Tensor qT = qtensor;
  bool transQ = qtensor.getStride(0) == 1 && qtensor.getStride(1) != 1;
  if (transQ) qT = qT.transpose(0, 1);

  std::vector<Tensor::ShapeType> shape = qT.getShape();
  Tensor dst = createCudaTensorHalf(shape);

  constexpr int blockSize = 256;
  int maxThreadsPerSM = getCudaDeviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  int numSM = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  int numBlock = (qT.getNumEl() / 2 + blockSize - 1) / blockSize;
  int deviceNumBlock = maxThreadsPerSM * numSM / blockSize;
  LOG(INFO) << "numBlock = " << numBlock;
  LOG(INFO) << "deviceNumBlock = " << deviceNumBlock;

  dim3 grid;
  grid.x = std::min(numBlock, deviceNumBlock);

  PackedSubtensor2DQ4 sA(qT);
  PackedSubtensor<half, 2> sC(dst);

  dequantTensor2DQ4<<<grid, blockSize>>>(sA, sC);
  cudaDeviceSynchronize();
  LL_CHECK_CUDA_STATUS(cudaGetLastError());

  if (transQ) dst = dst.transpose(0, 1);
  return dst;
}

}  // cuda
}  // op
}  // llyn
