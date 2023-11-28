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
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < qtensor.getNumCol() / 2 && y < qtensor.getNumRow()) {
    int rowGroup = y * qtensor.getNumCol() / 32;
    int rowOffset = y * qtensor.getNumCol() / 2;
    int elemOffset = rowOffset + x;
    int elemGroup = rowGroup + x / 16;

    uint8_t q4elem = qtensor.getData()[elemOffset];
    half scale = qtensor.getScale()[elemGroup];
    int8_t bias = qtensor.getBias()[elemGroup];

    destTensor[y][x * 2] = __hmul(scale, __int2half_rd(static_cast<int>(q4elem >> 4) - bias));
    destTensor[y][x * 2+ 1] = __hmul(scale, __int2half_rd(static_cast<int>(q4elem & 0xf) - bias));
  }
}

Tensor dequantQ4ToHalf(const Tensor &qtensor) {
  CHECK(qtensor.getDType() == DType::kQInt4Group32);

  std::vector<Tensor::ShapeType> shape = qtensor.getShape();
  Tensor dst = createCudaTensorHalf(shape);

  constexpr int blockSize = 256;
  dim3 d;
  d.y = dst.getShape(0);
  d.x = (dst.getShape(1) / 2 + blockSize - 1) / blockSize;

  PackedSubtensor2DQ4 sA(qtensor);
  PackedSubtensor<half, 2> sC(dst);

  dequantTensor2DQ4<<<d, blockSize>>>(sA, sC);
  LL_CHECK_CUDA_STATUS(cudaGetLastError());
  return dst;
}

}  // cuda
}  // op
}  // llyn
    
