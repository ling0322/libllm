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

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>
#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "llyn/tensor.h"
#include "llyn/internal/tensor_shape.h"
#include "llyn/operators/cuda/cuda_tensor_data.h"
#include "llyn/operators/cuda/subtensor.h"

#define LL_CHECK_CONTIGUOUS(x) { if (!x.isContiguous()) { \
    LOG(FATAL) << "contiguous is required for CUDA operators: " << #x; } }

namespace llyn {
namespace op {
namespace cuda {

/// @brief A q4 quantized constant matrix (2D tensor).
struct PackedSubtensor2DQ4 {
  int numRow;
  int numCol;

  const half *scale;
  const uint8_t *data;
  const int8_t *bias;

  PackedSubtensor2DQ4(const Tensor &tensor);
};

Tensor createCudaTensorHalf(ly::Span<const int> shape);
Tensor createCudaTensorLong(ly::Span<const int> shape);
Tensor createCudaTensorFloat(ly::Span<const int> shape);

void checkCudaError(cudaError_t err);

/// @brief Split a index into dim3 object according to the shape info in `size`.
/// @param index the index to split.
/// @param size the shape info. it should have at least 3 elements. size[0] is the shape and stride
//              info for axis `z`, size[1] for `y` and size[2] for `x`.
/// @return the dim3 object.
__device__ inline dim3 splitIndexToDim3(unsigned int index, const Size *size) {
  dim3 d;
  d.x = index % size[2].shape;
  d.y = (index / size[2].shape) % size[1].shape;
  d.z = index / (size[1].shape * size[2].shape);

  return d;
}

}  // cuda
}  // op
}  // llyn
