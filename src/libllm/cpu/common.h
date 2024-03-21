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

#include "libllm/tensor.h"
#include "libllm/cpu/accessor.h"
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/lut/span.h"

namespace libllm {
namespace op {
namespace cpu {

Tensor expandBatchDims(const Tensor &input, lut::Span<const Tensor::ShapeType> shape);
bool isShapeMatch(const Tensor &A, const Tensor &B);

template<typename T>
void copyVector(TensorAccessor<T, 1> dest, TensorAccessor<const T, 1> src) {
  CHECK(dest.getShape(0) == src.getShape(0));
  for (int i = 0; i < src.getShape(0); ++i) {
    dest[i] = src[i];
  }
}

template<typename T>
void applyDequant(int64_t offset, int n, const TensorData *data, float *tgt);

template<>
inline void applyDequant<Q4>(
  int64_t offset, int n, const TensorData *data, float *tgt) {
  kernel::dequantQ4(
      n,
      (const kernel::UInt4x2 *)data->getData<Q4>(),
      (const kernel::Float16 *)data->getSlot(1)->getData<Float16>(),
      (const kernel::UInt4x2 *)data->getSlot(2)->getData<UInt8>(),
      offset,
      tgt);
}

}  // cpu
}  // op
}  // ly
