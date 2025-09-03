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

#include "lutil/span.h"
#include "lynn/cpu/kernel/interface.h"
#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cpu {

#if LUT_CPU_ARCH == LUT_AARCH64
typedef Float16 DefaultFloatType;
#else
typedef float DefaultFloatType;
#endif

Tensor expandBatchDims(const Tensor &input, lut::Span<const Tensor::ShapeType> shape);
bool isShapeMatch(const Tensor &A, const Tensor &B);

// Check whether the given tensor can become contiguous by applying a permutation(transpose) of its
// dimensions.
bool isLooselyContiguous(const Tensor &tensor);

template<typename T>
inline T *getDataPtrCpu(const Tensor &input) {
  return input.getInternalData()->getData<T>(input.getInternalOffset());
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
