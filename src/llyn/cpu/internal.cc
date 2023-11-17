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

#include "llyn/cpu/internal.h"

#include "llyn/internal/cpu_tensor_data.h"
#include "llyn/internal/tensor_data.h"
#include "llyn/internal/tensor_shape.h"
#include "llyn/cpu/subtensor.h"
#include "lyutil/fixed_array.h"

namespace llyn {
namespace cpu {

using internal::TensorData;
using internal::TensorShape;

Tensor Internal::tensor(ly::Span<const int> shape, DType dtype) {
  CHECK(!dtype.isQuantized()) << "unable to create quantized tensor directly";
  Tensor x;

  x._shape = std::make_shared<TensorShape>(ly::makeConstSpan(shape));
  int64_t numel = x._shape->getNumEl();

  x._data = internal::CpuTensorData::create(numel, dtype);
  x._offset = 0;

  return x;
}

ly::Span<const Shape> Internal::getShapeData(const Tensor &input) {
  return ly::makeConstSpan(input._shape->_data);
}

Tensor Internal::tensorView(const Tensor &input, std::shared_ptr<TensorShape> shape) {
  Tensor x;
  x._data = input._data;
  x._offset = input._offset;
  x._shape = shape;

  return x;
}

}  // cpu
}  // flint

