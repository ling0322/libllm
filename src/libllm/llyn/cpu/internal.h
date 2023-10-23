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

#include <memory>
#include "llyn/internal/tensor_shape.h"
#include "llyn/internal/tensor_data.h"
#include "llyn/tensor.h"
#include "lyutil/span.h"

namespace llyn {
namespace cpu {

typedef internal::TensorShape::Elem Shape;
typedef Tensor::ShapeType ShapeType;

// Internal is a friend class of Tensor;
class Internal {
 public:
  // create a new tensor with given shape and dtype.
  static Tensor tensor(ly::Span<const int> shape, DType dtype);

  static ly::Span<const Shape> getShapeData(const Tensor &input);
  
  // return a new tensor with the data from input and a new shape. NOTE: this function will not
  // validate the shape.
  static Tensor tensorView(const Tensor &input, std::shared_ptr<internal::TensorShape> shape);
};

}  // cpu
}  // flint
