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

#include "lten/cpu/common.h"

namespace lten {
namespace op {
namespace cpu {

Tensor expandBatchDims(const Tensor &input, lut::Span<const Tensor::ShapeType> shape) {
  CHECK(static_cast<int>(shape.size()) >= input.getDim());
  if (input.getDim() == static_cast<int>(shape.size())) return input;
  int nBroadcastDim = static_cast<int>(shape.size()) - input.getDim();

  Tensor x = input;
  std::vector<TensorShape::Elem> broadcastShape;
  for (int i = 0; i < nBroadcastDim; ++i) {
    TensorShape::Elem shapeElem;
    shapeElem.stride = 0;
    shapeElem.shape = shape[i];
    broadcastShape.push_back(shapeElem);
  }

  for (int i = 0; i < input.getDim(); ++i) {
    TensorShape::Elem shapeElem;
    shapeElem.stride = input.getStride(i);
    shapeElem.shape = input.getShape(i);
    broadcastShape.push_back(shapeElem);
  }

  return Tensor::create(
      std::make_shared<TensorShape>(broadcastShape),
      input.getDataShared_(),
      input.getOffset_());
}

}  // namespace cpu
}  // namespace op
}  // namespace lten
