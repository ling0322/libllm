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

#pragma once

#include <stdint.h>

#include <limits>
#include <memory>

#include "lutil/fixed_array.h"
#include "lutil/reader.h"
#include "lutil/span.h"
#include "lynn/device.h"
#include "lynn/dtype.h"
#include "lynn/functional.h"

namespace ly {

// Stores shape and stride of a Tensor.
class TensorShape {
 public:
  typedef int32_t ShapeType;
  struct Elem {
    ShapeType shape;
    ShapeType stride;
  };

  // read tensor shape from file.
  static std::shared_ptr<TensorShape> read(lut::Reader *fp);

  // from shape.
  TensorShape(lut::Span<const ShapeType> shape);
  TensorShape(lut::Span<const Elem> shape);

  TensorShape(const TensorShape &size);
  TensorShape(TensorShape &&size) noexcept;
  TensorShape &operator=(const TensorShape &size);
  TensorShape &operator=(TensorShape &&size) noexcept;

  bool empty() const;
  int getDim() const;
  ShapeType getShape(int index) const;
  ShapeType getStride(int index) const;
  int64_t getNumEl() const;

  // Returns a sub-Size starting at specified dimension.
  std::shared_ptr<TensorShape> subsize(int d) const;

  // Returns a Size that is a transposed version of current size. The given
  // dimensions dim0 and dim1 are swapped.
  std::shared_ptr<TensorShape> transpose(int dim0, int dim1) const;

  // add or remove one shape=1 dimension at specified dimension.
  std::shared_ptr<TensorShape> unsqueeze(int dim) const;
  std::shared_ptr<TensorShape> squeeze(int dim) const;

  // set the value of shape(dim). Negative dim is allowed. new `shape` should be less or equal to
  // current size.
  void setShape(int dim, ShapeType shape);

  // return a new shape that expand singleton dimensions to a larger size.
  std::shared_ptr<TensorShape> expand(lut::Span<const int> shape) const;

  // convert negative dimension or index (in specific `dim`) to positive.
  int getRealDim(int dim) const;
  int getRealIndex(int dim, int index) const;

  lut::Span<const Elem> getData_() const {
    return lut::makeConstSpan(_data);
  }

  std::string toString() const;

 private:
  lut::FixedArray<Elem> _data;

  // an empty Tensor.
  TensorShape() = default;
};

}  // namespace ly
