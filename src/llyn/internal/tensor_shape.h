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

#include <stdint.h>
#include <string>
#include "lyutil/fixed_array.h"
#include "lyutil/reader.h"
#include "lyutil/span.h"

namespace llyn {

namespace cpu {
class Internal;
class CPUOperators;
}  // namespace cpu

namespace internal {

// Stores shape and stride of a Tensor.
class TensorShape {
 public:
  friend class cpu::Internal;
  friend class cpu::CPUOperators;

  typedef int32_t ShapeType;
  struct Elem {
    ShapeType shape;
    ShapeType stride;
  };

  // read tensor shape from file.
  static std::shared_ptr<TensorShape> read(ly::ReadableFile *fp);

  // from shape.
  TensorShape(ly::Span<const ShapeType> shape);
  TensorShape(ly::Span<const Elem> shape);

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
  std::shared_ptr<TensorShape> expand(ly::Span<const int> shape) const;

  // convert negative dimension or index (in specific `dim`) to positive.
  int getRealDim(int dim) const;
  int getRealIndex(int dim, int index) const;

  std::string toString() const;

 private:
  ly::FixedArray<Elem> _data;

  // an empty Tensor.
  TensorShape() = default;
};

}  // namespace internal
}  // namespace llyn
