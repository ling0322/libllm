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
#include <limits>
#include "llyn/device.h"
#include "llyn/dtype.h"
#include "llyn/internal/tensor_data.h"
#include "llyn/internal/tensor_shape.h"
#include "lyutil/fixed_array.h"
#include "lyutil/reader.h"
#include "lyutil/span.h"

namespace llyn {

namespace cpu {
class Internal;
class CPUOperators;
}  // namespace cpu

// Used in Tensor::index.
constexpr int None = std::numeric_limits<int>::min();
struct Index {
  int first;
  int second;
  constexpr Index(int index);
};
constexpr Index Slice(int start, int stop);

class Tensor {
 public:
  friend class cpu::CPUOperators;
  friend class cpu::Internal;

  // integer type for shape and stride
  typedef internal::TensorShape::ShapeType ShapeType;

  // rank for empty tansor.
  static constexpr int kEmptyRank = -1;

  // create a tensor in CPU storage. Size of `data` should be the same as `shape.numel()`.
  // Example:
  //   Tensor x = Tensor::FromData({2, 2}, {1.0f, 0.8f, 0.6f, 0.2f});
  template<typename T>
  static Tensor create(std::initializer_list<int> shape, ly::Span<const T> data);

  // constructor and destructor.
  Tensor();
  ~Tensor();

  // Read the tensor from fp.
  void read(ly::ReadableFile *fp);

  // copy and move constructors.
  Tensor(const Tensor &tensor);
  Tensor &operator=(const Tensor &tensor);
  Tensor(Tensor &&tensor) noexcept;
  Tensor &operator=(Tensor &&tensor);

  // get numebr of dimentsions.
  int getDim() const { return _shape->getDim(); }

  // get the size in dimention `d`. `d` supports positive number (index) and negative number (index
  // from back). Crash if `d` is out of boundary
  ShapeType getShape(int d) const { return _shape->getShape(d); }
  std::vector<int> getShape() const;
  std::string getShapeString() const;

  // get stride for dimension `d`. 
  ShapeType getStride(int d) const { return _shape->getStride(d); }

  // get number of elements in this tensor.
  int64_t getNumEl() const { return _shape->getNumEl(); }

  // return true if this tensor is empty.
  bool empty() const { return !_shape; }

  // get data type.
  DType getDType() const;

  // Get a new view of the tensor..
  Tensor view(ly::Span<const int> shape) const;

  // Get a new view of the tensor with singleton dimensions expanded to a larger size.
  Tensor expand(ly::Span<const int> shape) const;

  // Get slice of this tensor. `dim` is the dimension to slice. [begin, end) is the range. For
  // [begin, end) only version, dimension 0 is used. Negative `begin` and `end` is accepted. Crash
  // if dim or range out of boundary.
  // None could be used in both begin and end. (None, 5) means [: 5], (5, None) means [5: ].
  Tensor slice(int dim, std::pair<int, int> range) const;
  Tensor slice(std::pair<int, int> range) const;

  // Get subtensor at specified index of first dimension. Negative `index` is accepted. Crash if
  // `index` out of boundary.
  Tensor subtensor(int index) const;

  // add or remove an additional shape=1 dimension at specified position.
  Tensor unsqueeze(int dim) const;
  Tensor squeeze(int dim) const;

  Tensor transpose(int dim0, int dim1) const;

  // return true if the tensor is contigous.
  bool isContiguous() const;

  // pointer of data in this tensor
  template<typename T>
  T *getData() { 
    return _data->getData<0, T>(_offset);
  }
  template<typename T>
  const T *getData() const {
    return _data->getData<0, T>(_offset);
  }

  // get the internal TensorData object.
  const internal::TensorData *getDataObject() const { return _data.get(); }

  // return specific element at index. Size of `indices` should be the same as tensor dimension.
  // And the data should in CPU.
  template<typename T>
  T getElem(ly::Span<const int> indices);

  // Check the shape of a tensor. If shape of `tensor` does not match `shape`, return AbortedError
  // with message "invalid shape".
  void throwIfInvalidShape(std::initializer_list<int> shape);

 protected:
  std::shared_ptr<internal::TensorData> _data;
  std::shared_ptr<internal::TensorShape> _shape;
  int64_t _offset;
};

inline DType Tensor::getDType() const { 
  return _data ? _data->getDType() : DType(DType::kUnknown);
}

template<typename T>
inline T Tensor::getElem(ly::Span<const int> indices) {
  CHECK(indices.size() == getDim());

  const T *data = this->getData<T>();
  int64_t offset = 0;
  for (int d = 0; d < getDim(); ++d) {
    offset += indices[d] * getStride(d);
  }

  return data[offset];
}

namespace internal {
constexpr int Unused = std::numeric_limits<int>::min() + 1;
}  // namespace internal

constexpr Index::Index(int index) : first(index), second(internal::Unused) {}
constexpr Index Slice(int start, int stop) {
  Index index(start);
  index.second = stop;
  return index;
}


typedef const Tensor &TensorCRef;

}  // namespace llyn
