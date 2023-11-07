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

#include "llyn/tensor.h"

#include <stdlib.h>
#include <limits>
#include "llyn/cpu/view.h"
#include "lyutil/error.h"
#include "lyutil/strings.h"

namespace llyn {

using internal::TensorData;
using internal::TensorShape;

template<typename T>
Tensor Tensor::create(std::initializer_list<int> shape, ly::Span<const T> data) {
  Tensor tensor;

  tensor._shape = std::make_shared<TensorShape>(shape);
  int64_t numel = tensor._shape->getNumEl();

  DType dtype = DType::getType<T>();
  tensor._data = TensorData::create(numel, dtype);
  tensor._offset = 0;

  // fill data
  CHECK(numel == data.size()) << "data size and shape mismatch";
  std::copy(data.begin(), data.end(), tensor.getData<T>());

  return tensor;
}

template Tensor Tensor::create(std::initializer_list<int> shape, ly::Span<const float> data);
template Tensor Tensor::create(std::initializer_list<int> shape, ly::Span<const LongType> data);

Tensor Tensor::create(ly::Span<const int> shape, std::shared_ptr<internal::TensorData> data) {
  Tensor tensor;

  tensor._shape = std::make_shared<TensorShape>(shape);
  tensor._data = data;

  return tensor;
}

Tensor::Tensor() : _offset(0) {}
Tensor::~Tensor() {}

Tensor::Tensor(const Tensor &tensor) {
  _data = tensor._data;
  _shape = tensor._shape;
  _offset = tensor._offset;
}

Tensor &Tensor::operator=(const Tensor &tensor) {
  _data = tensor._data;
  _shape = tensor._shape;
  _offset = tensor._offset;

  return *this;
}

Tensor::Tensor(Tensor &&tensor) noexcept {
  _data = tensor._data;
  _shape = std::move(tensor._shape);
  _offset = tensor._offset;
}

Tensor &Tensor::operator=(Tensor &&tensor) {
  _data = tensor._data;
  _shape = std::move(tensor._shape);
  _offset = tensor._offset;

  return *this;
}

void Tensor::read(ly::ReadableFile *fp) {
  std::string s = fp->readString(4);
  if (s != "tnsr") {
    throw ly::AbortedError("bad tensor format");
  }

  _shape = TensorShape::read(fp);
  _data = TensorData::read(fp);
  _offset = 0;

  // check
  if (_shape->getNumEl() != _data->getNumEl())
    throw ly::AbortedError("tensor data and shape mismatch.");
  if (_data->getDType().isQuantized()) {
    if (_data->getNumEl() / _data->getDType().getGroupSize() != _data->getNumEl1())
      throw ly::AbortedError("tensor data and shape mismatch (scale).");
  }
}

Tensor Tensor::view(ly::Span<const int> shape) const {
  return cpu::view(*this, shape);
}

Tensor Tensor::expand(ly::Span<const int> shape) const {
  CHECK(!getDType().isQuantized());
  Tensor x;
  x._data = _data;
  x._offset = _offset;
  x._shape = _shape->expand(shape);

  return x;
}

std::vector<int> Tensor::getShape() const {
  std::vector<int> shape;
  for (int d = 0; d < getDim(); ++d) {
    shape.push_back(getShape(d));
  }
  return shape;
}

std::string Tensor::getShapeString() const {
  return _shape->toString();
}

bool Tensor::isContiguous() const {
  int numel = 1;
  for (int i = getDim() - 1; i >= 0; --i) {
    if (numel != getStride(i) && getShape(i) != 1) return false;
    numel *= getShape(i);
  }

  return true;
}

Tensor Tensor::slice(int dim, std::pair<int, int> range) const {
  CHECK(!getDType().isQuantized());

  dim = _shape->getRealDim(dim);
  CHECK(dim >= 0 && dim < this->getDim());

  int begin = range.first;
  int end = range.second;

  if (begin == None) begin = 0;
  if (end == None) end = getShape(dim);

  begin = _shape->getRealIndex(dim, begin);
  end = _shape->getRealIndex(dim, end);
  CHECK(begin >= 0 && begin < end && end <= getShape(dim));

  Tensor tensor;
  tensor._data = _data;
  tensor._shape = std::make_shared<TensorShape>(*_shape);
  tensor._shape->setShape(dim, end - begin);
  tensor._offset = _offset + _shape->getStride(dim) * begin;

  return tensor;
}

Tensor Tensor::slice(std::pair<int, int> range) const {
  CHECK(!getDType().isQuantized());

  int begin = range.first;
  int end = range.second;
  return slice(0, {begin, end});
}

Tensor Tensor::subtensor(int index) const {
  CHECK(!getDType().isQuantized());

  index = _shape->getRealIndex(0, index);
  CHECK(index >= 0 && index < getShape(0));

  Tensor tensor;
  tensor._data = _data;
  tensor._shape = _shape->subsize(1);
  tensor._offset = _offset + _shape->getStride(0) * index;

  return tensor;
}

Tensor Tensor::transpose(int dim0, int dim1) const {
  Tensor tensor;
  tensor._data = _data;
  tensor._offset = _offset;
  tensor._shape = _shape->transpose(dim0, dim1);

  return tensor;
}

Tensor Tensor::unsqueeze(int dim) const {
  Tensor tensor;
  tensor._data = _data;
  tensor._offset = _offset;
  tensor._shape = _shape->unsqueeze(dim);

  return tensor;
}

Tensor Tensor::squeeze(int dim) const {
  Tensor tensor;
  tensor._data = _data;
  tensor._offset = _offset;
  tensor._shape = _shape->squeeze(dim);

  return tensor;
}

void Tensor::throwIfInvalidShape(std::initializer_list<int> shape) {
  if (shape.size() != getDim()) {
    throw ly::AbortedError(ly::sprintf(
        "invalid shape. dim=%d expected, but %d got.", shape.size(), getDim()));
  }

  int i = 0;
  bool correct = true;
  for (int s : shape) {
    if (this->getShape(i) != s) {
      correct = false;
    }
    ++i;
  }

  if (!correct) {
    std::ostringstream actual;
    actual << "(";
    for (int i = 0; i < getDim(); ++i) {
      if (i) actual << ", ";
      actual << this->getShape(i);
    }
    actual << ")";

    std::ostringstream expected;
    bool first = true;
    expected << "(";
    for (int s : shape) {
      if (!first) expected << ", ";
      expected << s;
      first = false;
    }
    expected << ")";

    throw ly::AbortedError(ly::sprintf(
        "invalid shape: %s expected, but %s found.", expected.str(), actual.str()));
  }
}

}  // namespace llyn
