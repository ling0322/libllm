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

#include "libllm/tensor.h"

#include <stdlib.h>
#include <limits>
#include "libllm/cpu/view.h"
#include "libllm/cpu/cpu_tensor_data.h"
#include "libllm/functional.h"
#include "libllm/lut/error.h"
#include "libllm/lut/strings.h"

namespace libllm {

template<typename T>
Tensor Tensor::create(std::initializer_list<int> shape, lut::Span<const T> data) {
  Tensor tensor;

  tensor._shape = std::make_shared<TensorShape>(shape);
  int64_t numel = tensor._shape->getNumEl();

  DType dtype = DType::getType<T>();
  tensor._data = op::cpu::CpuTensorData::create(numel, dtype);
  tensor._offset = 0;

  // fill data
  CHECK(numel == data.size()) << "data size and shape mismatch";
  std::copy(data.begin(), data.end(), tensor.getData<T>());

  return tensor;
}

template Tensor Tensor::create(std::initializer_list<int> shape, lut::Span<const float> data);
template Tensor Tensor::create(std::initializer_list<int> shape, lut::Span<const LongType> data);


Tensor Tensor::create(std::shared_ptr<TensorShape> shape,
                      std::shared_ptr<TensorData> data,
                      int64_t offset) {

  Tensor tensor;
  tensor._shape = shape;
  tensor._data = data;
  tensor._offset = offset;

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

void Tensor::read(lut::Reader *fp) {
  std::string s = fp->readString(4);
  if (s != "tnsr") {
    throw lut::AbortedError("bad tensor format");
  }

  _shape = TensorShape::read(fp);
  _data = op::cpu::CpuTensorData::read(fp);
  _offset = 0;

  // check
  if (_shape->getNumEl() != _data->getNumEl())
    throw lut::AbortedError("tensor data and shape mismatch.");
}

Tensor Tensor::view(lut::Span<const int> shape) const {
  return op::cpu::view(*this, shape);
}

Tensor Tensor::expand(lut::Span<const int> shape) const {
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

void Tensor::throwIfInvalidShape(lut::Span<const int> shape) const {
  if (shape.size() != getDim()) {
    throw lut::AbortedError(lut::sprintf(
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

    throw lut::AbortedError(lut::sprintf(
        "invalid shape: %s expected, but %s found.", expected.str(), actual.str()));
  }
}

const TensorData *Tensor::getDataObject() const {
  return _data.get();
}

int Tensor::getDim() const {
  return _shape->getDim();
}

int Tensor::getShape(int d) const {
  return _shape->getShape(d);
}

const TensorShape *Tensor::getShape_() const {
  return _shape.get();
}

bool Tensor::empty() const {
  return !_shape;
}

Device Tensor::getDevice() const {
  return _data->getDevice();
}

Tensor::ShapeType Tensor::getStride(int d) const {
  return _shape->getStride(d);
}

int64_t Tensor::getNumEl() const {
  return _shape->getNumEl();
}

int64_t Tensor::getOffset_() const {
  return _offset;
}

std::shared_ptr<TensorData> Tensor::getDataShared_() const {
  return _data;
}


// -----------------------------------------------------------------------------------------------+
// TensorShaoe                                                                                    |
// -----------------------------------------------------------------------------------------------+

TensorShape::TensorShape(const TensorShape &size) : _data(size._data.copy()) {}
TensorShape::TensorShape(TensorShape &&size) noexcept : _data(std::move(size._data)) {}
TensorShape &TensorShape::operator=(const TensorShape &size) {
  _data = size._data.copy();
  return *this;
}
TensorShape &TensorShape::operator=(TensorShape &&size) noexcept {
  _data = std::move(size._data);
  return *this;
}

TensorShape::TensorShape(lut::Span<const ShapeType> shape) {
  _data = lut::FixedArray<Elem>(shape.size());
  lut::FixedArray<Elem>::iterator it = _data.begin();
  for (int n : shape) {
    it->shape = n;
    ++it;
  }

  int64_t stride = 1;
  for (int d = static_cast<int>(shape.size()) - 1; d >= 0; --d) {
    CHECK(stride < std::numeric_limits<ShapeType>::max());
    _data[d].stride = static_cast<ShapeType>(stride);
    stride *= _data[d].shape;
  }
}

TensorShape::TensorShape(lut::Span<const Elem> shape) {
  _data = lut::FixedArray<Elem>(shape.size());
  std::copy(shape.begin(), shape.end(), _data.begin());
}

std::shared_ptr<TensorShape> TensorShape::subsize(int d) const {
  CHECK(d < getDim());

  std::shared_ptr<TensorShape> subsize{new TensorShape()};
  subsize->_data = lut::FixedArray<Elem>(getDim() - d);
  std::copy(_data.begin() + d, _data.end(), subsize->_data.begin());

  return subsize;
}

std::shared_ptr<TensorShape> TensorShape::read(lut::Reader *fp) {
  // rank
  int16_t rank = fp->readValue<int16_t>();
  if (rank > 16 || rank < 0) {
    throw lut::AbortedError("invalid rank.");
  }

  // shape
  std::vector<ShapeType> shape;
  for (int16_t d = 0; d < rank; ++d) {
    int32_t size = fp->readValue<int32_t>();
    if (size >= 1048576 || size <= 0) throw lut::AbortedError("invalid size in shape.");

    shape.push_back(size);
  }

  return std::make_shared<TensorShape>(lut::makeConstSpan(shape));
}

std::shared_ptr<TensorShape> TensorShape::transpose(int dim0, int dim1) const {
  dim0 = getRealDim(dim0);
  dim1 = getRealDim(dim1);

  std::shared_ptr<TensorShape> size = std::make_shared<TensorShape>(*this);
  Elem dim0_elem = size->_data[dim0];
  size->_data[dim0] = size->_data[dim1];
  size->_data[dim1] = dim0_elem;

  return size;
}

std::shared_ptr<TensorShape> TensorShape::squeeze(int dim) const {
  CHECK(getShape(dim) == 1);

  dim = getRealDim(dim);
  std::shared_ptr<TensorShape> size{new TensorShape()};
  size->_data = lut::FixedArray<Elem>(getDim() - 1);
  for (int d = 0; d < dim; ++d) {
    size->_data[d] = _data[d];
  }
  for (int d = dim + 1; d < getDim(); ++d) {
    size->_data[d - 1] = _data[d];
  }

  return size;
}

std::shared_ptr<TensorShape> TensorShape::unsqueeze(int dim) const {
  dim = getRealDim(dim);

  std::shared_ptr<TensorShape> size{new TensorShape()};
  size->_data = lut::FixedArray<Elem>(getDim() + 1);
  for (int d = 0; d < dim; ++d) {
    size->_data[d] = _data[d];
  }
  size->_data[dim].shape = 1;
  size->_data[dim].stride = dim == 0 ? getStride(0) * getShape(0) : getStride(dim - 1);
  for (int d = dim; d < getDim(); ++d) {
    size->_data[d + 1] = _data[d];
  }

  return size;
}

int TensorShape::getRealDim(int d) const {
  CHECK(!empty());
  int rank = getDim();
  if (d < 0) {
    d = rank + d;
  }

  CHECK(d >= 0 && d < rank);
  return d;
}

int TensorShape::getRealIndex(int dim, int index) const {
  CHECK(!empty());
  dim = getRealDim(dim);

  int shape = _data[dim].shape;
  index = index >= 0 ? index : shape + index;

  CHECK(index >= 0 && index <= shape);
  return index;
}

int TensorShape::getDim() const {
  return static_cast<int>(_data.size());
}

bool TensorShape::empty() const {
  return _data.empty();
}

int TensorShape::getShape(int d) const {
  return _data[getRealDim(d)].shape;
}

int TensorShape::getStride(int d) const {
  return _data[getRealDim(d)].stride;
}

int64_t TensorShape::getNumEl() const {
  if (empty()) {
    return 0;
  }
  
  int64_t n = 1;
  for (const Elem &elem : _data) {
    n *= elem.shape;
  }
  return n;
}

void TensorShape::setShape(int dim, ShapeType shape) {
  dim = getRealDim(dim);
  CHECK(dim >= 0 && dim <= this->getDim());
  CHECK(shape <= _data[dim].shape);

  _data[dim].shape = shape;
}

std::shared_ptr<TensorShape> TensorShape::expand(lut::Span<const int> shape) const {
  CHECK(getDim() == shape.size());
  std::shared_ptr<TensorShape> view = std::make_shared<TensorShape>(lut::makeConstSpan(_data));
  int dim = getDim();
  for (int d = 0; d < dim; ++d) {
    if (shape[d] != getShape(d)) {
      CHECK(getShape(d) == 1) << "unable to expand a non-singleton dimension (size > 1).";
      view->_data[d].shape = shape[d];
      view->_data[d].stride = 0;
    }
  }

  return view;
}

std::string TensorShape::toString() const {
  std::ostringstream os;
  bool first = true;

  os << "(";
  for (Elem elem : _data) {
    if (first) {
      first = false;
    } else {
      os << ", ";
    }
    os << elem.shape;
  }
  os << ")";
  return os.str();
}

}  // namespace libllm
