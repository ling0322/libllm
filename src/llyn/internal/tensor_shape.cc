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

#include "llyn/internal/tensor_shape.h"

#include <limits>
#include <sstream>
#include "lyutil/error.h"
#include "lyutil/log.h"
#include "lyutil/reader.h"
#include "lyutil/span.h"

namespace llyn {
namespace internal {

// -- class TensorShape ----------

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

TensorShape::TensorShape(ly::Span<const ShapeType> shape) {
  _data = ly::FixedArray<Elem>(shape.size());
  ly::FixedArray<Elem>::iterator it = _data.begin();
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

TensorShape::TensorShape(ly::Span<const Elem> shape) {
  _data = ly::FixedArray<Elem>(shape.size());
  std::copy(shape.begin(), shape.end(), _data.begin());
}

std::shared_ptr<TensorShape> TensorShape::subsize(int d) const {
  CHECK(d < getDim());

  std::shared_ptr<TensorShape> subsize{new TensorShape()};
  subsize->_data = ly::FixedArray<Elem>(getDim() - d);
  std::copy(_data.begin() + d, _data.end(), subsize->_data.begin());

  return subsize;
}

std::shared_ptr<TensorShape> TensorShape::read(ly::ReadableFile *fp) {
  // rank
  int16_t rank = fp->readValue<int16_t>();
  if (rank > 16 || rank < 0) {
    throw ly::AbortedError("invalid rank.");
  }

  // shape
  std::vector<ShapeType> shape;
  for (int16_t d = 0; d < rank; ++d) {
    int32_t size = fp->readValue<int32_t>();
    if (size >= 65536 || size <= 0) 
      throw ly::AbortedError("invalid size in shape.");

    shape.push_back(size);
  }

  return std::make_shared<TensorShape>(ly::makeConstSpan(shape));
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
  size->_data = ly::FixedArray<Elem>(getDim() - 1);
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
  size->_data = ly::FixedArray<Elem>(getDim() + 1);
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

std::shared_ptr<TensorShape> TensorShape::expand(ly::Span<const int> shape) const {
  CHECK(getDim() == shape.size());
  std::shared_ptr<TensorShape> view = std::make_shared<TensorShape>(ly::makeConstSpan(_data));
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

}  // namespace internal
}  // namespace llyn
