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

#include <vector>

#include "lutil/span.h"
#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cpu {

template<typename T, int DIM>
class TensorAccessorBase {
 public:
  explicit TensorAccessorBase(Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getInternalData()->getData<T>();
    _size = tensor.getInternalShape()->getData_().data();
  }

  explicit TensorAccessorBase(const Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getInternalData()->getData<T>();
    _size = tensor.getInternalShape()->getData_().data();
  }

  int getShape(int d) const {
    CHECK(d < DIM);
    return _size[d].shape;
  }

  T *getData() const {
    return _data;
  }

  TensorAccessorBase(const TensorShape::Elem *size, T *data)
      : _size(size),
        _data(data) {
  }

 protected:
  const TensorShape::Elem *_size;
  T *_data;
};

template<typename T, int DIM>
class TensorAccessor : public TensorAccessorBase<T, DIM> {
 public:
  TensorAccessor(Tensor &tensor)
      : TensorAccessorBase<T, DIM>(tensor) {
  }
  TensorAccessor(const Tensor &tensor)
      : TensorAccessorBase<T, DIM>(tensor) {
  }

  TensorAccessor(const TensorShape::Elem *size, T *data)
      : TensorAccessorBase<T, DIM>(size, data) {
  }

  TensorAccessor<T, DIM - 1> operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return TensorAccessor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
  const TensorAccessor<T, DIM - 1> operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return TensorAccessor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
};

template<typename T>
class TensorAccessor<T, 1> : public TensorAccessorBase<T, 1> {
 public:
  TensorAccessor(Tensor &tensor)
      : TensorAccessorBase<T, 1>(tensor) {
  }
  TensorAccessor(const Tensor &tensor)
      : TensorAccessorBase<T, 1>(tensor) {
  }

  TensorAccessor(const TensorShape::Elem *size, T *data)
      : TensorAccessorBase<T, 1>(size, data) {
  }

  T &operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
  const T &operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
};

template<typename T, int DIM>
class TensorList {
 public:
  static TensorList<T, DIM> fromTensor(const Tensor &src);
  static TensorList<T, DIM> fromTensor(Tensor &src);

  lut::Span<const TensorShape::Elem> getShape() const {
    return lut::Span<const TensorShape::Elem>(_shape, DIM);
  }
  int getShape(int d) const {
    return _shape[d].shape;
  }
  int getLength() const {
    if (_basePtr) {
      return _size;
    } else {
      return static_cast<int>(_pointerList.size());
    }
  }
  lut::Span<T *const> getDataPtrList() {
    if (_basePtr && _pointerList.empty()) {
      for (int i = 0; i < _size; ++i) {
        _pointerList.push_back(_basePtr + i * _stride);
      }
    }
    return lut::makeConstSpan(_pointerList);
  }

  TensorAccessor<T, DIM> getTensor(int index) const {
    if (_basePtr) {
      return TensorAccessor<T, DIM>(_shape, _basePtr + index * _stride);
    } else {
      return TensorAccessor<T, DIM>(_shape, _pointerList[index]);
    }
  }

 private:
  const TensorShape::Elem *_shape;
  std::vector<T *> _pointerList;

  int64_t _stride;
  int _size;
  T *_basePtr;

  TensorList(const TensorShape::Elem *shape, std::vector<T *> &&pointerList)
      : _shape(shape),
        _pointerList(std::move(pointerList)),
        _stride(0),
        _size(0),
        _basePtr(nullptr) {
  }

  TensorList(const TensorShape::Elem *shape, T *data, int size, int stride)
      : _shape(shape),
        _basePtr(data),
        _size(size),
        _stride(stride) {
  }
};

template<typename T, int SUBTENSOR_DIM>
void getDataPointerList(
    T *base,
    lut::Span<const TensorShape::Elem> shape,
    std::vector<T *> &pointerList) {
  CHECK(shape.size() >= SUBTENSOR_DIM);

  if (shape.size() == SUBTENSOR_DIM) {
    pointerList.push_back(base);
  } else {
    int size = shape.front().shape;
    int stride = shape.front().stride;
    int dim = shape.size();

    for (int i = 0; i < size; ++i) {
      T *pointer = base + i * stride;
      if (dim - SUBTENSOR_DIM == 1) {
        pointerList.push_back(pointer);
      } else {
        getDataPointerList<T, SUBTENSOR_DIM>(pointer, shape.subspan(1), pointerList);
      }
    }
  }
}

template<typename T, int DIM>
TensorList<T, DIM> TensorList<T, DIM>::fromTensor(const Tensor &src) {
  std::vector<T *> pointerList;
  lut::Span<const TensorShape::Elem> shape = src.getInternalShape()->getData_();
  getDataPointerList<T, DIM>(src.getInternalData()->getData<T>(), shape, pointerList);

  const TensorShape::Elem *tensorShape = shape.data() + (shape.size() - DIM);
  if (src.isContiguous()) {
    int numTensor = 1;
    for (int i = 0; i < src.getDim() - DIM; ++i) {
      numTensor *= src.getShape(i);
    }

    int stride = 1;
    for (int i = 0; i < DIM; ++i) {
      stride *= tensorShape[i].shape;
    }

    return TensorList<T, DIM>(tensorShape, src.getInternalData()->getData<T>(), numTensor, stride);
  } else {
    return TensorList<T, DIM>(tensorShape, std::move(pointerList));
  }
}

template<typename T, int DIM>
TensorList<T, DIM> TensorList<T, DIM>::fromTensor(Tensor &src) {
  std::vector<T *> pointerList;
  lut::Span<const TensorShape::Elem> shape = src.getInternalShape()->getData_();
  getDataPointerList<T, DIM>(src.getInternalData()->getData<T>(), shape, pointerList);

  const TensorShape::Elem *tensorShape = shape.data() + (shape.size() - DIM);
  if (src.isContiguous()) {
    int numTensor = 1;
    for (int i = 0; i < src.getDim() - DIM; ++i) {
      numTensor *= src.getShape(i);
    }

    int stride = 1;
    for (int i = 0; i < DIM; ++i) {
      stride *= tensorShape[i].shape;
    }

    return TensorList<T, DIM>(tensorShape, src.getInternalData()->getData<T>(), numTensor, stride);
  } else {
    return TensorList<T, DIM>(tensorShape, std::move(pointerList));
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
