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
#include "libllm/lut/span.h"
#include "libllm/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

template<typename T, int DIM>
class TensorAccessorBase {
 public:
  explicit TensorAccessorBase(Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getData<T>();
    _size = tensor.getShape_()->getData_().data();
  }

  explicit TensorAccessorBase(const Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getData<T>();
    _size = tensor.getShape_()->getData_().data();
  }

  int getShape(int d) const {
    CHECK(d < DIM);
    return _size[d].shape;
  }

  TensorAccessorBase(const TensorShape::Elem *size, T *data) : _size(size), _data(data) {}

 protected:
  const TensorShape::Elem *_size;
  T *_data;
};

template<typename T, int DIM>
class TensorAccessor : public TensorAccessorBase<T, DIM> {
 public:
  TensorAccessor(Tensor &tensor) : TensorAccessorBase<T, DIM>(tensor) {}
  TensorAccessor(const Tensor &tensor) : TensorAccessorBase<T, DIM>(tensor) {}

  TensorAccessor(const TensorShape::Elem *size, T *data) : 
      TensorAccessorBase<T, DIM>(size, data) {}

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
  TensorAccessor(Tensor &tensor) : TensorAccessorBase<T, 1>(tensor) {}
  TensorAccessor(const Tensor &tensor) : TensorAccessorBase<T, 1>(tensor) {}

  TensorAccessor(const TensorShape::Elem *size, T *data) :
      TensorAccessorBase<T, 1>(size, data) {}

  T &operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
  T operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
};

template<typename T, int DIM>
class TensorList {
 public:
  static TensorList<T, DIM> fromTensor(const Tensor &src);
  static TensorList<T, DIM> fromTensor(Tensor &src);
  
  lut::Span<const TensorShape::Elem> getShape() const { return _shape; }
  int getShape(int d) const { return _shape[d].shape; }
  int getLength() const { return static_cast<int>(_pointerList.size()); }
  lut::Span<T *const> getDataPtrList() const { return lut::makeConstSpan(_pointerList); }

  TensorAccessor<T, DIM> getTensor(int index) const {
    return TensorAccessor<T, DIM>(_shape, _pointerList[index]);
  }

 private:
  const TensorShape::Elem *_shape;
  std::vector<T *> _pointerList;

  TensorList(const TensorShape::Elem *shape, std::vector<T *> &&pointerList):
      _shape(shape),
      _pointerList(std::move(pointerList)) {}
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
  lut::Span<const TensorShape::Elem> shape = src.getShape_()->getData_();
  getDataPointerList<T, DIM>(src.getData<T>(), shape, pointerList);

  const TensorShape::Elem *tensorShape = shape.data() + (shape.size() - DIM);
  return TensorList<T, DIM>(tensorShape, std::move(pointerList));
}

template<typename T, int DIM>
TensorList<T, DIM> TensorList<T, DIM>::fromTensor(Tensor &src) {
  std::vector<T *> pointerList;
  lut::Span<const TensorShape::Elem> shape = src.getShape_()->getData_();
  getDataPointerList<T, DIM>(src.getData<T>(), shape, pointerList);

  const TensorShape::Elem *tensorShape = shape.data() + (shape.size() - DIM);
  return TensorList<T, DIM>(tensorShape, std::move(pointerList));
}

}  // cpu
}  // op
}  // libllm

