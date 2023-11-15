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
#include "llyn/tensor.h"
#include "llyn/internal/tensor_shape.h"

#if defined(__CUDACC__) 
#define LLYN_DEVICE __device__
#define LLYN_HOST __host__
#else
#define LLYN_DEVICE
#define LLYN_HOST
#endif

namespace llyn {
namespace cuda {

template<int DIM>
LLYN_DEVICE int64_t getOffsetByFlatIndex(internal::TensorShape::Elem *shape, int64_t index);
template<>
LLYN_DEVICE int64_t getOffsetByFlatIndex<1>(internal::TensorShape::Elem *shape, int64_t index) {
  return index * shape[0].stride;
}
template<>
LLYN_DEVICE int64_t getOffsetByFlatIndex<2>(internal::TensorShape::Elem *shape, int64_t index) {
  int index1 = index % shape[1].shape;
  int numel0 = index / shape[1].shape;
  return numel0 * shape[0].stride + index1 * shape[1].stride;
}
template<>
LLYN_DEVICE int64_t getOffsetByFlatIndex<3>(internal::TensorShape::Elem *shape, int64_t index) {
  int index2 = index % shape[2].shape;
  int numel1 = index / shape[2].shape;
  int index1 = numel1 % shape[1].shape;
  int numel0 = numel1 / shape[1].shape;
  return numel0 * shape[0].stride + index1 * shape[1].stride + index2 * shape[1].stride;
}

template<typename T, int DIM>
class TensorAccessor {
 public:
  LLYN_DEVICE TensorAccessor(internal::TensorShape::Elem *shape, T *data) :
      _shape(shape),
      _data(data) {}

  LLYN_DEVICE TensorAccessor<T, DIM - 1> operator[](int index) {
    int64_t offset = index * this->_shape[0].stride;
    return TensorAccessor<T, DIM - 1>(_shape + 1, _data + offset);
  }

  LLYN_DEVICE const TensorAccessor<T, DIM - 1> operator[](int index) const {
    int64_t offset = index * this->_shape[0].stride;
    return TensorAccessor<T, DIM - 1>(_shape + 1, _data + offset);
  }

 private:
  internal::TensorShape::Elem *_shape;
  T *_data;
};

template<typename T>
class TensorAccessor<T, 1> {
 public:
  LLYN_DEVICE TensorAccessor(internal::TensorShape::Elem *shape, T *data) :
      _shape(shape),
      _data(data) {}

  LLYN_DEVICE T &operator[](int index) {
    int64_t offset = index * this->_shape[0].stride;
    return _data[offset];
  }

  LLYN_DEVICE T operator[](int index) const {
    int64_t offset = index * this->_shape[0].stride;
    return _data[offset];
  }

 private:
  internal::TensorShape::Elem *_shape;
  T *_data;
};

template<typename T, int DIM>
class PackedTensorAccessor {
 public:
  LLYN_DEVICE TensorAccessor<T, 1> getVectorByFlatIndex(int64_t index) {
    int64_t offset = getOffsetByFlatIndex<DIM - 1>(_shape, index);
    return TensorAccessor<T, 1>(this->_shape + DIM - 1, offset);
  }

  LLYN_DEVICE const TensorAccessor<T, 1> getVectorByFlatIndex(int64_t index) const {
    int64_t offset = getOffsetByFlatIndex<DIM - 1>(_shape, index);
    return TensorAccessor<T, 1>(this->_shape + DIM - 1, offset);
  }

 private:
  internal::TensorShape::Elem _shape[DIM];
  T *_data;
};

/// @brief A q4 quantized constant matrix (2D tensor).
struct Q4ConstMatrix {
  int stride;
  int shape0;
  int shape1;

  const half *scales;
  const uint8_t *data;
  const int8_t *zeroPoints;

  static Q4ConstMatrix fromTensor(const Tensor &tensor);
};

}  // namespace cuda
}  // namespace cuda
