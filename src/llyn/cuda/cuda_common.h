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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "llyn/tensor.h"
#include "llyn/internal/tensor_shape.h"


namespace llyn {
namespace cuda {

template<int DIM>
__device__ int64_t getOffsetByFlatIndex(internal::TensorShape::Elem *shape, int64_t index);
template<>
__device__ int64_t getOffsetByFlatIndex<1>(internal::TensorShape::Elem *shape, int64_t index) {
  return index * shape[0].stride;
}
template<>
__device__ int64_t getOffsetByFlatIndex<2>(internal::TensorShape::Elem *shape, int64_t index) {
  int index1 = index % shape[1].shape;
  int numel0 = index / shape[1].shape;
  return numel0 * shape[0].stride + index1 * shape[1].stride;
}
template<>
__device__ int64_t getOffsetByFlatIndex<3>(internal::TensorShape::Elem *shape, int64_t index) {
  int index2 = index % shape[2].shape;
  int numel1 = index / shape[2].shape;
  int index1 = numel1 % shape[1].shape;
  int numel0 = numel1 / shape[1].shape;
  return numel0 * shape[0].stride + index1 * shape[1].stride + index2 * shape[1].stride;
}

/// @brief Get number of tensors in a N-dim shape data.
/// @tparam DIM The dimension N.
/// @param shape Pointer to the shape data. It was expected to have at least N elements.
/// @return Number of tensors.
template<int DIM>
__host__ __device__ int64_t getNumTensors(internal::TensorShape::Elem *shape);
template<>
__host__ __device__ int64_t getNumTensors<1>(internal::TensorShape::Elem *shape) {
  return shape[0].shape;
}
template<>
__host__ __device__ int64_t getNumTensors<2>(internal::TensorShape::Elem *shape) {
  return shape[0].shape * shape[1].shape;
}
template<>
__host__ __device__ int64_t getNumTensors<3>(internal::TensorShape::Elem *shape) {
  return shape[0].shape * shape[1].shape * shape[2].shape;
}

template<typename T, int DIM>
class TensorAccessor {
 public:
  __device__ TensorAccessor(internal::TensorShape::Elem *shape, T *data) :
      _shape(shape),
      _data(data) {}

  __device__ TensorAccessor<T, DIM - 1> operator[](int index) {
    int64_t offset = index * this->_shape[0].stride;
    return TensorAccessor<T, DIM - 1>(_shape + 1, _data + offset);
  }

  __device__ const TensorAccessor<T, DIM - 1> operator[](int index) const {
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
  __device__ TensorAccessor(internal::TensorShape::Elem *shape, T *data) :
      _shape(shape),
      _data(data) {}

  __device__ T &operator[](int index) {
    int64_t offset = index * this->_shape[0].stride;
    return _data[offset];
  }

  __device__ T operator[](int index) const {
    int64_t offset = index * this->_shape[0].stride;
    return _data[offset];
  }

 private:
  internal::TensorShape::Elem *_shape;
  T *_data;
};

/// @brief A packed tensor accessor. `Packed` means the accessor also packed with the tensor 
/// metadata.
/// @tparam T Tensor data type.
/// @tparam DIM Dimension of this tensor.
template<typename T, int DIM>
class PackedTensorAccessor {
 public:
  __device__ TensorAccessor<T, 1> getVectorByFlatIndex(int64_t index) {
    int64_t offset = getOffsetByFlatIndex<DIM - 1>(_shape, index);
    return TensorAccessor<T, 1>(this->_shape + DIM - 1, offset);
  }

  /// @brief Get number of vectors in this tensor.
  /// @return Number of vectors.
  __host__ __device__ int64_t getNumVectors() {
    return getNumTensors<DIM - 1>(this->_shape);
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
