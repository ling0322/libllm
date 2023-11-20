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
#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "llyn/tensor.h"
#include "llyn/internal/cuda_tensor_data.h"
#include "llyn/internal/tensor_shape.h"

#define LL_CHECK_CONTIGUOUS(x) { if (!x.isContiguous()) { \
    LOG(FATAL) << "contiguous is required for CUDA operators: " << #x; } }

namespace llyn {
namespace cuda {


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

Tensor createCudaTensorHalf(ly::Span<const int> shape);
Tensor createCudaTensorLong(ly::Span<const int> shape);
Tensor createCudaTensorFloat(ly::Span<const int> shape);

void checkCudaError(cudaError_t err);

}  // namespace cuda
}  // namespace cuda
