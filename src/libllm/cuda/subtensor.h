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

#include <type_traits>

#include "libllm/cuda/cuda_tensor_data.h"
#include "libllm/tensor.h"
#include "lut/error.h"
#include "lut/strings.h"

namespace libllm {
namespace op {
namespace cuda {

struct Size {
  int32_t shape;
  int32_t stride;
};

template<typename T, int DIM>
class SubtensorBase {
 public:
  __device__ SubtensorBase(const Size *size, T *data)
      : _size(size),
        _data(data) {
  }

 protected:
  const Size *_size;
  T *_data;
};

template<typename T, int DIM>
class Subtensor : public SubtensorBase<T, DIM> {
 public:
  __device__ Subtensor(const Size *size, T *data)
      : SubtensorBase<T, DIM>(size, data) {
  }

  __device__ Subtensor<T, DIM - 1> operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return Subtensor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
  __device__ const Subtensor<T, DIM - 1> operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return Subtensor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
};

template<typename T>
class Subtensor<T, 1> : public SubtensorBase<T, 1> {
 public:
  __device__ Subtensor(const Size *size, T *data)
      : SubtensorBase<T, 1>(size, data) {
  }

  __device__ T &operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
  __device__ T operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
};

/// @brief A packed tensor accessor. `Packed` means the subtensor also packed with the tensor
/// metadata.
/// @tparam T Tensor data type.
/// @tparam DIM Dimension of this tensor.
template<typename T, int DIM>
class PackedSubtensorBase {
 public:
  __host__ explicit PackedSubtensorBase(Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getData<T>();
    for (int i = 0; i < DIM; ++i) {
      _size[i] = Size{tensor.getShape(i), tensor.getStride(i)};
    }
  }

  __host__ explicit PackedSubtensorBase(const Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getData<T>();
    for (int i = 0; i < DIM; ++i) {
      _size[i] = Size{tensor.getShape(i), tensor.getStride(i)};
    }
  }

  __device__ int getShape(int dim) const {
    return this->_size[dim].shape;
  }
  __device__ const Size *getSize() const {
    return _size;
  }
  __device__ T *getData() const {
    return _data;
  }

 protected:
  Size _size[DIM];
  T *_data;
};

template<typename T, int DIM>
class PackedSubtensor : public PackedSubtensorBase<T, DIM> {
 public:
  __host__ PackedSubtensor(Tensor &tensor)
      : PackedSubtensorBase<T, DIM>(tensor) {
  }
  __host__ PackedSubtensor(const Tensor &tensor)
      : PackedSubtensorBase<T, DIM>(tensor) {
  }

  __device__ Subtensor<T, DIM - 1> operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return Subtensor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
  __device__ const Subtensor<T, DIM - 1> operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return Subtensor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
};

template<typename T>
class PackedSubtensor<T, 1> : public PackedSubtensorBase<T, 1> {
 public:
  __host__ PackedSubtensor(Tensor &tensor)
      : PackedSubtensorBase<T, 1>(tensor) {
  }
  __host__ PackedSubtensor(const Tensor &tensor)
      : PackedSubtensorBase<T, 1>(tensor) {
  }

  __device__ T &operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
  __device__ T operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return this->_data[offset];
  }
};

}  // namespace cuda
}  // namespace op
}  // namespace libllm
