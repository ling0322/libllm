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

#include "lutil/error.h"
#include "lutil/strings.h"
#include "lynn/cuda/cuda_tensor_data.h"
#include "lynn/tensor.h"

namespace libllm {
namespace op {
namespace cuda {

struct Size {
  int32_t shape;
  int32_t stride;
};

template<typename T, int DIM>
class TensorAccessorBase {
 public:
  __device__ TensorAccessorBase(const Size *size, T *data)
      : _size(size),
        _data(data) {
  }

 protected:
  const Size *_size;
  T *_data;
};

template<typename T, int DIM>
class TensorAccessor : public TensorAccessorBase<T, DIM> {
 public:
  __device__ TensorAccessor(const Size *size, T *data)
      : TensorAccessorBase<T, DIM>(size, data) {
  }

  __device__ TensorAccessor<T, DIM - 1> operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return TensorAccessor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
  __device__ const TensorAccessor<T, DIM - 1> operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return TensorAccessor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
};

template<typename T>
class TensorAccessor<T, 1> : public TensorAccessorBase<T, 1> {
 public:
  __device__ TensorAccessor(const Size *size, T *data)
      : TensorAccessorBase<T, 1>(size, data) {
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

/// @brief A packed tensor accessor. `Packed` means the TensorAccessor also packed with the tensor
/// metadata.
/// @tparam T Tensor data type.
/// @tparam DIM Dimension of this tensor.
template<typename T, int DIM>
class PackedTensorAccessorBase {
 public:
  __host__ explicit PackedTensorAccessorBase(Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getData<T>();
    for (int i = 0; i < DIM; ++i) {
      _size[i] = Size{tensor.getShape(i), tensor.getStride(i)};
    }
  }

  __host__ explicit PackedTensorAccessorBase(const Tensor &tensor) {
    CHECK(tensor.getDim() == DIM);
    _data = tensor.getData<T>();
    for (int i = 0; i < DIM; ++i) {
      _size[i] = Size{tensor.getShape(i), tensor.getStride(i)};
    }
  }

  __forceinline__ __device__ int getShape(int dim) const {
    return this->_size[dim].shape;
  }
  __forceinline__ __device__ int getStride(int dim) const {
    return this->_size[dim].stride;
  }
  __forceinline__ __device__ const Size *getSize() const {
    return _size;
  }
  __forceinline__ __device__ T *getData() const {
    return _data;
  }

  __forceinline__ __device__ T &getElemByIndex(int idx) const {
    int stridedIdx = 0;
#pragma unroll
    for (int d = DIM - 1; d >= 0; --d) {
      stridedIdx += (idx % this->_size[d].shape) * this->_size[d].stride;
      idx /= this->_size[d].shape;
    }
    return _data[stridedIdx];
  }

 protected:
  Size _size[DIM];
  T *_data;
};

template<typename T, int DIM>
class PackedTensorAccessor : public PackedTensorAccessorBase<T, DIM> {
 public:
  __host__ PackedTensorAccessor(Tensor &tensor)
      : PackedTensorAccessorBase<T, DIM>(tensor) {
  }
  __host__ PackedTensorAccessor(const Tensor &tensor)
      : PackedTensorAccessorBase<T, DIM>(tensor) {
  }

  __device__ TensorAccessor<T, DIM - 1> operator[](int index) {
    int64_t offset = index * this->_size[0].stride;
    return TensorAccessor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
  __device__ const TensorAccessor<T, DIM - 1> operator[](int index) const {
    int64_t offset = index * this->_size[0].stride;
    return TensorAccessor<T, DIM - 1>(this->_size + 1, this->_data + offset);
  }
};

template<typename T>
class PackedTensorAccessor<T, 1> : public PackedTensorAccessorBase<T, 1> {
 public:
  __host__ PackedTensorAccessor(Tensor &tensor)
      : PackedTensorAccessorBase<T, 1>(tensor) {
  }
  __host__ PackedTensorAccessor(const Tensor &tensor)
      : PackedTensorAccessorBase<T, 1>(tensor) {
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
