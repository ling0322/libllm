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

#include <type_traits>
#include "lyutil/span.h"
#include "ly/internal/tensor_shape.h"
#include "ly/tensor.h"

namespace ly {
namespace op {
namespace cpu {

// sub-tensor. Stores the shape and data pointer to a sub region of the original
// Tensor. It's faster than Tensor when passing as parameters.
template<typename T>
struct Subtensor {
  lut::Span<const internal::TensorShape::Elem> shape;
  T *data;

  static Subtensor<T> fromTensor(Tensor &tensor) {
    return Subtensor<T>{tensor.getShape_()->getData_(), tensor.getData<T>()};
  }

  template <class U = std::enable_if<std::is_const<T>::value, Subtensor<T>>>
  static typename U::type fromTensor(const Tensor &tensor) {
    return Subtensor<T>{tensor.getShape_()->getData_(), tensor.getData<T>()};
  }

  // get sub-tensor of this Subtensor.
  Subtensor<T> subtensor(int index) {
    return Subtensor<T>{
      this->shape.subspan(1),
      data + index * this->shape[0].stride
    };
  }

  const Subtensor<T> subtensor(int index) const {
    return Subtensor<T>{
      this->shape.subspan(1),
    };
  }

  // get dimension or stride for an axis. NOTE: negative index is NOT supported.
  int dimension(int index) const { return shape[index].shape; }
  int stride(int index) const { return shape[index].stride; }

  // get element from 1D sub-tensor. NOTE: elem() will not check the shape.
  T &elem(int index) {
    return data[index * this->shape[0].stride];
  }
  const T &elem(int index) const {
    return data[index * this->shape[0].stride];
  }

  int64_t getNumVec() const {
    int64_t n = 1;
    for (int i = 0; i < rank() - 1; ++i) {
      n *= shape[i].shape;
    }
    return n;
  }

  // number of element.
  int64_t numel() const {
    int64_t n = 1;
    for (const internal::TensorShape::Elem &s : shape) {
      n *= s.shape;
    }
    return n;
  }

  std::vector<int> getShape() const {
    std::vector<int> shape;
    for (int d = 0; d < rank(); ++d) {
      shape.push_back(dimension(d));
    }
    return shape;
  }

  // get rank.
  int rank() const { return static_cast<int>(shape.size()); }
};

}  // cpu
}  // op
}  // ly
