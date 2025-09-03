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

#include <limits>
#include <memory>

#include "lutil/fixed_array.h"
#include "lutil/reader.h"
#include "lutil/span.h"
#include "lynn/device.h"
#include "lynn/dtype.h"
#include "lynn/functional.h"
#include "lynn/tensor_data.h"
#include "lynn/tensor_shape.h"

namespace ly {

class TensorData;
class Operators;

constexpr int None = std::numeric_limits<int>::min();

class Tensor {
 public:
  typedef int32_t ShapeType;
  // rank for empty tansor.
  static constexpr int kEmptyRank = -1;

  // create a tensor in CPU storage. Size of `data` should be the same as `shape.numel()`.
  // Example:
  //   Tensor x = Tensor::FromData({2, 2}, {1.0f, 0.8f, 0.6f, 0.2f});
  template<typename T>
  static Tensor create(std::initializer_list<int> shape, lut::Span<const T> data);

  /// @brief Create Tensor from TensorShape and TensorData.
  /// @param shape pointer to TensorShape.
  /// @param data pointer to TensorData.
  /// @return The Tensor created.
  static Tensor create(
      std::shared_ptr<TensorShape> shape,
      std::shared_ptr<TensorData> data,
      int64_t offset = 0);

  // constructor and destructor.
  Tensor();
  ~Tensor();

  // Read the tensor from fp.
  void read(lut::Reader *fp);

  // copy and move constructors.
  Tensor(const Tensor &tensor);
  Tensor &operator=(const Tensor &tensor);
  Tensor(Tensor &&tensor) noexcept;
  Tensor &operator=(Tensor &&tensor);

  // get numebr of dimentsions.
  int getDim() const;

  // get the size in dimention `d`. `d` supports positive number (index) and negative number (index
  // from back). Crash if `d` is out of boundary
  ShapeType getShape(int d) const;
  std::vector<int> getShape() const;
  std::string getShapeString() const;

  // get stride for dimension `d`.
  ShapeType getStride(int d) const;

  // get number of elements in this tensor.
  int64_t getNumEl() const;

  // return true if this tensor is empty.
  bool empty() const;

  // get data type.
  DType getDType() const;

  /// @brief Get storage device of this tensor.
  /// @return the device.
  Device getDevice() const;

  // Get a new view of the tensor..
  Tensor view(lut::Span<const int> shape) const;

  // Get a new view of the tensor with singleton dimensions expanded to a larger size.
  Tensor expand(lut::Span<const int> shape) const;

  // Get slice of this tensor. `dim` is the dimension to slice. [begin, end) is the range. For
  // [begin, end) only version, dimension 0 is used. Negative `begin` and `end` is accepted. Crash
  // if dim or range out of boundary.
  // None could be used in both begin and end. (None, 5) means [: 5], (5, None) means [5: ].
  Tensor slice(int dim, std::pair<int, int> range) const;
  Tensor slice(std::pair<int, int> range) const;

  // Get subtensor at specified index of first dimension. Negative `index` is accepted. Crash if
  // `index` out of boundary.
  Tensor subtensor(int index) const;

  // add or remove an additional shape=1 dimension at specified position.
  Tensor unsqueeze(int dim) const;
  Tensor squeeze(int dim) const;

  Tensor transpose(int dim0, int dim1) const;

  // return true if the tensor is contigous.
  bool isContiguous() const;

  // get operators for this tensor.
  Operators *getOperators() const;

  // Check the shape of a tensor. If shape of `tensor` does not match `shape`, return AbortedError
  // with message "invalid shape".
  void throwIfInvalidShape(lut::Span<const int> shape, const std::string &name) const;

  // low-level functions. DO NOT use them outside llyn.
  std::shared_ptr<TensorShape> getInternalShape() const;
  std::shared_ptr<TensorData> getInternalData() const;
  int64_t getInternalOffset() const;

  // ----------------- functional operators -----------------
  static Tensor randn(lut::Span<const int> shape, Device device = Device::getCpu());
  static Tensor arange(
      LongType begin,
      LongType end,
      LongType step = 1,
      Device device = Device::getCpu());
  Tensor div(float rhs) const;
  Tensor square() const;
  Tensor sum(int dim = None) const;
  Tensor mod(LongType rhs) const;
  Tensor eq(const Tensor &rhs) const;
  Tensor to(DType dtype) const;
  Tensor operator==(const Tensor &rhs) const;
  Tensor operator+(const Tensor &rhs) const;
  Tensor operator-(const Tensor &rhs) const;
  Tensor operator*(const Tensor &rhs) const;

  template<typename T>
  T elem() const;

 protected:
  std::shared_ptr<TensorData> _data;
  std::shared_ptr<TensorShape> _shape;
  int64_t _offset;
};

inline DType Tensor::getDType() const {
  return _data ? _data->getDType() : DType(DType::kUnknown);
}

}  // namespace ly
