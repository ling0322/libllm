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
#include "flint/device.h"
#include "flint/dtype.h"
#include "flint/functional.h"

namespace fl {

class Operators;

constexpr int None = std::numeric_limits<int>::min();

/// @brief holds the internal data of a Tensor.
class TensorData {
 public:
  static constexpr int64_t MaxNumEl = 1073741824;

  virtual ~TensorData() = default;

  // get the device of tensor data.
  virtual Device getDevice() const = 0;

  // get the pointer of raw data
  virtual std::byte *getRawData() const = 0;

  /// @brief Get data pointer of n-th element in slot[0] as type `T`.
  /// @tparam T the type of underlying data.
  /// @param offset the offset `n`.
  /// @return the pointer of type `T`.
  template<typename T>
  T *getData(int64_t offset) const {
    DType dtype = getDType();
    CHECK((std::is_same_v<T, void> || DType::getType<T>() == dtype));
    return reinterpret_cast<T *>(getRawData() + dtype.getTotalSize(offset));
  }

  /// @brief Get data type from slot[0]
  /// @return slot[0] data type.
  DType getDType() const {
    return _dtype;
  }

  /// @brief Get number of elements in slot[0]
  /// @return number of elements in slot[0].
  int64_t getNumEl() const {
    return _numel;
  }

  /// @brief Get total size in bytes of slot[0]
  /// @return slot[0] size in bytes.
  int64_t getSizeInBytes() const {
    return getDType().getTotalSize(getNumEl());
  }

 protected:
  int64_t _numel;
  DType _dtype;

  TensorData()
      : _numel(0),
        _dtype(DType::kUnknown) {
  }
};

// Stores shape and stride of a Tensor.
class TensorShape {
 public:
  typedef int32_t ShapeType;
  struct Elem {
    ShapeType shape;
    ShapeType stride;
  };

  // read tensor shape from file.
  static std::shared_ptr<TensorShape> read(lut::Reader *fp);

  // from shape.
  TensorShape(lut::Span<const ShapeType> shape);
  TensorShape(lut::Span<const Elem> shape);

  TensorShape(const TensorShape &size);
  TensorShape(TensorShape &&size) noexcept;
  TensorShape &operator=(const TensorShape &size);
  TensorShape &operator=(TensorShape &&size) noexcept;

  bool empty() const;
  int getDim() const;
  ShapeType getShape(int index) const;
  ShapeType getStride(int index) const;
  int64_t getNumEl() const;

  // Returns a sub-Size starting at specified dimension.
  std::shared_ptr<TensorShape> subsize(int d) const;

  // Returns a Size that is a transposed version of current size. The given
  // dimensions dim0 and dim1 are swapped.
  std::shared_ptr<TensorShape> transpose(int dim0, int dim1) const;

  // add or remove one shape=1 dimension at specified dimension.
  std::shared_ptr<TensorShape> unsqueeze(int dim) const;
  std::shared_ptr<TensorShape> squeeze(int dim) const;

  // set the value of shape(dim). Negative dim is allowed. new `shape` should be less or equal to
  // current size.
  void setShape(int dim, ShapeType shape);

  // return a new shape that expand singleton dimensions to a larger size.
  std::shared_ptr<TensorShape> expand(lut::Span<const int> shape) const;

  // convert negative dimension or index (in specific `dim`) to positive.
  int getRealDim(int dim) const;
  int getRealIndex(int dim, int index) const;

  lut::Span<const Elem> getData_() const {
    return lut::makeConstSpan(_data);
  }

  std::string toString() const;

 private:
  lut::FixedArray<Elem> _data;

  // an empty Tensor.
  TensorShape() = default;
};

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

 protected:
  std::shared_ptr<TensorData> _data;
  std::shared_ptr<TensorShape> _shape;
  int64_t _offset;
};

inline DType Tensor::getDType() const {
  return _data ? _data->getDType() : DType(DType::kUnknown);
}

}  // namespace fl
