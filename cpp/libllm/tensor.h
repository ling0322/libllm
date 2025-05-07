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

#include "libllm/device.h"
#include "libllm/dtype.h"
#include "lutil/fixed_array.h"
#include "lutil/reader.h"
#include "lutil/span.h"

namespace libllm {

class TensorShape;
class TensorData;

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

  // pointer of data in this tensor
  template<typename T>
  T *getData();
  template<typename T>
  const T *getData() const;

  // Check the shape of a tensor. If shape of `tensor` does not match `shape`, return AbortedError
  // with message "invalid shape".
  void throwIfInvalidShape(lut::Span<const int> shape, const std::string &name) const;

  // low-level functions. DO NOT use them outside llyn.
  const TensorData *getDataObject() const;
  const TensorShape *getShape_() const;
  std::shared_ptr<TensorData> getDataShared_() const;
  int64_t getOffset_() const;

 protected:
  std::shared_ptr<TensorData> _data;
  std::shared_ptr<TensorShape> _shape;
  int64_t _offset;
};

// Stores shape and stride of a Tensor.
class TensorShape {
 public:
  typedef Tensor::ShapeType ShapeType;
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

/// @brief A data record in TensorData object.
class SlotBase {
 public:
  virtual ~SlotBase() = default;

  /// @brief Get number of elements in this slot.
  /// @return number of elements.
  virtual int64_t getNumEl() const = 0;

  /// @brief Get data type of this slot.
  /// @return Data type.
  virtual DType getDType() const = 0;

  /// @brief Get the pointer to underlying data.
  /// @return data pointer.
  virtual Byte *getRawData() const = 0;

  /// @brief Get data pointer of n-th element in this slot as type `T`.
  /// @tparam T the type of underlying data. Use `void` to avoid the type checking.
  /// @param offset the offset `n`.
  /// @return the pointer of type `T`.
  template<typename T>
  T *getData(int64_t offset = 0) const;

  /// @brief Get total number of bytes in this slot.
  /// @return
  int64_t getSizeInBytes() const {
    return getDType().getTotalSize(getNumEl());
  }
};

template<typename T>
inline T *SlotBase::getData(int64_t offset) const {
  DType dtype = getDType();
  CHECK(DType::getType<T>() == dtype);
  return reinterpret_cast<T *>(getRawData() + dtype.getTotalSize(offset));
}

template<>
inline void *SlotBase::getData<void>(int64_t offset) const {
  DType dtype = getDType();
  return reinterpret_cast<void *>(getRawData() + dtype.getTotalSize(offset));
}

/// @brief holds the internal data of a Tensor.
class TensorData {
 public:
  static constexpr int MaxSlot = 3;
  static constexpr int64_t MaxNumEl = 1073741824;

  virtual ~TensorData() = default;

  // get the device of tensor data.
  virtual Device getDevice() const = 0;

  /// @brief Get number of slots in this tensor data.
  /// @return number of slots.
  virtual int getNumSlot() const = 0;

  /// @brief Get internal Slot by index.
  /// @param slot index of the slot. It should be less than getNumSlot();
  /// @return Slot object.
  virtual const SlotBase *getSlot(int slot) const = 0;

  /// @brief Get data pointer of n-th element in slot[0] as type `T`.
  /// @tparam T the type of underlying data.
  /// @param offset the offset `n`.
  /// @return the pointer of type `T`.
  template<typename T>
  T *getData(int64_t offset = 0) const {
    return getSlot(0)->getData<T>(offset);
  }

  /// @brief Get data type from slot[0]
  /// @return slot[0] data type.
  DType getDType() const {
    return getSlot(0)->getDType();
  }

  /// @brief Get number of elements in slot[0]
  /// @return number of elements in slot[0].
  int64_t getNumEl() const {
    return getSlot(0)->getNumEl();
  }

  /// @brief Get total size in bytes of slot[0]
  /// @return slot[0] size in bytes.
  int64_t getSizeInBytes() const {
    return getSlot(0)->getSizeInBytes();
  }
};

constexpr int None = std::numeric_limits<int>::min();

inline DType Tensor::getDType() const {
  return _data ? _data->getDType() : DType(DType::kUnknown);
}

template<typename T>
T *Tensor::getData() {
  return _data->getData<T>(_offset);
}

template<typename T>
const T *Tensor::getData() const {
  return _data->getData<T>(_offset);
}

}  // namespace libllm
