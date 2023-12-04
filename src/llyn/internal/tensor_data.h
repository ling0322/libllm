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
#include <memory>

#include "llyn/device.h"
#include "llyn/dtype.h"
#include "lyutil/log.h"
#include "lyutil/reader.h"

namespace llyn {
namespace internal {

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
  T *getData(int64_t offset = 0) const { return getSlot(0)->getData<T>(offset); }

  /// @brief Get data type from slot[0]
  /// @return slot[0] data type.
  DType getDType() const { return getSlot(0)->getDType(); }

  /// @brief Get number of elements in slot[0]
  /// @return number of elements in slot[0].
  int64_t getNumEl() const { return getSlot(0)->getNumEl(); }

  /// @brief Get total size in bytes of slot[0]
  /// @return slot[0] size in bytes.
  int64_t getSizeInBytes() const { return getSlot(0)->getSizeInBytes(); }

  /// @brief throw if the tensor data is invalid.
  void throwIfInvalid();
};

}  // namespace internal
}  // namespace llyn
