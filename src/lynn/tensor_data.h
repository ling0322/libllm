// The MIT License (MIT)
//
// Copyright (c) 2025 Xiaoyang Chen
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

namespace ly {

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

}  // namespace ly
