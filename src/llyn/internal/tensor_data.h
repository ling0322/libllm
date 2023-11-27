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

// contains dimension and stride information for an axis in tensor
class TensorData {
 public:
  static constexpr int MaxSlot = 3;
  static constexpr int64_t MaxNumEl = 1073741824;

  static std::shared_ptr<TensorData> read(ly::ReadableFile *fp);
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype);
  static std::shared_ptr<TensorData> create(
      int64_t numel, DType dtype, int64_t numel2, DType dtype2);

  virtual ~TensorData() = default;

  // get the device of tensor data.
  virtual Device getDevice() const = 0;

  // slot0
  template<int SLOT, typename T>
  T *getData(int64_t offset = 0) const;

  DType getDType() const { return _slots[0].dtype; }
  int64_t getNumEl() const { return _slots[0].numel; }
  int64_t getSizeInBytes() const { return _slots[0].dtype.getTotalSize(_slots[0].numel); }

  // slot1
  DType getDType1() const { CHECK(_numSlot > 1); return _slots[1].dtype; }
  int64_t getNumEl1() const { CHECK(_numSlot > 1); return _slots[1].numel; }
  int64_t getSizeInBytes1() const {
    CHECK(_numSlot > 1);
    return _slots[1].dtype.getTotalSize(_slots[1].numel);
  }

 protected:
  struct Slot {
    Byte *data;
    int64_t numel;
    DType dtype;

    Slot();
  };

  Slot _slots[3];
  int _numSlot;

  TensorData();
};

template<int SLOT, typename T>
T *TensorData::getData(int64_t offset) const {
  CHECK(_numSlot > SLOT && DType::getType<T>() == _slots[SLOT].dtype);
  return reinterpret_cast<T *>(_slots[SLOT].data + _slots[SLOT].dtype.getTotalSize(offset));
}

}  // namespace internal
}  // namespace llyn
