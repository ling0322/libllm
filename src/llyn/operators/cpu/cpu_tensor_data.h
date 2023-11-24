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


#include "lyutil/span.h"
#include "llyn/device.h"
#include "llyn/internal/tensor_data.h"

namespace llyn {
namespace internal {

class CpuTensorData : public TensorData {
 public:
  static std::shared_ptr<TensorData> create(int64_t numel0, DType dtype0);
  static std::shared_ptr<TensorData> create(ly::Span<const std::pair<int64_t, DType>> slots);
  static std::shared_ptr<TensorData> read(ly::ReadableFile *fp);

  /// @brief Create a new instance of CpuTensorData with the same size and slots as `tensorData`.
  /// @param tensorData The reference tensorData object.
  /// @return A new instance of CpuTensorData.
  static std::shared_ptr<TensorData> createLike(const TensorData *tensorData);

  CpuTensorData();
  ~CpuTensorData();

  Device getDevice() const override;
  int getNumSlot() const override;
  const SlotBase *getSlot(int slot) const override;

 private:
  struct Slot : public SlotBase {
    Byte *data;
    int64_t numel;
    DType dtype;

    Slot();

    int64_t getNumEl() const override;
    DType getDType() const override;
    Byte *getRawData() const override;
  };

  Slot _slots[TensorData::MaxSlot];
  int _numSlot;

  void readSlot(ly::ReadableFile *fp, int slotIdx);
};

}  // namespace internal
}  // namespace llyn
