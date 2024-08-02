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

#include <initializer_list>

#include "libllm/device.h"
#include "libllm/tensor.h"
#include "lut/span.h"

namespace libllm {
namespace op {
namespace cuda {

class CudaTensorData : public TensorData {
 public:
  static std::shared_ptr<TensorData> create(int64_t numel, DType dtype);
  static std::shared_ptr<TensorData> create(lut::Span<const std::pair<int64_t, DType>> slots);

  CudaTensorData();
  ~CudaTensorData();

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
};

}  // namespace cuda
}  // namespace op
}  // namespace libllm
