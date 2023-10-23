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

#include "llyn/internal/tensor_data.h"

#include <stdint.h>
#include <memory>

#include "llyn/device.h"
#include "llyn/dtype.h"
#include "lyutil/error.h"
#include "lyutil/platform.h"
#include "lyutil/span.h"

namespace llyn {
namespace internal {


// -- class TensorData ---------------------------------------------------------

TensorData::Slot::Slot()
    : data(nullptr),
      numel(0),
      dtype(DType::kUnknown) {}

TensorData::TensorData() : _numSlot(0) {}

// -- class CPUTensorData ---------------------------------------------------------

class CPUTensorData : public TensorData {
 public:
  ~CPUTensorData();

  Device getDevice() const override;
  void readSlot(ly::ReadableFile *fp, int slotIdx);
};

std::shared_ptr<TensorData> TensorData::create(int64_t numel, DType dtype) {
  auto tensorData = std::make_shared<CPUTensorData>();
  int64_t size = dtype.getTotalSize(numel);
  tensorData->_slots[0].data = reinterpret_cast<Byte *>(ly::alloc32ByteAlignedMem(size));
  tensorData->_slots[0].numel = numel;
  tensorData->_slots[0].dtype = dtype;

  tensorData->_numSlot = 1;
  return tensorData;
}

std::shared_ptr<TensorData> TensorData::create(
      int64_t numel, DType dtype, int64_t numel2, DType dtype2) {
  auto tensorData = std::make_shared<CPUTensorData>();

  int64_t size = dtype.getTotalSize(numel);
  tensorData->_slots[0].data = reinterpret_cast<Byte *>(ly::alloc32ByteAlignedMem(size));
  tensorData->_slots[0].numel = numel;
  tensorData->_slots[0].dtype = dtype;

  int64_t size2 = dtype2.getTotalSize(numel2);
  tensorData->_slots[1].data = reinterpret_cast<Byte *>(ly::alloc32ByteAlignedMem(size2));
  tensorData->_slots[1].numel = numel2;
  tensorData->_slots[1].dtype = dtype2;

  tensorData->_numSlot = 2;
  return tensorData;
}

void CPUTensorData::readSlot(ly::ReadableFile *fp, int slotIdx) {
  CHECK(slotIdx < MaxSlot);
  Slot &slot = _slots[slotIdx];

  CHECK(slot.data == nullptr);

  slot.dtype = fp->readValue<int16_t>();
  if (!slot.dtype.isValid())
    throw ly::AbortedError("invalid dtype.");

  slot.numel = fp->readValue<int64_t>();
  if (slot.numel > MaxNumEl)
    throw ly::AbortedError("tensor too big");
  
  int64_t size = slot.dtype.getTotalSize(slot.numel);
  slot.data = reinterpret_cast<Byte *>(ly::alloc32ByteAlignedMem(size));
  fp->readSpan(ly::makeSpan(reinterpret_cast<int8_t *>(slot.data), size));
  int magicNumber = fp->readValue<int16_t>();
  if (magicNumber != 0x55aa)
    throw ly::AbortedError("bad tensor data format (magic number).");
}

std::shared_ptr<TensorData> TensorData::read(ly::ReadableFile *fp) {
  std::shared_ptr<CPUTensorData> tensorData = std::make_shared<CPUTensorData>();

  if (fp->readString(4) != "tdat")
    throw ly::AbortedError("bad tensor data format.");

  int32_t numSlot = fp->readValue<int32_t>();
  if (numSlot <= 0 && numSlot > 2)
    throw ly::AbortedError("invalid num slot.");
  
  // slot 0
  tensorData->readSlot(fp, 0);

  // slot 1...N
  if (numSlot > 1) tensorData->readSlot(fp, 1);
  if (numSlot > 2) tensorData->readSlot(fp, 2);

  tensorData->_numSlot = numSlot;
  return tensorData;
}

CPUTensorData::~CPUTensorData() {
  if (_slots[0].data) {
    ly::free32ByteAlignedMem(_slots[0].data);
    _slots[0].data = nullptr;
  }

  if (_slots[1].data) {
    ly::free32ByteAlignedMem(_slots[1].data);
    _slots[1].data = nullptr;
  }
}

Device CPUTensorData::getDevice() const {
  return Device(Device::Type::kCpu);
}



}  // namespace internal
}  // namespace llyn
