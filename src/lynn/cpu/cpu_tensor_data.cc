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

#include "lynn/cpu/cpu_tensor_data.h"

#include "lutil/error.h"
#include "lutil/platform.h"
#include "lutil/span.h"
#include "lynn/device.h"
#include "lynn/dtype.h"

namespace ly {
namespace op {
namespace cpu {

CpuTensorData::Slot::Slot()
    : data(nullptr),
      numel(0),
      dtype(DType::kUnknown) {
}

int64_t CpuTensorData::Slot::getNumEl() const {
  return numel;
}
DType CpuTensorData::Slot::getDType() const {
  return dtype;
}
Byte *CpuTensorData::Slot::getRawData() const {
  return data;
}

CpuTensorData::CpuTensorData()
    : _numSlot(0) {
}

void CpuTensorData::readSlot(lut::Reader *fp, int slotIdx) {
  CHECK(slotIdx < MaxSlot);
  Slot &slot = _slots[slotIdx];

  CHECK(slot.data == nullptr);

  slot.dtype = fp->readValue<int16_t>();
  if (!slot.dtype.isValid()) THROW(Aborted, "invalid dtype.");

  slot.numel = fp->readValue<int64_t>();
  if (slot.numel > MaxNumEl) throw lut::AbortedError("tensor too big");

  int64_t size = slot.dtype.getTotalSize(slot.numel);
  slot.data = reinterpret_cast<Byte *>(lut::alloc32ByteAlignedMem(size));
  fp->readSpan(lut::makeSpan(reinterpret_cast<int8_t *>(slot.data), size));
  int magicNumber = fp->readValue<int16_t>();
  if (magicNumber != 0x55aa) throw lut::AbortedError("bad tensor data format (magic number).");
}

std::shared_ptr<TensorData> CpuTensorData::create(int64_t numel, DType dtype) {
  return create({{numel, dtype}});
}

std::shared_ptr<TensorData> CpuTensorData::create(
    lut::Span<const std::pair<int64_t, DType>> slots) {
  CHECK(slots.size() > 0 && slots.size() <= TensorData::MaxSlot);

  auto tensorData = std::make_shared<CpuTensorData>();
  for (const std::pair<int64_t, DType> &slotSpec : slots) {
    int64_t numel = slotSpec.first;
    DType dtype = slotSpec.second;

    CHECK(numel > 0);
    int64_t size = dtype.getTotalSize(numel);
    void *data = lut::alloc32ByteAlignedMem(size);
    tensorData->_slots[tensorData->_numSlot].data = reinterpret_cast<Byte *>(data);
    tensorData->_slots[tensorData->_numSlot].numel = numel;
    tensorData->_slots[tensorData->_numSlot].dtype = dtype;

    ++tensorData->_numSlot;
  }

  return tensorData;
}

std::shared_ptr<TensorData> CpuTensorData::read(lut::Reader *fp) {
  std::shared_ptr<CpuTensorData> tensorData = std::make_shared<CpuTensorData>();

  if (fp->readString(4) != "tdat") throw lut::AbortedError("bad tensor data format.");

  int32_t numSlot = fp->readValue<int32_t>();
  if (numSlot <= 0 || numSlot > 3) throw lut::AbortedError("invalid num slot.");

  // slot 0
  tensorData->readSlot(fp, 0);

  // slot 1...N
  if (numSlot > 1) tensorData->readSlot(fp, 1);
  if (numSlot > 2) tensorData->readSlot(fp, 2);

  tensorData->_numSlot = numSlot;
  return tensorData;
}

const SlotBase *CpuTensorData::getSlot(int slot) const {
  CHECK(slot < _numSlot);
  return &_slots[slot];
}

CpuTensorData::~CpuTensorData() {
  for (int i = 0; i < _numSlot; ++i) {
    if (_slots[i].data) {
      lut::free32ByteAlignedMem(_slots[i].data);
      _slots[i].data = nullptr;
    }
  }
}

Device CpuTensorData::getDevice() const {
  return Device(Device::Type::kCpu);
}

int CpuTensorData::getNumSlot() const {
  return _numSlot;
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
