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

CpuTensorData::CpuTensorData()
    : _data(nullptr) {
}

std::byte *CpuTensorData::getRawData() const {
  return reinterpret_cast<std::byte *>(_data);
}

void CpuTensorData::readSlot(lut::Reader *fp) {
  _dtype = fp->readValue<int16_t>();
  if (!_dtype.isValid()) THROW(Aborted, "invalid dtype.");

  _numel = fp->readValue<int64_t>();
  if (_numel > MaxNumEl) throw lut::AbortedError("tensor too big");

  int64_t size = _dtype.getTotalSize(_numel);

  CHECK(_data == nullptr);
  _data = lut::alloc32ByteAlignedMem(size);
  fp->readSpan(lut::makeSpan(reinterpret_cast<int8_t *>(_data), size));

  int magicNumber = fp->readValue<int16_t>();
  if (magicNumber != 0x55aa) throw lut::AbortedError("bad tensor data format (magic number).");
}

std::shared_ptr<TensorData> CpuTensorData::create(int64_t numel, DType dtype) {
  std::shared_ptr<CpuTensorData> tensorData = std::make_shared<CpuTensorData>();

  CHECK(numel > 0);
  int64_t size = dtype.getTotalSize(numel);
  tensorData->_data = lut::alloc32ByteAlignedMem(size);
  tensorData->_numel = numel;
  tensorData->_dtype = dtype;

  return tensorData;
}

std::shared_ptr<TensorData> CpuTensorData::read(lut::Reader *fp) {
  std::shared_ptr<CpuTensorData> tensorData = std::make_shared<CpuTensorData>();

  if (fp->readString(4) != "tdat") throw lut::AbortedError("bad tensor data format.");

  int32_t numSlot = fp->readValue<int32_t>();
  CHECK(numSlot == 0);

  // slot 0
  tensorData->readSlot(fp);
  return tensorData;
}

CpuTensorData::~CpuTensorData() {
  lut::free32ByteAlignedMem(_data);
  _data = nullptr;
}

Device CpuTensorData::getDevice() const {
  return Device(Device::Type::kCpu);
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
