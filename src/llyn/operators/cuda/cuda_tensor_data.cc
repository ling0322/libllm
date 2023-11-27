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

#include "llyn/operators/cuda/cuda_tensor_data.h"

#include <cuda_runtime.h>
#include "llyn/device.h"
#include "llyn/dtype.h"
#include "lyutil/error.h"
#include "lyutil/platform.h"
#include "lyutil/span.h"
#include "lyutil/strings.h"
#include "llyn/operators/cuda/common.h"

namespace llyn {
namespace op {
namespace cuda {

CudaTensorData::Slot::Slot()
    : data(nullptr),
      numel(0),
      dtype(DType::kUnknown) {}

int64_t CudaTensorData::Slot::getNumEl() const {
  return numel;
}
DType CudaTensorData::Slot::getDType() const {
  return dtype;
}
Byte *CudaTensorData::Slot::getRawData() const {
  return data;
}

CudaTensorData::CudaTensorData() : _numSlot(0) {}

std::shared_ptr<internal::TensorData> CudaTensorData::create(int64_t numel, DType dtype) {
  return create({{numel, dtype}});
}

std::shared_ptr<internal::TensorData> CudaTensorData::create(
    ly::Span<const std::pair<int64_t, DType>> slots) {
  CHECK(slots.size() > 0 && slots.size() <= TensorData::MaxSlot);

  auto tensorData = std::make_shared<CudaTensorData>();
  for (const std::pair<int64_t, DType> &slotSpec : slots) {
    int64_t numel = slotSpec.first;
    DType dtype = slotSpec.second;

    CHECK(numel > 0);
    int64_t size = dtype.getTotalSize(numel);
    void *data = nullptr;
    LL_CHECK_CUDA_STATUS(cudaMalloc(&data, size));

    tensorData->_slots[tensorData->_numSlot].data = reinterpret_cast<Byte *>(data);
    tensorData->_slots[tensorData->_numSlot].numel = numel;
    tensorData->_slots[tensorData->_numSlot].dtype = dtype;

    ++tensorData->_numSlot;
  }

  return tensorData;
}

CudaTensorData::~CudaTensorData() {
  for (int i = 0; i < _numSlot; ++i) {
    if (_slots[i].data) {
      llynCudaFree(_slots[i].data);
      _slots[i].data = nullptr;
    }
  }
}

const internal::SlotBase *CudaTensorData::getSlot(int slot) const {
  CHECK(slot < _numSlot);
  return &_slots[slot];
}

Device CudaTensorData::getDevice() const {
  return Device(Device::Type::kCuda);
}

int CudaTensorData::getNumSlot() const {
  return _numSlot;
}

}  // cuda
}  // op
}  // llyn
