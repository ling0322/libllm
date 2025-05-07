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

#include "libllm/cuda/to_device.h"

#include <cuda_runtime.h>
#include <vector>
#include "libllm/cpu/tensor.h"
#include "libllm/cuda/common.h"
#include "libllm/cpu/cpu_tensor_data.h"
#include "libllm/cuda/cuda_tensor_data.h"
#include "libllm/tensor.h"


namespace libllm {
namespace op {
namespace cuda {

template<Device::Type DEVICE>
std::shared_ptr<TensorData> createData(lut::Span<const std::pair<int64_t, DType>> slots);
template<>
std::shared_ptr<TensorData> createData<Device::kCpu>(
    lut::Span<const std::pair<int64_t, DType>> slots) {
  return op::cpu::CpuTensorData::create(slots);
}
template<>
std::shared_ptr<TensorData> createData<Device::kCuda>(
    lut::Span<const std::pair<int64_t, DType>> slots) {
  return CudaTensorData::create(slots);
}

template<Device::Type DEST_DEVICE>
void copyData(void *dest, const void *src, int64_t n);

template<>
inline void copyData<Device::kCpu>(void *dest, const void *src, int64_t n) {
  LL_CHECK_CUDA_STATUS(cudaMemcpy(dest, src, n, cudaMemcpyDeviceToHost));
}
template<>
inline void copyData<Device::kCuda>(void *dest, const void *src, int64_t n) {
  LL_CHECK_CUDA_STATUS(cudaMemcpy(dest, src, n, cudaMemcpyHostToDevice));
}

std::vector<std::pair<int64_t, DType>> getSlotSpec(const Tensor &srcTensor) {
  const TensorData *srcData = srcTensor.getDataObject();
  std::vector<std::pair<int64_t, DType>> slotSpec;

  if (srcData->getNumSlot() == 1) {
    // for single slot tensors, offset is allowed. the real numel is stored in shape object.
    const SlotBase *slot = srcData->getSlot(0);
    int64_t numel = srcTensor.getNumEl();
    CHECK(numel <= slot->getNumEl());
    slotSpec.emplace_back(numel, slot->getDType());
  } else {
    // for quantized tensors, offset is not allowed, the numel in slots could be used directly.
    CHECK(srcTensor.getOffset_() == 0);
    for (int i = 0; i < srcData->getNumSlot(); ++i) {
      const SlotBase *slot = srcData->getSlot(i);
      slotSpec.emplace_back(slot->getNumEl(), slot->getDType());
    }
  }

  return slotSpec;
}

template<Device::Type DEVICE>
Tensor toDevice(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() != DEVICE);
  CHECK(tensor.isContiguous()) << "only contiguous tensor is allowed to copy between devices";
  const TensorData *srcData = tensor.getDataObject();

  // create data object.
  std::vector<std::pair<int64_t, DType>> slotSpec = getSlotSpec(tensor);
  std::shared_ptr<TensorData> destData = createData<DEVICE>(slotSpec);

  // copy data.
  int64_t srcOffset = tensor.getOffset_();
  for (int i = 0; i < srcData->getNumSlot(); ++i) {
    const SlotBase *slot = srcData->getSlot(i);
    DType dtype = slot->getDType();

    void *src = srcData->getSlot(i)->getRawData() + dtype.getTotalSize(srcOffset);
    void *dest = destData->getSlot(i)->getRawData();
    int64_t nbytes = dtype.getTotalSize(destData->getSlot(i)->getNumEl());

    copyData<DEVICE>(dest, src, nbytes);
  }

  // create dest tesnor.
  auto shape = std::make_shared<TensorShape>(tensor.getShape());
  return Tensor::create(shape, destData);
}

Tensor toCpu(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCpu)
    return tensor;
  return toDevice<Device::kCpu>(tensor);
}

Tensor toCuda(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCuda)
    return tensor;
  return toDevice<Device::kCuda>(tensor);
}

Tensor toDevice(Device device, const Tensor &tensor) {
  if (Device::kCpu == device.getType()) return toCpu(tensor);
  if (Device::kCuda == device.getType()) return toCuda(tensor);
  
  NOT_IMPL();
  return Tensor();
}

}  // cuda
}  // op
}  // ly
