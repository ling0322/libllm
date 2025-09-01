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

#include "lynn/cuda/to_device.h"

#include <cuda_runtime.h>

#include <vector>

#include "lynn/cpu/cpu_tensor_data.h"
#include "lynn/cpu/tensor.h"
#include "lynn/cuda/common.h"
#include "lynn/cuda/cuda_tensor_data.h"
#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cuda {

template<Device::Type DEVICE>
std::shared_ptr<TensorData> createData(int64_t numel, DType dtype);
template<>
std::shared_ptr<TensorData> createData<Device::kCpu>(int64_t numel, DType dtype) {
  return op::cpu::CpuTensorData::create(numel, dtype);
}
template<>
std::shared_ptr<TensorData> createData<Device::kCuda>(int64_t numel, DType dtype) {
  return CudaTensorData::create(numel, dtype);
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

template<Device::Type DEVICE>
Tensor toDevice(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() != DEVICE);
  CHECK(tensor.isContiguous()) << "only contiguous tensor is allowed to copy between devices";
  std::shared_ptr<TensorData> srcData = tensor.getInternalData();

  // create data object.
  int64_t numel = srcData->getNumEl();
  DType dtype = srcData->getDType();

  std::shared_ptr<TensorData> destData = createData<DEVICE>(numel, dtype);

  // copy data.
  int64_t srcOffset = tensor.getInternalOffset();
  void *src = srcData->getRawData() + dtype.getTotalSize(srcOffset);
  void *dest = destData->getRawData();
  int64_t nbytes = dtype.getTotalSize(destData->getNumEl());
  copyData<DEVICE>(dest, src, nbytes);

  // create dest tesnor.
  auto shape = std::make_shared<TensorShape>(tensor.getShape());
  return Tensor::create(shape, destData);
}

Tensor toCpu(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCpu) return tensor;
  return toDevice<Device::kCpu>(tensor);
}

Tensor toCuda(const Tensor &tensor) {
  if (tensor.getDevice().getType() == Device::kCuda) return tensor;
  return toDevice<Device::kCuda>(tensor);
}

Tensor toDevice(Device device, const Tensor &tensor) {
  if (Device::kCpu == device.getType()) return toCpu(tensor);
  if (Device::kCuda == device.getType()) return toCuda(tensor);

  NOT_IMPL();
  return Tensor();
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
