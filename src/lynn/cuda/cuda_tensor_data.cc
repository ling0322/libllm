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

#include "lynn/cuda/cuda_tensor_data.h"

#include <cuda_runtime.h>

#include "lutil/error.h"
#include "lutil/platform.h"
#include "lutil/span.h"
#include "lutil/strings.h"
#include "lynn/cuda/common.h"
#include "lynn/device.h"
#include "lynn/dtype.h"

namespace ly {
namespace op {
namespace cuda {

std::shared_ptr<TensorData> CudaTensorData::create(int64_t numel, DType dtype) {
  auto tensorData = std::make_shared<CudaTensorData>();

  CHECK(numel > 0);
  int64_t size = dtype.getTotalSize(numel);
  void *data = nullptr;
  cudaError_t err = cudaMalloc(&data, size);
  if (err != cudaSuccess) {
    throw lut::AbortedError(cudaGetErrorString(err));
  }

  tensorData->_data = data;
  data = nullptr;

  tensorData->_numel = numel;
  tensorData->_dtype = dtype;

  return tensorData;
}

CudaTensorData::CudaTensorData()
    : _data(nullptr) {
}

CudaTensorData::~CudaTensorData() {
  if (_data) {
    llynCudaFree(_data);
    _data = nullptr;
  }
}

Device CudaTensorData::getDevice() const {
  return Device(Device::Type::kCuda);
}

std::byte *CudaTensorData::getRawData() const {
  return reinterpret_cast<std::byte *>(_data);
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
