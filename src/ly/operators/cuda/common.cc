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

#include "ly/operators/cuda/common.h"

#include <cuda_fp16.h>
#include "ly/tensor.h"

namespace ly {
namespace op {
namespace cuda {

PackedSubtensor2DQ4::PackedSubtensor2DQ4(const Tensor &tensor) {
  CHECK(tensor.getDType() == DType::kQInt4Group32);
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  CHECK(tensor.getStride(1) == 1);
  CHECK(tensor.getOffset_() == 0);
  CHECK(tensor.isContiguous());

  _numRow = tensor.getShape(0);
  _numCol = tensor.getShape(1);

  _data = (const uint8_t *)tensor.getDataObject()->getSlot(0)->getRawData();
  _scale = (const __half *)tensor.getDataObject()->getSlot(1)->getRawData();
  _bias = (const int8_t *)tensor.getDataObject()->getSlot(2)->getRawData();
}

Tensor createCudaTensorHalf(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat16);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorLong(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kLong);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorFloat(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat);

  return Tensor::create(tensorShape, data);
}

int getCudaDeviceAttribute(cudaDeviceAttr attr) {
  int value;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&value, attr, 0));
  return value;
}

int getCudaDeviceCount() {
  int value;
  LL_CHECK_CUDA_STATUS(cudaGetDeviceCount(&value));
  return value;
}

}  // cuda
}  // op
}  // ly
