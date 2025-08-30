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

#include "lynn/cuda/common.h"

#include <cuda_fp16.h>

#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cuda {

int gMaxThreadsPerSM = 0;
int gNumSM = 0;

PackedSubtensor2DQInt4x32::PackedSubtensor2DQInt4x32(const Tensor &tensor) {
  CHECK(tensor.getDType() == DType::kQInt4x32);
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  CHECK(tensor.getStride(1) == 1);
  CHECK(tensor.getOffset_() == 0);
  CHECK(tensor.isContiguous());

  _numRow = tensor.getShape(0);
  _numCol = tensor.getShape(1);
  _data = tensor.getDataObject()->getData<QInt4x32>();
}

Tensor createCudaTensorHalf(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat16);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorFp4x2(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kFp4E2M0x2);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorUInt8(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kUInt8);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorBool(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kBool);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorLong(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kLong);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorFloat(lut::Span<const int> shape) {
  auto tensorShape = std::make_shared<TensorShape>(shape);
  auto data = CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat);

  return Tensor::create(tensorShape, data);
}

Tensor tensorLike(const Tensor &tensor) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);

  if (tensor.getDType() == DType::kFloat16) return createCudaTensorHalf(tensor.getShape());
  if (tensor.getDType() == DType::kUInt8) return createCudaTensorUInt8(tensor.getShape());
  if (tensor.getDType() == DType::kLong) return createCudaTensorLong(tensor.getShape());

  NOT_IMPL();
}

template<typename T>
float elemImpl(const Tensor &tensor) {
  T v;
  LL_CHECK_CUDA_STATUS(cudaMemcpy(&v, tensor.getData<T>(), sizeof(T), cudaMemcpyDeviceToHost));
  return v;
}
float elem(const Tensor &tensor) {
  CHECK(tensor.getDim() == 1 && tensor.getShape(0) == 1);

  if (tensor.getDType() == DType::kFloat16) return elemImpl<half>(tensor);
  if (tensor.getDType() == DType::kFloat) return elemImpl<float>(tensor);

  NOT_IMPL();
}

bool elemBool(const Tensor &tensor) {
  CHECK(tensor.getDim() == 1 && tensor.getShape(0) == 1);
  CHECK(tensor.getDType() == DType::kBool);

  bool v;
  LL_CHECK_CUDA_STATUS(
      cudaMemcpy(&v, tensor.getData<BoolType>(), sizeof(BoolType), cudaMemcpyDeviceToHost));

  return v;
}

int getCudaDeviceAttribute(cudaDeviceAttr attr) {
  int value;
  LL_CHECK_CUDA_STATUS(cudaDeviceGetAttribute(&value, attr, 0));
  return value;
}

int getCudaDeviceCount() {
  int value;

  cudaError_t status = cudaGetDeviceCount(&value);
  if (status != cudaSuccess) {
    return 0;
  }

  return value;
}

dim3 getGrid1D(int numel, int blockSize) {
  if (!gMaxThreadsPerSM) {
    gMaxThreadsPerSM = getCudaDeviceAttribute(cudaDevAttrMaxThreadsPerMultiProcessor);
  }
  if (!gNumSM) {
    gNumSM = getCudaDeviceAttribute(cudaDevAttrMultiProcessorCount);
  }

  int maxThreadsPerSM = gMaxThreadsPerSM;
  int numSM = gNumSM;
  int numBlock = (numel + blockSize - 1) / blockSize;
  int deviceNumBlock = maxThreadsPerSM * numSM / blockSize;
  CHECK(deviceNumBlock > 0);

  dim3 grid(std::min(numBlock, deviceNumBlock));

  return grid;
}

}  // namespace cuda
}  // namespace op
}  // namespace ly
