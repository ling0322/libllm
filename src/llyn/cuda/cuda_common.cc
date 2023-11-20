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

#include "llyn/cuda/cuda_common.h"

#include "llyn/tensor.h"

namespace llyn {
namespace cuda {

Q4ConstMatrix Q4ConstMatrix::fromTensor(const Tensor &tensor) {

}

Tensor createCudaTensorHalf(ly::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = internal::CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat16);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorLong(ly::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = internal::CudaTensorData::create(tensorShape->getNumEl(), DType::kLong);

  return Tensor::create(tensorShape, data);
}

Tensor createCudaTensorFloat(ly::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = internal::CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat);

  return Tensor::create(tensorShape, data);
}

void checkCudaError(cudaError_t err) {
  if (err != cudaSuccess)
    throw ly::AbortedError(cudaGetErrorString(err));
}

}  // cuda
}  // llyn
    
