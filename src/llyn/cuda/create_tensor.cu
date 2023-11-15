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

#include "llyn/cuda/create_tensor.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include "llyn/internal/cuda_tensor_data.h"
#include "llyn/cuda/cuda_common.h"

namespace llyn {
namespace cuda {

template<int DIM>
__global__ void fillZero(PackedTensorAccessor<half, DIM> tensor) {
  int64_t numVec = tensor.getNumVectors();
  int64_t vecIdx = blockIdx.x * blockDim.y + threadIdx.y;
  if (vecIdx >= numVec)
    return;
  
  TensorAccessor<half, 1> vec = tensor.getVectorByFlatIndex(vecIdx);
  
}

Tensor createCudaTensorHalf(ly::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = internal::CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat16);

  return Tensor::create(tensorShape, data);
}

Tensor zeros(ly::Span<const int> shape) {
  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  auto data = internal::CudaTensorData::create(tensorShape->getNumEl(), DType::kFloat16);

  return Tensor::create(tensorShape, data);
}

}  // cuda
}  // llyn
    
    