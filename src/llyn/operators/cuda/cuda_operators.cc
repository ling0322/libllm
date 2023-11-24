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

#include "llyn/operators/cuda/cuda_operators.h"

#include "llyn/operators/cuda/cast.h"
#include "llyn/operators/cuda/copy.h"
#include "llyn/operators/cuda/create_tensor.h"
#include "llyn/operators/cuda/lookup.h"
#include "llyn/operators/cuda/to_device.h"
#include "llyn/operators/cuda/cudnn_operators.h"

namespace llyn {
namespace op {
namespace cuda {

internal::Operators *CudaOperators::create() {
  std::unique_ptr<CudaOperators> ops{new CudaOperators()};
  ops->_cudnnOperators = CudnnOperators::create();

  return ops.release();
}

Tensor CudaOperators::lookup(Tensor table, Tensor indices) {
  return cuda::lookup(table, indices);
}

Tensor CudaOperators::matmul(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor CudaOperators::mul(Tensor input, float other) {
  NOT_IMPL();
}

Tensor CudaOperators::mul(Tensor input, Tensor other) {
  NOT_IMPL();
}

Tensor CudaOperators::softmax(Tensor input) {
  NOT_IMPL();
}

Tensor CudaOperators::gelu(Tensor input) {
  NOT_IMPL();
}

Tensor CudaOperators::add(Tensor a, Tensor b) {
  NOT_IMPL();
}

Tensor CudaOperators::createTensor(std::initializer_list<int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor CudaOperators::createTensorLike(Tensor input) {
  return tensorLike(input);
}

Tensor CudaOperators::rand(std::initializer_list<int> shape, DType dtype) {
  NOT_IMPL();
}

Tensor CudaOperators::zeros(ly::Span<const int> shape, DType dtype) {
  NOT_IMPL();
}

bool CudaOperators::allClose(Tensor A, Tensor B) {
  NOT_IMPL();
}

void CudaOperators::print(Tensor tensor) {
  NOT_IMPL();
}

Tensor CudaOperators::layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  NOT_IMPL();
}

Tensor CudaOperators::rmsNorm(Tensor input, Tensor weight, float eps) {
  NOT_IMPL();
}

Tensor CudaOperators::causalMask(int max_len) {
  NOT_IMPL();
}

Tensor CudaOperators::cat(Tensor A, Tensor B, int dim) {
  NOT_IMPL();
}

Tensor CudaOperators::applRotaryPosEmb(Tensor A, Tensor roPE) {
  NOT_IMPL();
}

void CudaOperators::copy(Tensor src, Tensor dest) {
  CHECK(src.getDevice().getType() == Device::kCuda);
  CHECK(dest.getDevice().getType() == Device::kCuda);
  CHECK(src.getDType() == dest.getDType());
  src.throwIfInvalidShape(dest.getShape());

  if (src.isContiguous() && dest.isContiguous()) {
    copyContig(src, dest);
  } else if (src.getDim() <= 4 && src.getDType() == DType::kFloat16) {
    _cudnnOperators->copy(src, dest);
  } else {
    cuda::copy(src, dest);
  }
}

Tensor CudaOperators::attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
  NOT_IMPL();
}

Tensor CudaOperators::swiglu(Tensor A) {
  NOT_IMPL();
}

Tensor CudaOperators::toDevice(Tensor tensor, Device device) {
  return cuda::toDevice(tensor, device);
}

Tensor CudaOperators::cast(Tensor tensor, DType dtype) {
  CHECK(tensor.getDevice().getType() == Device::kCuda);
  return cuda::cast(tensor, dtype);
}

}  // cuda
}  // op
}  // llyn

llyn::internal::Operators *llynCreateCudaOperators() {
  return llyn::op::cuda::CudaOperators::create();
}