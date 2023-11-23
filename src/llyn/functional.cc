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

#include "llyn/functional.h"

#include "lyutil/error.h"
#include "lyutil/strings.h"
#include "llyn/internal/operators.h"

namespace llyn {
namespace functional {

using internal::gOperatorsForDevice;
using internal::getOperators;

Tensor lookup(Tensor table, Tensor indices) {
  switch (table.getDevice().getType()) {
    case Device::kCpu:
      return getOperators(Device::kCpu)->lookup(table, indices);
    case Device::kCuda:
      return getOperators(Device::kCuda)->lookup(table, indices);
    default:
      NOT_IMPL();
  }
}

Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  return gOperatorsForDevice[Device::kCpu]->layerNorm(input, weight, bias, eps);
}

Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
  return gOperatorsForDevice[Device::kCpu]->rmsNorm(input, weight, eps);
}

Tensor matmul(Tensor A, Tensor B) {
  CHECK(!A.empty());
  CHECK(!B.empty());
  return gOperatorsForDevice[Device::kCpu]->matmul(A, B);
}

Tensor mul(Tensor input, float other) {
  return gOperatorsForDevice[Device::kCpu]->mul(input, other);
}

Tensor mul(Tensor input, Tensor other) {
  return gOperatorsForDevice[Device::kCpu]->mul(input, other);
}

Tensor softmax(Tensor input) {
  return gOperatorsForDevice[Device::kCpu]->softmax(input);
}

Tensor add(Tensor input, Tensor other) {
  return gOperatorsForDevice[Device::kCpu]->add(input, other);
}

Tensor gelu(Tensor input) {
  return gOperatorsForDevice[Device::kCpu]->gelu(input);
}

Tensor createTensor(std::initializer_list<int> shape, DType dtype) {
  return gOperatorsForDevice[Device::kCpu]->createTensor(shape, dtype);
}

Tensor createTensorLike(Tensor input) {
  switch (input.getDevice().getType()) {
    case Device::kCpu:
      return getOperators(Device::kCpu)->createTensorLike(input);
    case Device::kCuda:
      return getOperators(Device::kCuda)->createTensorLike(input);
    default:
      NOT_IMPL();
  }
}

Tensor rand(std::initializer_list<int> shape, DType dtype) {
  return gOperatorsForDevice[Device::kCpu]->rand(shape, dtype);
}

Tensor zeros(ly::Span<const int> shape, DType dtype) {
  return gOperatorsForDevice[Device::kCpu]->zeros(shape, dtype);
}

Tensor contiguous(Tensor input) {
  Tensor x = createTensorLike(input);
  copy(input, x);
  
  return x;
}

bool allClose(Tensor A, Tensor B) {
  return gOperatorsForDevice[Device::kCpu]->allClose(A, B);
}

void print(Tensor tensor) {
  gOperatorsForDevice[Device::kCpu]->print(tensor);
}

Tensor causalMask(int maxLen) {
  return gOperatorsForDevice[Device::kCpu]->causalMask(maxLen);
}

Tensor cat(Tensor A, Tensor B, int dim) {
  return gOperatorsForDevice[Device::kCpu]->cat(A, B, dim);
}

Tensor applRotaryPosEmbd(Tensor A, Tensor roPE) {
  return gOperatorsForDevice[Device::kCpu]->applRotaryPosEmb(A, roPE);
}

void copy(Tensor src, Tensor dest) {
  CHECK(src.getDType() == dest.getDType());
  src.throwIfInvalidShape(dest.getShape());

  switch (src.getDevice().getType()) {
    case Device::kCpu:
      CHECK(dest.getDevice().getType() == Device::kCpu);
      getOperators(Device::kCpu)->copy(src, dest);
      break;
    case Device::kCuda:
      CHECK(dest.getDevice().getType() == Device::kCuda);
      getOperators(Device::kCuda)->copy(src, dest);
      break;
    default:
      NOT_IMPL();
  }
}

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
  return gOperatorsForDevice[Device::kCpu]->attention(q, k, v, mask);
}

Tensor swiglu(Tensor input) {
  return gOperatorsForDevice[Device::kCpu]->swiglu(input);
}

Tensor toDevice(Tensor tensor, Device device) {
  Device::Type src = tensor.getDevice().getType();
  Device::Type tgt = device.getType();

  Device::Type opDevice = Device::kCpu;
  if (src == Device::kCuda || tgt == Device::kCuda) opDevice = Device::kCuda;

  return getOperators(opDevice)->toDevice(tensor, device);
}

Tensor cast(Tensor tensor, DType dtype) {
  Device::Type opDevice = Device::kUnknown;
  if (tensor.getDevice().getType() == Device::kCuda) opDevice = Device::kCuda;
  if (tensor.getDevice().getType() == Device::kCpu) NOT_IMPL();

  return getOperators(opDevice)->cast(tensor, dtype);
}

}  // functional
}  // llyn
