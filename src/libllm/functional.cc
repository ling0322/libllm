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

#include "libllm/functional.h"

#include "libllm/lut/error.h"
#include "libllm/lut/strings.h"
#include "libllm/operators.h"

namespace libllm {
namespace F {

Tensor lookup(Tensor table, Tensor indices) {
  return getOperators(table.getDevice().getType())->lookup(table, indices);
}

Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) {
  return getOperators(input.getDevice().getType())->layerNorm(input, weight, bias, eps);
}

Tensor rmsNorm(Tensor input, Tensor weight, float eps) {
  return getOperators(input.getDevice().getType())->rmsNorm(input, weight, eps);

}

Tensor matmul(Tensor A, Tensor B) {
  CHECK(!A.empty());
  CHECK(!B.empty());

  return getOperators(A.getDevice().getType())->matmul(A, B);
}

Tensor mul(Tensor input, float other) {
  return getOperators(input.getDevice().getType())->mul(input, other);
}

Tensor mul(Tensor input, Tensor other) {
  return getOperators(input.getDevice().getType())->mul(input, other);
}

Tensor softmax(Tensor input) {
  return getOperators(input.getDevice().getType())->softmax(input);
}

Tensor add(Tensor input, Tensor other) {
  return getOperators(input.getDevice().getType())->add(input, other);
}

Tensor gelu(Tensor input) {
  return getOperators(input.getDevice().getType())->gelu(input);
}

Tensor createTensor(std::initializer_list<int> shape, DType dtype, Device device) {
  return getOperators(device.getType())->createTensor(shape, dtype);
}

Tensor createTensorLike(Tensor input) {
  return getOperators(input.getDevice().getType())->createTensorLike(input);
}

Tensor rand(lut::Span<const int> shape, DType dtype, Device device, lut::Random *generator,
            float min, float max) {
  if (generator) {
    return getOperators(device.getType())->rand(shape, dtype, generator, min, max);
  } else {
    lut::Random random;
    return getOperators(device.getType())->rand(shape, dtype, &random, min, max);
  }
}

Tensor zeros(lut::Span<const int> shape, DType dtype, Device device) {
  return getOperators(device.getType())->zeros(shape, dtype);
}

Tensor contiguous(Tensor input) {
  Tensor x = createTensorLike(input);
  F::copy(input, x);
  
  return x;
}

bool allClose(Tensor A, Tensor B, float atol, float rtol) {
  return getOperators(A.getDevice().getType())->allClose(A, B, atol, rtol);
}

void print(Tensor tensor) {
  getOperators(tensor.getDevice().getType())->print(tensor);
}

Tensor causalMask(int maxLen, Device device) {
  return getOperators(device.getType())->causalMask(maxLen);
}

Tensor cat(Tensor A, Tensor B, int dim) {
  return getOperators(A.getDevice().getType())->cat(A, B, dim);
}

Tensor applyRotaryPosEmb(Tensor A, Tensor roPE) {
  return getOperators(A.getDevice().getType())->applRotaryPosEmb(A, roPE);
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
  return getOperators(q.getDevice().getType())->attention(q, k, v, mask);
}

Tensor swiglu(Tensor input) {
  return getOperators(input.getDevice().getType())->swiglu(input);
}

Tensor to(Device device, Tensor tensor) {
  Device::Type src = tensor.getDevice().getType();
  Device::Type tgt = device.getType();

  Device srcDevice = tensor.getDevice();
  if (srcDevice.getType() == device.getType())
    return tensor;

  if (srcDevice.getType() == Device::kCuda || device.getType() == Device::kCuda)
    return getOperators(Device::kCuda)->toDevice(tensor, device);
  else
    NOT_IMPL();
}

Tensor cast(Tensor tensor, DType dtype) {
  return getOperators(tensor.getDevice().getType())->cast(tensor, dtype);
}

DType getDefaultFloatType(Device device) {
  return getOperators(device.getType())->getDefaultFloatType();
}

}  // F
}  // ly
