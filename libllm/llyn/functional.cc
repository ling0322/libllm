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

#include "llyn/internal/operators.h"

namespace llyn {
namespace functional {

using internal::gOperatorsForDevice;

Tensor lookup(Tensor table, Tensor indices) {
  return gOperatorsForDevice[Device::kCpu]->lookup(table, indices);
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
  return gOperatorsForDevice[Device::kCpu]->createTensorLike(input);
}

Tensor rand(std::initializer_list<int> shape, DType dtype) {
  return gOperatorsForDevice[Device::kCpu]->rand(shape, dtype);
}

Tensor zeros(ly::Span<const int> shape, DType dtype) {
  return gOperatorsForDevice[Device::kCpu]->zeros(shape, dtype);
}

Tensor contiguous(Tensor input) {
  return gOperatorsForDevice[Device::kCpu]->contiguous(input);
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
  return gOperatorsForDevice[Device::kCpu]->copy(src, dest);
}

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
  return gOperatorsForDevice[Device::kCpu]->attention(q, k, v, mask);
}

Tensor swiglu(Tensor input) {
  return gOperatorsForDevice[Device::kCpu]->swiglu(input);
}

}  // functional
}  // flint
