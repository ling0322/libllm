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

#pragma once

#include <stdint.h>
#include "llyn/device.h"
#include "llyn/tensor.h"

namespace llyn {
namespace internal {

// base functional interface to apply operators for Tensor
class Operators {
 public:
  virtual ~Operators() = default;

  virtual Tensor lookup(Tensor table, Tensor indices) = 0;
  virtual Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) = 0;
  virtual Tensor rmsNorm(Tensor input, Tensor weight, float eps) = 0;
  virtual Tensor matmul(Tensor A, Tensor B) = 0;
  virtual Tensor mul(Tensor input, float other) = 0;
  virtual Tensor mul(Tensor input, Tensor other) = 0;
  virtual Tensor softmax(Tensor input) = 0;
  virtual Tensor add(Tensor input, Tensor other) = 0;
  virtual Tensor gelu(Tensor input) = 0;
  virtual Tensor createTensor(std::initializer_list<int> shape, DType dtype) = 0;
  virtual Tensor createTensorLike(Tensor input) = 0;
  virtual Tensor rand(std::initializer_list<int> shape, DType dtype) = 0;
  virtual Tensor zeros(ly::Span<const int> shape, DType dtype) = 0;
  virtual Tensor contiguous(Tensor input) = 0;
  virtual bool allClose(Tensor A, Tensor B) = 0;
  virtual void print(Tensor tensor) = 0;
  virtual Tensor causalMask(int max_len) = 0;
  virtual Tensor cat(Tensor A, Tensor B, int dim) = 0;
  virtual Tensor applRotaryPosEmb(Tensor A, Tensor roPE) = 0;
  virtual void copy(Tensor src, Tensor dest) = 0;
  virtual Tensor attention(Tensor q, Tensor k, Tensor v, Tensor mask) = 0;
  virtual Tensor swiglu(Tensor A) = 0;
  virtual Tensor toDevice(Tensor tensor, Device device) = 0;
};

extern Operators *gOperatorsForDevice[Device::NumDeviceType];


Operators *getOperators(Device::Type deviceType);
void initOperators();
void destroyOperators();


}  // namespace internal
}  // namespace llyn
