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

#include "lten/device.h"
#include "lten/tensor.h"
#include "lutil/random.h"
#include "lutil/thread_pool.h"

namespace lten {

// base functional interface to apply operators for Tensor
class Operators {
 public:
  virtual ~Operators() = default;

  virtual Tensor lookup(Tensor table, Tensor indices);
  virtual Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps);
  virtual Tensor rmsNorm(Tensor input, Tensor weight, float eps);
  virtual Tensor matmul(Tensor A, Tensor B);
  virtual Tensor mul(Tensor input, float other);
  virtual Tensor mul(Tensor input, Tensor other);
  virtual Tensor softmax(Tensor input);
  virtual Tensor add(Tensor input, Tensor other);
  virtual Tensor sum(Tensor input);
  virtual Tensor max(Tensor input);
  virtual Tensor melFbank(Tensor input);
  virtual Tensor gelu(Tensor input);
  virtual void fill(Tensor input, float value);
  virtual Tensor tensor(lut::Span<const int> shape, DType dtype);
  virtual Tensor tensorLike(Tensor input);
  virtual Tensor zeros(lut::Span<const int> shape, DType dtype);
  virtual bool allClose(Tensor A, Tensor B, float rtol, float atol);
  virtual void print(Tensor tensor);
  virtual Tensor causalMask(int max_len);
  virtual Tensor applyRotaryPosEmb(Tensor A, Tensor roPE);
  virtual void copy(Tensor src, Tensor dest);
  virtual Tensor swiglu(Tensor A);
  virtual Tensor to(Device device, Tensor tensor);
  virtual Tensor unfold(Tensor input, int kernelSize, int stride);
  virtual void repetitionPenalty(Tensor logits, Tensor history, float weight);
  virtual Tensor cast(Tensor tensor, DType dtype);
  virtual Tensor logMelSpectrogram(Tensor wave);
  virtual Tensor rand(
      lut::Span<const int> shape,
      DType dtype,
      lut::Random *generator,
      float min,
      float max);

  virtual DType getDefaultFloatType();
};

extern Operators *gOperatorsForDevice[Device::NumDeviceType];

Operators *getOperators(Device::Type deviceType);
bool isOperatorsAvailable(Device::Type deviceType);
void initOperators();
void destroyOperators();

}  // namespace lten
