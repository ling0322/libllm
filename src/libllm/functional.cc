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

#include <math.h>

#include "libllm/operators.h"
#include "lutil/error.h"
#include "lutil/strings.h"

namespace libllm {
namespace F {

Tensor arange(LongType begin, LongType end, LongType step, Device device) {
  return getOperators(device.getType())->arangeLong(begin, end, step);
}

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
  CHECK(A.getDevice().getType() == B.getDevice().getType());
  CHECK(!A.empty());
  CHECK(!B.empty());

  return getOperators(A.getDevice().getType())->matmul(A, B);
}

Tensor mul(Tensor input, float other) {
  return getOperators(input.getDevice().getType())->mul(input, other);
}

Tensor div(Tensor input, float other) {
  return getOperators(input.getDevice().getType())->div(input, other);
}

Tensor mul(Tensor input, Tensor other) {
  return getOperators(input.getDevice().getType())->mul(input, other);
}

Tensor softmax(Tensor input) {
  return getOperators(input.getDevice().getType())->softmax(input);
}

Tensor square(Tensor input) {
  return getOperators(input.getDevice().getType())->square(input);
}

Tensor add(Tensor input, Tensor other) {
  return getOperators(input.getDevice().getType())->add(input, other);
}

Tensor sub(Tensor input, Tensor other) {
  return getOperators(input.getDevice().getType())->sub(input, other);
}

Tensor mod(Tensor input, LongType other) {
  return getOperators(input.getDevice().getType())->mod(input, other);
}

Tensor gelu(Tensor input) {
  return getOperators(input.getDevice().getType())->gelu(input);
}

Tensor tensor(lut::Span<const int> shape, DType dtype, Device device) {
  return getOperators(device.getType())->tensor(shape, dtype);
}

Tensor tensorLike(Tensor input) {
  return getOperators(input.getDevice().getType())->tensorLike(input);
}

Tensor rand(
    lut::Span<const int> shape,
    DType dtype,
    Device device,
    lut::Random *generator,
    float min,
    float max) {
  return getOperators(device.getType())->rand(shape, dtype, generator, min, max);
}

Tensor randn(lut::Span<const int> shape, Device device) {
  return getOperators(device.getType())->randNormal(shape);
}

Tensor zeros(lut::Span<const int> shape, DType dtype, Device device) {
  return getOperators(device.getType())->zeros(shape, dtype);
}

Tensor contiguous(Tensor input) {
  Tensor x = tensorLike(input);
  F::copy(input, x);

  return x;
}

bool allClose(Tensor A, Tensor B, float rtol, float atol) {
  return getOperators(A.getDevice().getType())->allClose(A, B, rtol, atol);
}

void print(Tensor tensor) {
  getOperators(tensor.getDevice().getType())->print(tensor);
}

Tensor causalMask(int maxLen, Device device) {
  return getOperators(device.getType())->causalMask(maxLen);
}

Tensor cat(Tensor A, Tensor B, int dim) {
  CHECK(A.getDType() == B.getDType());
  dim = A.getShape_()->getRealDim(dim);
  CHECK(A.getDim() == B.getDim() && dim < A.getDim());

  std::vector<int> shape = A.getShape();
  int dA = A.getShape(dim);
  int dB = B.getShape(dim);
  shape[dim] = dA + dB;

  Tensor C = tensor(shape, A.getDType(), A.getDevice());
  Tensor sA = C.slice(dim, {0, dA});
  Tensor sB = C.slice(dim, {dA, dA + dB});

  copy(A, sA);
  copy(B, sB);

  return C;
}

Tensor applyRotaryPosEmb(Tensor A, Tensor roPE) {
  return getOperators(A.getDevice().getType())->applyRotaryPosEmb(A, roPE);
}

void copy(Tensor src, Tensor dest) {
  CHECK(src.getDType() == dest.getDType());
  src.throwIfInvalidShape(dest.getShape(), "F::copy");

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

void repetitionPenalty(Tensor logits, Tensor history, float weight) {
  getOperators(logits.getDevice().getType())->repetitionPenalty(logits, history, weight);
}

Tensor sum(Tensor tensor, int dim) {
  return getOperators(tensor.getDevice().getType())->sum(tensor, dim);
}

Tensor max(Tensor tensor, int dim) {
  CHECK(dim == -1 || dim == tensor.getDim() - 1);
  return getOperators(tensor.getDevice().getType())->max(tensor);
}

void fill(Tensor tensor, float value) {
  getOperators(tensor.getDevice().getType())->fill(tensor, value);
}

Tensor attention(Tensor q, Tensor k, Tensor v, Tensor mask) {
  float dK = 1.0f / sqrtf(1.0f * q.getShape(-1));
  q = F::mul(q, sqrtf(dK));
  k = F::mul(k, sqrtf(dK));
  Tensor scores = F::matmul(q, k.transpose(-2, -1));

  if (!mask.empty()) {
    scores = F::add(scores, mask);
  }

  scores = F::softmax(scores);
  Tensor outputs = F::matmul(scores, v);

  return outputs;
}

Tensor swiglu(Tensor inputs) {
  return getOperators(inputs.getDevice().getType())->swiglu(inputs);
}

Tensor logMelSpectrogram(Tensor wave) {
  return getOperators(wave.getDevice().getType())->logMelSpectrogram(wave);
}

Tensor unfold(Tensor input, int kernelSize, int stride) {
  return getOperators(input.getDevice().getType())->unfold(input, kernelSize, stride);
}

Tensor to(Device device, Tensor tensor) {
  Device::Type src = tensor.getDevice().getType();
  Device::Type tgt = device.getType();

  Device srcDevice = tensor.getDevice();
  if (srcDevice.getType() == device.getType()) return tensor;

  if (srcDevice.getType() == Device::kCuda || device.getType() == Device::kCuda)
    return getOperators(Device::kCuda)->to(device, tensor);
  else
    NOT_IMPL();
}

Tensor cast(Tensor tensor, DType dtype) {
  return getOperators(tensor.getDevice().getType())->cast(tensor, dtype);
}

DType getDefaultFloatType(Device device) {
  return getOperators(device.getType())->getDefaultFloatType();
}

float elem(Tensor tensor) {
  return getOperators(tensor.getDevice().getType())->elem(tensor);
}

Tensor eq(Tensor tensor, Tensor other) {
  return getOperators(tensor.getDevice().getType())->eq(tensor, other);
}

bool all(Tensor tensor) {
  return getOperators(tensor.getDevice().getType())->all(tensor);
}

void manualSeed(Device device, uint64_t seed) {
  return getOperators(device.getType())->manualSeed(seed);
}

}  // namespace F
}  // namespace libllm
