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

#include <memory>

#include "lutil/random.h"
#include "flint/operators.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

constexpr float Pi = 3.14159f;

// the CPU implementation of Operators
class CPUOperators : public Operators {
 public:
  CPUOperators();

  // create a instance of CPUOperators
  static std::unique_ptr<Operators> create();
  static std::unique_ptr<Operators> createFp32Only();

  // implement interface Operators
  Tensor add(Tensor a, Tensor b) override;
  Tensor sub(Tensor a, Tensor b) override;
  Tensor subFloat(Tensor input, float other) override;
  bool allClose(Tensor A, Tensor B, float rtol, float atol) override;
  Tensor cast(Tensor tensor, DType dtype) override;
  Tensor causalMask(int max_len) override;
  void copy(Tensor src, Tensor dest) override;
  void fill(Tensor input, float value) override;
  Tensor lookup(Tensor table, Tensor indices) override;
  Tensor matmul(Tensor a, Tensor b) override;
  Tensor gatedDeltaNetPrefill(
      Tensor q,
      Tensor k,
      Tensor v,
      Tensor g,
      Tensor beta,
      Tensor cuSeqlens,
      Tensor stateSlots,
      Tensor state) override;
  Tensor max(Tensor inputs) override;
  Tensor min(Tensor inputs) override;
  Tensor square(Tensor input) override;
  Tensor divTensor(Tensor input, Tensor other) override;
  Tensor neg(Tensor input) override;
  Tensor abs(Tensor input) override;
  Tensor exp(Tensor input) override;
  Tensor sqrt(Tensor input) override;
  Tensor rsqrt(Tensor input) override;
  Tensor sigmoid(Tensor input) override;
  Tensor tanh(Tensor input) override;
  Tensor relu(Tensor input) override;
  Tensor gelu(Tensor input) override;
  Tensor silu(Tensor input) override;
  Tensor arangeLong(LongType begin, LongType end, LongType step) override;
  Tensor randNormal(lut::Span<const int> shape) override;
  Tensor div(Tensor input, float other) override;
  float elem(Tensor tensor) override;
  bool elemBool(Tensor tensor) override;
  Tensor mod(Tensor input, LongType other) override;
  Tensor eq(Tensor input, Tensor other) override;
  bool all(Tensor A) override;
  Tensor mul(Tensor input, float other) override;
  Tensor mul(Tensor input, Tensor other) override;
  void print(Tensor tensor) override;
  Tensor rand(lut::Span<const int> shape, DType dtype) override;
  void repetitionPenalty(Tensor logits, Tensor history, float weight) override;
  Tensor rmsNorm(Tensor input, Tensor weight, float eps) override;
  Tensor sample(Tensor logits, Tensor temperatures, Tensor topKs, Tensor topPs) override;
  Tensor softmax(Tensor input) override;
  Tensor sum(Tensor inputs, int dim) override;
  Tensor swiglu(Tensor A) override;
  Tensor tensor(lut::Span<const int> shape, DType dtype) override;
  Tensor tensorLike(Tensor input) override;
  Tensor to(Device device, Tensor tensor) override;
  Tensor zeros(lut::Span<const int> shape, DType dtype) override;
  void manualSeed(uint64_t seed) override;

  DType getDefaultFloatType() override;

  MemorySnapshot captureMemorySnapshot() override;
  void resetPeakMemoryStats() override;

 private:
  typedef TensorShape::Elem Shape;
  lut::Random _rand;
};

}  // namespace cpu
}  // namespace op
}  // namespace fl
