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

#include "libllm/operators.h"

namespace libllm {
namespace op {
namespace cuda {

class MatMul;

/// @brief Implementation of Operator interface with cuda device.
class CudaOperators : public Operators {
 public:
  ~CudaOperators() = default;

  /// @brief Returns true if the CudaOperators is available (CUDA device available in host).
  /// @return if CudaOperators available.
  static bool isAvailable();

  // create a instance of CPUOperators
  static Operators *create();

  // implement interface Operators
  Tensor cast(Tensor tensor, DType dtype) override;
  Tensor add(Tensor a, Tensor b) override;
  Tensor applyRotaryPosEmb(Tensor A, Tensor roPE) override;
  Tensor causalMask(int max_len) override;
  void copy(Tensor src, Tensor dest) override;
  void fill(Tensor input, float value) override;
  Tensor gelu(Tensor input) override;
  Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) override;
  Tensor lookup(Tensor table, Tensor indices) override;
  Tensor matmul(Tensor a, Tensor b) override;
  Tensor max(Tensor inputs) override;
  Tensor mul(Tensor input, float other) override;
  Tensor mul(Tensor input, Tensor other) override;
  void print(Tensor tensor) override;
  void repetitionPenalty(Tensor logits, Tensor history, float weight) override;
  Tensor rmsNorm(Tensor input, Tensor weight, float eps) override;
  Tensor softmax(Tensor input) override;
  Tensor sum(Tensor inputs) override;
  Tensor swiglu(Tensor A) override;
  Tensor tensor(lut::Span<const int> shape, DType dtype) override;
  Tensor tensorLike(Tensor input) override;
  Tensor to(Device device, Tensor tensor) override;
  Tensor unfold(Tensor input, int kernelSize, int stride) override;
  Tensor zeros(lut::Span<const int> shape, DType dtype) override;

  DType getDefaultFloatType() override;

 private:
  std::shared_ptr<MatMul> _matmul;

  CudaOperators() = default;
};

}  // namespace cuda
}  // namespace op
}  // namespace libllm

libllm::Operators *llynCreateCudaOperators();
