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
#include "llyn/cpu/subtensor.h"
#include "llyn/cpu/subtensor_list.h"
#include "llyn/internal/operators.h"
#include "llyn/tensor.h"

namespace llyn {
namespace cpu {

constexpr float Pi = 3.14159;

// the CPU implementation of Operators
class CPUOperators : public internal::Operators {
 public:
  CPUOperators();

  // create a instance of CPUOperators
  static std::unique_ptr<Operators> create();

  // implement interface Operators
  Tensor lookup(Tensor table, Tensor indices) override;
  Tensor matmul(Tensor a, Tensor b) override;
  Tensor mul(Tensor input, float other) override;
  Tensor mul(Tensor input, Tensor other) override;
  Tensor softmax(Tensor input) override;
  Tensor gelu(Tensor input) override;
  Tensor add(Tensor a, Tensor b) override;
  Tensor createTensor(std::initializer_list<int> shape, DType dtype) override;
  Tensor createTensorLike(Tensor input) override;
  Tensor rand(std::initializer_list<int> shape, DType dtype) override;
  Tensor zeros(ly::Span<const int> shape, DType dtype) override;
  Tensor contiguous(Tensor input) override;
  bool allClose(Tensor A, Tensor B) override;
  void print(Tensor tensor) override;
  Tensor layerNorm(Tensor input, Tensor weight, Tensor bias, float eps) override;
  Tensor rmsNorm(Tensor input, Tensor weight, float eps) override;
  Tensor causalMask(int max_len) override;
  Tensor cat(Tensor A, Tensor B, int dim) override;
  Tensor applRotaryPosEmb(Tensor A, Tensor roPE) override;
  void copy(Tensor src, Tensor dest) override;
  Tensor attention(Tensor q, Tensor k, Tensor v, Tensor mask) override;
  Tensor swiglu(Tensor A) override;
  Tensor toDevice(Tensor tensor, Device device) override;
  Tensor cast(Tensor tensor, DType dtype) override;

 private:
  typedef internal::TensorShape::Elem Shape;

  Tensor createTensor(ly::Span<const int> shape, DType dtype);
  Tensor createTensorLike(Subtensor<const float> A);

  Tensor addFp32(Subtensor<const float> A, Subtensor<const float> B);
  Tensor softmaxFp32(Subtensor<const float> A);
  Tensor geluFp32(Subtensor<const float> A);

  void randFp32(Tensor *tensor);
  void print1DFp32(Subtensor<const float> A);
  void printNDFp32(Subtensor<const float> A, int pad_space);
  void printFp32(Subtensor<const float> tensor);
  
  bool allCloseFp32(Subtensor<const float> A, Subtensor<const float> B, float rtol, float atol);

  Tensor lookupFp32(Subtensor<const float> table, Subtensor<const LongType> indices);
  Tensor layerNormFp32(
      Subtensor<const float> input,
      Subtensor<const float> weight,
      Subtensor<const float> bias,
      float eps);
  Tensor rmsNormFp32(
      Subtensor<const float> input,
      Subtensor<const float> weight,
      float eps);
  Tensor causalMaskFp32(int max_len);


  template<typename T>
  void getSubtensors(Subtensor<T> tensor, int subtensorDim, std::vector<T*>& l);

};

}  // cpu
}  // namespace llyn

