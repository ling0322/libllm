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

#include "llyn/cpu/tensor.h"

#include "llyn/cpu/internal.h"

namespace llyn {
namespace cpu {

Tensor tensor(ly::Span<const int> shape, DType dtype) {
  return Internal::tensor(shape, dtype);
}

Tensor tensorLike(const Tensor &input) {
  return Internal::tensor(input.getShape(), input.getDType());
}

Tensor zerosLike(const Tensor &input) {
  if (input.getDType() == DType::kFloat) return zerosLikeFp32(input);

  NOT_IMPL();
  return Tensor();
}

void zerosFp32(Subtensor<float> tensor) {
  // make sure tensor is contiguous.
  CHECK(tensor.numel() == tensor.stride(0) * tensor.dimension(0));

  float *data = tensor.data;
  int64_t numel = tensor.numel();

  for (int64_t i = 0; i < numel; ++i) {
    data[i] = 0.0f;
  }
}

Tensor zerosLikeFp32(const Tensor &input) {
  CHECK(input.getDType() == DType::kFloat);

  Tensor x = tensorLike(input);
  zerosFp32(Subtensor<float>::fromTensor(x));

  return x;
}

}  // cpu
}  // flint
