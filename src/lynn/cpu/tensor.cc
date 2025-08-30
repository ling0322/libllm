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

#include "lynn/cpu/tensor.h"

#include "lynn/cpu/cpu_tensor_data.h"
#include "lynn/cpu/print.h"

namespace ly {
namespace op {
namespace cpu {

Tensor tensor(lut::Span<const int> shape, DType dtype) {
  Tensor tensor;

  auto tensorShape = std::make_shared<TensorShape>(lut::makeConstSpan(shape));

  int64_t numel = tensorShape->getNumEl();
  auto tensorData = CpuTensorData::create(numel, dtype);

  return Tensor::create(tensorShape, tensorData);
}

Tensor tensorLike(const Tensor &input) {
  return tensor(input.getShape(), input.getDType());
}

template<typename T>
void fillZeroKernel(Tensor tensor) {
  // make sure tensor is contiguous.
  CHECK(tensor.isContiguous());

  T *data = tensor.getData<T>();
  int64_t numel = tensor.getNumEl();

  for (int64_t i = 0; i < numel; ++i) {
    data[i] = T(0);
  }
}

void fillZero(Tensor tensor) {
  if (tensor.getDType() == DType::kFloat) {
    fillZeroKernel<float>(tensor);
  }
#if LUT_CPU_ARCH == LUT_AARCH64
  else if (tensor.getDType() == DType::kFloat16) {
    fillZeroKernel<Float16>(tensor);
  }
#endif
  else {
    NOT_IMPL();
  }
}

Tensor zeros(lut::Span<const int> shape, DType dtype) {
  Tensor x = tensor(shape, dtype);
  fillZero(x);

  return x;
}

Tensor zerosLike(const Tensor &input) {
  Tensor x = tensorLike(input);
  fillZero(x);

  return x;
}

template<typename T>
Tensor causalMaskKernel(int length) {
  Tensor mask = tensor({length, length}, DType::getType<T>());

  T *data = mask.getData<T>();
  for (int i = 0; i < length; ++i) {
    T *row = data + i * length;
    for (int j = 0; j <= i; ++j) {
      row[j] = 0.0f;
    }
    for (int j = i + 1; j < length; ++j) {
      row[j] = -std::numeric_limits<float>::infinity();
    }
  }

  return mask;
}

Tensor causalMask(int length, DType dtype) {
  if (dtype == DType::kFloat) return causalMaskKernel<float>(length);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (dtype == DType::kFloat16) return causalMaskKernel<Float16>(length);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
