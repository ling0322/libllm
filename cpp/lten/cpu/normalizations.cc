// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
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

#include "lten/cpu/normalizations.h"

#include <cmath>

#include "lten/cpu/accessor.h"
#include "lten/cpu/common.h"
#include "lten/cpu/tensor.h"
#include "lten/mp.h"
#include "lten/tensor.h"

namespace lten {
namespace op {
namespace cpu {

template<typename T>
Tensor rmsNormKernel(const Tensor &tensor, const Tensor &weight, float eps) {
  CHECK(weight.getDim() == 1);
  CHECK(tensor.getShape(-1) == weight.getShape(0));

  Tensor C = tensorLike(tensor);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(tensor);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  TensorAccessor<const T, 1> w = weight;

  MP::parallelFor(vA.getLength(), [&vA, &vC, w, eps](MP::Context ctx) {
    TensorAccessor<const T, 1> a = vA.getTensor(ctx.getBlockIdx());
    TensorAccessor<T, 1> c = vC.getTensor(ctx.getBlockIdx());

    float sum = 0.0;
    for (int i = 0; i < a.getShape(0); ++i) {
      float va = a[i];
      sum += va * va;
    }
    float mean = sum / a.getShape(0);
    float rms = std::sqrt(mean + eps);

    // compute rms-norm
    for (int i = 0; i < a.getShape(0); ++i) {
      float va = a[i];
      float vw = w[i];
      c[i] = static_cast<T>(a[i] * w[i] / rms);
    }
  });

  return C;
}

template<typename T>
Tensor layerNormKernel(const Tensor &tensor, const Tensor &weight, const Tensor &bias, float eps) {
  CHECK(weight.getDim() == 1);
  CHECK(tensor.getShape(-1) == weight.getShape(0));

  Tensor C = tensorLike(tensor);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(tensor);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

  TensorAccessor<const T, 1> w = weight;
  TensorAccessor<const T, 1> b = bias;

  MP::parallelFor(vA.getLength(), [&vA, &vC, w, b, eps](MP::Context ctx) {
    TensorAccessor<const T, 1> a = vA.getTensor(ctx.getBlockIdx());
    TensorAccessor<T, 1> c = vC.getTensor(ctx.getBlockIdx());

    double sum = 0.0f;
    for (int i = 0; i < a.getShape(0); ++i) {
      sum += a[i];
    }
    double mean = sum / a.getShape(0);

    // var (unbiased)
    sum = 0.0;
    for (int i = 0; i < a.getShape(0); ++i) {
      double d = a[i] - mean;
      sum += d * d;
    }
    double var = sum / a.getShape(0);
    double sd = sqrt(var + eps);

    // compute layer-norm
    for (int i = 0; i < a.getShape(0); ++i) {
      float elem = static_cast<float>((a[i] - mean) / sd);
      c[i] = elem * w[i] + b[i];
    }
  });

  return C;
}

Tensor rmsNorm(Tensor tensor, Tensor weight, float eps) {
  if (tensor.getDType() == DType::kFloat) return rmsNormKernel<float>(tensor, weight, eps);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (tensor.getDType() == DType::kFloat16) return rmsNormKernel<Float16>(tensor, weight, eps);
#endif

  NOT_IMPL();
}

Tensor layerNorm(Tensor tensor, Tensor weight, Tensor bias, float eps) {
  if (tensor.getDType() == DType::kFloat) return layerNormKernel<float>(tensor, weight, bias, eps);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (tensor.getDType() == DType::kFloat16)
    return layerNormKernel<Float16>(tensor, weight, bias, eps);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace lten
