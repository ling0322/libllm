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

#include "libllm/cpu/rms_norm.h"

#include <cmath>

#include "libllm/cpu/accessor.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/tensor.h"
#include "libllm/mp.h"
#include "libllm/tensor.h"

namespace libllm {
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

  MP::parallelFor({vA.getLength()}, [&vA, &vC, w, eps](MP::Partition partition) {
    for (int j : partition.getRange()) {
      TensorAccessor<const T, 1> a = vA.getTensor(j);
      TensorAccessor<T, 1> c = vC.getTensor(j);

      double sum = 0.0;
      for (int i = 0; i < a.getShape(0); ++i) {
        double va = a[i];
        sum += va * va;
      }
      double mean = sum / a.getShape(0);
      double rms = std::sqrt(mean + eps);

      // compute rms-norm
      for (int i = 0; i < a.getShape(0); ++i) {
        double va = a[i];
        double vw = w[i];
        c[i] = static_cast<T>(a[i] * w[i] / rms);
      }
    }
  });

  return C;
}

Tensor rmsNorm(const Tensor &tensor, const Tensor &weight, float eps) {
  if (tensor.getDType() == DType::kFloat) return rmsNormKernel<float>(tensor, weight, eps);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (tensor.getDType() == DType::kFloat16) return rmsNormKernel<Float16>(tensor, weight, eps);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
