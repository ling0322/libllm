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

#include "flint/cpu/unary.h"

#include <math.h>
#include <omp.h>

#include "lutil/attributes.h"
#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/tensor.h"
#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

namespace {

/// Every function is evaluated in float regardless of the tensor's type, so that a half tensor
/// gets the same value the float one would, rounded once at the end rather than at each step.
inline float applyUnaryOpFloat(float x, UnaryOp op) {
  switch (op) {
    case UnaryOp::NEG:
      return -x;
    case UnaryOp::ABS:
      return fabsf(x);
    case UnaryOp::EXP:
      return expf(x);
    case UnaryOp::SQUARE:
      return x * x;
    case UnaryOp::SQRT:
      return sqrtf(x);
    case UnaryOp::RSQRT:
      return 1.0f / sqrtf(x);
    case UnaryOp::SIGMOID:
      return 1.0f / (1.0f + expf(-x));
    case UnaryOp::TANH:
      return tanhf(x);
    case UnaryOp::RELU:
      return x > 0.0f ? x : 0.0f;
    case UnaryOp::GELU:
      // The exact form, so that this matches torch.nn.GELU() rather than its tanh approximation.
      return x * 0.5f * (1.0f + erff(x * 0.70710678118654752f));
    case UnaryOp::SILU:
      return x / (1.0f + expf(-x));
    default:
      NOT_IMPL();
  }
}

}  // namespace

template<typename T>
Tensor unaryOpKernel(const Tensor &A, UnaryOp op) {
  Tensor C = tensorLike(A);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vC.getLength());

#pragma omp parallel for num_threads(omp_get_max_threads())
  for (int j = 0; j < vA.getLength(); ++j) {
    TensorAccessor<const T, 1> a = vA.getTensor(j);
    TensorAccessor<T, 1> c = vC.getTensor(j);

    for (int i = 0; i < a.getShape(0); ++i) {
      c[i] = static_cast<T>(applyUnaryOpFloat(static_cast<float>(a[i]), op));
    }
  }

  return C;
}

Tensor unaryOp(const Tensor &A, UnaryOp op) {
  if (A.getDType() == DType::kFloat) return unaryOpKernel<float>(A, op);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16) return unaryOpKernel<Float16>(A, op);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
