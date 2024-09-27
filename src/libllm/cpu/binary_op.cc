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

#include "libllm/cpu/binary_op.h"

#include "libllm/cpu/accessor.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/tensor.h"
#include "libllm/mp.h"
#include "libllm/tensor.h"
#include "lutil/attributes.h"

namespace libllm {
namespace op {
namespace cpu {

Tensor broadcastTensor(const Tensor &a, lut::Span<const Tensor::ShapeType> targetShape) {
  Tensor x = expandBatchDims(a, targetShape);
  return x.expand(targetShape);
}

template<typename T>
Tensor binaryOpKernel(const Tensor &A, const Tensor &B, BinaryOp op) {
  Tensor xB = broadcastTensor(B, A.getShape());
  Tensor C = tensorLike(A);

  TensorList<const T, 1> vA = TensorList<const T, 1>::fromTensor(A);
  TensorList<const T, 1> vB = TensorList<const T, 1>::fromTensor(xB);
  TensorList<T, 1> vC = TensorList<T, 1>::fromTensor(C);
  CHECK(vA.getLength() == vB.getLength() && vC.getLength() == vB.getLength());

  MP::parallelFor({vA.getLength()}, [&vA, &vB, &vC, op](MP::Partition partition) {
    for (int j : partition.getRange()) {
      TensorAccessor<const T, 1> a = vA.getTensor(j);
      TensorAccessor<const T, 1> b = vB.getTensor(j);
      TensorAccessor<T, 1> c = vC.getTensor(j);

      for (int i = 0; i < a.getShape(0); ++i) {
        if (op == BinaryOp::ADD) {
          c[i] = a[i] + b[i];
        } else if (op == BinaryOp::MUL) {
          c[i] = a[i] * b[i];
        } else {
          NOT_IMPL();
        }
      }
    }
  });

  return C;
}

// apply C <- BinaryOp(A, B)
Tensor binaryOp(const Tensor &A, const Tensor &B, BinaryOp op) {
  if (A.getDType() == DType::kFloat) return binaryOpKernel<float>(A, B, op);
#if LUT_CPU_ARCH == LUT_AARCH64
  if (A.getDType() == DType::kFloat16) return binaryOpKernel<Float16>(A, B, op);
#endif

  NOT_IMPL();
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
