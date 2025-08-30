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

#include "lynn/cpu/cast.h"

#include <math.h>
#include <string.h>

#include <algorithm>

#include "lutil/half.h"
#include "lynn/cpu/common.h"
#include "lynn/cpu/cpu_tensor_data.h"
#include "lynn/cpu/kernel/interface.h"
#include "lynn/cpu/lookup.h"
#include "lynn/cpu/tensor.h"
#include "lynn/tensor.h"

namespace libllm {
namespace op {
namespace cpu {

Tensor cast(Tensor A, DType dtype) {
  if (A.getDType() == dtype) {
    return A;
  } else if (A.getDType() == DType::kFloat && dtype == DType::kQInt4x32) {
    return castFp32ToQ4(A);
  } else if (A.getDType() == DType::kQInt4x32 && dtype == DType::kFloat) {
    return castQ4ToFp32(A);
  } else if (A.getDType() == DType::kFloat16 && dtype == DType::kFloat) {
    return castFp16ToFp32(A);
  } else if (A.getDType() == DType::kFloat && dtype == DType::kFloat16) {
    return castFp32ToFp16(A);
  } else {
    NOT_IMPL();
  }
}

Tensor castFp16ToFp32(Tensor A) {
  CHECK(A.isContiguous()) << "unable to cast a non-contiguous half tensor to float";
  Tensor C = op::cpu::tensor(A.getShape(), DType::kFloat);
  kernel::convertHalfToFloat(
      A.getNumEl(),
      reinterpret_cast<const kernel::Float16 *>(A.getData<Float16>()),
      C.getData<float>(),
      kernel::Mode::OMP,
      kernel::CpuMathBackend::DEFAULT);

  return C;
}

Tensor castFp32ToFp16(Tensor A) {
  CHECK(A.isContiguous()) << "unable to cast a non-contiguous half tensor to float";
  Tensor C = op::cpu::tensor(A.getShape(), DType::kFloat16);
  kernel::convertFloatToHalf(
      A.getNumEl(),
      A.getData<float>(),
      reinterpret_cast<kernel::Float16 *>(C.getData<Float16>()),
      kernel::Mode::OMP,
      kernel::CpuMathBackend::DEFAULT);

  return C;
}

Tensor castQ4ToFp32(Tensor A) {
  Tensor x = op::cpu::tensor(A.getShape(), DType::kFloat);
  op::cpu::applyDequant(0, A.getNumEl(), A.getDataObject(), x.getData<float>());

  return x;
}

Tensor castFp32ToQ4(Tensor A) {
  int64_t numel = A.getNumEl();
  int64_t groupSize = DType(DType::kQInt4x32).getGroupSize();
  CHECK(numel % groupSize == 0);

  auto tensorData = CpuTensorData::create({{numel, DType::kQInt4x32}});
  kernel::quantFloatToQInt4(
      numel,
      A.getData<float>(),
      0,
      (kernel::QInt4x32 *)tensorData->getData<QInt4x32>(0),
      kernel::Mode::OMP);
  auto tensorShape = std::make_shared<TensorShape>(A.getShape());
  return Tensor::create(tensorShape, tensorData);
}

}  // namespace cpu
}  // namespace op
}  // namespace libllm
