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

namespace ly {
namespace op {
namespace cpu {

Tensor cast(Tensor A, DType dtype) {
  if (A.getDType() == dtype) {
    return A;
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
      reinterpret_cast<const kernel::Float16 *>(A.getInternalData()->getData<Float16>()),
      C.getInternalData()->getData<float>(),
      kernel::Mode::OMP,
      kernel::CpuMathBackend::DEFAULT);

  return C;
}

Tensor castFp32ToFp16(Tensor A) {
  CHECK(A.isContiguous()) << "unable to cast a non-contiguous half tensor to float";
  Tensor C = op::cpu::tensor(A.getShape(), DType::kFloat16);
  kernel::convertFloatToHalf(
      A.getNumEl(),
      A.getInternalData()->getData<float>(),
      reinterpret_cast<kernel::Float16 *>(C.getInternalData()->getData<Float16>()),
      kernel::Mode::OMP,
      kernel::CpuMathBackend::DEFAULT);

  return C;
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
