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

#include "libllm/cpu/cast.h"

#include <math.h>
#include <string.h>
#include <algorithm>
#include "libllm/lut/half.h"
#include "libllm/cpu/kernel/kernel.h"
#include "libllm/tensor.h"
#include "libllm/cpu/common.h"
#include "libllm/cpu/cpu_tensor_data.h"
#include "libllm/cpu/lookup.h"
#include "libllm/cpu/tensor.h"


namespace libllm {
namespace op {
namespace cpu {

Tensor cast(Tensor A, DType dtype) {
  if (A.getDType() == dtype) {
    return A;
  } else if (A.getDType() == DType::kFloat && dtype == DType::kQ4) {
    return castFp32ToQ4(A);
  } else if (A.getDType() == DType::kQ4 && dtype == DType::kFloat) {
    return castQ4ToFp32(A);
  } else if (A.getDType() == DType::kFloat16 && dtype == DType::kFloat) {
    return castFp16ToFp32(A);
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
      C.getData<float>());
  
  return C;
}

Tensor castQ4ToFp32(Tensor A) {
  Tensor x = op::cpu::tensor(A.getShape(), DType::kFloat);
  op::cpu::applyDequant<Q4>(0, A.getNumEl(), A.getDataObject(), x.getData<float>());

  return x;
}

Tensor castFp32ToQ4(Tensor A) {
  CHECK(A.getDim() == 2);
  CHECK(A.getShape(1) % Q4::GroupSize == 0);
  CHECK(A.isContiguous()) << "unable to cast a non-contiguous tensor to Q4";

  int64_t numel = A.getNumEl();
  const float *dataA = A.getData<float>();
  std::vector<uint8_t> data(numel / 2);
  std::vector<uint16_t> qscale(numel / Q4::GroupSize);
  std::vector<uint8_t> qzero(numel / Q4::GroupSize / 2);

  int nb = numel / Q4::GroupSize;
  for (int i = 0; i < nb; ++i) {
    int begin = i * Q4::GroupSize;
    int end = (i + 1) * Q4::GroupSize;

    float minVal = *std::min_element(dataA + begin, dataA + end);
    float maxVal = *std::max_element(dataA + begin, dataA + end);

    float scale = (maxVal - minVal) / 15.0;
    float zeroFp32 = roundf(-minVal / scale);
    CHECK(zeroFp32 >= 0.0f && zeroFp32 <= 15.0f);
    uint8_t zero = static_cast<uint8_t>(zeroFp32);

    for (int j = 0; j < Q4::GroupSize; j += 2) {
      float dlFp32 = roundf((dataA[begin + j] - minVal) / scale);
      float dhFp32 = roundf((dataA[begin + j + 1] - minVal) / scale);
      CHECK(dlFp32 >= 0.0f && dlFp32 <= 15.0f && dhFp32 >= 0.0f && dhFp32 <= 15.0f);

      uint8_t dl = static_cast<uint8_t>(dlFp32);
      uint8_t dh = static_cast<uint8_t>(dhFp32);
      data[(begin + j) / 2] = (dh << 4) | dl;
    }

    if (i % 2 == 0) {
      qzero[i / 2] = 0;
      qzero[i / 2] |= zero;
    } else {
      qzero[i / 2] |= (zero << 4);
    }

    qscale[i] = lut::cvtss_sh(scale);
  }

  auto tensorData = CpuTensorData::create({
      {numel, DType::kQ4},
      {numel / Q4::GroupSize, DType::kFloat16},
      {numel / Q4::GroupSize / 2, DType::kUInt8}});
  memcpy(tensorData->getSlot(0)->getRawData(), data.data(), data.size());
  memcpy(tensorData->getSlot(1)->getRawData(), qscale.data(), qscale.size() * sizeof(uint16_t));
  memcpy(tensorData->getSlot(2)->getRawData(), qzero.data(), qzero.size());

  auto tensorShape = std::make_shared<TensorShape>(A.getShape());
  return Tensor::create(tensorShape, tensorData);
}

}  // cpu
}  // op
}  // ly
