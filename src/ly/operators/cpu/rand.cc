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

#include "ly/operators/cpu/rand.h"

#include <math.h>
#include <string.h>
#include <algorithm>
#include "ly/tensor.h"
#include "ly/internal/tensor_shape.h"
#include "ly/operators/cpu/tensor.h"
#include "ly/operators/cpu/cpu_tensor_data.h"
#include "lyutil/random.h"
#include "lyutil/half.h"

namespace ly {
namespace op {
namespace cpu {

Tensor randFp32(lut::Span<const int> shape, lut::Random *generator, float min, float max) {
  Tensor x = op::cpu::tensor(shape, DType::kFloat);
  lut::Span<float> tensorData(x.getData<float>(), x.getNumEl());
  generator->fill(tensorData, min, max);

  return x;
}




Tensor randQ4(lut::Span<const int> shape, lut::Random *generator, float min, float max) {
  constexpr int groupSize = ly::QInt4Group32::GroupSize;
  CHECK(shape.back() % groupSize == 0);
  CHECK(fabs(min + max) <= 1e-5);

  int64_t numel = 1;
  for (lut::Span<const int>::size_type d = 0; d < shape.size(); ++d) {
    numel *= shape[d];
  }

  std::vector<float> dataFloat(numel);
  generator->fill(lut::makeSpan(dataFloat), min, max);

  std::vector<uint8_t> data(numel / 2);
  std::vector<uint16_t> scale(numel / groupSize);
  std::vector<uint8_t> zero(numel / groupSize / 2);

  int nb = numel / groupSize;
  for (int i = 0; i < nb; ++i) {
    int begin = i * groupSize;
    int end = (i + 1) * groupSize;

    float minVal = *std::min_element(dataFloat.begin() + begin, dataFloat.begin() + end);
    float maxVal = *std::max_element(dataFloat.begin() + begin, dataFloat.begin() + end);

    float scale = (maxVal - minVal) / 15.0;
    float zeroFp32 = roundf(-minVal / scale);
    int8_t zero = static_cast<int8_t>(zeroFp32);
    CHECK(zero >= 0 && zero <= 15);

    for (int j = 0; j < 16; j += 2) {
      uint8_t dh = dataFloat
      data[(begin + j) / 2] = 
    }

  }

  generator->fillUInt8(lut::makeSpan(data));
  generator->fillUInt8(lut::makeSpan(zero));
  generator->fill(lut::makeSpan(scaleFloat), max / 16.0f, max / 15.0f);
  std::transform(scaleFloat.begin(), scaleFloat.end(), scale.begin(), lut::cvtss_sh);

  auto tensorData = CpuTensorData::create({
      {numel, DType::kQInt4Group32},
      {numel / groupSize, DType::kFloat16},
      {numel / groupSize / 2, DType::kUInt8}});
  memcpy(tensorData->getSlot(0)->getRawData(), data.data(), data.size());
  memcpy(tensorData->getSlot(1)->getRawData(), scale.data(), scale.size() * sizeof(uint16_t));
  memcpy(tensorData->getSlot(2)->getRawData(), zero.data(), zero.size());

  auto tensorShape = std::make_shared<internal::TensorShape>(shape);
  return Tensor::create(tensorShape, tensorData);
}

Tensor rand(lut::Span<const int> shape, DType dtype, lut::Random *generator, float min, float max) {
  switch (int16_t(dtype)) {
    case DType::kFloat:
      return randFp32(shape, generator, min, max);
    case DType::kQInt4Group32:
      return randQ4(shape, generator, min, max);
    default:
      NOT_IMPL();
  } 
}

}  // cpu
}  // op
}  // ly
