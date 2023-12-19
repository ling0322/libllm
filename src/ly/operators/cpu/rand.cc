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
#include "ly/operators/cpu/cast.h"
#include "ly/operators/cpu/tensor.h"
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
  Tensor x = randFp32(shape, generator, min, max);
  return op::cpu::castFp32ToQ4(x);
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
