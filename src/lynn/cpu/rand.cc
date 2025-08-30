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

#include "lynn/cpu/rand.h"

#include <math.h>
#include <string.h>

#include <algorithm>

#include "lutil/half.h"
#include "lutil/random.h"
#include "lutil/time.h"
#include "lynn/cpu/cast.h"
#include "lynn/cpu/tensor.h"
#include "lynn/mp.h"
#include "lynn/tensor.h"

namespace ly {
namespace op {
namespace cpu {

Tensor randFp32(lut::Span<const int> shape, lut::Random *generator, float min, float max) {
  Tensor x = op::cpu::tensor(shape, DType::kFloat);
  lut::Span<float> tensorData(x.getData<float>(), x.getNumEl());

  if (generator) {
    generator->fill(tensorData, min, max);
  } else {
    // if no generator specified, we could go parallel.
    std::vector<lut::Random> rs;
    lut::Random rseed;
    for (int i = 0; i < MP::getMaxThreads(); ++i) {
      rs.emplace_back(rseed.nextInt());
    }

    int blockSize = 1024;
    int nb = static_cast<int>((tensorData.size() + blockSize - 1) / blockSize);
    MP::parallelFor(nb, [&tensorData, &rs, min, max, blockSize](MP::Context ctx) {
      int64_t b = ctx.getBlockIdx();
      int64_t begin = b * blockSize;
      int64_t end = std::min(b * blockSize + blockSize, static_cast<int64_t>(tensorData.size()));
      for (int i = begin; i < end; ++i) {
        float nextR = rs[ctx.getAttachedThreadIdx()].nextFloat();
        tensorData[i] = min + (max - min) * nextR;
      }
    });
  }

  return x;
}

Tensor randFp16(lut::Span<const int> shape, lut::Random *generator, float min, float max) {
  Tensor x = randFp32(shape, generator, min, max);
  return castFp32ToFp16(x);
}

Tensor randQ4(lut::Span<const int> shape, lut::Random *generator, float min, float max) {
  Tensor x = randFp32(shape, generator, min, max);
  return op::cpu::castFp32ToQ4(x);
}

Tensor rand(lut::Span<const int> shape, DType dtype, lut::Random *generator, float min, float max) {
  switch (int16_t(dtype)) {
    case DType::kFloat:
      return randFp32(shape, generator, min, max);
    case DType::kFloat16:
      return randFp16(shape, generator, min, max);
    case DType::kQInt4x32:
      return randQ4(shape, generator, min, max);
    default:
      NOT_IMPL();
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace ly
