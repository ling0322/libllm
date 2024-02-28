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

#include "libllm/cpu/attention.h"

#include <math.h>
#include "libllm/functional.h"
#include "libllm/cpu/accessor.h"
#include "libllm/cpu/mul.h"
#include "libllm/lut/time.h"

namespace libllm {
namespace op {
namespace cpu {

Tensor attention(const Tensor &q, const Tensor &k, const Tensor &v, const Tensor &mask) {
  Tensor scores = F::matmul(q, k.transpose(-2, -1));
  scores = mul(scores,  1.0f / sqrtf(1.0f * q.getShape(-1)));

  if (!mask.empty()) {
    scores = F::add(scores, mask);
  }

  scores = F::softmax(scores);
  Tensor outputs = F::matmul(scores, v);  

  return outputs;
}

}  // cpu
}  // op
}  // ly
