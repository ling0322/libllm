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

#include "flint/cpu/triangular_solve.h"

#include "flint/cpu/accessor.h"
#include "flint/cpu/common.h"
#include "flint/cpu/copy.h"
#include "flint/cpu/tensor.h"

namespace fl {
namespace op {
namespace cpu {

void triangularSolveRowMajor(const float *l, int ldl, float *x, int ldx, int n, int m) {
  for (int i = 0; i < n; ++i) {
    float *xi = x + i * ldx;
    const float *li = l + i * ldl;
    for (int j = 0; j < i; ++j) {
      float lij = li[j];
      const float *xj = x + j * ldx;
      for (int c = 0; c < m; ++c) {
        xi[c] -= lij * xj[c];
      }
    }

    float inv = 1.0f / li[i];
    for (int c = 0; c < m; ++c) {
      xi[c] *= inv;
    }
  }
}

}  // namespace cpu
}  // namespace op
}  // namespace fl
