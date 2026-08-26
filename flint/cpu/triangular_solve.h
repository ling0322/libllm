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

#pragma once

#include "flint/tensor.h"

namespace fl {
namespace op {
namespace cpu {

/// Forward substitution for a single row-major lower triangular system, in place on `x`, which
/// holds B on entry and X on return. `ldl` and `ldx` are the row strides of `l` and `x` in
/// elements. Only the lower triangle of `l`, its diagonal included, is read.
///
/// This is the whole of what is left of a `triangularSolve` operator, which the gated DeltaNet
/// prefill was the only caller of. The CUDA paths that count now invert each chunk's system on the
/// tensor cores instead; this is the CPU reference's solve, and its only caller is
/// `cpu::gatedDeltaNetPrefill`.
void triangularSolveRowMajor(const float *l, int ldl, float *x, int ldx, int n, int m);

}  // namespace cpu
}  // namespace op
}  // namespace fl
