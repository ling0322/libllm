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
namespace cuda {

/// Solve the batch of lower triangular systems L X = B, writing X over B. `l` is <float>(..., N, N)
/// and `x` holds B on entry; it may be <float> or <half>, since the substitution accumulates in
/// float either way and the element type only says how wide the right hand side is stored. Only the
/// lower triangle of `l`, its diagonal included, is read. Both must be contiguous.
///
/// This is the whole of what is left of a `triangularSolve` operator, which the gated DeltaNet
/// prefill was the only caller of. Its only caller now is the chunked path of that prefill: the two
/// tensor core paths invert each chunk's system rather than substituting through it, because
/// substitution runs down the rows of a chunk in order and cannot be a tensor core instruction.
void triangularSolveInplace(const Tensor &l, Tensor &x);

}  // namespace cuda
}  // namespace op
}  // namespace fl
