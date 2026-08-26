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
namespace gdnwmma {

/// The single-launch, tensor core implementation of the gated DeltaNet prefill. One CTA owns a
/// (sequence, value head) and every product it does that is bigger than a vector goes through an
/// HMMA: the two score matrices, both passes over the state, the chunk's solve, the output and the
/// state update. See the file comment in gated_delta_net_wmma.cu for the arrangement.
///
/// Only the (I + A) inverse and the decays are computed outside the tensor cores.
///
/// Returns false, and leaves `smemOut` alone, if this device cannot run it at this head dimension:
/// it wants a multiple of 16, at most 128, whose working set fits in one block's shared memory.
bool fits(int headDim, int *smemOut);

void run(
    const Tensor &q,
    const Tensor &k,
    const Tensor &v,
    const Tensor &g,
    const Tensor &beta,
    const Tensor &cuSeqlens,
    const Tensor &stateSlots,
    Tensor &state,
    Tensor &o,
    int numKHead,
    int numVHead,
    int headDim,
    int numSeq);

}  // namespace gdnwmma
}  // namespace cuda
}  // namespace op
}  // namespace fl
