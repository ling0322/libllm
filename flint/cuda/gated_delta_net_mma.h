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
namespace gdnmma {

/// The gated DeltaNet prefill on `mma.sync`, with the state and every intermediate held in
/// registers. See the file comment in gated_delta_net_mma.cu.
///
/// Returns false, and leaves `smemOut` alone, if this device cannot run it at this head dimension.
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

}  // namespace gdnmma
}  // namespace cuda
}  // namespace op
}  // namespace fl
