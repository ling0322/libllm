// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
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

/// Scatter the keys and values of a packed batch into a paged KV cache. k and v are
/// <half>(numTokens, nKvHead, D), keyCache and valueCache are
/// <half>(nBlock, blockSize, nKvHead, D) and are written in place, and slotMapping is
/// <int>(numTokens) holding blockId * blockSize + offsetInBlock for each token. blockSize must be
/// a power of two.
void storeKVCache(
    const Tensor &k,
    const Tensor &v,
    Tensor &keyCache,
    Tensor &valueCache,
    const Tensor &slotMapping);

}  // namespace cuda
}  // namespace op
}  // namespace fl
