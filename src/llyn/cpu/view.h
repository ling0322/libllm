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

#pragma once

#include "llyn/tensor.h"
#include "llyn/cpu/internal.h"
#include "llyn/cpu/subtensor.h"
#include "lyutil/span.h"

namespace llyn {
namespace cpu {

Tensor view(const Tensor &src, ly::Span<const int> view);

// infer the -1 dimension in view.
std::vector<ShapeType> getRealShape(int64_t numEl, ly::Span<const int> view);

// infer the stride for new view, according to the original stride.
std::vector<Shape> getViewShapeStride(const Tensor &src, ly::Span<const int> view);

// merge contiguous dimensions in the shape of `src`.
std::vector<Shape> mergeContigShape(const Tensor &src);

}  // cpu
}  // flint
