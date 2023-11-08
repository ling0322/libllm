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

#include "llyn/cpu/view.h"

#include <algorithm>
#include "llyn/cpu/internal.h"
#include "llyn/internal/tensor_shape.h"

namespace llyn {
namespace cpu {

using internal::TensorShape;

std::vector<Tensor::ShapeType> getRealShape(int64_t numEl, ly::Span<const int> viewShape) {
  std::vector<Tensor::ShapeType> shape{viewShape.begin(), viewShape.end()};
  auto inferDim = shape.end();
  auto it = shape.begin();
  int64_t viewNumEl = 1;
  for (; it != shape.end(); ++it) {
    if (*it < 0) {
      CHECK(inferDim == shape.end()) << "more than 1 inferred dim";
      inferDim = it;
    } else {
      viewNumEl *= *it;
    }
  }

  // handle -1 shape
  if (inferDim != shape.end()) {
    CHECK(numEl % viewNumEl == 0) << "inferred shape is not a integer";
    *inferDim = static_cast<Tensor::ShapeType>(numEl / viewNumEl);
  } else {
    CHECK(numEl == viewNumEl) << "invalid view (element number mismatch)";
  }

  return shape;
}

std::vector<Shape> mergeContigShape(const Tensor &src) {
  std::vector<Shape> mergedShape;
  for (int64_t d = src.getDim() - 1; d >= 0; --d) {
    CHECK(src.getStride(d) != 0) << "unable to change view of expanded tensor.";

    if (d == src.getDim() - 1) {
      Shape s;
      s.shape = src.getShape(d);
      s.stride = src.getStride(d);
      mergedShape.push_back(s);
    } else if (src.getStride(d + 1) * src.getShape(d + 1) == src.getStride(d)) {
      // this dimension is contiguous.
      mergedShape.back().shape *= src.getShape(d);
    } else {
      Shape s;
      s.shape = src.getShape(d);
      s.stride = src.getStride(d);
      mergedShape.push_back(s);
    }
  }

  std::reverse(mergedShape.begin(), mergedShape.end());
  return mergedShape;
}

std::vector<Shape> getViewShapeStride(const Tensor &src, ly::Span<const int> view) {
  std::vector<Shape> mergedShape = mergeContigShape(src);
  std::vector<Shape> viewShape;
  auto vi = view.rbegin();
  for (int64_t d = mergedShape.size() - 1; d >= 0; --d) {
    Shape ms = mergedShape[d];
    int numel = 1;
    while (vi != view.rend() && *vi * numel <= ms.shape) {
      Shape s;
      s.shape = *vi;
      s.stride = numel * ms.stride;
      viewShape.push_back(s);

      numel *= *vi;
      ++vi;
    }

    CHECK(numel == ms.shape) << "unable to get view of tensor.";
  }

  std::reverse(viewShape.begin(), viewShape.end());
  return viewShape;
}

Tensor view(const Tensor &src, ly::Span<const int> view) {
  std::vector<Tensor::ShapeType> shape = getRealShape(src.getNumEl(), view);
  if (src.isContiguous()) {
    return Internal::tensorView(src, std::make_shared<TensorShape>(shape));
  } else {
    std::vector<Shape> viewShape = getViewShapeStride(src, view);
    return Internal::tensorView(src, std::make_shared<TensorShape>(viewShape));
  }
}

}  // cpu
}  // flint
