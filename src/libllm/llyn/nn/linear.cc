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

#include "llyn/nn/linear.h"

#include "llyn/functional.h"
#include "lyutil/error.h"

namespace llyn {
namespace nn {

namespace F = functional;

constexpr char Linear::kWeight[];
constexpr char Linear::kBias[];

Linear::Linear() : _inFeatures(0), _outFeatures(0) {}

std::unique_ptr<Linear> Linear::create(const Context &ctx, int inFeatures, int outFeatures) {
  std::unique_ptr<Linear> linear{new Linear()};
  if (inFeatures <= 0 || outFeatures <= 0) {
    throw ly::AbortedError("invalid d_model");
  }

  linear->_inFeatures = inFeatures;
  linear->_outFeatures = outFeatures;
  linear->_ctx = ctx;
  return linear;
}

void Linear::initParameters(const StateMap &stateDict) {
  std::string nameW = _ctx.name(kWeight);
  std::string nameB = _ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _b = stateDict.getTensor(nameB);

  _w.throwIfInvalidShape({_outFeatures, _inFeatures});
  _b.throwIfInvalidShape({_outFeatures});
}

Tensor Linear::forward(const Tensor &input) const {
  Tensor x;
  if (input.getDim() >= 2) {
    x = F::matmul(input, _w.transpose(0, 1));
  } else {
    NOT_IMPL();
  }
  x = F::add(x, _b);

  return x;
}

}  // namespace nn
}  // namespace llyn
