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

#include "llyn/nn/layer_norm.h"

#include "llyn/functional.h"
#include "lyutil/error.h"

namespace llyn {
namespace nn {

namespace F = functional;

constexpr char LayerNorm::kWeight[];
constexpr char LayerNorm::kBias[];

LayerNorm::LayerNorm() : _dModel(0), _eps(0.0f) {}

std::unique_ptr<LayerNorm> LayerNorm::create(const Context &ctx, int dModel, float eps) {
  std::unique_ptr<LayerNorm> layer{new LayerNorm()};
  if (dModel <= 0 || eps <= 0.0f) {
    throw ly::AbortedError("invalid dModel or eps");
  }

  layer->_dModel = dModel;
  layer->_eps = eps;
  layer->_ctx = ctx;
  return layer;
}

void LayerNorm::initParameters(const StateMap &stateDict) {
  std::string nameW = _ctx.name(kWeight);
  std::string nameB = _ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _b = stateDict.getTensor(nameB);

  _w.throwIfInvalidShape({_dModel});
  _b.throwIfInvalidShape({_dModel});
}

Tensor LayerNorm::forward(const Tensor &input) const {
  return F::layerNorm(input, _w, _b, _eps);
}

}  // namespace nn
}  // namespace llyn
