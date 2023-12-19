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

#include "ly/nn/rms_norm.h"

#include "ly/functional.h"

namespace F = ly::functional;

namespace ly {
namespace nn {

constexpr char RMSNorm::Weight[];

std::unique_ptr<RMSNorm> RMSNorm::create(const Context &ctx, int dModel, float eps) {
  std::unique_ptr<RMSNorm> layer{new RMSNorm()};
  layer->setCtx(ctx);

  layer->_dModel = dModel;
  layer->_eps = eps;

  return layer;
}

void RMSNorm::initParameters(const StateMap &stateDict) {
  std::string nameW = getCtx().name(Weight);

  _weight = stateDict.getTensor(nameW);
  _weight.throwIfInvalidShape({_dModel});
  _weight = moveAndCastFloat(_weight, getCtx());
}

void RMSNorm::initParameters(lut::Random *generator, DType _) {
  _weight = F::rand({_dModel}, DType::kFloat, Device::getCpu(), generator);
  _weight = moveAndCastFloat(_weight, getCtx());
}

Tensor RMSNorm::forward(const Tensor &input) const {
  Tensor x = F::rmsNorm(input, _weight, _eps);

  return x;
}

}  // namespace nn
}  // namespace ly
