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

#include "libllm/module.h"

#include <math.h>
#include "libllm/lut/error.h"
#include "libllm/functional.h"

namespace libllm {

// -----------------------------------------------------------------------------------------------+
//  Module                                                                                        |
// -----------------------------------------------------------------------------------------------+

Tensor Module::moveAndCastFloat(const Tensor &tensor, const Context &ctx) {
  Tensor x = tensor;
  x = F::to(ctx.getDevice(), x);

  if (x.getDType().isFloat()) x = F::cast(x, ctx.getFloatDType());
  return x;
}

void Module::initParameters(lut::Random *generator, DType quantType) {
  NOT_IMPL();
}

// -----------------------------------------------------------------------------------------------+
//  Embedding                                                                                     |
// -----------------------------------------------------------------------------------------------+

constexpr char Embedding::kWeight[];

std::unique_ptr<Embedding> Embedding::create(const Context &ctx, int dModel, int vocabSize) {
  std::unique_ptr<Embedding> layer{new Embedding()};
  layer->setCtx(ctx);

  layer->_dModel = dModel;
  layer->_vocabSize = vocabSize;

  return layer;
}

void Embedding::initParameters(const StateMap &stateDict) {
  std::string nameW = getCtx().name(kWeight);

  _wte = stateDict.getTensor(nameW);
  _wte.throwIfInvalidShape({_vocabSize, _dModel});
  _wte = moveAndCastFloat(_wte, getCtx());
}

void Embedding::initParameters(lut::Random *generator, DType weightType) {
  _wte = F::rand({_vocabSize, _dModel}, weightType, Device::getCpu(), generator);
  _wte = moveAndCastFloat(_wte, getCtx());
}

Tensor Embedding::forward(const Tensor &input) const {
  Tensor x = F::lookup(_wte, input);

  return x;
}

// -----------------------------------------------------------------------------------------------+
//  Linear                                                                                        |
// -----------------------------------------------------------------------------------------------+

constexpr char Linear::kWeight[];
constexpr char Linear::kBias[];

Linear::Linear() : _inFeatures(0), _outFeatures(0) {}

std::unique_ptr<Linear> Linear::create(const Context &ctx, int inFeatures, int outFeatures) {
  std::unique_ptr<Linear> linear{new Linear()};
  linear->setCtx(ctx);

  if (inFeatures <= 0 || outFeatures <= 0) {
    throw lut::AbortedError("invalid d_model");
  }

  linear->_inFeatures = inFeatures;
  linear->_outFeatures = outFeatures;
  return linear;
}

void Linear::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  std::string nameW = getCtx().name(kWeight);
  std::string nameB = ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _b = stateDict.getTensor(nameB);

  _w.throwIfInvalidShape({_outFeatures, _inFeatures});
  _b.throwIfInvalidShape({_outFeatures});

  _w = moveAndCastFloat(_w, ctx);
  _b = moveAndCastFloat(_b, ctx);

}

void Linear::initParameters(lut::Random *generator, DType weightType) {
  float xs = sqrtf(6.0f / (_outFeatures + _inFeatures));
  _w = F::rand({_outFeatures, _inFeatures}, weightType, Device::getCpu(), generator, -xs, xs);
  _w = moveAndCastFloat(_w, getCtx());

  _b = F::rand({_outFeatures}, DType::kFloat, Device::getCpu(), generator);
  _b = moveAndCastFloat(_b, getCtx());
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

// -----------------------------------------------------------------------------------------------+
//  RmsNorm                                                                                       |
// -----------------------------------------------------------------------------------------------+

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

// -----------------------------------------------------------------------------------------------+
//  LayerNorm                                                                                     |
// -----------------------------------------------------------------------------------------------+

constexpr char LayerNorm::kWeight[];
constexpr char LayerNorm::kBias[];

LayerNorm::LayerNorm() : _dModel(0), _eps(0.0f) {}

std::unique_ptr<LayerNorm> LayerNorm::create(const Context &ctx, int dModel, float eps) {
  std::unique_ptr<LayerNorm> layer{new LayerNorm()};
  layer->setCtx(ctx);

  if (dModel <= 0 || eps <= 0.0f) {
    throw lut::AbortedError("invalid dModel or eps");
  }

  layer->_dModel = dModel;
  layer->_eps = eps;
  return layer;
}

void LayerNorm::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  std::string nameW = ctx.name(kWeight);
  std::string nameB = ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _b = stateDict.getTensor(nameB);

  _w.throwIfInvalidShape({_dModel});
  _b.throwIfInvalidShape({_dModel});

  _w = moveAndCastFloat(_w, ctx);
  _b = moveAndCastFloat(_b, ctx);
}

Tensor LayerNorm::forward(const Tensor &input) const {
  return F::layerNorm(input, _w, _b, _eps);
}


}  // namespace libllm
