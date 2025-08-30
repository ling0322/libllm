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

#include "lynn/module.h"

#include <math.h>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "lynn/functional.h"

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
  _wte.throwIfInvalidShape({_vocabSize, _dModel}, nameW);
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

Linear::Linear()
    : _inDim(0),
      _outDim(0),
      _hasBias(true) {
}

std::unique_ptr<Linear> Linear::create(const Context &ctx, int inDim, int outDim, bool hasBias) {
  if (inDim <= 0 || outDim <= 0) throw lut::AbortedError("invalid d_model");

  std::unique_ptr<Linear> linear{new Linear()};
  linear->setCtx(ctx);
  linear->_inDim = inDim;
  linear->_outDim = outDim;
  linear->_hasBias = hasBias;
  return linear;
}

void Linear::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  std::string nameW = getCtx().name(kWeight);
  std::string nameB = ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _w.throwIfInvalidShape({_outDim, _inDim}, nameW);
  _w = moveAndCastFloat(_w, ctx);

  if (_hasBias) {
    _b = stateDict.getTensor(nameB);
    _b.throwIfInvalidShape({_outDim}, nameB);
    _b = moveAndCastFloat(_b, ctx);
  } else {
    if (stateDict.hasTensor(nameB)) {
      throw lut::AbortedError(
          lut::sprintf(
              "In module %s: hasBias=false but bias weight found in state_map.",
              ctx.name()));
    }
  }
}

void Linear::initParameters(lut::Random *generator, DType weightType) {
  float xs = sqrtf(3.0f / _inDim);
  _w = F::rand({_outDim, _inDim}, weightType, Device::getCpu(), generator, -xs, xs);
  _w = moveAndCastFloat(_w, getCtx());

  if (_hasBias) {
    _b = F::rand({_outDim}, DType::kFloat, Device::getCpu(), generator, -0.2f, 0.2f);
    _b = moveAndCastFloat(_b, getCtx());
  }
}

Tensor Linear::forward(const Tensor &input) const {
  Tensor x;
  if (input.getDim() >= 2) {
    x = F::matmul(input, _w.transpose(0, 1));
  } else {
    NOT_IMPL();
  }

  if (_hasBias) {
    x = F::add(x, _b);
  }

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
  _weight.throwIfInvalidShape({_dModel}, nameW);
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

LayerNorm::LayerNorm()
    : _dModel(0),
      _eps(0.0f) {
}

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

void LayerNorm::initParameters(lut::Random *generator, DType _) {
  _w = F::rand({_dModel}, DType::kFloat, Device::getCpu(), generator);
  _w = moveAndCastFloat(_w, getCtx());

  _b = F::rand({_dModel}, DType::kFloat, Device::getCpu(), generator);
  _b = moveAndCastFloat(_b, getCtx());
}

void LayerNorm::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  std::string nameW = ctx.name(kWeight);
  std::string nameB = ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _b = stateDict.getTensor(nameB);

  _w.throwIfInvalidShape({_dModel}, nameW);
  _b.throwIfInvalidShape({_dModel}, nameB);

  _w = moveAndCastFloat(_w, ctx);
  _b = moveAndCastFloat(_b, ctx);
}

Tensor LayerNorm::forward(const Tensor &input) const {
  return F::layerNorm(input, _w, _b, _eps);
}

// -----------------------------------------------------------------------------------------------+
//  Conv1D                                                                                        |
// -----------------------------------------------------------------------------------------------+

constexpr char Conv1D::kWeight[];
constexpr char Conv1D::kBias[];

Conv1D::Conv1D()
    : _inChannels(0),
      _outChannels(0),
      _kernelSize(0),
      _hasBias(false) {
}

std::shared_ptr<Conv1D> Conv1D::create(
    const Context &ctx,
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    bool bias) {
  std::shared_ptr<Conv1D> layer{new Conv1D()};
  layer->setCtx(ctx);

  if (kernelSize == 0 || kernelSize >= 16) {
    throw lut::AbortedError("invalid kernelSize");
  }

  layer->_hasBias = bias;
  layer->_inChannels = inChannels;
  layer->_outChannels = outChannels;
  layer->_kernelSize = kernelSize;
  layer->_stride = stride;
  return layer;
}

void Conv1D::initParameters(const StateMap &stateDict) {
  const Context &ctx = getCtx();

  std::string nameW = getCtx().name(kWeight);
  std::string nameB = ctx.name(kBias);

  _w = stateDict.getTensor(nameW);
  _w.throwIfInvalidShape({_outChannels, _inChannels, _kernelSize}, nameW);
  _w = moveAndCastFloat(_w, ctx);
  _w = _w.view({_outChannels, -1});

  if (_hasBias) {
    _b = stateDict.getTensor(nameB);
    _b.throwIfInvalidShape({_outChannels}, nameB);
    _b = moveAndCastFloat(_b, ctx);
  } else {
    if (stateDict.hasTensor(nameB)) {
      throw lut::AbortedError(
          lut::sprintf(
              "In module %s: hasBias=false but bias weight found in state_map.",
              ctx.name()));
    }
  }
}

void Conv1D::initParameters(lut::Random *generator, DType weightType) {
  float xs = sqrtf(3.0f / (_inChannels * _kernelSize));
  _w = F::rand(
      {_outChannels, _inChannels * _kernelSize},
      weightType,
      Device::getCpu(),
      generator,
      -xs,
      xs);
  _w = moveAndCastFloat(_w, getCtx());

  if (_hasBias) {
    _b = F::rand({_outChannels}, DType::kFloat, Device::getCpu(), generator, -0.2f, 0.2f);
    _b = moveAndCastFloat(_b, getCtx());
  }
}

Tensor Conv1D::forward(const Tensor &input) const {
  Tensor x = F::unfold(input, _kernelSize, _stride);
  if (input.getDim() >= 2) {
    x = F::matmul(x, _w.transpose(0, 1));
  } else {
    NOT_IMPL();
  }

  if (_hasBias) {
    x = F::add(x, _b);
  }

  return x;
}

}  // namespace libllm
