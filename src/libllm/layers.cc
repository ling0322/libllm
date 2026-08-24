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

#include "libllm/layers.h"

#include <math.h>

#include "lutil/error.h"
#include "lutil/strings.h"
#include "flint/functional.h"

namespace libllm {

// -----------------------------------------------------------------------------------------------+
//  Embedding                                                                                     |
// -----------------------------------------------------------------------------------------------+

constexpr char Embedding::kWeight[];

std::unique_ptr<Embedding> Embedding::build(int dModel, int vocabSize, const VarBuilder &vb) {
  std::unique_ptr<Embedding> layer{new Embedding()};
  layer->_wte = vb.get(kWeight, {vocabSize, dModel});

  return layer;
}

fl::Tensor Embedding::forward(const fl::Tensor &input) const {
  CHECK(input.getDim() == 1);

  return fl::F::lookup(_wte, input);
}

// -----------------------------------------------------------------------------------------------+
//  Linear                                                                                        |
// -----------------------------------------------------------------------------------------------+

constexpr char Linear::kWeight[];
constexpr char Linear::kBias[];

Linear::Linear()
    : _hasBias(true) {
}

std::unique_ptr<Linear> Linear::build(int inDim, int outDim, bool hasBias, const VarBuilder &vb) {
  if (inDim <= 0 || outDim <= 0) throw lut::AbortedError("invalid d_model");

  std::unique_ptr<Linear> linear{new Linear()};
  linear->_hasBias = hasBias;
  linear->_w = vb.get(kWeight, {outDim, inDim});

  if (hasBias) {
    linear->_b = vb.get(kBias, {outDim});
  } else if (vb.has(kBias)) {
    throw lut::AbortedError(
        lut::sprintf(
            "In module %s: hasBias=false but bias weight found in model.",
            vb.name()));
  }

  return linear;
}

fl::Tensor Linear::forward(const fl::Tensor &input) const {
  fl::Tensor x;
  if (input.getDim() >= 2) {
    x = fl::F::matmul(input, _w.transpose(0, 1));
  } else {
    NOT_IMPL();
  }

  if (_hasBias) {
    x = fl::F::add(x, _b);
  }

  return x;
}

// -----------------------------------------------------------------------------------------------+
//  RmsNorm                                                                                       |
// -----------------------------------------------------------------------------------------------+

constexpr char RMSNorm::Weight[];

std::unique_ptr<RMSNorm> RMSNorm::build(int dModel, float eps, const VarBuilder &vb) {
  std::unique_ptr<RMSNorm> layer{new RMSNorm()};
  layer->_eps = eps;
  layer->_weight = vb.get(Weight, {dModel});

  return layer;
}

fl::Tensor RMSNorm::forward(const fl::Tensor &input) const {
  fl::Tensor x = fl::F::rmsNorm(input, _weight, _eps);

  return x;
}

}  // namespace libllm
